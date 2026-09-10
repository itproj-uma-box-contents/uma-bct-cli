using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public static class FakeLauncherTool
{
    private static string Role()
    {
        return Path.GetFileName(Environment.GetCommandLineArgs()[0]).ToLowerInvariant();
    }

    private static Uri Endpoint()
    {
        return new Uri(Environment.GetEnvironmentVariable("OLLAMA_HOST"));
    }

    private static void WriteEvent(string name, int processId = 0)
    {
        string path = Environment.GetEnvironmentVariable("UMA_BCT_FAKE_OLLAMA_EVENTS");
        if (!String.IsNullOrEmpty(path))
        {
            File.AppendAllText(
                path,
                name + " " + Environment.GetEnvironmentVariable("OLLAMA_HOST") + " " +
                    (processId == 0 ? Process.GetCurrentProcess().Id : processId) + Environment.NewLine
            );
        }
    }

    private static int AllocatePort()
    {
        TcpListener listener = new TcpListener(IPAddress.Loopback, 0);
        try
        {
            listener.Start();
            Console.WriteLine(((IPEndPoint)listener.LocalEndpoint).Port);
            return 0;
        }
        finally
        {
            listener.Stop();
        }
    }

    private static int RunApplication()
    {
        File.WriteAllText(
            Environment.GetEnvironmentVariable("UMA_BCT_FAKE_APP_HOST"),
            Environment.GetEnvironmentVariable("OLLAMA_HOST") + Environment.NewLine
        );
        int exitCode;
        return Int32.TryParse(Environment.GetEnvironmentVariable("UMA_BCT_FAKE_APP_EXIT"), out exitCode)
            ? exitCode
            : 0;
    }

    private static int ListModels()
    {
        WriteEvent("unexpected-list");
        return 1;
    }

    private static bool FailThisStart()
    {
        string counterPath = Environment.GetEnvironmentVariable("UMA_BCT_FAKE_OLLAMA_COUNTER");
        if (String.IsNullOrEmpty(counterPath))
        {
            return false;
        }
        int count = File.Exists(counterPath) ? Int32.Parse(File.ReadAllText(counterPath)) : 0;
        count += 1;
        File.WriteAllText(counterPath, count.ToString());
        int failures;
        Int32.TryParse(Environment.GetEnvironmentVariable("UMA_BCT_FAKE_OLLAMA_FAIL_STARTS"), out failures);
        return count <= failures;
    }

    private static int Serve()
    {
        if (FailThisStart())
        {
            WriteEvent("bind-failed");
            return 98;
        }

        Uri endpoint = Endpoint();
        TcpListener listener = new TcpListener(IPAddress.Parse(endpoint.Host), endpoint.Port);
        listener.Start();
        ProcessStartInfo runnerStart = new ProcessStartInfo(Environment.GetCommandLineArgs()[0], "runner");
        runnerStart.UseShellExecute = false;
        Process runner = Process.Start(runnerStart);
        WriteEvent("runner-started", runner.Id);
        runner.Dispose();
        WriteEvent("started");
        while (true)
        {
            using (TcpClient client = listener.AcceptTcpClient())
            using (NetworkStream stream = client.GetStream())
            {
                byte[] request = new byte[4096];
                stream.Read(request, 0, request.Length);
                byte[] response = Encoding.ASCII.GetBytes(
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n{\"models\":[]}"
                );
                stream.Write(response, 0, response.Length);
            }
        }
    }

    public static int Main(string[] args)
    {
        string role = Role();
        if (role == "uv.exe")
        {
            if (args.Length > 0 && args[0] == "--version")
            {
                Console.WriteLine("uv 0.12.9");
            }
            return 0;
        }
        if (role == "python.exe")
        {
            return AllocatePort();
        }
        if (role == "uma-bct.exe")
        {
            return RunApplication();
        }
        if (role != "ollama.exe" || args.Length == 0)
        {
            return 1;
        }
        if (args[0] == "--version")
        {
            Console.Error.WriteLine("Warning: client version is 0.30.8");
            return 0;
        }
        if (args[0] == "list")
        {
            return ListModels();
        }
        if (args[0] == "runner")
        {
            while (true)
            {
                Thread.Sleep(1000);
            }
        }
        return args[0] == "serve" ? Serve() : 1;
    }
}
