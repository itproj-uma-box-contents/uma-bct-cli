using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;

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

    private static void WriteEvent(string name)
    {
        string path = Environment.GetEnvironmentVariable("UMA_BCT_FAKE_OLLAMA_EVENTS");
        if (!String.IsNullOrEmpty(path))
        {
            File.AppendAllText(
                path,
                name + " " + Environment.GetEnvironmentVariable("OLLAMA_HOST") + " " +
                    System.Diagnostics.Process.GetCurrentProcess().Id + Environment.NewLine
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
        Uri endpoint = Endpoint();
        try
        {
            using (TcpClient client = new TcpClient())
            {
                client.ReceiveTimeout = 1000;
                client.SendTimeout = 1000;
                client.Connect(endpoint.Host, endpoint.Port);
                using (NetworkStream stream = client.GetStream())
                {
                    byte[] request = Encoding.ASCII.GetBytes("GET /api/tags HTTP/1.0\r\n\r\n");
                    stream.Write(request, 0, request.Length);
                    byte[] response = new byte[128];
                    int length = stream.Read(response, 0, response.Length);
                    string text = Encoding.ASCII.GetString(response, 0, length);
                    return text.Contains("200 OK") ? 0 : 1;
                }
            }
        }
        catch
        {
            return 1;
        }
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
        return args[0] == "serve" ? Serve() : 1;
    }
}
