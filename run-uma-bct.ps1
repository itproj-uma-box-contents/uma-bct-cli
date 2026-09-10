$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Inputs = Join-Path $Root "inputs"
$Outputs = Join-Path $Root "outputs"
$Runtime = Join-Path $Root ".uma-bct-runtime"
$Constraints = Join-Path $Root "client-constraints.txt"
$AppEnvironment = Join-Path $Runtime "app"
$AppPython = Join-Path $AppEnvironment "Scripts\python.exe"
$AppCli = Join-Path $AppEnvironment "Scripts\uma-bct.exe"
$UvVersion = "0.12.9"
$UvArchiveUrl = "https://releases.astral.sh/github/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$UvArchiveSha256 = "ddbfcee1ac615a0499f6aa97b5ec8ebdf3ee4a7714a48055ec2ba0030e3cf810"
$OllamaVersion = "0.30.8"
$MinimumOllamaVersion = [version] $OllamaVersion
$OllamaArchiveUrl = "https://github.com/ollama/ollama/releases/download/v$OllamaVersion/ollama-windows-amd64.zip"
$OllamaArchiveSha256 = "c2d26d97e698027329c252629d7113bbc05d874b49960cbb03e93a39ae9fd95c"
$OllamaDirectory = Join-Path $Runtime "ollama"
$OllamaLog = Join-Path $Runtime "ollama.log"
$OllamaOutputLog = Join-Path $Runtime "ollama.stdout.log"
$OllamaProcess = $null
$SupportedExtensions = @(".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")

function Get-InputImages {
    @(Get-ChildItem -LiteralPath $Inputs -File -ErrorAction SilentlyContinue |
        Where-Object { $SupportedExtensions -contains $_.Extension.ToLowerInvariant() })
}

function Test-PrivateData {
    foreach ($Directory in @($Inputs, $Outputs)) {
        if ($null -ne (Get-ChildItem -LiteralPath $Directory -Force |
            Select-Object -First 1)) {
            return $true
        }
    }
    return $false
}

function Assert-NoPrivateData {
    if (Test-PrivateData) {
        throw "Private files appeared before online preparation finished."
    }
}

function Open-LocalFolder([string] $Path) {
    if ($env:UMA_BCT_NO_OPEN -ne "1") {
        Start-Process -FilePath "explorer.exe" -ArgumentList "`"$Path`""
    }
}

function Find-Ollama {
    $Candidates = @((Join-Path $OllamaDirectory "ollama.exe"))
    foreach ($Candidate in $Candidates) {
        if ((Test-Path -LiteralPath $Candidate -PathType Leaf) -and
            ((Get-Item -LiteralPath $Candidate -Force).Attributes -band
                [IO.FileAttributes]::ReparsePoint) -eq 0) {
            return $Candidate
        }
    }
    return $null
}

function Find-LlamaServer([string] $Executable) {
    if ([string]::IsNullOrWhiteSpace($Executable)) {
        return $null
    }
    $ExecutableDirectory = Split-Path -Parent $Executable
    $Candidates = @(
        (Join-Path $ExecutableDirectory "lib\ollama\llama-server.exe"),
        (Join-Path $ExecutableDirectory "..\lib\ollama\llama-server.exe")
    )
    foreach ($Candidate in $Candidates) {
        if ((Test-Path -LiteralPath $Candidate -PathType Leaf) -and
            ((Get-Item -LiteralPath $Candidate -Force).Attributes -band
                [IO.FileAttributes]::ReparsePoint) -eq 0) {
            return $Candidate
        }
    }
    return $null
}

function Test-UvVersion([string] $Executable) {
    try {
        $VersionOutput = & $Executable --version 2>$null
        return $LASTEXITCODE -eq 0 -and $VersionOutput -match "^uv $UvVersion(?:\s|$)"
    } catch {
        return $false
    }
}

function Assert-InstallerHash([string] $Path, [string] $ExpectedHash) {
    $ActualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $ExpectedHash) {
        throw "Download checksum verification failed."
    }
}

function Assert-OllamaSignature([string] $Path) {
    $Signature = Get-AuthenticodeSignature -FilePath $Path
    $Subject = if ($null -eq $Signature.SignerCertificate) {
        ""
    } else {
        $Signature.SignerCertificate.Subject
    }
    if ($Signature.Status -ne "Valid" -or
        $Subject -notmatch "(^|, )O=Ollama Inc\.(,|$)") {
        throw "Ollama runtime signature verification failed."
    }
}

function Test-OllamaVersion([string] $Executable) {
    $Process = $null
    try {
        $StartInfo = [Diagnostics.ProcessStartInfo]::new()
        $StartInfo.FileName = $Executable
        $StartInfo.Arguments = "--version"
        $StartInfo.UseShellExecute = $false
        $StartInfo.CreateNoWindow = $true
        $StartInfo.RedirectStandardOutput = $true
        $StartInfo.RedirectStandardError = $true
        $StartInfo.EnvironmentVariables["OLLAMA_HOST"] = "http://127.0.0.1:1"
        $Process = [Diagnostics.Process]::Start($StartInfo)
        if (-not $Process.WaitForExit(5000)) {
            $Process.Kill()
            $Process.WaitForExit()
            return $false
        }
        $VersionOutput = $Process.StandardOutput.ReadToEnd() + $Process.StandardError.ReadToEnd()
        $VersionMatches = [regex]::Matches($VersionOutput, "\d+\.\d+\.\d+")
        if ($VersionMatches.Count -eq 0) {
            return $false
        }
        $ClientVersion = $VersionMatches[$VersionMatches.Count - 1].Value
        return [version] $ClientVersion -ge $MinimumOllamaVersion
    } catch {
        return $false
    } finally {
        if ($null -ne $Process) {
            $Process.Dispose()
        }
    }
}

function Test-OllamaHealth {
    $Client = $null
    try {
        $Endpoint = [Uri] $env:OLLAMA_HOST
        $Client = [Net.Sockets.TcpClient]::new()
        $Connection = $Client.ConnectAsync($Endpoint.Host, $Endpoint.Port)
        if (-not $Connection.Wait(500)) {
            return $false
        }
        $Client.ReceiveTimeout = 500
        $Client.SendTimeout = 500
        $Stream = $Client.GetStream()
        $Request = [Text.Encoding]::ASCII.GetBytes(
            "GET /api/version HTTP/1.0`r`nHost: 127.0.0.1`r`n`r`n"
        )
        $Stream.Write($Request, 0, $Request.Length)
        $Response = [byte[]]::new(64)
        $Length = $Stream.Read($Response, 0, $Response.Length)
        $Status = [Text.Encoding]::ASCII.GetString($Response, 0, $Length)
        return $Status -match "^HTTP/1\.[01] 200 "
    } catch {
        return $false
    } finally {
        if ($null -ne $Client) {
            $Client.Dispose()
        }
    }
}

function Get-FreeLoopbackPort {
    $Listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    try {
        $Listener.Start()
        return ([Net.IPEndPoint] $Listener.LocalEndpoint).Port
    } finally {
        $Listener.Stop()
    }
}

function Stop-OwnedOllama {
    if ($null -eq $script:OllamaProcess) {
        return
    }
    $OwnedProcess = $script:OllamaProcess
    $script:OllamaProcess = $null
    try {
        $OwnedProcess.Refresh()
        if (-not $OwnedProcess.HasExited) {
            $TaskKill = Join-Path $env:SystemRoot "System32\taskkill.exe"
            try {
                & $TaskKill /PID $OwnedProcess.Id /T /F *> $null
            } catch {
                # Fall through to the exact-process fallback below.
            }
            $OwnedProcess.Refresh()
            if (-not $OwnedProcess.HasExited) {
                Stop-Process -Id $OwnedProcess.Id -Force -ErrorAction SilentlyContinue
            }
        }
        $OwnedProcess.WaitForExit()
    } catch {
        # The exact child may already have exited and been reaped.
    } finally {
        $OwnedProcess.Dispose()
    }
}

function Start-OwnedOllama([string] $Executable) {
    foreach ($LogPath in @($OllamaLog, $OllamaOutputLog)) {
        if (Test-Path -LiteralPath $LogPath) {
            $LogItem = Get-Item -LiteralPath $LogPath -Force
            if ($LogItem.PSIsContainer -or
                ($LogItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "The Ollama runtime log must be a regular file and not a link: $LogPath"
            }
        }
    }
    foreach ($Attempt in 1..3) {
        try {
            $Port = Get-FreeLoopbackPort
            $env:OLLAMA_HOST = "http://127.0.0.1:$Port"
            Write-Host "Starting the isolated Ollama service (attempt $Attempt of 3)..."
            $script:OllamaProcess = Start-Process -FilePath $Executable `
                -ArgumentList "serve" -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput $OllamaOutputLog -RedirectStandardError $OllamaLog
        } catch {
            Stop-OwnedOllama
            continue
        }

        $Deadline = [DateTime]::UtcNow.AddSeconds(30)
        while ([DateTime]::UtcNow -lt $Deadline) {
            $script:OllamaProcess.Refresh()
            if ($script:OllamaProcess.HasExited) {
                break
            }
            if (Test-OllamaHealth) {
                Start-Sleep -Milliseconds 200
                $script:OllamaProcess.Refresh()
                if ([DateTime]::UtcNow -lt $Deadline -and
                    -not $script:OllamaProcess.HasExited -and (Test-OllamaHealth)) {
                    return
                }
            }
            Start-Sleep -Milliseconds 500
        }
        Stop-OwnedOllama
    }
    throw "Ollama did not start after three attempts. Review the runtime log at $OllamaLog"
}

function Install-Ollama {
    Write-Host "Installing the verified private Ollama runtime..."
    Assert-NoPrivateData
    $OllamaArchive = Join-Path $Runtime "ollama-windows-amd64.zip"
    $OllamaStaging = Join-Path $Runtime "ollama-install"
    try {
        Invoke-WebRequest -Uri $OllamaArchiveUrl -OutFile $OllamaArchive -UseBasicParsing
        Assert-InstallerHash $OllamaArchive $OllamaArchiveSha256
        Remove-Item -LiteralPath $OllamaStaging -Recurse -Force -ErrorAction SilentlyContinue
        Microsoft.PowerShell.Archive\Expand-Archive `
            -LiteralPath $OllamaArchive -DestinationPath $OllamaStaging
        $StagedOllama = Join-Path $OllamaStaging "ollama.exe"
        $StagedLlamaServer = Join-Path $OllamaStaging "lib\ollama\llama-server.exe"
        if (-not (Test-Path -LiteralPath $StagedOllama -PathType Leaf) -or
            -not (Test-Path -LiteralPath $StagedLlamaServer -PathType Leaf)) {
            throw "The verified Ollama archive is missing required executables."
        }
        Assert-OllamaSignature $StagedOllama
        Assert-OllamaSignature $StagedLlamaServer
        Remove-Item -LiteralPath $OllamaDirectory -Recurse -Force -ErrorAction SilentlyContinue
        Move-Item -LiteralPath $OllamaStaging -Destination $OllamaDirectory
    } finally {
        Remove-Item -LiteralPath $OllamaArchive -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $OllamaStaging -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Assert-LocalDirectory([string] $Path) {
    if (Test-Path -LiteralPath $Path) {
        $Item = Get-Item -LiteralPath $Path -Force
        if (-not $Item.PSIsContainer) {
            throw "$Path must be a directory."
        }
        if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Path must not be a symbolic link or junction."
        }
    }
}

function Assert-NoReparseAncestor([string] $Path) {
    $Current = Get-Item -LiteralPath $Path -Force
    while ($null -ne $Current) {
        if (($Current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Path must not be inside a symbolic link or junction."
        }
        $Current = $Current.Parent
    }
}

$RunStatus = 2
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    if ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne
        [Runtime.InteropServices.Architecture]::X64) {
        throw "UMA BCT requires an x86-64 Windows computer."
    }
    if ([Environment]::OSVersion.Version.Build -lt 19045) {
        throw "UMA BCT requires Windows 10 22H2 or newer."
    }
    Assert-NoReparseAncestor $Root
    Assert-LocalDirectory $Inputs
    Assert-LocalDirectory $Outputs
    Assert-LocalDirectory $Runtime
    Assert-LocalDirectory (Join-Path $Runtime "bin")
    Assert-LocalDirectory (Join-Path $Runtime "uv-cache")
    Assert-LocalDirectory (Join-Path $Runtime "python")
    Assert-LocalDirectory $AppEnvironment
    Assert-LocalDirectory $OllamaDirectory
    Assert-LocalDirectory (Join-Path $Runtime "ollama-install")
    New-Item -ItemType Directory -Force -Path $Inputs, $Outputs, $Runtime | Out-Null
    $PrivateDataPresent = Test-PrivateData

    $Wheels = @(Get-ChildItem -LiteralPath $Root -Filter "uma_bct_cli-*.whl" -File)
    if ($Wheels.Count -ne 1) {
        throw "The supplied folder must contain exactly one UMA BCT wheel."
    }
    if (($Wheels[0].Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The supplied wheel must not be a symbolic link."
    }
    if (-not (Test-Path -LiteralPath $Constraints -PathType Leaf) -or
        ((Get-Item -LiteralPath $Constraints -Force).Attributes -band
            [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The supplied dependency lock is missing or redirected."
    }

    Get-ChildItem Env: | Where-Object { $_.Name -like "UV_*" } |
        ForEach-Object { Remove-Item "Env:$($_.Name)" }
    foreach ($Variable in @(
        "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL", "OLLAMA_HOST", "OLLAMA_DEBUG", "OLLAMA_DOWNLOAD_URL",
        "OLLAMA_INSTALL_DIR", "OLLAMA_NO_START", "OLLAMA_UNINSTALL",
        "OLLAMA_VERSION")) {
        Remove-Item "Env:$Variable" -ErrorAction SilentlyContinue
    }

    $UvDirectory = Join-Path $Runtime "bin"
    $Uv = Join-Path $UvDirectory "uv.exe"
    if (Test-Path -LiteralPath $Uv) {
        $UvItem = Get-Item -LiteralPath $Uv -Force
        if ($UvItem.PSIsContainer -or
            ($UvItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Uv must be a regular file and not a link."
        }
    }
    if (-not (Test-UvVersion $Uv)) {
        if ($PrivateDataPresent) {
            throw "Move private files out of inputs and outputs before repairing the local runtime."
        }
        Assert-NoPrivateData
        Write-Host "Preparing the private local Python runtime..."
        New-Item -ItemType Directory -Force -Path $UvDirectory | Out-Null
        $UvArchive = Join-Path $Runtime "uv.zip"
        $UvExtract = Join-Path $Runtime "uv-extract"
        try {
            Invoke-WebRequest -Uri $UvArchiveUrl -OutFile $UvArchive -UseBasicParsing
            Assert-InstallerHash $UvArchive $UvArchiveSha256
            Remove-Item -LiteralPath $UvExtract -Recurse -Force -ErrorAction SilentlyContinue
            Microsoft.PowerShell.Archive\Expand-Archive `
                -LiteralPath $UvArchive -DestinationPath $UvExtract
            Copy-Item -LiteralPath (Join-Path $UvExtract "uv.exe") -Destination $Uv
            Copy-Item -LiteralPath (Join-Path $UvExtract "uvx.exe") -Destination $UvDirectory
        } finally {
            Remove-Item -LiteralPath $UvArchive -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $UvExtract -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    if (-not (Test-UvVersion $Uv)) {
        throw "The pinned local Python runtime could not be installed."
    }

    $env:UV_CACHE_DIR = Join-Path $Runtime "uv-cache"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime "python"
    $env:UV_DEFAULT_INDEX = "https://pypi.org/simple"
    $env:UV_NO_CONFIG = "1"
    $env:OLLAMA_NO_CLOUD = "1"
    Remove-Item Env:UMA_BCT_ALLOW_REMOTE_OLLAMA -ErrorAction SilentlyContinue
    Remove-Item Env:UMA_BCT_ALLOW_CONTAINER_OLLAMA -ErrorAction SilentlyContinue

    $Ollama = Find-Ollama
    $LlamaServer = Find-LlamaServer $Ollama
    if ($null -eq $Ollama -or -not (Test-OllamaVersion $Ollama) -or $null -eq $LlamaServer) {
        if ($PrivateDataPresent) {
            throw "The local Ollama runtime is incomplete or unsupported; both ollama.exe and llama-server.exe are required. Move private files out of inputs and outputs, then run this launcher again to repair it. Runtime log: $OllamaLog"
        }
        Install-Ollama
        $Ollama = Find-Ollama
        $LlamaServer = Find-LlamaServer $Ollama
    }
    if ($null -eq $Ollama) {
        throw "Ollama could not be installed. Runtime log: $OllamaLog"
    }
    if (-not (Test-OllamaVersion $Ollama)) {
        throw "Ollama is too old and could not be updated. Runtime log: $OllamaLog"
    }
    if ($null -eq $LlamaServer) {
        throw "Ollama's llama-server.exe companion is missing. Runtime log: $OllamaLog"
    }

    if (-not $PrivateDataPresent) {
        Assert-NoPrivateData
        Write-Host "Installing the verified application dependencies..."
        & $Uv venv --no-config --clear --managed-python --python 3.12 $AppEnvironment
        if ($LASTEXITCODE -ne 0) {
            throw "The managed Python environment could not be created."
        }
        & $Uv pip sync --no-config --require-hashes `
            --python $AppPython $Constraints
        if ($LASTEXITCODE -ne 0) {
            throw "The verified application dependencies could not be installed."
        }
        & $Uv pip install --no-config --offline --no-deps `
            --python $AppPython $Wheels[0].FullName
        if ($LASTEXITCODE -ne 0) {
            throw "UMA BCT could not be installed."
        }
    } else {
        if (-not (Test-Path -LiteralPath $AppPython -PathType Leaf) -or
            -not (Test-Path -LiteralPath $AppCli -PathType Leaf)) {
            throw "Move private files out of inputs and outputs before repairing the application environment."
        }
        & $Uv pip install --no-config --offline --no-deps --reinstall `
            --python $AppPython $Wheels[0].FullName
        if ($LASTEXITCODE -ne 0) {
            throw "Move private files out of inputs and outputs before repairing the application environment."
        }
    }
    if (-not (Test-Path -LiteralPath $AppCli -PathType Leaf)) {
        throw "The application could not be installed."
    }

    Start-OwnedOllama $Ollama

    Push-Location $Root
    try {
        if (-not $PrivateDataPresent) {
            Assert-NoPrivateData
            Write-Host "Checking the application and model. The first model download is large..."
            $SetupLog = Join-Path $Runtime "setup.json"
            & $AppCli setup |
                Set-Content -LiteralPath $SetupLog -Encoding UTF8
            if ($LASTEXITCODE -ne 0) {
                throw "Automatic setup failed. See the message above and run this launcher again."
            }
        }

        if (@(Get-InputImages).Count -eq 0) {
            Write-Host "Add JPG, PNG, TIF, or WEBP photos to the folder that just opened."
            Open-LocalFolder $Inputs
            if ([Environment]::UserInteractive) {
                Read-Host "Press Enter after the photos have finished copying"
            }
        }
        if (@(Get-InputImages).Count -eq 0) {
            throw "No supported photos were found in $Inputs"
        }

        Write-Host "Transcribing the photos..."
        & $AppCli --force
        $RunStatus = $LASTEXITCODE
    } finally {
        Pop-Location
    }

    if ($RunStatus -eq 0) {
        Open-LocalFolder $Outputs
        Write-Host "`nFinished. Review $Outputs\catalogue.csv"
    } elseif ($RunStatus -eq 1) {
        Open-LocalFolder $Outputs
        [Console]::Error.WriteLine(
            "`nProcessing was incomplete. Review the newest run manifest in $Outputs."
        )
        [Console]::Error.WriteLine(
            "The catalogue may be partial, absent, or retained from the previous run."
        )
    }
} catch {
    [Console]::Error.WriteLine("`nUMA BCT could not continue: $($_.Exception.Message)")
    $RunStatus = 2
} finally {
    Stop-OwnedOllama
}
exit $RunStatus
