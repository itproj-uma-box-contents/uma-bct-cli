$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
# Direct PowerShell calls can supply numbers as objects; native CLI arguments
# are strings. Capture them once before any helper's local $args is used.
$CliArguments = @($args | ForEach-Object { [string] $_ })

# powershell.exe -File splits attached values such as -mmodel:8b before
# filling $args. A calling script's arguments must never replace its callee's.
if ([string]::IsNullOrEmpty($MyInvocation.ScriptName)) {
    $NativeArguments = [Environment]::GetCommandLineArgs()
    for ($Index = 1; $Index -lt $NativeArguments.Length - 1; $Index++) {
        if ($NativeArguments[$Index] -iin @("-Command", "-EncodedCommand")) { break }
        if ($NativeArguments[$Index] -ieq "-File") {
            $InvokedScript = Get-Item -LiteralPath $NativeArguments[$Index + 1] -ErrorAction SilentlyContinue
            if ($null -ne $InvokedScript -and $InvokedScript.FullName -ieq $MyInvocation.MyCommand.Path) {
                $CliArguments = @($NativeArguments | Select-Object -Skip ($Index + 2))
            }
            break
        }
    }
}

# Keep runtime, input, and result paths anchored to this extracted bundle.
# The launcher reserves it before any repair or model setup begins.
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$FriendlyInputs = Join-Path $Root "Add Photos Here"
$FriendlyOutputs = Join-Path $Root "Results"
$LegacyInputs = Join-Path $Root "inputs"
$LegacyOutputs = Join-Path $Root "outputs"
$Inputs = $null
$Outputs = $null
$Runtime = Join-Path $Root ".uma-bct-runtime"
$Constraints = Join-Path $Root "client-constraints.txt"
$AppEnvironment = Join-Path $Runtime "app"
$AppPython = Join-Path $AppEnvironment "Scripts\python.exe"
$AppCli = Join-Path $AppEnvironment "Scripts\uma-bct.exe"
$UvVersion = "0.12.9"
$UvArchiveUrl = "https://releases.astral.sh/github/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$UvArchiveSha256 = "ddbfcee1ac615a0499f6aa97b5ec8ebdf3ee4a7714a48055ec2ba0030e3cf810"
$OllamaVersion = "0.32.9"
$MinimumOllamaVersion = [version] $OllamaVersion
$OllamaArchiveUrl = "https://github.com/ollama/ollama/releases/download/v$OllamaVersion/ollama-windows-amd64.zip"
$OllamaArchiveSha256 = "7b4f6ce09c1f2c3b21561b323779beaf3ca3c7012f8e4522605a13cbbb19f0b8"
$OllamaDirectory = Join-Path $Runtime "ollama"
$OllamaPrevious = Join-Path $Runtime "ollama-previous"
$OllamaLog = Join-Path $Runtime "ollama.log"
$OllamaOutputLog = Join-Path $Runtime "ollama.stdout.log"
$OllamaProcess = $null
$AppProcess = $null
$LaunchLockPath = Join-Path $Root ".uma-bct-launcher.lock"
$LaunchLockStream = $null
$SupportedExtensions = @(
    ".avif", ".bmp", ".dng", ".gif", ".heic", ".heif",
    ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
)

function Show-LauncherHelp {
    [Console]::WriteLine(@'
UMA BCT ZIP launcher
Usage: .\run-uma-bct.ps1 [processing options]

Photos: Add Photos Here   Results: Results
Existing inputs/outputs folders are kept when they already contain data.

  --format, --file-type csv|excel   Save CSV (.csv) or Excel (.xlsx); default CSV
  --reprocess                     Create a new batch; keep previous results
  --verification ask|always|never  Select the second-reading accuracy check
  --hosted                        Use the configured hosted model
  -d, --detector qwen|opencv       Select the box detector
  -t, --transcriber qwen          Select the text transcriber
  -m, --model NAME                Select the local model
  --ai-dates                      Use the selected model to extract dates
  -c, --min-transcription-confidence INTEGER  Set the review threshold (0-100)
  --min-confidence, --min-detector-confidence FLOAT  Set OpenCV threshold (0-1)
  --request-timeout SECONDS        Set the model request timeout
  --no-missing-box, --no-missing-box-detection  Disable missing-box suggestions
  --json                          Write one JSON summary; never prompt
  -h, --help                      Show this help without setting up the runtime
  --version                       Show the bundled application version

Interactive launches open the photo folder and wait for Enter every time.
Completed batches stay unchanged; use --reprocess to apply a different format.
Use direct uma-bct for custom paths, --force, or setup. See README.md for details.
'@)
}

function Get-BundledWheel {
    $Wheels = @(Get-ChildItem -LiteralPath $Root -Filter "uma_bct_cli-*.whl" -File)
    if ($Wheels.Count -ne 1) {
        throw "The supplied folder must contain exactly one UMA BCT wheel."
    }
    if (($Wheels[0].Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The supplied wheel must not be a symbolic link."
    }
    return $Wheels[0]
}

function Test-VerifiedResultsLock([IO.FileInfo] $Item) {
    if ($Item.Name -ne ".uma-bct.lock" -or
        ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        return $false
    }
    $LinkType = $Item.PSObject.Properties["LinkType"]
    if ($null -eq $LinkType -or -not [string]::IsNullOrEmpty("$($LinkType.Value)")) {
        return $false
    }
    try {
        $Raw = Get-Content -LiteralPath $Item.FullName -Raw -Encoding UTF8
        return $Raw -cmatch '^\{"schema_version": 1, "pid": [1-9][0-9]*\}\r?\n?$'
    } catch {
        return $false
    }
}

function Test-VerifiedSettings([IO.FileInfo] $Item) {
    if ($Item.Name -ne ".uma-bct-settings.json" -or $Item.Length -gt 4096 -or
        ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        return $false
    }
    $LinkType = $Item.PSObject.Properties["LinkType"]
    if ($null -eq $LinkType -or -not [string]::IsNullOrEmpty("$($LinkType.Value)")) {
        return $false
    }
    try {
        $Raw = Get-Content -LiteralPath $Item.FullName -Raw -Encoding UTF8
        return $Raw -cmatch '^\s*\{\s*"verification"\s*:\s*"(ask|always|never)"\s*\}\s*$'
    } catch {
        return $false
    }
}

function Test-DirectoryData([string] $Path) {
    try {
        $Item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    } catch [Management.Automation.ItemNotFoundException] {
        return $false
    } catch {
        throw "$Path could not be inspected safely: $($_.Exception.Message)"
    }
    if (-not $Item.PSIsContainer -or
        ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Path must be a directory and not a symbolic link or junction."
    }
    foreach ($Entry in @(Get-ChildItem -LiteralPath $Path -Force)) {
        if ($Entry -is [IO.FileInfo] -and ((Test-VerifiedResultsLock $Entry) -or (Test-VerifiedSettings $Entry))) {
            continue
        }
        return $true
    }
    return $false
}

function Select-DataDirectories {
    # Respect an existing legacy layout in place. Mixed data layouts need an
    # explicit human choice so a resume never points at the wrong photos.
    $FriendlyData = (Test-DirectoryData $FriendlyInputs) -or
        (Test-DirectoryData $FriendlyOutputs)
    $LegacyData = (Test-DirectoryData $LegacyInputs) -or
        (Test-DirectoryData $LegacyOutputs)
    if ($FriendlyData -and $LegacyData) {
        throw "Private data exists in both '$FriendlyInputs' / '$FriendlyOutputs' and " +
            "'$LegacyInputs' / '$LegacyOutputs'. Keep only one layout; files were not merged or moved."
    }
    if ($LegacyData) {
        $script:Inputs = $LegacyInputs
        $script:Outputs = $LegacyOutputs
    } else {
        $script:Inputs = $FriendlyInputs
        $script:Outputs = $FriendlyOutputs
    }
}

function Reserve-Bundle {
    # The open file handle serializes launches. An existing lock file is
    # reused only when its saved owner text has the expected shape.
    $Created = $false
    try {
        $script:LaunchLockStream = [IO.File]::Open(
            $LaunchLockPath,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None
        )
        $Created = $true
    } catch [IO.IOException] {
        $Item = Get-Item -LiteralPath $LaunchLockPath -Force -ErrorAction Stop
        $LinkType = $Item.PSObject.Properties["LinkType"]
        if ($Item.PSIsContainer -or
            ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            ($null -ne $LinkType -and -not [string]::IsNullOrEmpty("$($LinkType.Value)"))) {
            throw "The launcher reservation must be a regular file and not a link: $LaunchLockPath"
        }
        try {
            $script:LaunchLockStream = [IO.File]::Open(
                $LaunchLockPath,
                [IO.FileMode]::Open,
                [IO.FileAccess]::ReadWrite,
                [IO.FileShare]::None
            )
        } catch [IO.IOException] {
            throw "This extracted UMA BCT folder is already running. Wait for the other launcher to finish: $Root"
        }
    }
    try {
        if (-not $Created) {
            if ($script:LaunchLockStream.Length -gt 64) {
                throw "The launcher reservation is malformed and was not recovered: $LaunchLockPath"
            }
            $ExistingBytes = [byte[]]::new([int] $script:LaunchLockStream.Length)
            [void] $script:LaunchLockStream.Read($ExistingBytes, 0, $ExistingBytes.Length)
            $ExistingOwner = [Text.Encoding]::UTF8.GetString($ExistingBytes)
            if ($ExistingOwner -cnotmatch '^[1-9][0-9]*\r?\n$') {
                throw "The launcher reservation is malformed and was not recovered: $LaunchLockPath"
            }
        }
        $Owner = [Text.Encoding]::UTF8.GetBytes("$PID`n")
        $script:LaunchLockStream.SetLength(0)
        $script:LaunchLockStream.Write($Owner, 0, $Owner.Length)
        $script:LaunchLockStream.Flush($true)
    } catch {
        Release-Bundle
        throw
    }
}

function Release-Bundle {
    if ($null -ne $script:LaunchLockStream) {
        $script:LaunchLockStream.Dispose()
        $script:LaunchLockStream = $null
    }
}

function Get-InputImages {
    @(Get-ChildItem -LiteralPath $Inputs -File -ErrorAction SilentlyContinue |
        Where-Object { $SupportedExtensions -contains $_.Extension.ToLowerInvariant() })
}

function Test-PrivateData {
    foreach ($Directory in @($FriendlyInputs, $FriendlyOutputs, $LegacyInputs, $LegacyOutputs)) {
        if (Test-DirectoryData $Directory) {
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
        try {
            Start-Process -FilePath "explorer.exe" -ArgumentList "`"$Path`""
        } catch {
            # Folder opening is a convenience. Keep the processing outcome
            # intact and give the operator the path, as the Unix launcher does.
            [Console]::Error.WriteLine("Could not open the folder automatically. Open it in your file manager: $Path")
        }
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

function Stop-OwnedProcess([Diagnostics.Process] $OwnedProcess, [bool] $RequestStop) {
    try {
        $OwnedProcess.Refresh()
        $TaskKill = Join-Path $env:SystemRoot "System32\taskkill.exe"
        if (-not $OwnedProcess.HasExited) {
            if ($RequestStop) {
                try {
                    & $TaskKill /PID $OwnedProcess.Id /T *> $null
                } catch {
                    # The verified child may not own a window.
                }
            }
            [void] $OwnedProcess.WaitForExit(5000)
        }
        $OwnedProcess.Refresh()
        if (-not $OwnedProcess.HasExited) {
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
        [void] $OwnedProcess.WaitForExit(5000)
    } catch {
        # The verified child may already have exited and been reaped.
    } finally {
        $OwnedProcess.Dispose()
    }
}

function Stop-OwnedOllama {
    if ($null -eq $script:OllamaProcess) {
        return
    }
    $OwnedProcess = $script:OllamaProcess
    $script:OllamaProcess = $null
    Stop-OwnedProcess $OwnedProcess $true
}

function Stop-OwnedApp {
    if ($null -eq $script:AppProcess) {
        return
    }
    $OwnedProcess = $script:AppProcess
    $script:AppProcess = $null
    Stop-OwnedProcess $OwnedProcess $false
}

function Quote-NativeArgument([string] $Value) {
    # Windows consumes backslashes before quotes. Double those backslashes,
    # escape the quote itself, then double any backslashes at the end.
    $EscapedQuotes = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $EscapedEnding = [regex]::Replace($EscapedQuotes, '\\+$', '$0$0')
    return '"' + $EscapedEnding + '"'
}

function Invoke-OwnedApp([string] $Arguments, [string] $OutputPath = "") {
    $StartInfo = [Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $AppCli
    $StartInfo.Arguments = $Arguments
    $StartInfo.UseShellExecute = $false
    if (-not [string]::IsNullOrEmpty($OutputPath)) {
        $StartInfo.RedirectStandardOutput = $true
    }
    $script:AppProcess = [Diagnostics.Process]::Start($StartInfo)
    try {
        if ($StartInfo.RedirectStandardOutput) {
            $Output = $script:AppProcess.StandardOutput.ReadToEnd()
        }
        $script:AppProcess.WaitForExit()
        $Status = $script:AppProcess.ExitCode
        if ($StartInfo.RedirectStandardOutput) {
            [IO.File]::WriteAllText($OutputPath, $Output, [Text.UTF8Encoding]::new($false))
        }
        if ($Status -eq -1073741510) {
            return 130
        }
        return $Status
    } finally {
        Stop-OwnedApp
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
    # Use a fresh numeric loopback port and confirm the owned process remains
    # healthy twice before the CLI sends it any private image.
    foreach ($Attempt in 1..3) {
        try {
            $Port = Get-FreeLoopbackPort
            $env:OLLAMA_HOST = "http://127.0.0.1:$Port"
            [Console]::Error.WriteLine("Starting the isolated Ollama service (attempt $Attempt of 3)...")
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
    [Console]::Error.WriteLine("Installing the verified private Ollama runtime...")
    Assert-NoPrivateData
    $OllamaArchive = Join-Path $Runtime "ollama-windows-amd64.zip"
    $OllamaStaging = Join-Path $Runtime "ollama-install"
    $Promoted = $false
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
        if (-not (Test-OllamaVersion $StagedOllama)) {
            throw "The verified Ollama archive has an unsupported executable version."
        }
        # Verify both executables before promotion, retaining the prior
        # runtime until the new directory is in place.
        Remove-Item -LiteralPath $OllamaPrevious -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $OllamaDirectory) {
            Move-Item -LiteralPath $OllamaDirectory -Destination $OllamaPrevious
        }
        try {
            Move-Item -LiteralPath $OllamaStaging -Destination $OllamaDirectory
            $Promoted = $true
            Remove-Item -LiteralPath $OllamaPrevious -Recurse -Force -ErrorAction SilentlyContinue
        } catch {
            if (-not (Test-Path -LiteralPath $OllamaDirectory) -and
                (Test-Path -LiteralPath $OllamaPrevious -PathType Container)) {
                Move-Item -LiteralPath $OllamaPrevious -Destination $OllamaDirectory
            }
            throw
        }
    } finally {
        Remove-Item -LiteralPath $OllamaArchive -Force -ErrorAction SilentlyContinue
        if ($Promoted) {
            Remove-Item -LiteralPath $OllamaStaging -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Recover-OllamaPromotion {
    if (-not (Test-Path -LiteralPath $OllamaDirectory) -and
        (Test-Path -LiteralPath $OllamaPrevious -PathType Container)) {
        Move-Item -LiteralPath $OllamaPrevious -Destination $OllamaDirectory
    }
}

function Assert-LocalDirectory([string] $Path) {
    try {
        $Item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    } catch [Management.Automation.ItemNotFoundException] {
        return
    } catch {
        throw "$Path could not be inspected safely: $($_.Exception.Message)"
    }
    if (-not $Item.PSIsContainer) {
        throw "$Path must be a directory."
    }
    if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Path must not be a symbolic link or junction."
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
$Interactive = $false
try {
    # Check only the launcher's fixed-folder boundary. Python remains the
    # single place that validates processing choices, numbers, and values.
    $JsonOutput = $false
    $HostedRequested = $false
    $MetadataOption = ""
    $ExpectsValue = $false
    $ValueOption = ""
    $SetupModel = $null
    $ValueOptions = @(
        "--format", "--file-type", "--verification", "-d", "--detector",
        "-t", "--transcriber", "-m", "--model", "-c", "--min-transcription-confidence",
        "--min-confidence", "--min-detector-confidence", "--request-timeout"
    )
    foreach ($Argument in $CliArguments) {
        if ($ExpectsValue) {
            $ExpectsValue = $false
            if (-not $Argument.StartsWith("-")) {
                if ($ValueOption -ceq "--model" -or $ValueOption -ceq "-m") {
                    $SetupModel = $Argument
                }
                continue
            }
        }
        if ($Argument -cmatch '^--(path-to-inputs|path-to-outputs?|path-to-box-images|work-dir|results-directory|force)(=|$)' -or
            $Argument -cmatch '^-[iobf]' -or $Argument -ceq "setup") {
            throw "This ZIP launcher uses its managed photo and results folders. Use direct uma-bct for '$Argument'; see README.md."
        }
        if ($Argument -ceq "--help" -or $Argument -ceq "-h") { $MetadataOption = "help" }
        if ($Argument -ceq "--version" -and $MetadataOption -ne "help") { $MetadataOption = "version" }
        if ($Argument -ceq "--json") { $JsonOutput = $true }
        if ($Argument -ceq "--hosted") { $HostedRequested = $true }
        if ($Argument.StartsWith("--model=")) { $SetupModel = $Argument.Substring(8) }
        elseif ($Argument.StartsWith("-m=")) { $SetupModel = $Argument.Substring(3) }
        elseif ($Argument.StartsWith("-m") -and $Argument.Length -gt 2) { $SetupModel = $Argument.Substring(2) }
        $ExpectsValue = $ValueOptions -ccontains $Argument
        $ValueOption = $Argument
    }

    # Metadata exits before folders, locks, installers, or model setup.
    if ($MetadataOption -eq "help") {
        Show-LauncherHelp
        exit 0
    }
    if ($MetadataOption -eq "version") {
        $Wheel = Get-BundledWheel
        [Console]::WriteLine($Wheel.Name.Split('-')[1])
        exit 0
    }
    if ($HostedRequested -and [string]::IsNullOrWhiteSpace($env:UMA_BCT_API_KEY)) {
        throw "UMA_BCT_API_KEY is required when --hosted is selected."
    }
    $Interactive = -not $JsonOutput -and [Environment]::UserInteractive -and
        -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected -and
        -not [Console]::IsErrorRedirected

    # Reject redirected workspace paths before creating folders or repairing
    # the local runtime beside any client photographs.
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    if ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne
        [Runtime.InteropServices.Architecture]::X64) {
        throw "UMA BCT requires an x86-64 Windows computer."
    }
    if ([Environment]::OSVersion.Version.Build -lt 19045) {
        throw "UMA BCT requires Windows 10 22H2 or newer."
    }
    Assert-NoReparseAncestor $Root
    Select-DataDirectories
    Reserve-Bundle
    Assert-LocalDirectory $Inputs
    Assert-LocalDirectory $Outputs
    Assert-LocalDirectory $Runtime
    Assert-LocalDirectory (Join-Path $Runtime "bin")
    Assert-LocalDirectory (Join-Path $Runtime "uv-cache")
    Assert-LocalDirectory (Join-Path $Runtime "python")
    Assert-LocalDirectory $AppEnvironment
    Assert-LocalDirectory $OllamaDirectory
    Assert-LocalDirectory (Join-Path $Runtime "ollama-install")
    Assert-LocalDirectory $OllamaPrevious
    New-Item -ItemType Directory -Force -Path $Inputs, $Outputs, $Runtime | Out-Null
    $PrivateDataPresent = Test-PrivateData

    $Wheel = Get-BundledWheel
    if (-not (Test-Path -LiteralPath $Constraints -PathType Leaf) -or
        ((Get-Item -LiteralPath $Constraints -Force).Attributes -band
            [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "The supplied dependency lock is missing or redirected."
    }

    # Inherited package and endpoint settings must not redirect the pinned
    # installer or the launcher-owned Ollama service.
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
            throw "Move private files out of '$Inputs' and '$Outputs' before repairing the local runtime."
        }
        Assert-NoPrivateData
        [Console]::Error.WriteLine("Preparing the private local Python runtime...")
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
    $HostedMode = -not [string]::IsNullOrWhiteSpace($env:UMA_BCT_API_KEY)
    if ($HostedMode -and [string]::IsNullOrWhiteSpace($env:UMA_BCT_API_MODEL)) {
        throw "UMA_BCT_API_MODEL is required when UMA_BCT_API_KEY is set."
    }
    Remove-Item Env:UMA_BCT_ALLOW_REMOTE_OLLAMA -ErrorAction SilentlyContinue
    Remove-Item Env:UMA_BCT_ALLOW_CONTAINER_OLLAMA -ErrorAction SilentlyContinue

    if (-not $HostedMode) {
        Recover-OllamaPromotion
        $Ollama = Find-Ollama
        $LlamaServer = Find-LlamaServer $Ollama
        if ($null -eq $Ollama -or -not (Test-OllamaVersion $Ollama) -or $null -eq $LlamaServer) {
            if ($PrivateDataPresent) {
                throw "The local Ollama runtime is incomplete or unsupported; both ollama.exe and llama-server.exe are required. Move private files out of '$Inputs' and '$Outputs', then run this launcher again to repair it. Runtime log: $OllamaLog"
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
    }

    if (-not $PrivateDataPresent) {
        # Finish verified downloads and dependency installation before the
        # client is prompted to place private photos in the input folder.
        Assert-NoPrivateData
        [Console]::Error.WriteLine("Installing the verified application dependencies...")
        & $Uv venv --no-config --clear --managed-python --python 3.12 $AppEnvironment |
            ForEach-Object { [Console]::Error.WriteLine($_) }
        if ($LASTEXITCODE -ne 0) {
            throw "The managed Python environment could not be created."
        }
        & $Uv pip sync --no-config --require-hashes `
            --python $AppPython $Constraints | ForEach-Object { [Console]::Error.WriteLine($_) }
        if ($LASTEXITCODE -ne 0) {
            throw "The verified application dependencies could not be installed."
        }
        & $Uv pip install --no-config --offline --no-deps `
            --python $AppPython $Wheel.FullName | ForEach-Object { [Console]::Error.WriteLine($_) }
        if ($LASTEXITCODE -ne 0) {
            throw "UMA BCT could not be installed."
        }
    } else {
        if (-not (Test-Path -LiteralPath $AppPython -PathType Leaf) -or
            -not (Test-Path -LiteralPath $AppCli -PathType Leaf)) {
            throw "Move private files out of '$Inputs' and '$Outputs' before repairing the application environment."
        }
        & $Uv pip install --no-config --offline --no-deps --reinstall `
            --python $AppPython $Wheel.FullName | ForEach-Object { [Console]::Error.WriteLine($_) }
        if ($LASTEXITCODE -ne 0) {
            throw "Move private files out of '$Inputs' and '$Outputs' before repairing the application environment."
        }
    }
    if (-not (Test-Path -LiteralPath $AppCli -PathType Leaf)) {
        throw "The application could not be installed."
    }

    if (-not $HostedMode) {
        Start-OwnedOllama $Ollama
    }

    Push-Location $Root
    try {
        if (-not $PrivateDataPresent) {
            Assert-NoPrivateData
            if ($HostedMode) {
                [Console]::Error.WriteLine("Checking the hosted client and private folders...")
            } else {
                [Console]::Error.WriteLine("Checking the application and model. The first model download is large...")
            }
            $SetupLog = Join-Path $Runtime "setup.json"
            $SetupArguments = "setup --path-to-inputs " + (Quote-NativeArgument $Inputs) +
                " --results-directory " + (Quote-NativeArgument $Outputs)
            if ($HostedMode) { $SetupArguments += " --hosted" }
            # Setup must prepare the same model that processing will use.
            if ($null -ne $SetupModel) {
                $SetupArguments += " " + (Quote-NativeArgument "--model=$SetupModel")
            }
            $SetupStatus = Invoke-OwnedApp $SetupArguments $SetupLog
            if ($SetupStatus -ne 0) {
                throw "Automatic setup failed. See the message above and run this launcher again."
            }
        }

        [Console]::Error.WriteLine("Photos folder: $Inputs")
        [Console]::Error.WriteLine("Results folder: $Outputs")
        if ($Interactive) {
            # Give users the same chance to add photos before every run,
            # including a launch that will reuse an already completed batch.
            [Console]::Error.WriteLine("Add supported photo files (including JPG, PNG, HEIC, or DNG) to the photos folder.")
            Open-LocalFolder $Inputs
            [Console]::Error.Write("Press Enter after the photos have finished copying: ")
            if ($null -eq [Console]::ReadLine()) {
                throw "No input was received. Run this launcher again when you are ready."
            }
        }
        if (@(Get-InputImages).Count -eq 0) {
            throw "No supported photos were found in $Inputs"
        }

        # Keep the CLI's progress and final report in this terminal. The
        # launcher opens Results only after that foreground run returns.
        $Arguments = "--path-to-inputs " + (Quote-NativeArgument $Inputs) +
            " --results-directory " + (Quote-NativeArgument $Outputs)
        if ($HostedMode) { $Arguments += " --hosted" }
        foreach ($Argument in $CliArguments) {
            $Arguments += " " + (Quote-NativeArgument $Argument)
        }
        $RunStatus = Invoke-OwnedApp $Arguments
    } finally {
        Pop-Location
    }

    if ($Interactive -and ($RunStatus -eq 0 -or $RunStatus -eq 1)) {
        Open-LocalFolder $Outputs
    }
} catch [Management.Automation.PipelineStoppedException] {
    [Console]::Error.WriteLine("`nUMA BCT was interrupted.")
    $RunStatus = 130
} catch {
    [Console]::Error.WriteLine("`nUMA BCT could not continue: $($_.Exception.Message)")
    $RunStatus = 2
} finally {
    Stop-OwnedApp
    Stop-OwnedOllama
    Release-Bundle
}

# A double-clicked console would otherwise close before the error can be
# read. Release owned processes and locks first; never delay automation.
if ($Interactive -and $RunStatus -gt 0 -and $RunStatus -lt 128) {
    try {
        [Console]::Error.Write("Press Enter to close this window: ")
        [void] [Console]::ReadLine()
    } catch {
        # A closing console must not change the processing exit code.
    }
}
exit $RunStatus
