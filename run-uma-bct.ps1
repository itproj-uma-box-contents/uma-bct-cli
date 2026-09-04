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
$OllamaInstallerUrl = "https://github.com/ollama/ollama/releases/download/v$OllamaVersion/OllamaSetup.exe"
$OllamaInstallerSha256 = "d7cfbb2509d9e9464972d162899a5f085c1802619b6bbe0c4e210b47bcf2d446"
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
    $Candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
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
        throw "Ollama installer signature verification failed."
    }
}

function Test-OllamaVersion([string] $Executable) {
    try {
        $VersionOutput = & $Executable --version 2>$null | Out-String
        if ($VersionOutput -notmatch "(\d+\.\d+\.\d+)") {
            return $false
        }
        return [version] $Matches[1] -ge $MinimumOllamaVersion
    } catch {
        return $false
    }
}

function Test-Ollama([string] $Executable) {
    try {
        & $Executable list *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Install-Ollama {
    Write-Host "Installing the verified local Ollama service..."
    Assert-NoPrivateData
    $OllamaInstaller = Join-Path $Runtime "OllamaSetup.exe"
    Invoke-WebRequest -Uri $OllamaInstallerUrl -OutFile $OllamaInstaller -UseBasicParsing
    Assert-InstallerHash $OllamaInstaller $OllamaInstallerSha256
    Assert-OllamaSignature $OllamaInstaller
    try {
        $Installer = Start-Process -FilePath $OllamaInstaller `
            -ArgumentList "/VERYSILENT /NORESTART /SUPPRESSMSGBOXES" `
            -PassThru
        $Installer.WaitForExit()
        if ($Installer.ExitCode -ne 0) {
            throw "Ollama installation failed with exit code $($Installer.ExitCode)."
        }
    } finally {
        Remove-Item -LiteralPath $OllamaInstaller -Force -ErrorAction SilentlyContinue
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

function Preserve-ExistingResults {
    $Timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    $Results = @(
        @{
            Source = Join-Path $Outputs "catalogue.csv"
            Saved = Join-Path $Outputs "catalogue.previous-$Timestamp.csv"
        },
        @{
            Source = Join-Path $Outputs "catalogue.run.json"
            Saved = Join-Path $Outputs "catalogue.previous-$Timestamp.run.json"
        }
    )
    foreach ($Result in $Results) {
        if (Test-Path -LiteralPath $Result.Source) {
            $Item = Get-Item -LiteralPath $Result.Source -Force
            if (-not $Item.PSIsContainer -and
                ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
                Copy-Item -LiteralPath $Result.Source -Destination $Result.Saved
                Write-Host "Preserved the previous result as $($Result.Saved)"
            } else {
                throw "$($Result.Source) must be a regular file and not a link."
            }
        }
    }
}

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
        "PIP_EXTRA_INDEX_URL", "OLLAMA_DEBUG", "OLLAMA_DOWNLOAD_URL",
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
    $env:OLLAMA_HOST = "http://127.0.0.1:11434"
    $env:OLLAMA_NO_CLOUD = "1"
    Remove-Item Env:UMA_BCT_ALLOW_REMOTE_OLLAMA -ErrorAction SilentlyContinue
    Remove-Item Env:UMA_BCT_ALLOW_CONTAINER_OLLAMA -ErrorAction SilentlyContinue

    $Ollama = Find-Ollama
    if ($null -eq $Ollama -or -not (Test-OllamaVersion $Ollama)) {
        if ($PrivateDataPresent) {
            throw "Move private files out of inputs and outputs before installing or updating Ollama."
        }
        Install-Ollama
        $Ollama = Find-Ollama
    }
    if ($null -eq $Ollama) {
        throw "Ollama could not be installed."
    }
    if (-not (Test-OllamaVersion $Ollama)) {
        throw "Ollama is too old and could not be updated."
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

    if (-not (Test-Ollama $Ollama)) {
        Write-Host "Starting Ollama..."
        Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden
        $Started = $false
        foreach ($Attempt in 1..30) {
            Start-Sleep -Seconds 1
            if (Test-Ollama $Ollama) {
                $Started = $true
                break
            }
        }
        if (-not $Started) {
            throw "Ollama did not start."
        }
    }

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

        if ((Get-InputImages).Count -eq 0) {
            Write-Host "Add JPG, PNG, TIF, or WEBP photos to the folder that just opened."
            Open-LocalFolder $Inputs
            if ([Environment]::UserInteractive) {
                Read-Host "Press Enter after the photos have finished copying"
            }
        }
        if ((Get-InputImages).Count -eq 0) {
            throw "No supported photos were found in $Inputs"
        }

        Write-Host "Transcribing the photos..."
        Preserve-ExistingResults
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
    exit $RunStatus
} catch {
    [Console]::Error.WriteLine("`nUMA BCT could not continue: $($_.Exception.Message)")
    exit 2
}
