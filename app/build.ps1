# Builds a release into app\release\ :
#   LocationLabel.exe          small online installer (send this to people)
#   LocationLabel-app-<ver>.zip      the program (upload to your download server)
#   latest.txt                       tells the installer and installed apps what the newest version is (upload too)
#
# First build:      .\build.ps1 -BaseUrl "https://huggingface.co/USER/REPO/resolve/main"
# Only new URL:     .\build.ps1 -BaseUrl "..." -SkipApp      (reuses the zip and the PyInstaller output; takes seconds)
# New version:      change APP_VERSION in gui\version.py, then build again and upload the zip and latest.txt.

param(
    [string]$BaseUrl = "https://huggingface.co/REPLACE_ME/LocationLabel/resolve/main",
    [string]$Notes = "",
    [switch]$SkipApp
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot "..\env\Scripts\python.exe"
$BaseUrl = $BaseUrl.TrimEnd("/")
$version = ((Get-Content gui\version.py -Raw) -replace '(?s).*"([\d\.]+)".*', '$1').Trim()
$zipName = "LocationLabel-app-$version.zip"
$release = Join-Path $PSScriptRoot "release"
New-Item -ItemType Directory -Force $release | Out-Null

if (-not $SkipApp) {
    & $python -m PyInstaller --noconfirm --clean LocationLabel.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
}
$zip = Join-Path $release $zipName
if ((-not $SkipApp) -or (-not (Test-Path $zip))) {
    if (-not (Test-Path dist\LocationLabel\LocationLabel.exe)) { throw "No app build found; run once without -SkipApp" }
    Remove-Item $zip -ErrorAction SilentlyContinue
    # Windows' own tar (bsdtar) writes zip64 archives; the tar that comes with Git does not.
    & (Join-Path $env:SystemRoot "System32\tar.exe") -a -cf $zip -C dist\LocationLabel .
    if ($LASTEXITCODE -ne 0) { throw "Could not create the package zip" }
}

$sha = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
$manifest = @(
    "version=$version",
    "package_url=$BaseUrl/$zipName",
    "package_sha256=$sha",
    "installer_url=$BaseUrl/LocationLabel.exe",
    "notes=$Notes"
)
[System.IO.File]::WriteAllLines((Join-Path $release "latest.txt"), $manifest, (New-Object System.Text.UTF8Encoding($false)))

$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 is not installed (https://jrsoftware.org/isdl.php)" }
& $iscc "/DManifestUrl=$BaseUrl/latest.txt" "/DAppVersion=$version" installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

"{0:N2} GB package, sha256 {1}" -f ((Get-Item $zip).Length / 1GB), $sha
Write-Host "Done. Upload these to $BaseUrl : $zipName, latest.txt, LocationLabel.exe"
