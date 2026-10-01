; Inno Setup script for a small online installer. The wizard downloads the app package named in latest.txt,
; verifies its SHA-256, unpacks it, and adds shortcuts. Build with build.ps1, which passes the defines below.

#ifndef ManifestUrl
  #define ManifestUrl "https://huggingface.co/REPLACE_ME/LocationLabel/resolve/main/latest.txt"
#endif
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "LocationLabel"
#define AppExe "LocationLabel.exe"

[Setup]
AppId={{B2E1A6F0-5C1D-4B57-9E0B-6F7A6C0D1A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Ikarus Delta
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=release
OutputBaseFilename=LocationLabel
SetupIconFile=assets\app.ico
UninstallDisplayIcon={app}\{#AppExe}
WizardStyle=modern
WizardImageFile=assets\wizard_large.bmp
WizardSmallImageFile=assets\wizard_small.bmp
InfoBeforeFile=installer_notes.txt
ExtraDiskSpaceRequired=5000000000
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName} now"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{localappdata}\LocationLabel\logs"

[Code]
var
  DownloadPage: TDownloadWizardPage;
  PackageUrl, PackageSha: String;
  DownloadStart: Cardinal;

function GetTickCount: Cardinal;
  external 'GetTickCount@kernel32.dll stdcall';

function ProgressOk(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  Result := True;
end;

function FormatSize(Bytes: Int64): String;
var
  Tenths: Int64;
begin
  if Bytes >= 1073741824 then
  begin
    Tenths := (Bytes * 10) div 1073741824;
    Result := IntToStr(Tenths div 10) + '.' + IntToStr(Tenths mod 10) + ' GB';
  end
  else
  begin
    Tenths := (Bytes * 10) div 1048576;
    Result := IntToStr(Tenths div 10) + '.' + IntToStr(Tenths mod 10) + ' MB';
  end;
end;

function FormatDuration(Seconds: Int64): String;
begin
  if Seconds >= 3600 then
    Result := IntToStr(Seconds div 3600) + ' h ' + IntToStr((Seconds mod 3600) div 60) + ' min'
  else if Seconds >= 60 then
    Result := IntToStr(Seconds div 60) + ' min ' + IntToStr(Seconds mod 60) + ' s'
  else
    Result := IntToStr(Seconds) + ' s';
end;

{ Shows percent, size, speed and time remaining on the download page. }
function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
var
  Elapsed, BytesPerSecond, Remaining: Int64;
  Percent: Integer;
  Detail: String;
begin
  Result := True;
  if ProgressMax <= 0 then Exit;
  try
    Percent := Integer((Progress * 100) div ProgressMax);
    Elapsed := Int64(GetTickCount - DownloadStart);
    Detail := FormatSize(Progress) + ' of ' + FormatSize(ProgressMax);
    if (Elapsed > 2000) and (Progress > 0) then
    begin
      BytesPerSecond := (Progress * 1000) div Elapsed;
      if BytesPerSecond > 0 then
      begin
        Remaining := (ProgressMax - Progress) div BytesPerSecond;
        Detail := Detail + '   |   ' + FormatSize(BytesPerSecond) + '/s   |   about ' + FormatDuration(Remaining) + ' remaining';
      end;
    end;
    DownloadPage.SetText('Downloading LocationLabel... ' + IntToStr(Percent) + '%', Detail);
  except
    { progress text must never stop the download }
  end;
end;

function ManifestValue(Lines: TStringList; const Key: String): String;
var
  I: Integer;
  Line: String;
begin
  Result := '';
  for I := 0 to Lines.Count - 1 do
  begin
    Line := Trim(Lines[I]);
    if Copy(Line, 1, Length(Key) + 1) = Key + '=' then
    begin
      Result := Trim(Copy(Line, Length(Key) + 2, MaxInt));
      Exit;
    end;
  end;
end;

function LoadManifest: Boolean;
var
  Manifest: TStringList;
begin
  Result := False;
  try
    DownloadTemporaryFile('{#ManifestUrl}', 'latest.txt', '', @ProgressOk);
    Manifest := TStringList.Create;
    try
      Manifest.LoadFromFile(ExpandConstant('{tmp}\latest.txt'));
      PackageUrl := ManifestValue(Manifest, 'package_url');
      PackageSha := ManifestValue(Manifest, 'package_sha256');
    finally
      Manifest.Free;
    end;
    Result := (PackageUrl <> '') and (PackageSha <> '');
  except
    Log(GetExceptionMessage);
  end;
  if not Result then
    SuppressibleMsgBox('Could not read the download information. Check your internet connection and try again.', mbError, MB_OK, IDOK);
end;

{ The program is 4.7 GB once unpacked; the download (2.9 GB) is kept in the Windows temp folder until it is unpacked. }
const
  InstallNeedMB = 5000;
  DownloadNeedMB = 3000;

function FreeMBOnDriveOf(const Path: String; var FreeMB: Cardinal): Boolean;
var
  TotalMB: Cardinal;
begin
  Result := GetSpaceOnDisk(ExtractFileDrive(Path) + '\', True, FreeMB, TotalMB);
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage('Downloading LocationLabel', 'Downloading the program files. This is a large download and can take a while.', @OnDownloadProgress);
  WizardForm.SelectDirBrowseLabel.Caption :=
    'Make sure the drive has enough free space. LocationLabel needs about 5 GB on this drive (4.7 GB once installed), ' +
    'plus about 3 GB of temporary space while it downloads.' + #13#10#13#10 +
    'To continue, click Next. If you would like to select a different folder, click Browse.';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  FreeMB: Cardinal;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    if FreeMBOnDriveOf(WizardDirValue, FreeMB) and (FreeMB < InstallNeedMB) then
    begin
      SuppressibleMsgBox('There is not enough free space on drive ' + ExtractFileDrive(WizardDirValue) + '.' + #13#10#13#10 +
        'Free: ' + FormatSize(Int64(FreeMB) * 1048576) + #13#10 + 'Needed: about 5 GB' + #13#10#13#10 +
        'Free up some space or choose a folder on another drive.', mbError, MB_OK, IDOK);
      Result := False;
    end;
  end;
  if CurPageID = wpReady then
  begin
    if FreeMBOnDriveOf(ExpandConstant('{tmp}'), FreeMB) and (FreeMB < DownloadNeedMB) then
    begin
      SuppressibleMsgBox('There is not enough temporary space to download LocationLabel.' + #13#10#13#10 +
        'Free on drive ' + ExtractFileDrive(ExpandConstant('{tmp}')) + ': ' + FormatSize(Int64(FreeMB) * 1048576) + #13#10 + 'Needed: about 3 GB' + #13#10#13#10 +
        'Free up some space on that drive and try again.', mbError, MB_OK, IDOK);
      Result := False;
      Exit;
    end;
    Result := LoadManifest;
    if not Result then Exit;
    DownloadPage.Clear;
    DownloadPage.Add(PackageUrl, 'LocationLabel-app.zip', PackageSha);
    DownloadStart := GetTickCount;
    DownloadPage.SetText('Starting download...', '');
    DownloadPage.Show;
    try
      try
        DownloadPage.Download;
      except
        Result := False;
        if not DownloadPage.AbortedByUser then
          SuppressibleMsgBox('The download failed: ' + GetExceptionMessage + #13#10#13#10 + 'Check your internet connection and try again.', mbCriticalError, MB_OK, IDOK);
      end;
    finally
      DownloadPage.Hide;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ZipPath, AppDir: String;
  ExitCode: Integer;
begin
  if CurStep <> ssInstall then Exit;

  ZipPath := ExpandConstant('{tmp}\LocationLabel-app.zip');
  if not FileExists(ZipPath) then
  begin
    { silent installs skip the wizard pages, so download here }
    if not LoadManifest then RaiseException('Could not get the program files.');
    DownloadTemporaryFile(PackageUrl, 'LocationLabel-app.zip', PackageSha, @ProgressOk);
  end;

  AppDir := ExpandConstant('{app}');
  ForceDirectories(AppDir);
  if DirExists(AppDir + '\_internal') then DelTree(AppDir + '\_internal', True, True, True);

  WizardForm.StatusLabel.Caption := 'Unpacking the program files. This usually takes 1 to 3 minutes...';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  if not Exec(ExpandConstant('{sys}\tar.exe'), '-xf "' + ZipPath + '" -C "' + AppDir + '"', '', SW_HIDE, ewWaitUntilTerminated, ExitCode) or (ExitCode <> 0) then
    RaiseException('The program files could not be unpacked (code ' + IntToStr(ExitCode) + '). Free some disk space and try again.');
  WizardForm.ProgressGauge.Style := npbstNormal;

  SaveStringToFile(AppDir + '\update_url.txt', '{#ManifestUrl}', False);
end;
