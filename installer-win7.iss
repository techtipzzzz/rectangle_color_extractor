[Setup]
AppName=Smart Passport Cropper
AppVersion=1.0
DefaultDirName={pf}\Smart Passport Cropper
DefaultGroupName=Smart Passport Cropper
UninstallDisplayIcon={app}\smart-passport-cropper.exe
OutputBaseFilename=SmartPassportCropper-Win7-Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\passport_extractor_win7_x64.exe"; DestDir: "{app}"; DestName: "smart-passport-cropper.exe"
; Add models and data if needed
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Smart Passport Cropper"; Filename: "{app}\smart-passport-cropper.exe"; IconFilename: "icon\passport_extractor.ico"
Name: "{userdesktop}\Smart Passport Cropper"; Filename: "{app}\smart-passport-cropper.exe"; IconFilename: "icon\passport_extractor.ico"
