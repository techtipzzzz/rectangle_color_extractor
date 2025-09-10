[Setup]
AppName=Smart Passport Cropper
AppVersion=1.0
DefaultDirName={pf}\Smart Passport Cropper
DefaultGroupName=Smart Passport Cropper
UninstallDisplayIcon={app}\smart-passport-cropper.exe
OutputBaseFilename=SmartPassportCropper-Win10-Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\smart-passport-cropper-win10-v1.0.exe"; DestDir: "{app}"; DestName: "smart-passport-cropper.exe"

[Icons]
Name: "{group}\Smart Passport Cropper"; Filename: "{app}\smart-passport-cropper.exe"; IconFilename: "icon\passport_extractor.ico"
Name: "{userdesktop}\Smart Passport Cropper"; Filename: "{app}\smart-passport-cropper.exe"; IconFilename: "icon\passport_extractor.ico"
