[Setup]
AppName=Smart Passport Cropper
AppVersion=1.0
DefaultDirName={pf}\Smart Passport Cropper
DefaultGroupName=Smart Passport Cropper
UninstallDisplayIcon={app}\smart-passport-cropper.exe
OutputBaseFilename=SmartPassportCropper-Win7-Setup
Compression=lzma
SolidCompression=yes
OutputDir=Output

[Files]
Source: "dist\smart-passport-cropper-win7-v1.0.exe"; DestDir: "{app}"; DestName: "smart-passport-cropper.exe"

[Icons]
Name: "{group}\Smart Passport Cropper"; Filename: "{app}\smart-passport-cropper.exe"; IconFilename: "icon\passport_extractor.ico"
Name: "{userdesktop}\Smart Passport Cropper"; Filename: "{app}\smart-passport-cropper.exe"; IconFilename: "icon\passport_extractor.ico"
