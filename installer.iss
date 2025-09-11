; installer.iss - Inno Setup script for Smart Passport Photo

[Setup]
AppName=Smart Passport Photo
AppVersion=1.0
AppPublisher=Kasim K, HSST Computer Science, Govt. Seethi Haji Memorial HSS, Edavanna
DefaultDirName={pf}\\SmartPassportPhoto
DefaultGroupName=Smart Passport Photo
OutputBaseFilename=setup
Compression=lzma
SolidCompression=yes
SetupIconFile=icon\\passport_extractor.ico
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main EXE (PyInstaller output)
Source: "dist\\main.exe"; DestDir: "{app}"; Flags: ignoreversion
; Custom icon
Source: "icon\\passport_extractor.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu shortcut
Name: "{group}\\Smart Passport Photo"; Filename: "{app}\\main.exe"; IconFilename: "{app}\\passport_extractor.ico"
; Desktop shortcut
Name: "{commondesktop}\\Smart Passport Photo"; Filename: "{app}\\main.exe"; IconFilename: "{app}\\passport_extractor.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\\main.exe"; Description: "{cm:LaunchProgram,Smart Passport Photo}"; Flags: nowait postinstall skipifsilent
