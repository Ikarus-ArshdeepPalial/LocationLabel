Here are the steps to build and publish a new update, run from PowerShell.

1. Change the version
Edit D:\Ikarus\LocationLabel\app\gui\version.py:
APP_VERSION = "1.1.0"
It must be higher than the last release, because that's how installed apps know an update exists.

2. Build
cd D:\Ikarus\LocationLabel\app
powershell -ExecutionPolicy Bypass -File .\build.ps1 -BaseUrl "https://huggingface.co/arsh208/LocationLabel/resolve/main" -Notes "What changed in this version"
- This takes about 10–15 minutes and needs about 15 GB of free disk space.
- Close any running LocationLabel first.
- The build uses the project's environment in ..\env.
- It creates three files in app\release\:
  - LocationLabel-app-1.1.0.zip (the program, about 2.7 GB)
  - latest.txt (version, download link, checksum and your notes)
  - LocationLabel.exe (the small installer)

3. Test the built app before uploading
Try it on a photo, and on a ramp set if you have one:
$env:LOCATIONLABEL_HOME = "D:\test_home"
.\dist\LocationLabel\LocationLabel.exe --selftest D:\test_out D:\photos\image.png --ramp D:\photos\ramp\1.png D:\photos\ramp\2.png D:\photos\ramp\3.png
type D:\test_out\selftest.txt
- It should start with OK. If it says FAILED, don't upload.
- The ramp part downloads the 4.7 GB VGGT model into D:\test_home the first time.
- The frozen build with VGGT bundled has never been run, so this check matters for this release.

4. Upload (the zip first, then the rest)
..\env\Scripts\hf upload arsh208/LocationLabel .\release\LocationLabel-app-1.1.0.zip LocationLabel-app-1.1.0.zip --repo-type model
..\env\Scripts\hf upload arsh208/LocationLabel .\release\latest.txt latest.txt --repo-type model
..\env\Scripts\hf upload arsh208/LocationLabel .\release\LocationLabel.exe LocationLabel.exe --repo-type model
- Upload latest.txt last. It points at the zip, so if it went up first, people could start an install that fails.
- Use the hf tool inside ..\env. Don't run pip install -U huggingface_hub, which broke the environment earlier.
- If it says you're not logged in, run ..\env\Scripts\hf auth login with a write token.

5. Check it
- Open https://huggingface.co/arsh208/LocationLabel/resolve/main/latest.txt. It should show the new version and link.
- Run the new LocationLabel.exe installer once on your machine.