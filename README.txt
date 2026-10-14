PHOTO EDITOR - WINDOWS FINAL APPLICATION
=========================================

Purpose
-------
This package prepares the current Photo Editor Python application as a
standalone Windows executable.

Files
-----
photo_studio_bright.py  - current Photo Editor application
PhotoEditor.spec        - PyInstaller build configuration
build_windows.bat       - one-click Windows build script

How to create the EXE
---------------------
1. Copy this folder to a Windows PC.
2. Make sure Python is installed and available in PATH.
3. Double-click build_windows.bat.
4. Wait for the build to finish.
5. Open the "dist" folder.
6. Double-click PhotoEditor.exe.

The final executable is:
    dist\PhotoEditor.exe

Notes
-----
- The build script installs PyInstaller, Pillow, OpenCV and NumPy.
- The application is configured as a GUI application, so no console
  window is intended to appear when PhotoEditor.exe is launched.
- If Windows SmartScreen shows a warning for an unsigned locally-built
  executable, choose the option to view more information and run it
  only if you trust the build source.
