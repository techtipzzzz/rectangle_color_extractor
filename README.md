# Smart Passport Photo Cropper v4

A Python-based GUI application to extract and crop passport photos from images or scanned documents with multiple detection methods and scanner integration for Windows and Linux.

---

## Features

- **User-friendly GUI** using Tkinter
- Supports **scanner integration** with TWAIN (Windows) and SANE (Linux)
- Multiple detection methods: rectangle, grid, color, template, watershed, original
- Adjustable output photo size, padding, and file size limits
- Preview cropped photos within the application
- Cross-platform builds: Windows 7/10/11 and Ubuntu 18.04+ (via GitHub Actions)
- Standalone executables created using PyInstaller

---

## Installation

### Prerequisites

- Python 3.7 (for Windows 7 builds) or 3.10+ (for Windows 10/Ubuntu builds)
- Pip package manager

### Installing Dependencies

For Windows 7:

