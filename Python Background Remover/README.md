# Background Remover

Background Remover is a small CLI tool that batch-processes images, removes a flat background color, centers the object and exports square PNG icons with transparency.  
Optionally, it can add a Photoshop-style drop shadow and prints average CPU / RAM / GPU usage at the end.

---

## Requirements

- Python 3.8+
- Packages:
  - numpy
  - opencv-python
  - Pillow
  - psutil

Install dependencies:

> pip install numpy opencv-python pillow psutil

---

## Basic usage

1. Place your source images (renders with a flat background color) in a folder, for example `files/`.
2. Open a terminal in the directory where `script.py` is located.
3. Run the tool:

> python script.py

4. Follow the interactive wizard:
   - Input folder with renders: path to your source images (for example `files`)
   - Output folder for PNG icons: where processed images will be saved (for example `exports`)
   - Icon size (square, px): final size like `512` or `1024`
   - Background color to remove: the flat background color (`#336699`, `rgb(0 255 0)`, etc.)
   - Add drop shadow (Photoshop style)? [Y/n]: enable or disable the shadow

5. The script will:
   - Remove the selected background color
   - Crop to the visible object
   - Resize and center it in a square canvas
   - Optionally apply drop shadow
   - Export transparent PNGs into the chosen output folder

You can stop processing at any time with `Ctrl + C`. The script exits cleanly and prints the average CPU, RAM and GPU usage collected so far.
