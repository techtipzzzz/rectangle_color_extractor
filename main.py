import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import sys
import tempfile
from PIL import Image, ImageTk

# Multi-backend scanner support
SCANNER_BACKENDS = []
WIA_METHOD = None

try:
    import wia_scan
    SCANNER_BACKENDS.append("wia_scan")
    WIA_METHOD = "wia_scan"
except ImportError:
    try:
        import win32com.client
        SCANNER_BACKENDS.append("wia_com")
        WIA_METHOD = "wia_com"
    except ImportError:
        pass

try:
    import twain
    SCANNER_BACKENDS.append("twain")
except ImportError:
    pass

if not sys.platform.startswith("win"):
    try:
        import pyinsane2
        SCANNER_BACKENDS.append("sane")
    except ImportError:
        pass

from utils import process_passport_photos_v3, process_passport_photos_v4


def browse_input():
    filenames = filedialog.askopenfilenames(
        title="Select Input Images",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if filenames:
        input_listbox.delete(0, tk.END)
        for f in filenames:
            input_listbox.insert(tk.END, f)


def scan_images():
    if "wia_scan" in SCANNER_BACKENDS:
        scan_with_wia_scan()
    elif "wia_com" in SCANNER_BACKENDS:
        scan_with_wia_com()
    elif "twain" in SCANNER_BACKENDS:
        scan_with_twain()
    elif "sane" in SCANNER_BACKENDS:
        scan_with_sane()
    else:
        messagebox.showerror("Error", "No scanner backend available.")


def scan_with_wia_scan():
    try:
        import wia_scan
        device = wia_scan.prompt_choose_device_and_connect()
        scans = []
        while True:
            image = wia_scan.scan_side(device=device)
            if image is None:
                break
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(tmp.name)
            scans.append(tmp.name)
            if not messagebox.askyesno("Scan more?", "Scan another page?"):
                break
        device.disconnect()
        input_listbox.delete(0, tk.END)
        for f in scans:
            input_listbox.insert(tk.END, f)
    except Exception as e:
        messagebox.showerror("WIA Scan Error", str(e))


def scan_with_wia_com():
    try:
        import win32com.client
        wia = win32com.client.Dispatch("WIA.CommonDialog")
        device = wia.ShowSelectDevice()
        if device is None:
            messagebox.showinfo("Info", "No device selected.")
            return
        image = wia.ShowAcquire()
        tmp = tempfile.NamedTemporaryFile(suffix=".bmp", delete=False)
        image.SaveFile(tmp.name)
        input_listbox.delete(0, tk.END)
        input_listbox.insert(tk.END, tmp.name)
    except Exception as e:
        messagebox.showerror("WIA COM Error", str(e))


def scan_with_twain():
    try:
        import twain
        sm = twain.SourceManager(0)
        src = sm.OpenSource()
        src.RequestAcquire(0, 0)
        scanned = []
        info = src.XferImageNatively()
        while info:
            handle, _ = info
            img = twain.DIBToBMFile(handle)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            scanned.append(tmp.name)
            info = src.XferImageNatively()
        src.CloseSource()
        sm.Destroy()
        input_listbox.delete(0, tk.END)
        for f in scanned:
            input_listbox.insert(tk.END, f)
    except Exception as e:
        messagebox.showerror("Scan Error", str(e))


def scan_with_sane():
    try:
        import pyinsane2
        pyinsane2.init()
        devices = pyinsane2.get_devices()
        if not devices:
            raise RuntimeError("No scanners found")
        scanner = devices[0]
        scanner.options["resolution"].value = 300
        session = scanner.scan(multiple=True)
        scanned = []
        for scan in session:
            pil = scan.to_pil()
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil.save(tmp.name)
            scanned.append(tmp.name)
        pyinsane2.exit()
        input_listbox.delete(0, tk.END)
        for f in scanned:
            input_listbox.insert(tk.END, f)
    except Exception as e:
        messagebox.showerror("Scan Error", str(e))


def browse_output():
    folder = filedialog.askdirectory(title="Select Output Folder")
    if folder:
        output_entry.delete(0, tk.END)
        output_entry.insert(tk.END, folder)


def run_extraction():
    input_paths = list(input_listbox.get(0, tk.END))
    output_path = output_entry.get()
    try:
        width = int(width_entry.get())
        height = int(height_entry.get())
        max_kb = int(size_entry.get())
        padx = float(padx_entry.get()) / 100
        pady = float(pady_entry.get()) / 100
    except ValueError:
        messagebox.showerror("Error", "Check numerical fields!")
        return

    if not input_paths:
        messagebox.showerror("Error", "No input images selected.")
        return
    if not os.path.exists(output_path):
        messagebox.showerror("Error", "Invalid output folder selected.")
        return

    total = 0
    all_saved = []
    for inp in input_paths:
        base = os.path.splitext(os.path.basename(inp))[0]

        if save_mode_var.get() == "subfolder":
            subfolder = os.path.join(output_path, base)
        else:
            subfolder = output_path
        os.makedirs(subfolder, exist_ok=True)

        try:
            if method_var.get() == 'original':
                cnt, saved = process_passport_photos_v3(
                    inp, subfolder, width, height, max_kb, padx=padx, pady=pady
                )
            elif method_var.get() == 'rectangle':  # maps to tilted_rectangle
                cnt, saved = process_passport_photos_v4(
                    inp, subfolder, width, height, max_kb, padx=padx, pady=pady,
                    method="tilted_rectangle"
                )
            elif method_var.get() == 'color':  # maps to color_tilted
                cnt, saved = process_passport_photos_v4(
                    inp, subfolder, width, height, max_kb, padx=padx, pady=pady,
                    method="color_tilted"
                )
            else:
                cnt, saved = (0, [])

            if save_mode_var.get() == "single_folder":
                renamed = []
                for i, path in enumerate(saved, start=1):
                    new_name = f"{base}_photo_{i}.jpg"
                    new_path = os.path.join(output_path, new_name)
                    os.rename(path, new_path)
                    renamed.append(new_path)
                saved = renamed

            total += cnt
            all_saved.extend(saved)

        except Exception as e:
            messagebox.showerror("Error", f"{base}: {e}")

    messagebox.showinfo("Done", f"Extracted {total} photos.")
    if all_saved:
        show_preview_window(all_saved)


def show_preview_window(paths):
    win = tk.Toplevel()
    win.title("Preview")
    win.geometry("500x600")

    canvas = tk.Canvas(win)
    scrollbar = tk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame = tk.Frame(canvas)
    canvas.create_window((0,0), window=frame, anchor=tk.NW)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    for path in paths:
        try:
            img = Image.open(path)
            img.thumbnail((150, 200))
            imgtk = ImageTk.PhotoImage(img)
            label = tk.Label(frame, image=imgtk, text=os.path.basename(path),
                             compound="top", font=("Arial", 9))
            label.image = imgtk
            label.pack(pady=5)
        except:
            tk.Label(frame, text=f"Error loading {os.path.basename(path)}").pack()


def show_about():
    messagebox.showinfo("About",
        "Smart Passport Photo Cropper\n"
        "Scanner support enabled\n"
        "Methods: original, rectangle, color\n"
        "Developed by Kasim K\n"
        "HSST Computer Science\n"
        "Govt.Seethi Haji Memorial HSS, Edavanna"
    )


# ---------------- GUI ----------------
root = tk.Tk()
root.title("Smart Passport Photo Cropper")

mb = tk.Menu(root)
root.config(menu=mb)
file_menu = tk.Menu(mb, tearoff=0)
file_menu.add_command(label="Exit", command=root.quit)
mb.add_cascade(label="File", menu=file_menu)
help_menu = tk.Menu(mb, tearoff=0)
help_menu.add_command(label="About", command=show_about)
mb.add_cascade(label="Help", menu=help_menu)

tk.Label(root, text="Input Images:").grid(row=0, column=0, sticky="nw")
input_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50, height=5)
input_listbox.grid(row=0, column=1, sticky="nwes", padx=5, pady=5)
tk.Button(root, text="Browse...", command=browse_input).grid(row=0, column=2, padx=5)
tk.Button(root, text="Scan...", command=scan_images).grid(row=0, column=3, padx=5)

tk.Label(root, text="Output Folder:").grid(row=1, column=0, sticky="w")
output_entry = tk.Entry(root, width=50)
output_entry.grid(row=1, column=1, sticky="w", padx=5)
tk.Button(root, text="Browse...", command=browse_output).grid(row=1, column=2, padx=5)

tk.Label(root, text="Width:").grid(row=2, column=0, sticky="w")
width_entry = tk.Entry(root, width=10); width_entry.insert(0,"150"); width_entry.grid(row=2, column=1, sticky="w", padx=5)
tk.Label(root, text="Height:").grid(row=3, column=0, sticky="w")
height_entry = tk.Entry(root, width=10); height_entry.insert(0,"200"); height_entry.grid(row=3, column=1, sticky="w", padx=5)

tk.Label(root, text="Max KB:").grid(row=4, column=0, sticky="w")
size_entry = tk.Entry(root, width=10); size_entry.insert(0,"40"); size_entry.grid(row=4, column=1, sticky="w", padx=5)

tk.Label(root, text="Pad X:").grid(row=5, column=0, sticky="w")
padx_entry = tk.Entry(root, width=10); padx_entry.insert(0,"0"); padx_entry.grid(row=5, column=1, sticky="w", padx=5)

tk.Label(root, text="Pad Y:").grid(row=6, column=0, sticky="w")
pady_entry = tk.Entry(root, width=10); pady_entry.insert(0,"0"); pady_entry.grid(row=6, column=1, sticky="w", padx=5)

tk.Label(root, text="Method:").grid(row=7, column=0, sticky="w")
method_var = tk.StringVar(value="color")
ttk.Combobox(root, textvariable=method_var, values=[
    "rectangle","color","original"
], width=15).grid(row=7, column=1, sticky="w")

tk.Label(root, text="Save Mode:").grid(row=9, column=0, sticky="w")
save_mode_var = tk.StringVar(value="subfolder")
ttk.Combobox(root, textvariable=save_mode_var,
             values=["subfolder", "single_folder"], width=15).grid(row=9, column=1, sticky="w")

tk.Button(root, text="Run", command=run_extraction).grid(row=10, column=0, columnspan=4, pady=10)

root.mainloop()

