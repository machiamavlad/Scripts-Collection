import math
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import subprocess
import psutil
import shutil

INPUT_DIR = Path("files")
OUTPUT_DIR = Path("exports")

TARGET_SIZE = 512
PAD_RATIO = 0.02

# Background color in BGR, default #336699
BG_BGR = np.array([153, 102, 51], dtype=np.uint8)
BG_TOL = 2.0           # per-channel tolerance (simplu, ca la inceput)

# alpha processing
FEATHER = 2.0          # blur radius on alpha
SCALE_FACTOR = 0.9     # cat din icon sa ocupe obiectul (0..1)

# drop shadow config (Photoshop-like)
APPLY_SHADOW = False
SHADOW_OPACITY = 1.0   # 1.0 = 100%
SHADOW_ANGLE = 30      # degrees
SHADOW_DISTANCE = 7    # px
SHADOW_SIZE = 30       # blur radius (approx)

# Resource stats
CPU_SAMPLES = []
RAM_SAMPLES = []
GPU_SAMPLES = []  # MB VRAM


def gpu_usage_value():
    """Returns used VRAM in MB or None if nvidia-smi is not available."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        )
        first_line = out.decode().strip().split("\n")[0]
        return int(first_line)
    except Exception:
        return None


def sample_usage():
    """Save a CPU/RAM/GPU usage sample for the final averages."""
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().used / (1024 * 1024)  # MB
    gpu_val = gpu_usage_value()

    CPU_SAMPLES.append(cpu)
    RAM_SAMPLES.append(ram)
    if gpu_val is not None:
        GPU_SAMPLES.append(gpu_val)


def print_usage_summary():
    print("\n=== Background Remover average resource usage ===")
    if CPU_SAMPLES:
        avg_cpu = sum(CPU_SAMPLES) / len(CPU_SAMPLES)
        print(f" Average CPU: {avg_cpu:.1f}%")
    else:
        print(" Average CPU: no samples")

    if RAM_SAMPLES:
        avg_ram = sum(RAM_SAMPLES) / len(RAM_SAMPLES)
        print(f" Average RAM: {avg_ram:.0f} MB")
    else:
        print(" Average RAM: no samples")

    if GPU_SAMPLES:
        avg_gpu = sum(GPU_SAMPLES) / len(GPU_SAMPLES)
        print(f" Average GPU VRAM: {avg_gpu:.0f} MB")
    else:
        print(" Average GPU VRAM: no data (nvidia-smi not available)")


def print_progress(current, total, bar_len=40):
    """Simple pip style progress bar."""
    if total <= 0:
        return
    frac = current / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    percent = int(frac * 100)
    sys.stdout.write(f"\rProcessing images: [{bar}] {current}/{total} ({percent}%)")
    sys.stdout.flush()


def read_image(p: Path):
    bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return bgr


def parse_color_flexible(s: str) -> np.ndarray:
    """
    Accepts:
      - '#336699'
      - '336699'
      - 'rgb(0 255 0)', 'rgb(0,255,0)'
      - '0 255 0', '0,255,0'
    Returns color in BGR (np.uint8[3]) for OpenCV.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty color string")

    lower = s.lower()

    if lower.startswith("rgb"):
        if "(" in s and ")" in s:
            inside = s[s.find("(") + 1:s.rfind(")")]
        else:
            inside = s[3:]
        parts = [p for p in inside.replace(",", " ").split() if p]
        if len(parts) != 3:
            raise ValueError("rgb() must have 3 components")
        try:
            r, g, b = map(int, parts)
        except ValueError:
            raise ValueError("rgb components must be integers")
        for v in (r, g, b):
            if not (0 <= v <= 255):
                raise ValueError("rgb values must be between 0 and 255")
        return np.array([b, g, r], dtype=np.uint8)

    if "," in s or " " in s:
        parts = [p for p in s.replace(",", " ").split() if p]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            r, g, b = map(int, parts)
            for v in (r, g, b):
                if not (0 <= v <= 255):
                    raise ValueError("rgb values must be between 0 and 255")
            return np.array([b, g, r], dtype=np.uint8)

    hex_s = s.lstrip("#")
    if len(hex_s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in hex_s):
        raise ValueError("Invalid hex color (expected #RRGGBB)")
    r = int(hex_s[0:2], 16)
    g = int(hex_s[2:4], 16)
    b = int(hex_s[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)


def bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(x) for x in bgr]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def build_alpha_from_color(bgr):
    """
    Simple color-threshold background removal,
    exactly ca in varianta initiala: comparam fiecare pixel cu BG_BGR
    si facem alpha 0 pentru ce e „aproape” de culoarea de background.
    """
    diff = bgr.astype(np.int16) - BG_BGR.reshape(1, 1, 3).astype(np.int16)
    mask_bg = np.all(np.abs(diff) <= BG_TOL, axis=2)
    alpha = np.where(mask_bg, 0, 255).astype(np.uint8)
    return alpha


def feather_alpha(alpha, radius):
    if radius <= 0:
        return alpha
    k = int(max(1, round(radius * 2 + 1)))
    return cv2.GaussianBlur(alpha, (k | 1, k | 1), radius)


def crop_subject(alpha):
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        h, w = alpha.shape
        return (0, 0, w, h)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = alpha.shape
    pad = int(max(w, h) * PAD_RATIO)

    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w - 1, x2 + pad) - (x1 - pad) + 1,
        min(h - 1, y2 + pad) - (y1 - pad) + 1
    )


def resize_center_square(rgba, size):
    """Scale the object to fit in a square and center it."""
    h, w = rgba.shape[:2]
    s = min(size / w, size / h) * SCALE_FACTOR
    nw, nh = int(math.ceil(w * s)), int(math.ceil(h * s))

    resized = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def add_drop_shadow(bgra):
    """Add Photoshop-style drop shadow behind the object."""
    h, w = bgra.shape[:2]
    alpha = bgra[:, :, 3]

    if alpha.max() == 0:
        return bgra

    shadow_alpha = cv2.GaussianBlur(alpha, (0, 0), SHADOW_SIZE / 2.0)
    shadow_alpha = (shadow_alpha.astype(np.float32) * SHADOW_OPACITY)
    shadow_alpha = np.clip(shadow_alpha, 0, 255).astype(np.uint8)

    angle_rad = math.radians(SHADOW_ANGLE)
    dx = SHADOW_DISTANCE * math.cos(angle_rad)
    dy = SHADOW_DISTANCE * math.sin(angle_rad)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shadow_alpha_shifted = cv2.warpAffine(
        shadow_alpha, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    )

    shadow = np.zeros_like(bgra)
    shadow[:, :, 3] = shadow_alpha_shifted

    result = shadow.copy()
    obj_mask = alpha > 0

    for c in range(3):
        result[:, :, c][obj_mask] = bgra[:, :, c][obj_mask]

    result[:, :, 3] = np.maximum(result[:, :, 3], alpha)
    return result


def process_one(p: Path, out_dir: Path):
    sample_usage()

    bgr = read_image(p)
    if bgr is None:
        print(f"\nSkip {p.name}: unsupported format or read error")
        return False

    alpha = build_alpha_from_color(bgr)

    alpha = feather_alpha(alpha, FEATHER)

    x, y, w, h = crop_subject(alpha)
    bgr_c = bgr[y:y + h, x:x + w]
    alpha_c = alpha[y:y + h, x:x + w]

    bgra = np.dstack([bgr_c, alpha_c])
    out = resize_center_square(bgra, TARGET_SIZE)

    if APPLY_SHADOW:
        out = add_drop_shadow(out)

    out_rgba = cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA)
    out_path = out_dir / (p.stem + ".png")
    Image.fromarray(out_rgba).save(out_path, "PNG", compress_level=9)
    return True


def interactive_setup():
    global INPUT_DIR, OUTPUT_DIR, BG_BGR, TARGET_SIZE, APPLY_SHADOW, BG_TOL, FEATHER, SCALE_FACTOR

    print("=== Background Remover ===")
    print("Simple solid-color background removal (first-version logic),")
    print("with centering, optional drop shadow and resource stats.\n")

    default_input = "files"
    default_output = "exports"
    default_color = "#336699"
    default_size = 512
    default_tol = 2.0
    default_feather = 2.0
    default_scale = 0.9

    inp = input(f"Input folder with renders [{default_input}]: ").strip()
    if not inp:
        inp = default_input
    INPUT_DIR = Path(inp)

    out = input(f"Output folder for PNG icons [{default_output}]: ").strip()
    if not out:
        out = default_output
    OUTPUT_DIR = Path(out)

    size_str = input(f"Icon size (square, px) [{default_size}]: ").strip()
    if size_str.isdigit():
        TARGET_SIZE = int(size_str)
    else:
        TARGET_SIZE = default_size

    while True:
        c = input(f"Background color to remove [{default_color}]: ").strip()
        if not c:
            c = default_color
        try:
            BG = parse_color_flexible(c)
            BG_BGR[:] = BG
            break
        except ValueError as e:
            print(f"  Invalid color: {e}")
            print("  Accepted formats: #RRGGBB, rgb(r g b), r g b, r,g,b")

    tol_str = input(f"Color tolerance per channel (0–50) [{default_tol}]: ").strip()
    try:
        BG_TOL = float(tol_str) if tol_str else default_tol
    except ValueError:
        BG_TOL = default_tol

    feather_str = input(f"Feather radius for edges (0–10) [{default_feather}]: ").strip()
    try:
        FEATHER = float(feather_str) if feather_str else default_feather
    except ValueError:
        FEATHER = default_feather

    scale_str = input(f"Object scale inside icon (0.5–1.0) [{default_scale}]: ").strip()
    try:
        SCALE_FACTOR = float(scale_str) if scale_str else default_scale
    except ValueError:
        SCALE_FACTOR = default_scale
    SCALE_FACTOR = max(0.5, min(1.0, SCALE_FACTOR))

    shadow_ans = input("Add drop shadow (Photoshop style)? [Y/n]: ").strip().lower()
    APPLY_SHADOW = (shadow_ans == "" or shadow_ans.startswith("y"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Final configuration ===")
    print(f" Input folder : {INPUT_DIR.resolve()}")
    print(f" Output folder: {OUTPUT_DIR.resolve()}")
    print(f" Icon size    : {TARGET_SIZE} px")
    print(f" BG color     : {bgr_to_hex(BG_BGR)} (BGR={BG_BGR.tolist()})")
    print(f" Color tol    : {BG_TOL}")
    print(f" Feather      : {FEATHER}")
    print(f" Scale factor : {SCALE_FACTOR}")
    print(f" Drop shadow  : {'ON' if APPLY_SHADOW else 'OFF'}\n")

    confirm = input("Proceed with these settings? [Y/n]: ").strip().lower()
    if confirm.startswith("n"):
        print("Cancelled by user.")
        sys.exit(0)



def main():
    interactive_setup()

    files = [p for p in INPUT_DIR.iterdir() if p.is_file()]
    total = len(files)
    ok, fail = 0, 0

    if not files:
        print("\nNo files found in input folder. Nothing to do.")
        return

    print(f"\nStarting processing of {total} file(s)...")

    for i, p in enumerate(files, start=1):
        try:
            if process_one(p, OUTPUT_DIR):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"\nError while processing {p.name}: {e}")
            fail += 1

        print_progress(i, total)

    print()
    print(f"Done: processed {ok}, failed {fail} -> {OUTPUT_DIR.resolve()}")
    print_usage_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C). Exiting gracefully.")
        print_usage_summary()
        sys.exit(0)
