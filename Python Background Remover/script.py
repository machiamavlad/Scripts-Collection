import math
import sys
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import subprocess
import psutil
import shutil

if os.name == "nt":
    os.system("")

INPUT_DIR = Path("files")
OUTPUT_DIR = Path("exports")

TARGET_SIZE = 512
PAD_RATIO = 0.02

BG_BGR = np.array([153, 102, 51], dtype=np.uint8)
BG_HSV = None
BG_TOL = 2.0

FEATHER = 2.0
SCALE_FACTOR = 0.9

APPLY_SHADOW = False
SHADOW_OPACITY = 1.0
SHADOW_ANGLE = 30
SHADOW_DISTANCE = 7
SHADOW_SIZE = 30

EDGE_BLUR = False
EDGE_BLUR_RADIUS = 1.5

GPU_ENABLED = False

CPU_SAMPLES = []
RAM_SAMPLES = []
GPU_SAMPLES = []

RAM_TOTAL_MB = psutil.virtual_memory().total / (1024 * 1024)
GPU_TOTAL_MB = None

LAST_CPU = None
LAST_RAM = None
LAST_GPU = None

PROGRESS_LINES_INIT = False

COLOR_RESET = "\033[0m"
COLOR_MAIN = "\033[33m"
COLOR_CPU = "\033[31m"
COLOR_RAM = "\033[36m"
COLOR_GPU = "\033[35m"


def gpu_usage_value():
    global GPU_TOTAL_MB
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        )
        first_line = out.decode().strip().split("\n")[0]
        parts = [x.strip() for x in first_line.split(",")]
        if len(parts) < 2:
            return None
        used_str, total_str = parts[0], parts[1]
        used = int(used_str)
        if GPU_TOTAL_MB is None:
            GPU_TOTAL_MB = int(total_str)
        return used
    except Exception:
        return None


def sample_usage():
    global LAST_CPU, LAST_RAM, LAST_GPU
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().used / (1024 * 1024)
    gpu_val = gpu_usage_value()
    LAST_CPU = cpu
    LAST_RAM = ram
    LAST_GPU = gpu_val
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


def build_bar(label, frac, length, color, suffix=""):
    frac = max(0.0, min(1.0, frac))
    filled = int(length * frac)
    bar = "#" * filled + "-" * (length - filled)
    percent = int(frac * 100)
    return f"{label}: {color}[{bar}]{COLOR_RESET} {percent:3d}% {suffix}"


def print_progress(current, total, bar_len=40):
    global PROGRESS_LINES_INIT
    if total <= 0:
        return
    frac = current / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    percent = int(frac * 100)
    if not PROGRESS_LINES_INIT:
        sys.stdout.write("\n" * 4)
        PROGRESS_LINES_INIT = True
    sys.stdout.write("\033[4F")
    main_line = f"{COLOR_MAIN}Processing images:{COLOR_RESET} [{bar}] {current}/{total} ({percent}%)"
    main_line = main_line.ljust(bar_len + 40)
    cpu = LAST_CPU if LAST_CPU is not None else 0.0
    ram_used = LAST_RAM if LAST_RAM is not None else 0.0
    gpu_used = LAST_GPU
    cpu_frac = cpu / 100.0
    ram_frac = min(1.0, ram_used / RAM_TOTAL_MB) if RAM_TOTAL_MB > 0 else 0.0
    cpu_line = build_bar("CPU", cpu_frac, bar_len, COLOR_CPU, f"{cpu:.1f}%")
    ram_line = build_bar("RAM", ram_frac, bar_len, COLOR_RAM, f"{ram_used:.0f} MB/{RAM_TOTAL_MB:.0f} MB")
    if gpu_used is not None and GPU_TOTAL_MB:
        gpu_frac = min(1.0, gpu_used / GPU_TOTAL_MB)
        gpu_line = build_bar("GPU", gpu_frac, bar_len, COLOR_GPU, f"{gpu_used:.0f} MB/{GPU_TOTAL_MB:.0f} MB")
    else:
        gpu_line = "GPU: [no data or nvidia-smi missing]"
    sys.stdout.write(main_line + "\n")
    sys.stdout.write(cpu_line + "\n")
    sys.stdout.write(ram_line + "\n")
    sys.stdout.write(gpu_line + "\n")
    sys.stdout.flush()


def read_image(p: Path):
    bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return bgr


def parse_color_flexible(s: str) -> np.ndarray:
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
        r, g, b = map(int, parts)
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
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if BG_HSV is None:
        bg_hsv = cv2.cvtColor(BG_BGR.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    else:
        bg_hsv = BG_HSV
    h_bg, s_bg, v_bg = [int(x) for x in bg_hsv]
    h_tol = int(8 + BG_TOL * 0.3)
    h_tol = max(5, min(30, h_tol))
    s_min = max(60, int(s_bg * 0.5))
    v_min = max(60, int(v_bg * 0.5))
    lower = np.array([max(0, h_bg - h_tol), s_min, v_min], dtype=np.uint8)
    upper = np.array([min(179, h_bg + h_tol), 255, 255], dtype=np.uint8)
    bg_mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    bg_mask = cv2.dilate(bg_mask, kernel, iterations=1)
    alpha = 255 - bg_mask
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
        return 0, 0, w, h
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


def apply_edge_blur(bgra, radius):
    if radius <= 0:
        return bgra
    bgr = bgra[:, :, :3]
    alpha = bgra[:, :, 3]
    k = int(max(1, round(radius * 2 + 1)))
    alpha_blur = cv2.GaussianBlur(alpha, (k | 1, k | 1), radius)
    edge_mask = ((alpha_blur > 0) & (alpha_blur < 255)).astype(np.uint8) * 255
    if not np.any(edge_mask):
        return bgra
    bgr_blur = cv2.GaussianBlur(bgr, (k | 1, k | 1), radius)
    m = edge_mask.astype(np.float32) / 255.0
    m3 = m[..., None]
    out_bgr = bgr.astype(np.float32) * (1.0 - m3) + bgr_blur.astype(np.float32) * m3
    out_bgr = np.clip(out_bgr, 0, 255).astype(np.uint8)
    return np.dstack([out_bgr, alpha])


def process_one(p: Path, out_dir: Path):
    sample_usage()
    bgr = read_image(p)
    if bgr is None:
        return False, "unsupported format or read error"
    alpha_raw = build_alpha_from_color(bgr)
    fg_mask = (alpha_raw > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    if FEATHER > 0:
        alpha_soft = feather_alpha(fg_mask, FEATHER)
    else:
        alpha_soft = fg_mask
    alpha = np.where(alpha_soft > 160, 255, 0).astype(np.uint8)
    x, y, w, h = crop_subject(alpha)
    bgr_c = bgr[y:y + h, x:x + w]
    alpha_c = alpha[y:y + h, x:x + w]
    alpha_raw_c = alpha_raw[y:y + h, x:x + w]
    spill_mask = ((alpha_c == 255) & (alpha_raw_c < 240)).astype(np.uint8) * 255
    if np.any(spill_mask):
        bgr_fixed = cv2.inpaint(bgr_c, spill_mask, 3, cv2.INPAINT_TELEA)
    else:
        bgr_fixed = bgr_c
    bgra = np.dstack([bgr_fixed, alpha_c])
    out = resize_center_square(bgra, TARGET_SIZE)
    if EDGE_BLUR:
        out = apply_edge_blur(out, EDGE_BLUR_RADIUS)
        if APPLY_SHADOW:
            out = add_drop_shadow(out)
    else:
        if APPLY_SHADOW:
            out = add_drop_shadow(out)
    out_rgba = cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA)
    out_path = out_dir / (p.stem + ".png")
    try:
        Image.fromarray(out_rgba).save(out_path, "PNG", compress_level=9)
    except Exception as e:
        return False, str(e)
    return True, None


def interactive_setup():
    global INPUT_DIR, OUTPUT_DIR, BG_BGR, BG_HSV, TARGET_SIZE, APPLY_SHADOW, BG_TOL, FEATHER, SCALE_FACTOR, EDGE_BLUR, EDGE_BLUR_RADIUS, GPU_ENABLED

    print("=== Background Remover ===")
    print("Solid-color background removal with centering, stats and options.\n")

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
            BG_HSV = cv2.cvtColor(BG_BGR.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            break
        except ValueError as e:
            print(f"  Invalid color: {e}")
            print("  Accepted formats: #RRGGBB, rgb(r g b), r g b, r,g,b")

    tol_str = input(f"Color tolerance (0–50) [{default_tol}]: ").strip()
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

    blur_ans = input("Apply subtle edge blur on object edges? [y/N]: ").strip().lower()
    EDGE_BLUR = blur_ans.startswith("y")
    if EDGE_BLUR:
        blur_r = input(" Edge blur radius (0.5–3.0) [1.5]: ").strip()
        try:
            EDGE_BLUR_RADIUS = float(blur_r) if blur_r else 1.5
        except ValueError:
            EDGE_BLUR_RADIUS = 1.5
        EDGE_BLUR_RADIUS = max(0.5, min(5.0, EDGE_BLUR_RADIUS))

    gpu_ans = input("Use GPU acceleration where available? [y/N]: ").strip().lower()
    GPU_ENABLED = gpu_ans.startswith("y")

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
    print(f" Drop shadow  : {'ON' if APPLY_SHADOW else 'OFF'}")
    print(f" Edge blur    : {'ON' if EDGE_BLUR else 'OFF'}")
    print(f" GPU usage    : {'ON' if GPU_ENABLED else 'OFF'}\n")

    confirm = input("Proceed with these settings? [Y/n]: ").strip().lower()
    if confirm.startswith("n"):
        print("Cancelled by user.")
        sys.exit(0)


def notify_done(ok, fail, errors):
    if os.name != "nt":
        return
    try:
        import ctypes
        if fail == 0:
            msg = f"Background Remover finished.\n\nProcessed {ok} file(s) with no errors."
            icon = 0x40
        else:
            msg = f"Background Remover finished.\n\nProcessed {ok} file(s), {fail} failed."
            if errors:
                n, m = errors[0]
                msg += f"\n\nFirst error:\n{n}: {m[:160]}"
            icon = 0x30
        ctypes.windll.user32.MessageBoxW(0, msg, "Background Remover", icon)
    except Exception:
        pass


def main():
    interactive_setup()

    files = [p for p in INPUT_DIR.iterdir() if p.is_file()]
    total = len(files)
    ok = 0
    fail = 0
    errors = []

    if not files:
        print("\nNo files found in input folder. Nothing to do.")
        return

    print(f"\nStarting processing of {total} file(s)...")

    for i, p in enumerate(files, start=1):
        success = False
        err_msg = None
        try:
            success, err_msg = process_one(p, OUTPUT_DIR)
        except Exception as e:
            success = False
            err_msg = str(e)
        if success:
            ok += 1
        else:
            fail += 1
            errors.append((p.name, err_msg or "unknown error"))
        print_progress(i, total)

    print()
    print(f"Done: processed {ok}, failed {fail} -> {OUTPUT_DIR.resolve()}")
    print_usage_summary()

    if errors:
        print("\nFiles with errors:")
        for name, msg in errors:
            print(f" - {name}: {msg}")

    notify_done(ok, fail, errors)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C). Exiting gracefully.")
        print_usage_summary()
        sys.exit(0)
