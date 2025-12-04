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

EDGE_BLUR_ENABLED = False
EDGE_BLUR_RADIUS = 1.0

USE_GPU = False
HAS_CUDA = False

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

ERRORS = []


def init_gpu():
    global HAS_CUDA, GPU_TOTAL_MB
    if not hasattr(cv2, "cuda"):
        HAS_CUDA = False
        return
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        HAS_CUDA = False
        return
    if count <= 0:
        HAS_CUDA = False
        return
    HAS_CUDA = True
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            )
            first_line = out.decode().strip().split("\n")[0]
            GPU_TOTAL_MB = int(first_line.strip())
        except Exception:
            GPU_TOTAL_MB = None


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
    if ERRORS:
        print("\nFiles with errors:")
        for name, msg in ERRORS:
            print(f" - {name}: {msg}")


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


def build_alpha_from_color_cpu(bgr):
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


def build_alpha_from_color_gpu(bgr):
    if not HAS_CUDA:
        return build_alpha_from_color_cpu(bgr)
    gpu_bgr = cv2.cuda_GpuMat()
    gpu_bgr.upload(bgr)
    gpu_hsv = cv2.cuda.cvtColor(gpu_bgr, cv2.COLOR_BGR2HSV)
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
    gpu_lower = cv2.cuda_GpuMat()
    gpu_upper = cv2.cuda_GpuMat()
    gpu_lower.upload(lower.reshape(1, 1, 3))
    gpu_upper.upload(upper.reshape(1, 1, 3))
    bg_mask_gpu = cv2.cuda.inRange(gpu_hsv, gpu_lower, gpu_upper)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, bg_mask_gpu.type(), kernel)
    bg_mask_gpu = morph.apply(bg_mask_gpu)
    bg_mask = bg_mask_gpu.download()
    alpha = 255 - bg_mask
    return alpha


def build_alpha_from_color(bgr):
    if USE_GPU and HAS_CUDA:
        try:
            return build_alpha_from_color_gpu(bgr)
        except Exception:
            return build_alpha_from_color_cpu(bgr)
    return build_alpha_from_color_cpu(bgr)


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
    x0 = max(0, x1 - pad)
    y0 = max(0, y1 - pad)
    x2p = min(w - 1, x2 + pad)
    y2p = min(h - 1, y2 + pad)
    return x0, y0, x2p - x0 + 1, y2p - y0 + 1


def resize_center_square(rgba, size):
    h, w = rgba.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 4), dtype=np.uint8)
    s = min(size / w, size / h) * SCALE_FACTOR
    nw = int(math.ceil(w * s))
    nh = int(math.ceil(h * s))
    nw = max(1, min(size, nw))
    nh = max(1, min(size, nh))
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


def apply_edge_blur(bgr, alpha):
    edges = cv2.Canny(alpha, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    mask = edges.astype(np.float32) / 255.0
    if mask.max() == 0:
        return bgr
    blurred = cv2.GaussianBlur(bgr, (5, 5), EDGE_BLUR_RADIUS)
    mask3 = mask[..., None]
    out = blurred.astype(np.float32) * mask3 + bgr.astype(np.float32) * (1.0 - mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def process_one(p: Path, out_dir: Path):
    sample_usage()
    bgr = read_image(p)
    if bgr is None:
        return False
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

    if EDGE_BLUR_ENABLED:
        bgr_post = apply_edge_blur(bgr_fixed, alpha_c)
    else:
        bgr_post = bgr_fixed

    bgra = np.dstack([bgr_post, alpha_c])

    if APPLY_SHADOW:
        bgra = add_drop_shadow(bgra)

    out = resize_center_square(bgra, TARGET_SIZE)
    out_rgba = cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA)
    out_path = out_dir / (p.stem + ".png")
    Image.fromarray(out_rgba).save(out_path, "PNG", compress_level=9)
    return True


def interactive_setup():
    global INPUT_DIR, OUTPUT_DIR, BG_BGR, BG_HSV, TARGET_SIZE, APPLY_SHADOW, BG_TOL, FEATHER, SCALE_FACTOR, EDGE_BLUR_ENABLED, EDGE_BLUR_RADIUS, USE_GPU
    print("=== Background Remover ===")
    print("Simple solid-color background removal,")
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
            bg = parse_color_flexible(c)
            BG_BGR[:] = bg
            BG_HSV = cv2.cvtColor(BG_BGR.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            break
        except ValueError as e:
            print(f"  Invalid color: {e}")
            print("  Accepted formats: #RRGGBB, rgb(r g b), r g b, r,g,b")
    tol_str = input(f"Color tolerance (0-50) [{default_tol}]: ").strip()
    try:
        BG_TOL = float(tol_str) if tol_str else default_tol
    except ValueError:
        BG_TOL = default_tol
    feather_str = input(f"Feather radius for edges (0-10) [{default_feather}]: ").strip()
    try:
        FEATHER = float(feather_str) if feather_str else default_feather
    except ValueError:
        FEATHER = default_feather
    scale_str = input(f"Object scale inside icon (0.5-1.0) [{default_scale}]: ").strip()
    try:
        SCALE_FACTOR = float(scale_str) if scale_str else default_scale
    except ValueError:
        SCALE_FACTOR = default_scale
    SCALE_FACTOR = max(0.5, min(1.0, SCALE_FACTOR))
    shadow_ans = input("Add drop shadow (Photoshop style)? [Y/n]: ").strip().lower()
    APPLY_SHADOW = (shadow_ans == "" or shadow_ans.startswith("y"))
    edge_ans = input("Apply subtle edge blur? [y/N]: ").strip().lower()
    EDGE_BLUR_ENABLED = edge_ans.startswith("y")
    if EDGE_BLUR_ENABLED:
        radius_str = input(f"Edge blur strength (0.5-3.0) [{EDGE_BLUR_RADIUS}]: ").strip()
        try:
            EDGE_BLUR_RADIUS = float(radius_str) if radius_str else EDGE_BLUR_RADIUS
        except ValueError:
            EDGE_BLUR_RADIUS = 1.0
        EDGE_BLUR_RADIUS = max(0.5, min(3.0, EDGE_BLUR_RADIUS))
    gpu_ans = input("Use GPU acceleration if available? [y/N]: ").strip().lower()
    if gpu_ans.startswith("y"):
        init_gpu()
        USE_GPU = HAS_CUDA
        if USE_GPU:
            print("GPU acceleration enabled (if OpenCV CUDA is available).")
        else:
            print("GPU acceleration not available in this OpenCV build. Using CPU.")
    else:
        USE_GPU = False
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
    print(f" Edge blur    : {'ON' if EDGE_BLUR_ENABLED else 'OFF'}")
    print(f" GPU usage    : {'ON' if USE_GPU else 'OFF'}\n")
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
            ERRORS.append((p.name, str(e)))
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
