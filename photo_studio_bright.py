import os
import sys
import math
import shutil
import subprocess
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import numpy as np
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

RS = Image.Resampling
TR = Image.Transpose

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
# Bright phone-style interface
BG = "#f5f7fb"
PANEL = "#ffffff"
CARD = "#eef2ff"
HOVER = "#e2e7ff"
ACCENT = "#6c5ce7"
ACCENT_H = "#8577f2"
SLIDER = "#3b82f6"
SLIDER_H = "#60a5fa"
FG = "#202124"
MUTED = "#6b7280"
CANVAS_BG = "#e9edf5"
BORDER = "#dfe3ec"
SUCCESS = "#22c55e"
DANGER = "#ef4444"

if sys.platform.startswith("win"):
    FAMILY = "Segoe UI"
elif sys.platform == "darwin":
    FAMILY = "Helvetica Neue"
else:
    FAMILY = "DejaVu Sans"

FONT = (FAMILY, 10)
FONT_S = (FAMILY, 9)
FONT_B = (FAMILY, 10, "bold")
FONT_H = (FAMILY, 14, "bold")

PROXY_MAX = 1200          # preview resolution for live sliders
UNDO_LIMIT = 25
PALETTE = ["#ffffff", "#000000", "#ff3b30", "#ff9500", "#ffd60a",
           "#34c759", "#0a84ff", "#bf5af2", "#ff2d92"]


# --------------------------------------------------------------------------
# Image helpers (pure functions)
# --------------------------------------------------------------------------
def load_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf", "LiberationSans-Bold.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def inscribed_size(w, h, angle):
    """Largest axis-aligned rectangle inside a w x h rectangle rotated by angle (radians)."""
    angle = abs(angle) % math.pi
    if angle > math.pi / 2:
        angle = math.pi - angle
    if angle < 1e-9:
        return w, h
    long_side, short_side = (w, h) if w >= h else (h, w)
    sin_a, cos_a = math.sin(angle), math.cos(angle)
    if short_side <= 2 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side
        wr, hr = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


def straighten(img, deg):
    """Rotate by `deg` and auto-crop so no empty corners remain."""
    if abs(deg) < 0.01:
        return img
    w, h = img.size
    rot = img.rotate(deg, resample=Image.Resampling.BICUBIC, expand=True)
    wr, hr = inscribed_size(w, h, math.radians(deg))
    wr, hr = max(2, wr - 2), max(2, hr - 2)
    cx, cy = rot.width / 2, rot.height / 2
    box = (int(cx - wr / 2), int(cy - hr / 2), int(cx + wr / 2), int(cy + hr / 2))
    return rot.crop(box)


def _luma(a):
    return (a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114)[..., None]


ADJ_ITEMS = [
    ("exposure", "Exposure"), ("brightness", "Brightness"), ("contrast", "Contrast"),
    ("highlights", "Highlights"), ("shadows", "Shadows"), ("saturation", "Saturation"),
    ("warmth", "Warmth"), ("tint", "Tint"), ("sharpness", "Sharpness"),
    ("vignette", "Vignette"),
]


def apply_adjustments(img, adj):
    """adj: dict of slider values in -100..100."""
    if not any(adj.values()):
        return img
    a = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    if adj["exposure"]:
        a = a * (2.0 ** (adj["exposure"] / 100.0 * 1.5))
    if adj["brightness"]:
        a = a + adj["brightness"] / 100.0 * 0.3
    if adj["contrast"]:
        a = (a - 0.5) * (1.0 + adj["contrast"] / 100.0 * 0.8) + 0.5
    if adj["highlights"]:
        m = np.clip((np.clip(_luma(a), 0, 1) - 0.5) * 2.0, 0, 1)
        a = a + m * (adj["highlights"] / 100.0) * 0.35
    if adj["shadows"]:
        m = np.clip(1.0 - np.clip(_luma(a), 0, 1) * 2.0, 0, 1)
        a = a + m * (adj["shadows"] / 100.0) * 0.35
    if adj["saturation"]:
        g = _luma(a)
        a = g + (a - g) * (1.0 + adj["saturation"] / 100.0)
    if adj["warmth"]:
        w = adj["warmth"] / 100.0
        a[..., 0] += w * 0.10
        a[..., 2] -= w * 0.10
    if adj["tint"]:
        t = adj["tint"] / 100.0
        a[..., 1] -= t * 0.08
        a[..., 0] += t * 0.03
        a[..., 2] += t * 0.03
    if adj["vignette"]:
        h, w = a.shape[:2]
        xx = ((np.arange(w, dtype=np.float32) - w / 2) / (w / 2)) ** 2
        yy = ((np.arange(h, dtype=np.float32) - h / 2) / (h / 2)) ** 2
        d2 = ((yy[:, None] + xx[None, :]) / 2.0)[..., None]
        v = adj["vignette"] / 100.0
        if v > 0:
            a = a * (1.0 - v * 0.9 * d2)
        else:
            a = a + (-v) * 0.6 * d2

    out = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    s = adj["sharpness"]
    if s:
        m = max(out.size)
        if s > 0:
            out = out.filter(ImageFilter.UnsharpMask(radius=1 + m / 1500.0,
                                                     percent=int(s * 2.5), threshold=2))
        else:
            out = out.filter(ImageFilter.GaussianBlur(-s / 100.0 * 3 * m / 1000.0))
    return out


def _scale_channels(img, r=1.0, g=1.0, b=1.0):
    a = np.asarray(img, dtype=np.float32) * np.array([r, g, b], dtype=np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _vivid(i):
    return ImageEnhance.Contrast(ImageEnhance.Color(i).enhance(1.6)).enhance(1.12)


def _sepia(i):
    return ImageOps.colorize(ImageOps.grayscale(i), black=(45, 30, 15),
                             white=(255, 235, 180)).convert("RGB")


def _vintage(i):
    i = ImageEnhance.Color(i).enhance(0.65)
    i = _scale_channels(i, 1.06, 1.0, 0.86)
    return ImageEnhance.Contrast(i).enhance(0.92)


def _fade(i):
    a = np.asarray(i, dtype=np.float32) * 0.85 + 32
    return ImageEnhance.Color(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))).enhance(0.8)


def _noir(i):
    return ImageEnhance.Contrast(ImageOps.grayscale(i).convert("RGB")).enhance(1.5)


def _dramatic(i):
    i = ImageEnhance.Contrast(i).enhance(1.5)
    i = ImageEnhance.Color(i).enhance(0.8)
    return ImageEnhance.Sharpness(i).enhance(1.6)


def _blur(i):
    return i.filter(ImageFilter.GaussianBlur(max(i.size) / 250.0))


def _sharpen(i):
    return i.filter(ImageFilter.UnsharpMask(radius=1 + max(i.size) / 700.0, percent=180, threshold=2))


FILTERS = {
    "Original": lambda i: i,
    "Vivid": _vivid,
    "Warm": lambda i: _scale_channels(i, 1.12, 1.0, 0.88),
    "Cool": lambda i: _scale_channels(i, 0.88, 1.0, 1.12),
    "Mono": lambda i: ImageOps.grayscale(i).convert("RGB"),
    "Noir": _noir,
    "Sepia": _sepia,
    "Vintage": _vintage,
    "Fade": _fade,
    "Dramatic": _dramatic,
    "Posterize": lambda i: ImageOps.posterize(i, 3),
    "Sharpen": _sharpen,
    "Blur": _blur,
    "Emboss": lambda i: i.filter(ImageFilter.EMBOSS).convert("RGB"),
    "Edges": lambda i: ImageOps.grayscale(i).filter(ImageFilter.FIND_EDGES).convert("RGB"),
}


def run_filter(img, name, strength=100):
    img = img.convert("RGB")
    out = FILTERS[name](img)
    if strength < 100:
        out = Image.blend(img, out, strength / 100.0)
    return out


# --------------------------------------------------------------------------
# Scrollable side panel
# --------------------------------------------------------------------------
class ScrollFrame(tk.Frame):
    def __init__(self, parent, bg):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, width=300)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self.win, width=e.width))
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
class PhotoStudio:
    NAV = [("crop", "✂", "Crop"), ("rotate", "⟳", "Rotate"), ("adjust", "☀", "Adjust"),
           ("filters", "✦", "Filters"), ("draw", "✎", "Draw"), ("text", "T", "Text"),
           ("eraser", "⌫", "Eraser"), ("more", "⋯", "More")]

    RATIOS = [("Free", None), ("Original", "orig"), ("1:1", 1.0), ("4:3", 4 / 3),
              ("3:4", 3 / 4), ("16:9", 16 / 9), ("9:16", 9 / 16), ("3:2", 3 / 2),
              ("2:3", 2 / 3), ("5:4", 5 / 4)]

    def __init__(self, root):
        self.root = root
        root.title("Photo Editor")
        root.geometry("1280x820")
        root.minsize(980, 640)
        root.configure(bg=BG)

        # image state
        self.image = None
        self.original = None
        self.file_path = None
        self.save_path = None
        self.version = 0
        self.dirty = False
        self.undo_stack, self.redo_stack = [], []
        self._proxy = None
        self._proxy_ver = -1

        # view state
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.map = (0.0, 0.0, 1.0, 1.0)  # ox, oy, scale, fit
        self.photo = None
        self.preview = None              # (PIL image, logical size) or None
        self._jobs = {}
        self._pan_drag = None

        # tool state
        self.mode = "adjust"
        self.adj = {k: 0 for k, _ in ADJ_ITEMS}
        self.filter_name = None
        self.filter_strength = 100.0
        self.angle = 0.0
        self.crop = None
        self.crop_ratio = None
        self.crop_state = None
        self.color = "#ff3b30"
        self.draw_tool = tk.StringVar(value="pen")
        self.fill_shapes = tk.BooleanVar(value=False)
        self.draw_size = 8.0
        self.text_var = tk.StringVar(value="Hello")
        self.text_size = 50.0
        self.text_outline = tk.BooleanVar(value=True)
        self.erase_size = 30.0
        self._stroke = None
        self._mask = None
        self._mdraw = None
        self._drawer = None
        self._thumbs = []
        self._cv_warned = False

        self.setup_style()
        self.build_ui()
        self.bind_events()
        self.select_mode("adjust")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(150, self.refresh)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("Photo.Horizontal.TScale", background=SLIDER, troughcolor="#dbeafe",
                    bordercolor="#dbeafe", lightcolor=SLIDER_H, darkcolor=SLIDER,
                    sliderlength=18)
        s.configure("Vertical.TScrollbar", background="#d8deea", troughcolor="#f4f6fa",
                    bordercolor="#f4f6fa", arrowcolor=MUTED)
        s.configure("TCheckbutton", background=PANEL, foreground=FG)

    def mkbtn(self, parent, text, cmd, accent=False, **kw):
        normal = ACCENT if accent else CARD
        hover = ACCENT_H if accent else HOVER
        fg = "white" if accent else FG
        b = tk.Button(parent, text=text, command=cmd,
                      bg=normal, fg=fg,
                      activebackground=hover, activeforeground="white" if accent else FG,
                      relief="flat", bd=0, padx=13, pady=8,
                      font=FONT, cursor="hand2", **kw)
        # Small hover response makes the interface feel more alive.
        b.bind("<Enter>", lambda e: b.config(bg=hover))
        b.bind("<Leave>", lambda e: b.config(bg=normal))
        return b

    def build_ui(self):
        r = self.root
        r.grid_rowconfigure(1, weight=1)
        r.grid_columnconfigure(1, weight=1)

        # Top app bar -----------------------------------------------------
        top = tk.Frame(r, bg=PANEL, height=68,
                       highlightbackground=BORDER, highlightthickness=1)
        top.grid(row=0, column=0, columnspan=3, sticky="ew")
        top.grid_propagate(False)

        brand = tk.Frame(top, bg=PANEL)
        brand.pack(side="left", padx=16, pady=10)

        logo = tk.Label(brand, text="✦", bg=ACCENT, fg="white",
                        font=(FAMILY, 14, "bold"), width=2, height=1)
        logo.pack(side="left", padx=(0, 9))

        titles = tk.Frame(brand, bg=PANEL)
        titles.pack(side="left")
        tk.Label(titles, text="Photo Editor", bg=PANEL, fg=FG,
                 font=(FAMILY, 15, "bold")).pack(anchor="w")
        tk.Label(titles, text="Photo editor", bg=PANEL, fg=MUTED,
                 font=(FAMILY, 8)).pack(anchor="w")

        left = tk.Frame(top, bg=PANEL)
        left.pack(side="left", padx=8, pady=10)
        self.mkbtn(left, "＋  Open", self.open_image, accent=True).pack(side="left", padx=3)
        self.mkbtn(left, "↶", self.undo).pack(side="left", padx=3)
        self.mkbtn(left, "↷", self.redo).pack(side="left", padx=3)

        cmp_btn = self.mkbtn(left, "◐  Compare", lambda: None)
        cmp_btn.pack(side="left", padx=(7, 3))
        cmp_btn.bind("<ButtonPress-1>", lambda e: self.show_original())
        cmp_btn.bind("<ButtonRelease-1>", lambda e: self.refresh())

        mid = tk.Frame(top, bg=PANEL)
        mid.pack(side="left", padx=8)
        self.mkbtn(mid, "−", lambda: self.zoom_by(1 / 1.25)).pack(side="left", padx=2)
        self.mkbtn(mid, "Fit", self.fit_view).pack(side="left", padx=2)
        self.mkbtn(mid, "+", lambda: self.zoom_by(1.25)).pack(side="left", padx=2)

        right = tk.Frame(top, bg=PANEL)
        right.pack(side="right", padx=14)
        self.mkbtn(right, "↺  Revert", self.revert).pack(side="left", padx=3)
        self.mkbtn(right, "Save As", self.save_as).pack(side="left", padx=3)
        self.mkbtn(right, "✓  Save", self.save_image, accent=True).pack(side="left", padx=3)

        # Left editing navigation ---------------------------------------
        nav = tk.Frame(r, bg=PANEL, width=96,
                       highlightbackground=BORDER, highlightthickness=1)
        nav.grid(row=1, column=0, sticky="ns")
        nav.grid_propagate(False)

        nav_title = tk.Label(nav, text="EDIT", bg=PANEL, fg=MUTED,
                             font=(FAMILY, 8, "bold"))
        nav_title.pack(fill="x", pady=(12, 7))

        self.nav_btns = {}
        for key, icon, label in self.NAV:
            b = tk.Button(nav, text=f"{icon}\n{label}", bg=PANEL, fg=MUTED,
                          activebackground=CARD, activeforeground=ACCENT,
                          relief="flat", bd=0, font=FONT_S, pady=9,
                          cursor="hand2", command=lambda k=key: self.select_mode(k))
            b.pack(fill="x", padx=7, pady=2)
            self.nav_btns[key] = b

            b.bind("<Enter>", lambda e, btn=b: btn.config(
                bg=HOVER if btn.cget("bg") != ACCENT else ACCENT_H))
            b.bind("<Leave>", lambda e, btn=b: btn.config(
                bg=ACCENT if self.mode == next(k for k,v in self.nav_btns.items() if v is btn)
                else PANEL))

        # Main image canvas ----------------------------------------------
        canvas_wrap = tk.Frame(r, bg=CANVAS_BG)
        canvas_wrap.grid(row=1, column=1, sticky="nsew")
        self.canvas = tk.Canvas(canvas_wrap, bg=CANVAS_BG,
                                highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Right controls -------------------------------------------------
        self.panel = ScrollFrame(r, PANEL)
        self.panel.grid(row=1, column=2, sticky="ns")
        self.p = self.panel.inner

        # Bottom status / action bar ------------------------------------
        bar = tk.Frame(r, bg=PANEL, height=46,
                       highlightbackground=BORDER, highlightthickness=1)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew")
        bar.grid_propagate(False)

        self.status_l = tk.Label(bar, text="Open a photo to start  •  Ctrl+O",
                                 bg=PANEL, fg=MUTED, font=FONT_S)
        self.status_l.pack(side="left", padx=14)

        # Phone-like quick actions remain available without changing
        # the underlying editor functions.
        quick = tk.Frame(bar, bg=PANEL)
        quick.pack(side="left", padx=25)
        self.mkbtn(quick, "♡ Favourite", self.favorite_image).pack(side="left", padx=3)
        self.mkbtn(quick, "✎ Edit", lambda: self.select_mode("adjust"),
                   accent=True).pack(side="left", padx=3)
        self.mkbtn(quick, "↗ Share", self.share_image).pack(side="left", padx=3)

        self.status_r = tk.Label(bar, text="", bg=PANEL, fg=MUTED, font=FONT_S)
        self.status_r.pack(side="right", padx=14)

    def bind_events(self):
        c = self.canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Motion>", self.on_motion)
        c.bind("<Leave>", lambda e: c.delete("ghost"))
        for b in (2, 3):
            c.bind(f"<ButtonPress-{b}>", self.pan_start)
            c.bind(f"<B{b}-Motion>", self.pan_move)
        c.bind("<Configure>", lambda e: self.debounce("cfg", self.refresh, 60))

        r = self.root
        r.bind_all("<MouseWheel>", self.on_wheel)
        r.bind_all("<Button-4>", self.on_wheel)
        r.bind_all("<Button-5>", self.on_wheel)
        r.bind("<Control-o>", lambda e: self.open_image())
        r.bind("<Control-s>", lambda e: self.save_image())
        r.bind("<Control-S>", lambda e: self.save_as())
        r.bind("<Control-z>", lambda e: self.undo())
        r.bind("<Control-y>", lambda e: self.redo())
        r.bind("<Control-Z>", lambda e: self.redo())
        r.bind("<Return>", self.key(self.on_enter))
        r.bind("<Escape>", self.key(self.on_escape))
        r.bind("<plus>", self.key(lambda: self.zoom_by(1.25)))
        r.bind("<equal>", self.key(lambda: self.zoom_by(1.25)))
        r.bind("<minus>", self.key(lambda: self.zoom_by(1 / 1.25)))
        r.bind("<Key-0>", self.key(self.fit_view))

    def key(self, fn):
        def handler(e):
            if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
                return
            fn()
        return handler

    # ------------------------------------------------------------------
    # Generic widgets for the side panel
    # ------------------------------------------------------------------
    def clear_panel(self):
        for w in self.p.winfo_children():
            w.destroy()
        self.panel.canvas.yview_moveto(0)

    def head(self, text, sub=None):
        tk.Label(self.p, text=text, bg=PANEL, fg=FG, font=(FAMILY, 15, "bold"), anchor="w").pack(
            fill="x", padx=14, pady=(14, 2))
        if sub:
            tk.Label(self.p, text=sub, bg=PANEL, fg=MUTED, font=FONT_S, anchor="w",
                     justify="left", wraplength=250).pack(fill="x", padx=14, pady=(0, 8))

    def label(self, text, muted=True):
        tk.Label(self.p, text=text, bg=PANEL, fg=MUTED if muted else FG, font=FONT_S,
                 anchor="w", justify="left", wraplength=250).pack(fill="x", padx=14, pady=(8, 2))

    def grid_buttons(self, items, cols=2):
        f = tk.Frame(self.p, bg=PANEL)
        f.pack(fill="x", padx=10, pady=4)
        for i in range(cols):
            f.grid_columnconfigure(i, weight=1, uniform="g")
        for i, (text, cmd) in enumerate(items):
            self.mkbtn(f, text, cmd).grid(row=i // cols, column=i % cols,
                                          sticky="ew", padx=4, pady=4)
        return f

    def slider(self, label, lo, hi, init, command, fmt="{:.0f}"):
        f = tk.Frame(self.p, bg=PANEL)
        f.pack(fill="x", padx=14, pady=(6, 0))
        row = tk.Frame(f, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text=label, bg=PANEL, fg=FG, font=FONT_S).pack(side="left")
        val = tk.Label(row, text=fmt.format(init), bg=PANEL, fg=MUTED, font=FONT_S)
        val.pack(side="right")
        var = tk.DoubleVar(value=init)

        def cb(v):
            v = float(v)
            val.config(text=fmt.format(v))
            command(v)
        ttk.Scale(f, from_=lo, to=hi, variable=var, command=cb, style="Photo.Horizontal.TScale").pack(fill="x", pady=(4, 2))
        return var

    def action_row(self, left_text, left_cmd, right_text, right_cmd):
        f = tk.Frame(self.p, bg=PANEL)
        f.pack(fill="x", padx=10, pady=14)
        f.grid_columnconfigure(0, weight=1, uniform="a")
        f.grid_columnconfigure(1, weight=1, uniform="a")
        self.mkbtn(f, left_text, left_cmd).grid(row=0, column=0, sticky="ew", padx=4)
        self.mkbtn(f, right_text, right_cmd, accent=True).grid(row=0, column=1, sticky="ew", padx=4)

    def color_row(self):
        self.label("Colour")
        f = tk.Frame(self.p, bg=PANEL)
        f.pack(fill="x", padx=14)
        self.swatches = []
        for col in PALETTE:
            sw = tk.Label(f, bg=col, width=2, height=1, relief="flat", bd=2, cursor="hand2",
                          highlightthickness=2,
                          highlightbackground=ACCENT if col == self.color else PANEL)
            sw.pack(side="left", padx=2, pady=2)
            sw.bind("<Button-1>", lambda e, c=col: self.set_color(c))
            self.swatches.append((col, sw))
        self.mkbtn(self.p, "Custom colour…", self.pick_color).pack(fill="x", padx=14, pady=6)

    def set_color(self, col):
        self.color = col
        for c, sw in getattr(self, "swatches", []):
            try:
                sw.config(highlightbackground=ACCENT if c == col else PANEL)
            except tk.TclError:
                pass

    def pick_color(self):
        res = colorchooser.askcolor(color=self.color, title="Choose colour")
        if res and res[1]:
            self.set_color(res[1])

    # ------------------------------------------------------------------
    # Modes / panels
    # ------------------------------------------------------------------
    def select_mode(self, mode):
        self.finalize_mode()
        self.mode = mode
        self.preview = None
        self.crop_state = None
        self.reset_tool_state()
        for k, b in self.nav_btns.items():
            b.config(fg="white" if k == mode else MUTED,
                     bg=ACCENT if k == mode else PANEL)
        self.canvas.config(cursor={"crop": "crosshair", "draw": "pencil", "text": "xterm",
                                   "eraser": "circle"}.get(mode, "arrow"))
        self.canvas.delete("ghost")
        self.build_panel()
        if mode == "crop" and self.image is not None:
            self.crop_init()
        self.refresh()

    def build_panel(self):
        self.clear_panel()
        getattr(self, f"panel_{self.mode}")()

    def reset_tool_state(self):
        self.adj = {k: 0 for k, _ in ADJ_ITEMS}
        self.filter_name = None
        self.filter_strength = 100.0
        self.angle = 0.0

    def finalize_mode(self):
        """Keep any pending (previewed) edit when the user switches tabs."""
        if self.image is None:
            return
        if self.mode == "adjust":
            self.apply_adjust()
        elif self.mode == "filters":
            self.apply_filter()
        elif self.mode == "rotate":
            self.apply_straighten()

    # ---- Crop panel ----------------------------------------------------
    def panel_crop(self):
        self.head("Crop", "Drag the corners or edges, drag inside to move. "
                          "Press Enter to apply.")
        self.label("Aspect ratio")
        self.grid_buttons([(n, lambda r=r: self.set_crop_ratio(r)) for n, r in self.RATIOS], 2)
        self.mkbtn(self.p, "⇄  Swap orientation", self.swap_crop_ratio).pack(
            fill="x", padx=14, pady=(8, 0))
        self.action_row("Reset", self.crop_reset, "Apply crop", self.apply_crop)

    # ---- Rotate panel --------------------------------------------------
    def panel_rotate(self):
        self.head("Rotate & flip")
        self.grid_buttons([
            ("⟲  90° left", lambda: self.transpose(TR.ROTATE_90)),
            ("⟳  90° right", lambda: self.transpose(TR.ROTATE_270)),
            ("↻  180°", lambda: self.transpose(TR.ROTATE_180)),
            ("⇋  Flip H", lambda: self.transpose(TR.FLIP_LEFT_RIGHT)),
            ("⇅  Flip V", lambda: self.transpose(TR.FLIP_TOP_BOTTOM)),
        ], 2)
        self.label("Straighten (auto-crops empty corners)")
        self.straight_var = self.slider("Angle", -45, 45, 0, self.on_angle, "{:.1f}°")
        self.action_row("Reset", self.reset_straighten, "Apply", self.apply_straighten)

    # ---- Adjust panel --------------------------------------------------
    def panel_adjust(self):
        self.head("Adjust", "Move sliders to preview. Changes are kept when you press Apply "
                            "or switch tool.")
        self.mkbtn(self.p, "✨  Auto enhance", self.auto_enhance).pack(fill="x", padx=14, pady=4)
        for key, label in ADJ_ITEMS:
            self.slider(label, -100, 100, 0, lambda v, k=key: self.on_adjust(k, v))
        self.action_row("Reset", self.reset_adjust, "Apply", self.apply_adjust)

    # ---- Filters panel -------------------------------------------------
    def panel_filters(self):
        self.head("Filters")
        self._thumbs = []
        self._thumb_btns = {}
        if self.image is not None:
            base = ImageOps.fit(self.proxy(), (88, 88), RS.BILINEAR)
            grid = tk.Frame(self.p, bg=PANEL)
            grid.pack(fill="x", padx=10)
            for i in range(3):
                grid.grid_columnconfigure(i, weight=1, uniform="t")
            for i, name in enumerate(FILTERS):
                th = ImageTk.PhotoImage(run_filter(base, name))
                self._thumbs.append(th)
                b = tk.Button(grid, image=th, text=name, compound="top", fg=FG, bg=CARD,
                              activebackground=HOVER, activeforeground="white",
                              relief="flat", bd=0, font=FONT_S, cursor="hand2",
                              highlightthickness=2, highlightbackground=PANEL,
                              command=lambda n=name: self.choose_filter(n))
                b.grid(row=i // 3, column=i % 3, padx=3, pady=3, sticky="nsew")
                self._thumb_btns[name] = b
        else:
            self.label("Open a photo to see filter previews.")
        self.strength_var = self.slider("Strength", 0, 100, 100, self.on_strength)
        self.action_row("Cancel", self.cancel_filter, "Apply", self.apply_filter)

    # ---- Draw panel ----------------------------------------------------
    def panel_draw(self):
        self.head("Draw", "Draw directly on the photo. Esc stops a stroke; Ctrl+Z undoes it.")
        f = tk.Frame(self.p, bg=PANEL)
        f.pack(fill="x", padx=10)
        for i in range(2):
            f.grid_columnconfigure(i, weight=1, uniform="d")
        tools = [("✎ Pen", "pen"), ("╱ Line", "line"), ("▭ Rectangle", "rect"), ("◯ Ellipse", "oval")]
        for i, (txt, val) in enumerate(tools):
            tk.Radiobutton(f, text=txt, variable=self.draw_tool, value=val, indicatoron=False,
                           bg=CARD, fg=FG, selectcolor=ACCENT, activebackground=HOVER,
                           activeforeground="white", relief="flat", bd=0, font=FONT,
                           padx=8, pady=8, cursor="hand2").grid(
                row=i // 2, column=i % 2, sticky="ew", padx=4, pady=4)
        self.color_row()
        self.slider("Brush size", 1, 100, self.draw_size, self.set_draw_size)
        tk.Checkbutton(self.p, text="Fill shapes", variable=self.fill_shapes, bg=PANEL, fg=FG,
                       selectcolor=CARD, activebackground=PANEL, activeforeground=FG,
                       font=FONT_S).pack(anchor="w", padx=14, pady=8)

    def set_draw_size(self, v):
        self.draw_size = v

    # ---- Text panel ----------------------------------------------------
    def panel_text(self):
        self.head("Text", "Type your text, then click on the photo to place it.")
        e = tk.Entry(self.p, textvariable=self.text_var, bg=CARD, fg=FG, insertbackground=FG,
                     relief="flat", font=(FAMILY, 12))
        e.pack(fill="x", padx=14, pady=6, ipady=6)
        self.slider("Size", 5, 150, self.text_size, self.set_text_size)
        self.color_row()
        tk.Checkbutton(self.p, text="Outline", variable=self.text_outline, bg=PANEL, fg=FG,
                       selectcolor=CARD, activebackground=PANEL, activeforeground=FG,
                       font=FONT_S).pack(anchor="w", padx=14, pady=8)

    def set_text_size(self, v):
        self.text_size = v

    # ---- Eraser panel --------------------------------------------------
    def panel_eraser(self):
        self.head("Object eraser", "Paint over an unwanted object; it is removed when you "
                                   "release the mouse.")
        if not CV_AVAILABLE:
            self.label("OpenCV is not installed.\nRun:  pip install opencv-python", muted=False)
        self.slider("Brush size", 5, 120, self.erase_size, self.set_erase_size)

    def set_erase_size(self, v):
        self.erase_size = v

    # ---- More panel ----------------------------------------------------
    def panel_more(self):
        self.head("More")
        self.grid_buttons([
            ("Resize…", self.resize_dialog),
            ("Image info", self.show_info),
            ("Fit to screen", self.fit_view),
            ("Actual size", self.actual_size),
            ("★ Favourite", self.favorite_image),
            ("Show in folder", self.share_image),
            ("Set wallpaper", self.set_wallpaper),
            ("🗑 Delete file", self.delete_image),
        ], 2)

    # ------------------------------------------------------------------
    # Undo / redo / commit
    # ------------------------------------------------------------------
    def push_undo(self):
        if self.image is not None:
            self.undo_stack.append(self.image)
            if len(self.undo_stack) > UNDO_LIMIT:
                self.undo_stack.pop(0)
            self.redo_stack.clear()

    def commit(self, img):
        old = self.image
        self.push_undo()
        self.image = img
        self.version += 1
        self.dirty = True
        self.preview = None
        if old is None or old.size != img.size:
            self.zoom, self.pan = 1.0, [0.0, 0.0]
        if self.mode == "crop":
            self.crop_init()
        self.refresh()

    def set_current(self, img):
        old = self.image
        self.image = img
        self.version += 1
        self.dirty = True
        self.preview = None
        if old is None or old.size != img.size:
            self.zoom, self.pan = 1.0, [0.0, 0.0]
        self.reset_tool_state()
        if self.mode in ("adjust", "filters", "rotate"):
            self.build_panel()
        if self.mode == "crop":
            self.crop_init()
        self.refresh()

    def undo(self):
        if self.image is None or not self.undo_stack:
            return
        self.redo_stack.append(self.image)
        self.set_current(self.undo_stack.pop())
        self.status("Undo")

    def redo(self):
        if self.image is None or not self.redo_stack:
            return
        self.undo_stack.append(self.image)
        self.set_current(self.redo_stack.pop())
        self.status("Redo")

    def revert(self):
        if self.original is None or self.image is self.original:
            return
        self.commit(self.original)
        self.status("Reverted to original")

    def proxy(self):
        if self._proxy_ver != self.version or self._proxy is None:
            w, h = self.image.size
            m = max(w, h)
            if m > PROXY_MAX:
                k = PROXY_MAX / m
                self._proxy = self.image.resize((max(1, round(w * k)), max(1, round(h * k))),
                                                RS.BILINEAR, reducing_gap=2.0)
            else:
                self._proxy = self.image
            self._proxy_ver = self.version
        return self._proxy

    def rel(self, v):
        """Convert a slider value to pixels relative to the image size."""
        return max(1.0, v * max(self.image.size) / 1000.0)

    # ------------------------------------------------------------------
    # Rendering / view
    # ------------------------------------------------------------------
    def debounce(self, key, fn, ms=30):
        old = self._jobs.get(key)
        if old:
            try:
                self.root.after_cancel(old)
            except Exception:
                pass
        self._jobs[key] = self.root.after(ms, fn)

    def refresh(self):
        if self.preview is not None:
            self.render(*self.preview)
        else:
            self.render()

    def show_original(self):
        if self.original is not None:
            self.render(self.original)
            self.status("Showing original")

    def render(self, img=None, logical=None):
        c = self.canvas
        c.delete("all")
        if img is None:
            img = self.image
        if img is None:
            self.draw_empty()
            return
        lw, lh = logical if logical else img.size
        cw, ch = max(c.winfo_width(), 2), max(c.winfo_height(), 2)
        fit = max(min((cw - 48) / lw, (ch - 48) / lh, 4.0), 0.0001)
        s = fit * self.zoom
        ox = cw / 2 + self.pan[0] - lw * s / 2
        oy = ch / 2 + self.pan[1] - lh * s / 2
        self.map = (ox, oy, s, fit)

        k = s * lw / img.width
        l = max(0, math.floor(-ox / k))
        t = max(0, math.floor(-oy / k))
        r = min(img.width, math.ceil((cw - ox) / k))
        b = min(img.height, math.ceil((ch - oy) / k))
        if r > l and b > t:
            region = img.crop((l, t, r, b))
            size = (max(1, round((r - l) * k)), max(1, round((b - t) * k)))
            if k >= 3:
                region = region.resize(size, RS.NEAREST)
            elif k >= 1:
                region = region.resize(size, RS.BILINEAR)
            else:
                region = region.resize(size, RS.BILINEAR, reducing_gap=2.0)
            self.photo = ImageTk.PhotoImage(region)
            c.create_image(ox + l * k, oy + t * k, image=self.photo, anchor="nw")
        self.draw_overlay(lw, lh)
        self.update_status()

    def draw_empty(self):
        c = self.canvas
        cw, ch = c.winfo_width(), c.winfo_height()
        c.create_text(cw // 2, ch // 2 - 35, text="✦",
                      fill=ACCENT, font=(FAMILY, 34, "bold"))
        c.create_text(cw // 2, ch // 2 + 8, text="Open a photo to start editing",
                      fill=FG, font=(FAMILY, 20, "bold"))
        c.create_text(cw // 2, ch // 2 + 40, text="Click here or press Ctrl+O",
                      fill=MUTED, font=(FAMILY, 11))
        self.status_r.config(text="")

    def update_status(self):
        if self.image is not None:
            w, h = self.image.size
            self.status_r.config(text=f"{w} × {h} px    {self.map[2] * 100:.0f}%")

    def status(self, msg):
        self.status_l.config(text=msg)

    def fit_view(self):
        self.zoom, self.pan = 1.0, [0.0, 0.0]
        self.refresh()

    def actual_size(self):
        if self.image is None:
            return
        fit = self.map[3]
        self.zoom, self.pan = 1.0 / fit, [0.0, 0.0]
        self.refresh()

    def zoom_by(self, f, cx=None, cy=None):
        if self.image is None:
            return
        ox, oy, s, _ = self.map
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cx is None:
            cx, cy = cw / 2, ch / 2
        new = min(max(self.zoom * f, 0.1), 40.0)
        f2 = new / self.zoom
        lw, lh = (self.preview[1] if self.preview else self.image.size)
        self.pan[0] = cx - (cx - ox) * f2 - cw / 2 + lw * s * f2 / 2
        self.pan[1] = cy - (cy - oy) * f2 - ch / 2 + lh * s * f2 / 2
        self.zoom = new
        self.refresh()

    def i2c(self, x, y):
        ox, oy, s, _ = self.map
        return ox + x * s, oy + y * s

    def c2i(self, x, y):
        ox, oy, s, _ = self.map
        return (x - ox) / s, (y - oy) / s

    def is_inside(self, w, ancestor):
        while w is not None:
            if w is ancestor:
                return True
            w = getattr(w, "master", None)
        return False

    def on_wheel(self, e):
        try:
            w = self.root.winfo_containing(e.x_root, e.y_root)
        except Exception:
            return
        if w is None:
            return
        if e.num == 4:
            d = 1
        elif e.num == 5:
            d = -1
        else:
            d = 1 if e.delta > 0 else -1
        if w is self.canvas:
            self.zoom_by(1.12 ** d, e.x_root - self.canvas.winfo_rootx(),
                         e.y_root - self.canvas.winfo_rooty())
        elif self.is_inside(w, self.panel):
            self.panel.canvas.yview_scroll(-d * 2, "units")

    def pan_start(self, e):
        self._pan_drag = (e.x, e.y, self.pan[0], self.pan[1])

    def pan_move(self, e):
        if self._pan_drag and self.image is not None:
            x0, y0, px, py = self._pan_drag
            self.pan = [px + e.x - x0, py + e.y - y0]
            self.refresh()

    # ------------------------------------------------------------------
    # Mouse dispatch
    # ------------------------------------------------------------------
    def on_press(self, e):
        if self.image is None:
            self.open_image()
            return
        self.canvas.focus_set()
        m = self.mode
        if m == "crop":
            self.crop_press(e)
        elif m == "draw":
            self.draw_press(e)
        elif m == "text":
            self.text_click(e)
        elif m == "eraser":
            self.erase_press(e)
        else:
            self.pan_start(e)

    def on_drag(self, e):
        if self.image is None:
            return
        m = self.mode
        if m == "crop":
            self.crop_drag(e)
        elif m == "draw":
            self.draw_drag(e)
        elif m == "eraser":
            self.erase_drag(e)
        elif m in ("text",):
            self.text_ghost(e)
        else:
            self.pan_move(e)

    def on_release(self, e):
        if self.image is None:
            return
        m = self.mode
        if m == "crop":
            self.crop_release(e)
        elif m == "draw":
            self.draw_release(e)
        elif m == "eraser":
            self.erase_release(e)
        self._pan_drag = None

    def on_motion(self, e):
        if self.image is None:
            return
        if self.mode == "text":
            self.text_ghost(e)
        elif self.mode == "crop" and self.crop:
            h = self.crop_hit(e.x, e.y)
            self.canvas.config(cursor="fleur" if h == "move" else "crosshair")

    def on_enter(self):
        if self.mode == "crop":
            self.apply_crop()
        elif self.mode == "rotate":
            self.apply_straighten()
        elif self.mode == "adjust":
            self.apply_adjust()
        elif self.mode == "filters":
            self.apply_filter()

    def on_escape(self):
        if self.mode == "crop":
            self.crop_reset()
        self._stroke = None
        self.canvas.delete("stroke", "shape", "ghost")

    # ------------------------------------------------------------------
    # Crop
    # ------------------------------------------------------------------
    def crop_init(self):
        w, h = self.image.size
        self.crop = [0.0, 0.0, float(w), float(h)]
        self.crop_ratio = None
        self.crop_state = None

    def crop_reset(self):
        if self.image is not None:
            self.crop_init()
            self.refresh()

    def set_crop_ratio(self, r):
        if self.image is None:
            return
        W, H = self.image.size
        if r == "orig":
            r = W / H
        self.crop_ratio = r
        if r is None:
            self.refresh()
            return
        x1, y1, x2, y2 = self.crop
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w = min(x2 - x1, (y2 - y1) * r)
        h = w / r
        x1 = min(max(cx - w / 2, 0), W - w)
        y1 = min(max(cy - h / 2, 0), H - h)
        self.crop = [x1, y1, x1 + w, y1 + h]
        self.refresh()

    def swap_crop_ratio(self):
        if self.crop_ratio:
            self.set_crop_ratio(1.0 / self.crop_ratio)

    def crop_hit(self, ex, ey):
        x1, y1 = self.i2c(self.crop[0], self.crop[1])
        x2, y2 = self.i2c(self.crop[2], self.crop[3])
        pts = {"nw": (x1, y1), "ne": (x2, y1), "sw": (x1, y2), "se": (x2, y2)}
        if self.crop_ratio is None:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            pts.update({"n": (mx, y1), "s": (mx, y2), "w": (x1, my), "e": (x2, my)})
        for name, (px, py) in pts.items():
            if abs(ex - px) <= 18 and abs(ey - py) <= 18:
                return name
        if x1 < ex < x2 and y1 < ey < y2:
            return "move"
        return "new"

    def crop_press(self, e):
        if not self.crop:
            self.crop_init()
        h = self.crop_hit(e.x, e.y)
        p = self.c2i(e.x, e.y)
        x1, y1, x2, y2 = self.crop
        W, H = self.image.size
        orig = list(self.crop)
        if h == "new":
            ax = min(max(p[0], 0), W)
            ay = min(max(p[1], 0), H)
            self.crop = [ax, ay, ax, ay]
            self.crop_state = ("corner", (ax, ay), orig, p)
        elif h in ("nw", "ne", "sw", "se"):
            anchor = {"nw": (x2, y2), "ne": (x1, y2), "sw": (x2, y1), "se": (x1, y1)}[h]
            self.crop_state = ("corner", anchor, orig, p)
        else:
            self.crop_state = (h, None, orig, p)

    def crop_drag(self, e):
        if not self.crop_state:
            return
        kind, anchor, orig, start = self.crop_state
        W, H = self.image.size
        px, py = self.c2i(e.x, e.y)
        px = min(max(px, 0), W)
        py = min(max(py, 0), H)
        MIN = 20.0
        if kind == "move":
            dx, dy = px - start[0], py - start[1]
            w, h = orig[2] - orig[0], orig[3] - orig[1]
            x1 = min(max(orig[0] + dx, 0), W - w)
            y1 = min(max(orig[1] + dy, 0), H - h)
            self.crop = [x1, y1, x1 + w, y1 + h]
        elif kind == "corner":
            ax, ay = anchor
            dx, dy = px - ax, py - ay
            sx = 1 if dx >= 0 else -1
            sy = 1 if dy >= 0 else -1
            w, h = abs(dx), abs(dy)
            r = self.crop_ratio
            if r:
                w = max(w, h * r)
                h = w / r if r else h
                max_w = (W - ax) if sx > 0 else ax
                max_h = (H - ay) if sy > 0 else ay
                if w > 0 and h > 0:
                    k = min(1.0, max_w / w, max_h / h)
                    w, h = w * k, h * k
            xa, xb = ax, ax + sx * w
            ya, yb = ay, ay + sy * h
            self.crop = [min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)]
        else:  # edges
            x1, y1, x2, y2 = orig
            if kind == "n":
                y1 = min(py, y2 - MIN)
            elif kind == "s":
                y2 = max(py, y1 + MIN)
            elif kind == "w":
                x1 = min(px, x2 - MIN)
            elif kind == "e":
                x2 = max(px, x1 + MIN)
            self.crop = [x1, y1, x2, y2]
        self.canvas.delete("crop")
        self.draw_crop_overlay()

    def crop_release(self, e):
        if not self.crop_state:
            return
        x1, y1, x2, y2 = self.crop
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            self.crop = list(self.crop_state[2])
        self.crop_state = None
        self.refresh()

    def apply_crop(self):
        if self.image is None or not self.crop:
            return
        box = tuple(int(round(v)) for v in self.crop)
        W, H = self.image.size
        if box == (0, 0, W, H) or box[2] - box[0] < 2 or box[3] - box[1] < 2:
            self.status("Nothing to crop")
            return
        self.commit(self.image.crop(box))
        self.status(f"Cropped to {box[2] - box[0]} × {box[3] - box[1]}")

    # ---- overlays -------------------------------------------------------
    def draw_overlay(self, lw=None, lh=None):
        if self.mode == "crop" and self.preview is None and self.crop and self.image is not None:
            self.draw_crop_overlay()
        elif self.mode == "rotate" and self.preview is not None and lw:
            ox, oy, s, _ = self.map
            self.draw_grid(ox, oy, ox + lw * s, oy + lh * s, "grid")

    def draw_grid(self, x1, y1, x2, y2, tag):
        c = self.canvas
        for i in (1, 2):
            x = x1 + (x2 - x1) * i / 3
            y = y1 + (y2 - y1) * i / 3
            c.create_line(x, y1, x, y2, fill="#ffffff", width=1, tags=tag)
            c.create_line(x1, y, x2, y, fill="#ffffff", width=1, tags=tag)

    def draw_crop_overlay(self):
        c = self.canvas
        cw, ch = c.winfo_width(), c.winfo_height()
        x1, y1 = self.i2c(self.crop[0], self.crop[1])
        x2, y2 = self.i2c(self.crop[2], self.crop[3])
        dim = {"fill": "black"}
        if sys.platform != "darwin":
            dim["stipple"] = "gray50"
        if sys.platform != "darwin":
            for box in ((0, 0, cw, y1), (0, y2, cw, ch), (0, y1, x1, y2), (x2, y1, cw, y2)):
                c.create_rectangle(*box, outline="", tags="crop", **dim)
        c.create_rectangle(x1, y1, x2, y2, outline="white", width=2, tags="crop")
        self.draw_grid(x1, y1, x2, y2, "crop")
        hs = 6
        pts = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        if self.crop_ratio is None:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            pts += [(mx, y1), (mx, y2), (x1, my), (x2, my)]
        for px, py in pts:
            c.create_rectangle(px - hs, py - hs, px + hs, py + hs, fill="white",
                               outline=ACCENT, width=1, tags="crop")
        w, h = int(self.crop[2] - self.crop[0]), int(self.crop[3] - self.crop[1])
        c.create_text((x1 + x2) / 2, max(12, y1 - 14), text=f"{w} × {h}", fill="white",
                      font=FONT_S, tags="crop")

    # ------------------------------------------------------------------
    # Rotate / straighten
    # ------------------------------------------------------------------
    def transpose(self, method):
        if self.image is None:
            return
        self.commit(self.image.transpose(method))

    def on_angle(self, v):
        self.angle = v
        self.debounce("angle", self.update_straighten, 20)

    def update_straighten(self):
        if self.image is None:
            return
        if abs(self.angle) < 0.05:
            self.preview = None
            self.refresh()
            return
        p = self.proxy()
        st = straighten(p, self.angle)
        k = self.image.width / p.width
        self.preview = (st, (max(1, round(st.width * k)), max(1, round(st.height * k))))
        self.refresh()

    def apply_straighten(self):
        if self.image is None or abs(self.angle) < 0.05:
            return
        out = straighten(self.image, self.angle)
        self.angle = 0.0
        self.commit(out)
        if self.mode == "rotate":
            self.build_panel()

    def reset_straighten(self):
        self.angle = 0.0
        self.preview = None
        if self.mode == "rotate":
            self.build_panel()
        self.refresh()

    # ------------------------------------------------------------------
    # Adjust
    # ------------------------------------------------------------------
    def on_adjust(self, key, v):
        self.adj[key] = v
        self.debounce("adj", self.update_adjust, 25)

    def update_adjust(self):
        if self.image is None:
            return
        if not any(self.adj.values()):
            self.preview = None
        else:
            self.preview = (apply_adjustments(self.proxy(), self.adj), self.image.size)
        self.refresh()

    def apply_adjust(self):
        if self.image is None or not any(self.adj.values()):
            return
        out = apply_adjustments(self.image, self.adj)
        self.adj = {k: 0 for k, _ in ADJ_ITEMS}
        self.commit(out)
        if self.mode == "adjust":
            self.build_panel()
        self.status("Adjustments applied")

    def reset_adjust(self):
        self.adj = {k: 0 for k, _ in ADJ_ITEMS}
        self.preview = None
        if self.mode == "adjust":
            self.build_panel()
        self.refresh()

    def auto_enhance(self):
        if self.image is None:
            return
        self.adj = {k: 0 for k, _ in ADJ_ITEMS}
        out = ImageOps.autocontrast(self.image, cutoff=1)
        out = ImageEnhance.Color(out).enhance(1.12)
        self.commit(out)
        if self.mode == "adjust":
            self.build_panel()
        self.status("Auto enhance applied")

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    def choose_filter(self, name):
        self.filter_name = name
        for n, b in getattr(self, "_thumb_btns", {}).items():
            b.config(highlightbackground=ACCENT if n == name else PANEL)
        self.update_filter_preview()

    def on_strength(self, v):
        self.filter_strength = v
        self.debounce("flt", self.update_filter_preview, 25)

    def update_filter_preview(self):
        if self.image is None:
            return
        if self.filter_name in (None, "Original"):
            self.preview = None
        else:
            out = run_filter(self.proxy(), self.filter_name, self.filter_strength)
            self.preview = (out, self.image.size)
        self.refresh()

    def apply_filter(self):
        if self.image is None or self.filter_name in (None, "Original"):
            return
        out = run_filter(self.image, self.filter_name, self.filter_strength)
        self.filter_name = None
        self.filter_strength = 100.0
        self.commit(out)
        if self.mode == "filters":
            self.build_panel()
        self.status("Filter applied")

    def cancel_filter(self):
        self.filter_name = None
        self.filter_strength = 100.0
        self.preview = None
        if self.mode == "filters":
            self.build_panel()
        self.refresh()

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    def begin_edit_copy(self):
        """Undo-safe in-place editing: keep the old image in the undo stack."""
        self.push_undo()
        self.image = self.image.copy()
        self.version += 1
        self.dirty = True

    def draw_press(self, e):
        p = self.c2i(e.x, e.y)
        tool = self.draw_tool.get()
        self._stroke = {"start": p, "last": p, "tool": tool, "moved": False}
        if tool == "pen":
            self.begin_edit_copy()
            self._drawer = ImageDraw.Draw(self.image)
            w = self.rel(self.draw_size)
            r = w / 2
            self._drawer.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=self.color)
            cx, cy = e.x, e.y
            rr = max(1, w * self.map[2] / 2)
            self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill=self.color,
                                    outline=self.color, tags="stroke")

    def draw_drag(self, e):
        st = self._stroke
        if not st:
            return
        p = self.c2i(e.x, e.y)
        w = self.rel(self.draw_size)
        s = self.map[2]
        st["moved"] = True
        if st["tool"] == "pen":
            d = self._drawer
            d.line([st["last"], p], fill=self.color, width=max(1, int(round(w))))
            r = w / 2
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=self.color)
            x0, y0 = self.i2c(*st["last"])
            self.canvas.create_line(x0, y0, e.x, e.y, fill=self.color, width=max(1, w * s),
                                    capstyle="round", tags="stroke")
            st["last"] = p
        else:
            self.canvas.delete("shape")
            x0, y0 = self.i2c(*st["start"])
            wd = max(1, w * s)
            fill = self.color if self.fill_shapes.get() else ""
            if st["tool"] == "line":
                self.canvas.create_line(x0, y0, e.x, e.y, fill=self.color, width=wd,
                                        capstyle="round", tags="shape")
            elif st["tool"] == "rect":
                self.canvas.create_rectangle(x0, y0, e.x, e.y, outline=self.color, width=wd,
                                             fill=fill, tags="shape")
            else:
                self.canvas.create_oval(x0, y0, e.x, e.y, outline=self.color, width=wd,
                                        fill=fill, tags="shape")

    def draw_release(self, e):
        st = self._stroke
        self._stroke = None
        if not st:
            return
        if st["tool"] == "pen":
            self._drawer = None
            self.canvas.delete("stroke")
            self.refresh()
            return
        self.canvas.delete("shape")
        if not st["moved"]:
            return
        x0, y0 = st["start"]
        x1, y1 = self.c2i(e.x, e.y)
        if abs(x1 - x0) < 2 and abs(y1 - y0) < 2:
            return
        self.begin_edit_copy()
        d = ImageDraw.Draw(self.image)
        w = max(1, int(round(self.rel(self.draw_size))))
        fill = self.color if self.fill_shapes.get() else None
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        if st["tool"] == "line":
            d.line([(x0, y0), (x1, y1)], fill=self.color, width=w)
            r = w / 2
            for (px, py) in ((x0, y0), (x1, y1)):
                d.ellipse([px - r, py - r, px + r, py + r], fill=self.color)
        elif st["tool"] == "rect":
            d.rectangle(box, outline=self.color, width=w, fill=fill)
        else:
            d.ellipse(box, outline=self.color, width=w, fill=fill)
        self.refresh()

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------
    def text_ghost(self, e):
        c = self.canvas
        c.delete("ghost")
        txt = self.text_var.get()
        if self.image is None or not txt:
            return
        px = max(6, int(self.rel(self.text_size) * self.map[2]))
        c.create_text(e.x, e.y, text=txt, fill=self.color, font=(FAMILY, -px, "bold"),
                      tags="ghost")

    def text_click(self, e):
        txt = self.text_var.get()
        if not txt:
            return
        x, y = self.c2i(e.x, e.y)
        W, H = self.image.size
        if not (0 <= x < W and 0 <= y < H):
            return
        size = int(self.rel(self.text_size))
        font = load_font(size)
        self.begin_edit_copy()
        d = ImageDraw.Draw(self.image)
        kwargs = {}
        if self.text_outline.get():
            kwargs = dict(stroke_width=max(1, size // 16),
                          stroke_fill="black" if self.color.lower() != "#000000" else "white")
        try:
            d.text((x, y), txt, fill=self.color, font=font, anchor="mm", **kwargs)
        except Exception:
            d.text((x, y), txt, fill=self.color, font=font, **kwargs)
        self.canvas.delete("ghost")
        self.refresh()

    # ------------------------------------------------------------------
    # Object eraser (OpenCV inpainting)
    # ------------------------------------------------------------------
    def erase_press(self, e):
        if not CV_AVAILABLE:
            if not self._cv_warned:
                self._cv_warned = True
                messagebox.showwarning("OpenCV required",
                                       "Install OpenCV first:\n\npip install opencv-python")
            return
        p = self.c2i(e.x, e.y)
        self._mask = Image.new("L", self.image.size, 0)
        self._mdraw = ImageDraw.Draw(self._mask)
        self._stroke = {"last": p}
        w = self.rel(self.erase_size)
        r = w / 2
        self._mdraw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
        rr = max(1, w * self.map[2] / 2)
        self.canvas.create_oval(e.x - rr, e.y - rr, e.x + rr, e.y + rr, fill="#ff3b30",
                                outline="#ff3b30", tags="stroke")

    def erase_drag(self, e):
        if not self._stroke or self._mdraw is None:
            return
        p = self.c2i(e.x, e.y)
        w = self.rel(self.erase_size)
        self._mdraw.line([self._stroke["last"], p], fill=255, width=max(1, int(round(w))))
        r = w / 2
        self._mdraw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
        x0, y0 = self.i2c(*self._stroke["last"])
        self.canvas.create_line(x0, y0, e.x, e.y, fill="#ff3b30", width=max(1, w * self.map[2]),
                                capstyle="round", tags="stroke")
        self._stroke["last"] = p

    def erase_release(self, e):
        if not self._stroke or self._mask is None:
            return
        self._stroke = None
        self.canvas.delete("stroke")
        mask = np.array(self._mask)
        self._mask = self._mdraw = None
        if mask.max() == 0:
            return
        self.status("Removing object…")
        self.root.update_idletasks()
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        bgr = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
        self.commit(Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
        self.status("Object removed")

    # ------------------------------------------------------------------
    # More tools
    # ------------------------------------------------------------------
    def resize_dialog(self):
        if self.image is None:
            return
        w0, h0 = self.image.size
        top = tk.Toplevel(self.root)
        top.title("Resize")
        top.configure(bg=PANEL)
        top.transient(self.root)
        top.resizable(False, False)
        wv, hv = tk.StringVar(value=str(w0)), tk.StringVar(value=str(h0))
        lock = tk.BooleanVar(value=True)
        busy = [False]

        def sync(src, dst, ratio):
            if busy[0] or not lock.get():
                return
            try:
                v = int(src.get())
            except ValueError:
                return
            busy[0] = True
            dst.set(str(max(1, round(v * ratio))))
            busy[0] = False

        wv.trace_add("write", lambda *a: sync(wv, hv, h0 / w0))
        hv.trace_add("write", lambda *a: sync(hv, wv, w0 / h0))

        for i, (txt, var) in enumerate((("Width (px)", wv), ("Height (px)", hv))):
            tk.Label(top, text=txt, bg=PANEL, fg=FG, font=FONT).grid(
                row=i, column=0, padx=14, pady=8, sticky="w")
            tk.Entry(top, textvariable=var, bg=CARD, fg=FG, insertbackground=FG,
                     relief="flat", width=10, font=FONT).grid(row=i, column=1, padx=14, pady=8)
        tk.Checkbutton(top, text="Keep aspect ratio", variable=lock, bg=PANEL, fg=FG,
                       selectcolor=CARD, activebackground=PANEL, activeforeground=FG).grid(
            row=2, column=0, columnspan=2, pady=4)

        def ok():
            try:
                w, h = int(wv.get()), int(hv.get())
                if w < 1 or h < 1 or w * h > 200_000_000:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Resize", "Enter valid sizes.", parent=top)
                return
            top.destroy()
            self.commit(self.image.resize((w, h), RS.LANCZOS))
            self.status(f"Resized to {w} × {h}")

        self.mkbtn(top, "Resize", ok, accent=True).grid(row=3, column=0, columnspan=2,
                                                        padx=14, pady=14, sticky="ew")
        top.grab_set()

    def show_info(self):
        if self.image is None:
            return
        w, h = self.image.size
        lines = [f"Size: {w} × {h} px  ({w * h / 1e6:.1f} MP)"]
        if self.file_path and os.path.exists(self.file_path):
            lines.append(f"File: {self.file_path}")
            lines.append(f"On disk: {os.path.getsize(self.file_path) / 1024:.0f} KB")
        lines.append(f"Undo steps: {len(self.undo_stack)}")
        messagebox.showinfo("Image info", "\n".join(lines))

    def _pictures_dir(self):
        d = os.path.join(os.path.expanduser("~"), "Pictures")
        os.makedirs(d, exist_ok=True)
        return d

    def favorite_image(self):
        if self.image is None:
            return
        d = os.path.join(self._pictures_dir(), "PhotoStudioFavorites")
        os.makedirs(d, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.save_path or self.file_path or "photo"))[0]
        dest = os.path.join(d, f"{base}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        try:
            self.image.save(dest, quality=95)
            messagebox.showinfo("Favourites", f"Added to favourites:\n\n{dest}")
        except Exception as ex:
            messagebox.showerror("Favourites", str(ex))

    def share_image(self):
        if self.image is None:
            return
        path = self.save_path or self.file_path
        if not path or self.dirty:
            path = os.path.join(self._pictures_dir(), "Shared_Image.jpg")
            try:
                self.image.save(path, quality=95)
            except Exception as ex:
                messagebox.showerror("Share", str(ex))
                return
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(path)])
        except Exception as ex:
            messagebox.showerror("Share", str(ex))

    def set_wallpaper(self):
        if self.image is None:
            return
        if not sys.platform.startswith("win"):
            messagebox.showinfo("Wallpaper", "Set-as-wallpaper is implemented for Windows only.")
            return
        import ctypes
        path = os.path.join(self._pictures_dir(), "PhotoStudioWallpaper.jpg")
        try:
            self.image.save(path, "JPEG", quality=95)
            ctypes.windll.user32.SystemParametersInfoW(20, 0, path, 3)
            self.status("Wallpaper set")
        except Exception as ex:
            messagebox.showerror("Wallpaper", str(ex))

    def delete_image(self):
        if not self.file_path or not os.path.exists(self.file_path):
            messagebox.showwarning("Delete", "There is no file on disk to delete.")
            return
        if not messagebox.askyesno("Delete file",
                                   f"Permanently delete this file from disk?\n\n{self.file_path}"):
            return
        try:
            os.remove(self.file_path)
        except Exception as ex:
            messagebox.showerror("Delete", str(ex))
            return
        self.file_path = self.save_path = None
        self.image = self.original = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.preview = None
        self.dirty = False
        self.root.title("Photo Editor")
        self.refresh()
        self.status("File deleted")

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def confirm_discard(self):
        if self.image is not None and self.dirty:
            ans = messagebox.askyesnocancel("Unsaved changes", "Save changes before continuing?")
            if ans is None:
                return False
            if ans:
                self.save_image()
                return not self.dirty
        return True

    def open_image(self, path=None):
        if not self.confirm_discard():
            return
        if not path:
            path = filedialog.askopenfilename(
                title="Open image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.gif"),
                           ("All files", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as ex:
            messagebox.showerror("Open error", str(ex))
            return
        self.file_path = path
        self.save_path = None
        self.original = img
        self.image = img
        self.version += 1
        self.dirty = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.preview = None
        self.zoom, self.pan = 1.0, [0.0, 0.0]
        self.reset_tool_state()
        self.root.title(f"Photo Editor - {os.path.basename(path)}")
        self.build_panel()
        if self.mode == "crop":
            self.crop_init()
        self.refresh()
        self.status(f"Opened {os.path.basename(path)}")

    def _write(self, path):
        ext = os.path.splitext(path)[1].lower()
        kw = {}
        if ext in (".jpg", ".jpeg"):
            kw = dict(quality=95, subsampling=0)
        try:
            self.image.save(path, **kw)
        except Exception as ex:
            messagebox.showerror("Save error", str(ex))
            return False
        self.save_path = path
        self.dirty = False
        self.status(f"Saved {path}")
        return True

    def save_image(self):
        if self.image is None:
            return
        if self.save_path:
            self._write(self.save_path)
        else:
            self.save_as()

    def save_as(self):
        if self.image is None:
            return
        base = os.path.splitext(os.path.basename(self.file_path or "photo"))[0] + "_edited"
        path = filedialog.asksaveasfilename(
            title="Save As", initialfile=base,
            initialdir=os.path.dirname(self.file_path) if self.file_path else None,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("WEBP", "*.webp"), ("BMP", "*.bmp")])
        if path:
            self._write(path)

    def on_close(self):
        if self.confirm_discard():
            self.root.destroy()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:  # crisp text on high-DPI laptops
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    app = PhotoStudio(root)
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(250, lambda: app.open_image(sys.argv[1]))
    root.mainloop()
