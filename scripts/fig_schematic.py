"""Schematic figures drawn to scale on a synthetic axial CT phantom.

  figures/teaser.pdf      Fig. 1  problem (two explanations) + intervention (ball / band / full)
  figures/visibility.pdf  Fig. 2  the five visibility conditions + mask-before-interpolation

The phantom is illustrative (not patient data). All radii follow Sec. 3 exactly:
ball 10 mm; probe radii r_N = 10 + max(1.9, d + 0.05), r_F = r_N + 1.5; band [r_N - d, r_F + d],
with an illustrative native spacing of 0.70 x 0.70 x 1.25 mm (d = voxel diagonal).
CPM values in Fig. 1 come from scripts/data.py.
"""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter

import style
from style import C, plt, save
import data as D

ROOT = D.ROOT
PX = 0.08                       # phantom pixel (mm)
HALF = 16.2                     # half of the 32.4 mm field
SPACING = np.array([0.70, 0.70, 1.25])
DIAG = float(np.linalg.norm(SPACING))
R_BALL = 10.0
R_N = R_BALL + max(1.9, DIAG + 0.05)
R_F = R_N + 1.5
BAND = (R_N - DIAG, R_F + DIAG)
WIN = (-1000, 250)

ax1 = np.arange(-HALF, HALF + 1e-9, PX)
X, Y = np.meshgrid(ax1, -ax1)   # y up
R = np.hypot(X, Y)


def seg_dist(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    t = np.clip(((X - p[0]) * d[0] + (Y - p[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(X - p[0] - t * d[0], Y - p[1] - t * d[1])


def curve(points, r0, r1, hu=60, n=40):
    """Smooth vessel along a quadratic Bezier, radius tapering r0 -> r1."""
    P = np.asarray(points, float)
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * P[0] + 2 * (1 - t) * t * P[1] + t ** 2 * P[2]
    out = np.full(X.shape, -np.inf)
    for i in range(n - 1):
        r = r0 + (r1 - r0) * (i / (n - 1))
        prof = hu - (hu + 850) * np.clip(seg_dist(pts[i], pts[i + 1]) / r, 0, 1.6) ** 4 / 1.6 ** 4
        out = np.maximum(out, prof)
    return out


def lung(seed):
    rng = np.random.default_rng(seed)
    base = -860 + gaussian_filter(rng.normal(0, 1, X.shape), 6) * 260 + rng.normal(0, 14, X.shape)
    return base


def blob(cx, cy, r, hu=30, lob=(0.07, 0.05), seed=0, soft=0.6):
    th = np.arctan2(Y - cy, X - cx)
    rr = r * (1 + lob[0] * np.sin(3 * th + 0.4) + lob[1] * np.sin(5 * th + 1.3))
    d = np.hypot(X - cx, Y - cy) - rr
    w = 1 / (1 + np.exp(d / soft))
    rng = np.random.default_rng(seed)
    tex = gaussian_filter(rng.normal(0, 1, X.shape), 3) * 60
    return w, hu + tex


def compose(layers, seed):
    img = lung(seed)
    for lay in layers:
        if isinstance(lay, tuple):
            w, val = lay
            img = img * (1 - w) + val * w
        else:
            img = np.maximum(img, lay)
    return gaussian_filter(img, 0.8)


def vessel_case():
    """A vessel running obliquely through the slice: inside 10 mm it is a compact bright blob,
    outside it continues as a tube in both directions and branches."""
    # oblique section: an ellipse elongated along the vessel, same width as the tube
    th = np.deg2rad(34)
    xr, yr = X * np.cos(th) + Y * np.sin(th), -X * np.sin(th) + Y * np.cos(th)
    d = np.hypot(xr / 4.6, yr / 2.5) - 1
    knot = (1 / (1 + np.exp(d / 0.12)), 45 + gaussian_filter(np.random.default_rng(3).normal(0, 1, X.shape), 3) * 40)
    # within ~10 mm the vessel runs out of the slice (only faint partial-volume traces);
    # beyond the ball it returns into the slice as a bright tube
    v = [curve([(-9.0, -6.1), (-6.2, -4.2), (-3.6, -2.4)], 1.3, 1.6, -560),
         curve([(3.6, 2.4), (6.2, 4.2), (9.0, 6.1)], 1.6, 1.3, -560),
         curve([(-17, -11.5), (-13, -8.8), (-9.0, -6.1)], 2.0, 1.9, 50),
         curve([(9.0, 6.1), (13, 8.8), (17, 11.5)], 1.9, 2.0, 50),
         curve([(10, 7), (13.5, 2), (17, -2)], 0.9, 0.7, 40),
         curve([(-17, 9), (-11, 11), (-6, 17)], 0.8, 0.6, 30),
         curve([(9, -17), (11, -11), (17, -9)], 0.7, 0.6, 30)]
    return compose([*v, knot], 11)


def nodule_case():
    v = [curve([(-17, -4), (-9, -10), (-3, -17)], 1.1, 0.9, 40),
         curve([(17, -15), (13, -9), (9.5, -3.5)], 1.2, 1.0, 50),
         curve([(-17, 12), (-10, 9), (-5, 17)], 0.8, 0.6, 30)]
    nod = blob(5.6, 3.2, 7.1, hu=25, seed=5, soft=0.45)
    return compose([*v, nod], 23)


def masks():
    ball = R <= R_BALL
    band = (R >= BAND[0]) & (R <= BAND[1])
    return ball, band


def probe_geometry(n_axes=12):
    th = np.arange(n_axes) * np.pi / n_axes + 0.13
    u = np.column_stack((np.cos(th), np.sin(th)))
    q = np.r_[R_N * u, R_F * u, -R_N * u, -R_F * u]
    return u, q


def source_mask(q):
    """In-plane trilinear (bilinear) footprints of the probes on a 0.7 mm native lattice."""
    s = SPACING[0]
    m = np.zeros(X.shape, bool)
    off = 0.21
    for qx, qy in q:
        ix, iy = np.floor((qx - off) / s), np.floor((qy - off) / s)
        for a in (0, 1):
            for b in (0, 1):
                cx, cy = (ix + a) * s + off, (iy + b) * s + off
                m |= (np.abs(X - cx) <= s / 2) & (np.abs(Y - cy) <= s / 2)
    return m


def show(ax, img, visible=None, hatch=True):
    im = img.copy()
    if visible is not None:
        im[~visible] = -1000
    ax.imshow(im, cmap="gray", vmin=WIN[0], vmax=WIN[1], extent=(-HALF, HALF, -HALF, HALF),
              interpolation="bilinear")
    if visible is not None and hatch:
        stripes = (np.mod(X - Y, 2.4) < 0.45) & ~visible
        rgba = np.zeros(X.shape + (4,))
        rgba[stripes] = (0.42, 0.42, 0.42, 1.0)
        ax.imshow(rgba, extent=(-HALF, HALF, -HALF, HALF), interpolation="nearest")
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#333333")
        sp.set_linewidth(0.6)


def band_fill(ax, alpha=0.30):
    from matplotlib.patches import Wedge
    ax.add_patch(Wedge((0, 0), BAND[1], 0, 360, width=BAND[1] - BAND[0], facecolor=C["band"],
                       alpha=alpha, edgecolor="none", zorder=4))
    ring(ax, BAND[0], C["band"], lw=0.7)
    ring(ax, BAND[1], C["band"], lw=0.7)


def cross(ax, color="#ffd23f", s=2.0, lw=1.1):
    ax.plot([-s, s], [0, 0], color=color, lw=lw, zorder=6, solid_capstyle="butt")
    ax.plot([0, 0], [-s, s], color=color, lw=lw, zorder=6, solid_capstyle="butt")


def ring(ax, r, color, lw=0.9, ls="-", alpha=1):
    ax.add_patch(Circle((0, 0), r, fill=False, ec=color, lw=lw, ls=ls, alpha=alpha, zorder=5))


def scalebar(ax, color="white"):
    ax.plot([-14.6, -9.6], [14.4, 14.4], color=color, lw=1.4, solid_capstyle="butt", zorder=7)
    ax.text(-12.1, 13.6, "5 mm", color=color, ha="center", va="top", fontsize=6.2, zorder=7)


# ----------------------------------------------------------------------------- Fig. 1
def teaser(path):
    vessel, nodule = vessel_case(), nodule_case()
    ball, band = masks()
    W, H = style.TEXT, 2.55
    fig = plt.figure(figsize=(W, H))
    sz = 0.98                                   # image size (in)
    w, h = sz / W, sz / H
    # (a) 2 x 2: rows = ball only / full view, cols = vessel / nodule
    xa = [0.085, 0.085 + w + 0.008]
    ya = [0.475, 0.475 - h - 0.035]
    fig.text(0.005, 0.975, "(a) Inside the ball, a vessel and a nodule edge look alike", fontsize=8.5,
             fontweight="bold", va="top")
    for j, (img, t) in enumerate(((vessel, "Vessel"), (nodule, "Off-centre nodule"))):
        for i, vis in enumerate((ball, None)):
            ax = fig.add_axes((xa[j], ya[i], w, h))
            show(ax, img, vis)
            ring(ax, R_BALL, C["ball"], lw=1.0)
            cross(ax)
            if i == 0:
                ax.set_title(t, fontsize=7.8, pad=2.5)
            if i == 1 and j == 0:
                scalebar(ax)
            if i == 1:
                ax.text(0.5, 0.04, ["tube continues: vessel", "round mass: nodule"][j], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=6.4, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.65))
    fig.text(xa[0] - 0.008, ya[0] + h / 2, "10 mm ball\nonly", ha="right", va="center", fontsize=7.4,
             color=C["ball"], fontweight="bold", linespacing=1.05)
    fig.text(xa[0] - 0.008, ya[1] + h / 2, "Full\nview", ha="right", va="center", fontsize=7.4,
             color=C["full"], fontweight="bold", linespacing=1.05)
    sep = xa[1] + w + 0.022
    fig.add_artist(plt.Line2D([sep, sep], [0.03, 0.95], color="#bbbbbb", lw=0.6))

    # (b) the intervention
    xb0 = sep + 0.02
    fig.text(xb0, 0.975, "(b) Fixed crop, network and training; only the visible native voxels vary",
             fontsize=8.5, fontweight="bold", va="top")
    bs = 1.30
    wb, hb = bs / W, bs / H
    gap = (0.995 - xb0 - 3 * wb) / 2
    xb = [xb0 + i * (wb + gap) for i in range(3)]
    yb = 0.30
    cpm = D.arch_absolute("3691")
    vis = [ball, ball | band, None]
    titles = ["Ball", "Ball + band", "Full field"]
    cols = [C["ball"], C["band"], C["full"]]
    for x, v, t, col, c in zip(xb, vis, titles, cols, cpm):
        ax = fig.add_axes((x, yb, wb, hb))
        show(ax, nodule, v)
        if v is not None and v is not ball:
            band_fill(ax, alpha=0.22)
        ring(ax, R_BALL, C["ball"], lw=1.0)
        cross(ax)
        ax.set_title(t, fontsize=8, pad=2.5, color=col, fontweight="bold")
        ax.text(0.5, -0.05, f"CPM {c:.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
                color=col, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(1.1)
    # bracket from ball to band
    gain = D.restore("3691", "cpm_adjusted")[0]
    rec = 100 * (cpm[1] - cpm[0]) / (cpm[2] - cpm[0])
    xl, xr = xb[0] + wb / 2, xb[1] + wb / 2
    yy = 0.085
    fig.add_artist(plt.Line2D([xl, xl, xr, xr], [yy + 0.035, yy, yy, yy + 0.035], color=C["band"], lw=0.9))
    fig.text((xl + xr) / 2, yy - 0.012, f"+{gain:.1f} points: {rec:.0f}% of the ball-to-full gap",
             ha="center", va="top", fontsize=7.4, color=C["band"], fontweight="bold")
    fig.text(xb[2] + wb / 2, yy - 0.012, "CPM: mean sensitivity at 7 FP rates,\n353 test scans",
             ha="center", va="top", fontsize=6.6, color="#555555", linespacing=1.05)
    # hatch key
    kx = xa[0]
    ky = 0.005
    fig.add_artist(Rectangle((kx, ky), 0.016, 0.04, transform=fig.transFigure, facecolor="black",
                             edgecolor="#444444", hatch="////", lw=0.5))
    fig.text(kx + 0.022, ky + 0.02, "hidden: set to air before resampling", va="center", fontsize=6.8)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 2
def visibility(path):
    import json
    from matplotlib.image import imread
    img = vessel_case()
    ball, band = masks()
    u, q = probe_geometry()
    src = source_mask(q)
    W, H = style.TEXT, 2.95
    fig = plt.figure(figsize=(W, H))
    sz = 1.02
    w, h = sz / W, sz / H
    y0 = 0.50
    xs = [0.008 + i * (w + 0.012) for i in range(4)]
    fig.text(0.008, 0.985, "(a) What the image network sees (axial section, to scale)", fontsize=8.5,
             fontweight="bold", va="top")
    titles = ["Ball", "Ball + sources", "Ball + band", "Full field"]
    notes = ["$\\|v-p\\|\\leq 10$ mm", "voxels feeding the\n384 samples, $U(p)$",
             "whole shell $A(p)$,\n$r = %.1f$\u2013$%.1f$ mm" % BAND, "all voxels"]
    cols = [C["ball"], C["sources"], C["band"], C["full"]]
    vis = [ball, ball | src, ball | band, None]
    for i, (x, t, n, col, v) in enumerate(zip(xs, titles, notes, cols, vis)):
        ax = fig.add_axes((x, y0, w, h))
        show(ax, img, v)
        if i == 1:
            ring(ax, BAND[0], C["band"], lw=0.6, ls=(0, (2, 1.5)))
            ring(ax, BAND[1], C["band"], lw=0.6, ls=(0, (2, 1.5)))
        if i == 2:
            band_fill(ax, alpha=0.22)
        ring(ax, R_BALL, C["ball"], lw=0.9)
        cross(ax, s=1.6, lw=0.9)
        ax.set_title(t, fontsize=8, pad=2.5, color=col, fontweight="bold")
        ax.text(0.5, -0.04, n, transform=ax.transAxes, ha="center", va="top", fontsize=6.9,
                linespacing=1.1)
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(1.0)
        if i == 0:
            scalebar(ax)
    kx, ky = xs[0], 0.245
    fig.add_artist(Rectangle((kx, ky), 0.014, 0.035, transform=fig.transFigure, facecolor="black",
                             edgecolor="#444444", hatch="////", lw=0.5))
    fig.text(kx + 0.02, ky + 0.0175, "hidden voxels (set to air)", va="center", fontsize=6.8)
    fig.text(kx + 0.205, ky + 0.0175, "dashed orange: band outline, for reference", va="center",
             fontsize=6.8, color=C["band"])

    # (b) BioRender schematic of the sparse samples
    xb = xs[-1] + w + 0.035
    fig.add_artist(plt.Line2D([xb - 0.017, xb - 0.017], [0.27, 0.95], color="#bbbbbb", lw=0.6))
    fig.text(xb, 0.985, "(b) Sparse samples (schematic)", fontsize=8.5, fontweight="bold", va="top")
    art = imread(str(ROOT / "figures" / "biorender" / "probes_clean.png"))
    meta = json.load(open(ROOT / "figures" / "biorender" / "probes_beads.json"))
    bw = 1.30 / W
    bh = 1.30 / H
    ax = fig.add_axes((xb + 0.005, 0.37, bw, bh))
    ax.imshow(art[..., ::-1] if art.shape[-1] == 3 and art.max() > 1 else art)
    ax.axis("off")
    rb = meta["bead_r"]
    side = 0.55 * rb
    for bx, by in meta["beads"]:
        for ang in np.deg2rad([45, 135, 225, 315]):
            px, py = bx + 1.25 * rb * np.cos(ang), by + 1.25 * rb * np.sin(ang)
            ax.add_patch(Rectangle((px - side / 2, py - side / 2), side, side, facecolor=C["sources"],
                                   edgecolor="none", zorder=2))
    ax.set_xlim(0, meta["size"][0])
    ax.set_ylim(meta["size"][1], 0)
    tx0 = xb + bw + 0.012
    lines = [("o", C["spatial"], C["spatial"], "sample at $r_N$"), ("o", "white", C["spatial"], "sample at $r_F$"),
             ("s", C["sources"], C["sources"], "voxels it reads\n(native sources)"),
             ("o", C["ball"], C["ball"], "10 mm ball"), ("o", "#f6d9c6", "#e8b08a", "band: envelope\nof all sources")]
    for k, (m, fc, ec, txt) in enumerate(lines):
        yy = 0.83 - k * 0.105
        fig.add_artist(plt.Line2D([tx0 + 0.006], [yy], marker=m, ms=5.5, mfc=fc, mec=ec, mew=1.1, ls="",
                                  transform=fig.transFigure))
        fig.text(tx0 + 0.018, yy, txt, fontsize=6.8, va="center", linespacing=1.05)
    fig.text(xb, 0.33, "12 of 96 axes shown. The 384 sample values go to\nseparate heads, not to the image network.",
             fontsize=6.6, va="top", color="#444444", linespacing=1.1)

    # (c) mask before interpolation, 1D, neutral colours
    fig.add_artist(plt.Line2D([0.008, 0.992], [0.205, 0.205], color="#cccccc", lw=0.5))
    fig.text(0.008, 0.19, "(c) Masking before vs. after resampling (1D example)", fontsize=8.5,
             fontweight="bold", va="top")
    ax = fig.add_axes((0.0, 0.0, 1.0, 0.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bad, good = "#b2182b", "#1a1a1a"
    for row, (yy, hidden_val, res, txt, col, mark) in enumerate((
            (0.70, "+100 HU", "$-275$ HU", "Resample, then mask: the hidden value leaks into the sample", bad, r"$\times$"),
            (0.22, "air", "$-687.5$ HU", "Mask, then resample (ours): the same for any hidden value", good, r"$\checkmark$"))):
        xin, xout = 0.03, 0.27
        xq = xin + 0.375 * (xout - xin)
        ax.hlines(yy, 0.015, 0.285, color="#999999", lw=0.7)
        ax.vlines(0.165, yy - 0.2, yy + 0.2, color="#999999", lw=0.6, ls=":")
        ax.scatter([xin], [yy], marker="s", s=30, color="#888888", zorder=3)
        ax.scatter([xout], [yy], marker="s", s=30, facecolor="black" if row else "#d9d9d9",
                   edgecolor="#444444", hatch="////" if row else None, lw=0.6, zorder=3)
        ax.scatter([xq], [yy], marker="D", s=16, facecolor="white", edgecolor=col, lw=0.9, zorder=3)
        if row == 0:
            ax.text(xq, yy + 0.13, "sample", ha="center", fontsize=6.5)
            ax.text(0.168, yy + 0.13, "ball edge", ha="left", fontsize=6.0, color="#777777")
        ax.text(xin, yy + 0.13, "visible $-500$ HU", ha="center", fontsize=6.5)
        ax.text(xout, yy + 0.13, f"hidden: {hidden_val}", ha="center", fontsize=6.5)
        ax.text(0.31, yy, f"{mark}", color=col, fontsize=9, va="center", fontweight="bold")
        ax.text(0.335, yy, f"sample = {res}.", color=col, fontsize=7.4, va="center", fontweight="bold")
        ax.text(0.505, yy, txt, fontsize=7.4, va="center", color=col)
    save(fig, path)


if __name__ == "__main__":
    out = ROOT / "figures"
    teaser(out / "teaser.pdf")
    visibility(out / "visibility.pdf")
    print("band", BAND, "rN", R_N, "rF", R_F)
