"""Schematic figures drawn to scale on a synthetic axial CT phantom.

  figures/teaser.pdf      Fig. 1  problem (two explanations) + intervention (ball / band / full)
  figures/visibility.pdf  Fig. 2  the four image conditions (one row)

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
    ax.add_patch(Rectangle((-16.2, 10.6), 7.6, 5.6, facecolor="black", edgecolor="none", alpha=0.75, zorder=6))
    ax.plot([-14.9, -9.9], [14.6, 14.6], color=color, lw=1.4, solid_capstyle="butt", zorder=7)
    ax.text(-12.4, 13.8, "5 mm", color=color, ha="center", va="top", fontsize=6.2, zorder=7)


# ----------------------------------------------------------------------------- Fig. 1
def _frame(ax, col, lw=1.0):
    for sp in ax.spines.values():
        sp.set_color(col)
        sp.set_linewidth(lw)


def teaser(path):
    """One-column Fig. 1: (a) the ambiguity as one row of four panels, (b) the intervention (1 x 3)."""
    vessel, nodule = vessel_case(), nodule_case()
    ball, band = masks()
    W, H = style.COL, 3.02
    fig = plt.figure(figsize=(W, H))

    def box(x_in, y_in, w_in, h_in):
        return fig.add_axes((x_in / W, y_in / H, w_in / W, h_in / H))

    # (a) one row: vessel ball / vessel full / nodule ball / nodule full
    fig.text(0.02 / W, 1 - 0.02 / H, "(a) A vessel and a nodule edge look alike inside the ball",
             fontsize=7.6, fontweight="bold", va="top")
    s_a, g_in, g_out = 0.74, 0.04, 0.14
    x0 = (W - 4 * s_a - 2 * g_in - g_out) / 2
    xa = [x0, x0 + s_a + g_in, x0 + 2 * s_a + g_in + g_out, x0 + 3 * s_a + 2 * g_in + g_out]
    ya = H - 0.42 - s_a
    cases = [(vessel, ball, "ball only", C["ball"]), (vessel, None, "full view", C["full"]),
             (nodule, ball, "ball only", C["ball"]), (nodule, None, "full view", C["full"])]
    under = ["", "tube: vessel", "", "round: nodule"]
    for k, (img, vis, t, col) in enumerate(cases):
        ax = box(xa[k], ya, s_a, s_a)
        show(ax, img, vis)
        ring(ax, R_BALL, C["ball"], lw=0.8)
        cross(ax, s=1.8, lw=0.9)
        ax.set_title(t, fontsize=6.6, pad=1.5, color=col, fontweight="bold")
        if under[k]:
            ax.text(0.5, -0.03, under[k], transform=ax.transAxes, ha="center", va="top", fontsize=6.2)
    for k0, name in ((0, "Vessel"), (2, "Off-centre nodule")):
        xc = (xa[k0] + xa[k0 + 1] + s_a) / 2
        fig.text(xc / W, (ya + s_a + 0.15) / H, name, ha="center", va="bottom", fontsize=7.0)
        fig.add_artist(plt.Line2D([(xa[k0] + 0.05) / W, (xa[k0 + 1] + s_a - 0.05) / W],
                                  [(ya + s_a + 0.135) / H] * 2, color="#888888", lw=0.5))
    ysep = ya - 0.22
    fig.add_artist(plt.Line2D([0.02, 0.98], [ysep / H] * 2, color="#bbbbbb", lw=0.5))

    # (b) intervention on the off-centre nodule
    fig.text(0.02 / W, (ysep - 0.04) / H, "(b) Same crop, network and training; only visible voxels vary",
             fontsize=7.6, fontweight="bold", va="top")
    s_b = 0.90
    gap = (W - 0.10 - 3 * s_b) / 2
    xb = [0.05 + k * (s_b + gap) for k in range(3)]
    yb = 0.42
    cpm = D.arch_absolute("3691")
    vis = [ball, ball | band, None]
    titles = ["Ball", "Ball + band", "Full field"]
    cols = [C["ball"], C["band"], C["full"]]
    for k, (x, v, t, col, c) in enumerate(zip(xb, vis, titles, cols, cpm)):
        ax = box(x, yb, s_b, s_b)
        show(ax, nodule, v)
        if v is not None and v is not ball:
            band_fill(ax, alpha=0.22)
        ring(ax, R_BALL, C["ball"], lw=0.9)
        cross(ax, s=1.8, lw=1.0)
        ax.set_title(t, fontsize=7.2, pad=1.5, color=col, fontweight="bold")
        ax.text(0.5, -0.03, f"{c:.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=7.4,
                color=col, fontweight="bold")
        _frame(ax, col)
        if k == 0:
            scalebar(ax)
    gain = D.restore("3691", "cpm_adjusted")[0]
    rec = 100 * (cpm[1] - cpm[0]) / (cpm[2] - cpm[0])
    xl, xr = (xb[0] + s_b / 2) / W, (xb[1] + s_b / 2) / W
    yy = 0.17 / H
    fig.add_artist(plt.Line2D([xl, xl, xr, xr], [yy + 0.06 / H, yy, yy, yy + 0.06 / H], color="#222222", lw=0.8))
    fig.text((xl + xr) / 2, yy - 0.02 / H, f"+{gain:.1f} CPM points ({rec:.0f}% of the gap)", ha="center",
             va="top", fontsize=6.8, color="#222222", fontweight="bold")
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 2
def visibility(path):
    """One-column Fig. 2: the four image conditions as one row of four panels."""
    img = vessel_case()
    ball, band = masks()
    u, q = probe_geometry()
    src = source_mask(q)
    W, H = style.COL, 1.30
    fig = plt.figure(figsize=(W, H))

    def box(x_in, y_in, w_in, h_in):
        return fig.add_axes((x_in / W, y_in / H, w_in / W, h_in / H))

    titles = ["Ball", "Ball + sources", "Ball + band", "Full field"]
    notes = ["$B(p)$, 10 mm", r"$B(p)\cup U(p)$", r"$B(p)\cup A(p)$", "all voxels"]
    cols = [C["ball"], C["sources"], C["band"], C["full"]]
    vis = [ball, ball | src, ball | band, None]
    sz, g = 0.755, 0.06
    x0 = (W - 4 * sz - 3 * g) / 2
    yb = 0.27
    for k, (t, n, col, v) in enumerate(zip(titles, notes, cols, vis)):
        ax = box(x0 + k * (sz + g), yb, sz, sz)
        show(ax, img, v)
        if k == 1:
            ring(ax, BAND[0], C["band"], lw=0.5, ls=(0, (2, 1.5)))
            ring(ax, BAND[1], C["band"], lw=0.5, ls=(0, (2, 1.5)))
        if k == 2:
            band_fill(ax, alpha=0.22)
        ring(ax, R_BALL, C["ball"], lw=0.8)
        cross(ax, s=1.6, lw=0.9)
        ax.set_title(t, fontsize=6.8, pad=1.5, color=col, fontweight="bold")
        ax.text(0.5, -0.04, n, transform=ax.transAxes, ha="center", va="top", fontsize=6.4)
        _frame(ax, col)
    save(fig, path)


if __name__ == "__main__":
    out = ROOT / "figures"
    teaser(out / "teaser.pdf")
    visibility(out / "visibility.pdf")
    print("band", BAND, "rN", R_N, "rF", R_F)
