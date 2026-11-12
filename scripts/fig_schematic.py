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
    v = [curve([(-17, -13), (-8, -7), (0, 0)], 2.1, 1.9, seed_hu := 70),
         curve([(0, 0), (8, -1), (17, -6)], 1.7, 1.2),
         curve([(0, 0), (2, 9), (7, 17)], 1.6, 1.1),
         curve([(-17, 9), (-11, 11), (-6, 17)], 0.9, 0.7, 30),
         curve([(9, 17), (12, 10), (17, 9)], 0.8, 0.6, 30)]
    knot = blob(0, 0, 2.9, hu=60, lob=(0.04, 0.03), seed=3)
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


def show(ax, img, visible=None):
    im = img.copy()
    if visible is not None:
        im[~visible] = -1000
    ax.imshow(im, cmap="gray", vmin=WIN[0], vmax=WIN[1], extent=(-HALF, HALF, -HALF, HALF),
              interpolation="bilinear")
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#333333")
        s.set_linewidth(0.6)


def cross(ax, color="#ffd23f", s=2.0, lw=1.1):
    ax.plot([-s, s], [0, 0], color=color, lw=lw, zorder=6, solid_capstyle="butt")
    ax.plot([0, 0], [-s, s], color=color, lw=lw, zorder=6, solid_capstyle="butt")


def ring(ax, r, color, lw=0.9, ls="-", alpha=1):
    ax.add_patch(Circle((0, 0), r, fill=False, ec=color, lw=lw, ls=ls, alpha=alpha, zorder=5))


def scalebar(ax, color="white"):
    ax.plot([-14.6, -9.6], [-14.4, -14.4], color=color, lw=1.4, solid_capstyle="butt", zorder=7)
    ax.text(-12.1, -13.6, "5 mm", color=color, ha="center", va="bottom", fontsize=6.2, zorder=7)


# ----------------------------------------------------------------------------- Fig. 1
def teaser(path):
    vessel, nodule = vessel_case(), nodule_case()
    ball, band = masks()
    fig = plt.figure(figsize=(style.TEXT, 2.12))
    W, H = style.TEXT, 2.12
    size = 1.28 / W                      # image width as fraction of figure width
    h = 1.28 / H
    y0 = 0.205
    xa = [0.012, 0.012 + size + 0.012]
    xb0 = 0.405
    xb = [xb0 + i * (size + 0.020) for i in range(3)]

    # (a) the problem
    fig.text(xa[0], 0.975, "(a) Two explanations for the same candidate", fontsize=8.5,
             fontweight="bold", va="top")
    titles = ["Vessel branching", "Off-centre nodule"]
    notes = ["beyond the ball: the\nvessel continues", "beyond the ball: the\nrest of the nodule"]
    for x, img, t, n in zip(xa, (vessel, nodule), titles, notes):
        ax = fig.add_axes((x, y0, size, h))
        show(ax, img)
        ring(ax, R_BALL, "#ffd23f", lw=1.0, ls=(0, (3, 2)))
        cross(ax)
        ax.set_title(t, fontsize=8, pad=2.5, fontweight="normal")
        ax.text(0.5, -0.04, n, transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
                linespacing=1.05)
        if x == xa[0]:
            scalebar(ax)
    # separator
    fig.add_artist(plt.Line2D([xb0 - 0.03, xb0 - 0.03], [0.04, 0.93], color="#bbbbbb", lw=0.6))

    # (b) the intervention, applied to the off-centre nodule
    fig.text(xb0, 0.975, "(b) Our test: same crop, network and training; change only the visible "
             "native CT", fontsize=8.5, fontweight="bold", va="top")
    cpm = [D.arch_absolute("3691")[i] for i in range(3)]
    vis = [ball, ball | band, None]
    titles = ["Ball (10 mm)", "Ball + band", "Full field"]
    cols = [C["ball"], C["band"], C["full"]]
    axes = []
    for x, v, t, col, c in zip(xb, vis, titles, cols, cpm):
        ax = fig.add_axes((x, y0, size, h))
        show(ax, nodule, v)
        if v is not None:
            ring(ax, R_BALL, C["ball"], lw=0.9)
        if v is not None and v is not ball:
            ring(ax, BAND[0], C["band"], lw=0.8)
            ring(ax, BAND[1], C["band"], lw=0.8)
        cross(ax)
        ax.set_title(t, fontsize=8, pad=2.5, color=col, fontweight="bold")
        ax.text(0.5, -0.045, f"CPM {c:.1f}%", transform=ax.transAxes, ha="center", va="top",
                fontsize=7.6, color=col, fontweight="bold")
        for s in ax.spines.values():
            s.set_color(col)
            s.set_linewidth(1.1)
        axes.append(ax)
    gain = D.restore("3691", "cpm_adjusted")[0]
    rec = 100 * (cpm[1] - cpm[0]) / (cpm[2] - cpm[0])
    fig.text((xb[0] + xb[1] + size) / 2, 0.035,
             f"band: +{gain:.1f} CPM points, {rec:.0f}% of the ball-to-full gap",
             ha="center", va="bottom", fontsize=7.4, color=C["band"], style="italic")
    for i in range(2):
        a = FancyArrowPatch((xb[i] + size + 0.002, y0 + h / 2), (xb[i + 1] - 0.002, y0 + h / 2),
                            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                            lw=0.8, color="#444444")
        fig.add_artist(a)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 2
def visibility(path):
    img = vessel_case()
    ball, band = masks()
    u, q = probe_geometry()
    src = source_mask(q)
    W, H = style.TEXT, 2.62
    fig = plt.figure(figsize=(W, H))
    size = 1.215 / W
    h = 1.215 / H
    y0 = 0.40
    gap = (1 - 0.01 - 5 * size) / 4
    xs = [0.005 + i * (size + gap) for i in range(5)]
    titles = ["Ball", "Sparse readings", "Native sources", "Ball + band", "Full field"]
    notes = ["$\\|v-p\\|\\leq 10$ mm", "96 axes $\\times$ 2 signs $\\times$\n2 radii = 384 values",
             "$B(p)\\cup U(p)$: voxels\nfeeding the 384 values", "$B(p)\\cup A(p)$:\ncomplete shell",
             "original field"]
    cols = [C["ball"], C["spatial"], C["sources"], C["band"], C["full"]]
    vis = [ball, ball, ball | src, ball | band, None]
    fig.text(0.005, 0.985, "(a) Visibility conditions at one proposal (axial section, to scale)",
             fontsize=8.5, fontweight="bold", va="top")
    for i, (x, t, n, col, v) in enumerate(zip(xs, titles, notes, cols, vis)):
        ax = fig.add_axes((x, y0, size, h))
        show(ax, img, v)
        if v is not None:
            ring(ax, R_BALL, C["ball"], lw=0.8)
        if i == 1:
            for uu in u:
                ax.plot([-R_F * uu[0], R_F * uu[0]], [-R_F * uu[1], R_F * uu[1]], color="#d9c9ef",
                        lw=0.35, zorder=4)
            nq = len(q) // 4
            near = np.r_[q[:nq], q[2 * nq:3 * nq]]
            far = np.r_[q[nq:2 * nq], q[3 * nq:]]
            ax.scatter(*near.T, s=6, color="#cbb3ef", lw=0, zorder=6)
            ax.scatter(*far.T, s=6, facecolor="none", edgecolor="#cbb3ef", lw=0.6, zorder=6)
        if i == 3:
            ring(ax, BAND[0], C["band"], lw=0.7)
            ring(ax, BAND[1], C["band"], lw=0.7)
        cross(ax, s=1.6, lw=0.9)
        ax.set_title(t, fontsize=8, pad=2.5, color=col, fontweight="bold")
        ax.text(0.5, -0.04, n, transform=ax.transAxes, ha="center", va="top", fontsize=7.0,
                linespacing=1.1)
        for s in ax.spines.values():
            s.set_color(col)
            s.set_linewidth(1.0)
        if i == 0:
            scalebar(ax)
        if i == 3:
            a0, a1 = BAND
            ang = np.deg2rad(-135)
            ax.annotate("", xy=(a1 * np.cos(ang), a1 * np.sin(ang)), xytext=(a0 * np.cos(ang), a0 * np.sin(ang)),
                        arrowprops=dict(arrowstyle="<->", color="white", lw=0.7, shrinkA=0, shrinkB=0),
                        zorder=8)
            ax.text(-15.6, -15.4, f"{a0:.1f}\u2013{a1:.1f} mm", color="white", fontsize=6.2,
                    ha="left", va="bottom", zorder=8)

    # (b) mask before interpolation, 1D
    ax = fig.add_axes((0.005, 0.0, 0.99, 0.20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(0.005, 0.215, "(b) Hidden voxels become air before resampling "
             "(1D example; the query lies 37.5% of the way to the outside voxel)", fontsize=8.5,
             fontweight="bold", va="bottom")
    xin, xout = 0.05, 0.33
    xq = xin + 0.375 * (xout - xin)
    ax.hlines(0.5, 0.02, 0.36, color="#999999", lw=0.8)
    ax.vlines((xin + xout) / 2 + 0.02, 0.15, 0.85, color="#999999", lw=0.7, ls=":")
    ax.text((xin + xout) / 2 + 0.024, 0.86, "ball boundary", fontsize=6.4, color="#666666", va="top")
    ax.scatter([xin], [0.5], marker="s", s=26, color=C["ball"], zorder=3)
    ax.scatter([xout], [0.5], marker="s", s=26, color=C["band"], zorder=3)
    ax.scatter([xq], [0.5], marker="D", s=16, facecolor="white", edgecolor="#222222", lw=0.8, zorder=3)
    ax.text(xin, 0.12, "inside voxel\n$-500$ HU", ha="center", fontsize=6.8, color=C["ball"],
            va="bottom", linespacing=1.0)
    ax.text(xout, 0.12, "outside voxel\n$+100$ HU", ha="center", fontsize=6.8, color=C["band"],
            va="bottom", linespacing=1.0)
    ax.text(xq, 0.70, "query", ha="center", fontsize=6.8)
    ax.text(0.40, 0.66, "Interpolate first:", fontsize=7.6, fontweight="bold")
    ax.text(0.40, 0.22, "Mask first:", fontsize=7.6, fontweight="bold")
    ax.text(0.535, 0.66, "$0.625\\,(-500) + 0.375\\,(+100) = -275$ HU, so outside signal leaks in",
            fontsize=7.6, color=C["band"])
    ax.text(0.535, 0.22, "$0.625\\,(-500) + 0.375\\,(-1000) = -687.5$ HU, so only fixed air enters",
            fontsize=7.6, color=C["ball"])
    fig.add_artist(plt.Line2D([0.005, 0.995], [0.255, 0.255], color="#cccccc", lw=0.5))
    save(fig, path)


if __name__ == "__main__":
    out = ROOT / "figures"
    teaser(out / "teaser.pdf")
    visibility(out / "visibility.pdf")
    print("band", BAND, "rN", R_N, "rF", R_F)
