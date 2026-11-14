"""Results figures (all numbers via scripts/data.py, i.e. repro/aggregate.json).

  figures/restoration.pdf   Fig. 3  band-minus-ball effects across architectures, matched FROC,
                                    effect at each of the seven CPM operating points
  figures/coverage.pdf      Fig. 4  native query sources vs dense band in the same encoder
  figures/sparse.pdf        Fig. 5  sparse readouts of the same 384 values + compact controls
  figures/anatomy.pdf       Fig. 6  gains by whether the band can reach annotated lesion tissue

Intervals are paired-patient bootstrap 95% intervals (2,000 draws); the primary fit's CPM
restoration interval is the multiplicity-adjusted one (marked with a dagger).
"""
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

import style
from style import C, plt, save, ci_bar, zero_line, panel, SEED_MARK, SEED_NAME
import data as D

OUT = D.ROOT / "figures"


def rate_axis(ax, label=True, sparse=False):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(D.RATES))
    ax.set_xticklabels(["1/8", "", "1/2", "", "2", "", "8"] if sparse else D.RATE_LABELS)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.105, 9.5)
    if label:
        ax.set_xlabel("False positives per scan")


def forest_axis(ax, labels, ylim_pad=0.55):
    n = len(labels)
    ax.set_yticks(range(n - 1, -1, -1), labels)
    ax.set_ylim(-ylim_pad, n - 1 + ylim_pad)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    zero_line(ax)


def seed_handles(color="#444444", extra=()):
    h = [Line2D([], [], color=color, marker=SEED_MARK[s][0], ls="",
                mfc=color if SEED_MARK[s][1] else "white", mec=color, ms=4, label=SEED_NAME[s])
         for s in D.SEEDS]
    return h + list(extra)


# ----------------------------------------------------------------------------- Fig. 3
def restoration(path):
    fig = plt.figure(figsize=(style.COL, 3.75))
    a1 = fig.add_axes((0.355, 0.635, 0.29, 0.30))
    a2 = fig.add_axes((0.69, 0.635, 0.29, 0.30))
    b = fig.add_axes((0.13, 0.11, 0.36, 0.33))
    c = fig.add_axes((0.625, 0.11, 0.36, 0.33))
    rows = [("Two-scale 3D, seed 1", lambda f: D.restore("3691", "cpm_adjusted" if f == "cpm" else f)),
            ("Two-scale 3D, seed 2", lambda f: D.restore("9143", f)),
            ("Single-stream 3D", lambda f: D.single_stream(f)),
            ("Three-view 2D", lambda f: D.three_view(f))]
    for ax, field, title in ((a1, "cpm", "CPM"), (a2, "at_1", "$S_1$ (at 1 FP/scan)")):
        for i, (name, fn) in enumerate(rows):
            e, lo, hi = fn(field)
            ci_bar(ax, len(rows) - 1 - i, e, lo, hi, C["band"])
            dag = "$^\\dagger$" if (i == 0 and field == "cpm") else ""
            ax.text(hi + 0.25, len(rows) - 1 - i, f"{e:+.1f}{dag}", ha="left", va="center", fontsize=6.6,
                    color=C["band"])
        forest_axis(ax, [r[0] for r in rows])
        ax.set_xlim(-3.2, 10.5)
        ax.set_xticks([0, 4, 8])
        ax.set_title(title, fontsize=7.8, pad=2, fontweight="normal")
        ax.set_xlabel("Band $-$ ball (points)")
    a2.set_yticklabels([])
    panel(a1, "a", "Band $-$ ball, every architecture", x=-1.15, y=1.13)
    a2.text(1.0, -0.47, "$\\dagger$adjusted CI", transform=a2.transAxes, ha="right",
            fontsize=6.3, color="#555555")

    rates, _ = D.froc("3691")
    for seed, ls in zip(D.SEEDS, ("-", (0, (3, 1.6)))):
        r, curves = D.froc(seed)
        for k in ("ball", "full", "band"):
            b.plot(r, curves[k], color=C[k], lw=1.25 if k == "band" else 1.0, ls=ls,
                   zorder=3 if k == "band" else 2)
    rate_axis(b, sparse=True)
    b.set_xlabel("FP per scan")
    b.axvline(1, color="#bbbbbb", lw=0.6, ls=":")
    b.text(1.06, 96.0, "$S_1$", fontsize=6.6, color="#777777", va="top")
    b.set_ylim(64, 98)
    b.set_ylabel("Sensitivity (%)")
    b.grid(axis="y")
    panel(b, "b", "FROC of each model", x=-0.30, y=1.06)
    r1, c1 = D.froc("3691")
    hb = [Line2D([], [], color=C[k], lw=1.4, label=l) for k, l in
          (("full", "Full"), ("band", "Band"), ("ball", "Ball"))]
    b.legend(handles=hb, loc="lower right", ncol=1, fontsize=6.2, handlelength=1.3, labelspacing=0.15,
             borderaxespad=0.15)

    for j, seed in enumerate(D.SEEDS):
        m, fill = SEED_MARK[seed]
        xs = np.array(D.RATES) * (0.93 if j == 0 else 1.075)
        for x, (e, lo, hi) in zip(xs, D.restore_rates(seed)):
            c.vlines(x, lo, hi, color=C["band"], lw=1.0)
            c.hlines([lo, hi], x / 1.045, x * 1.045, color=C["band"], lw=0.9)
            c.plot(x, e, marker=m, ms=3.6, mec=C["band"], mfc=C["band"] if fill else "white", mew=0.9,
                   zorder=3)
    rate_axis(c, sparse=True)
    c.set_xlabel("FP per scan")
    c.axhline(0, color="#444444", lw=0.8, zorder=1)
    c.set_ylabel("Band $-$ ball (points)")
    c.set_ylim(-1.5, 13.5)
    c.grid(axis="y")
    panel(c, "c", "Gain per operating point", x=-0.30, y=1.06)
    c.legend(handles=seed_handles(C["band"]), loc="upper right", ncol=1, fontsize=6.2,
             handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 4
def coverage(path):
    """Single panel: paired effects of native sources and band in the same two-scale network.
    The voxel counts (2,849 vs 16,672) are reported in the text and caption, not plotted."""
    fig = plt.figure(figsize=(style.COL, 1.22))
    b = fig.add_axes((0.30, 0.27, 0.68, 0.66))
    rows = [("Sources $-$ ball", "sources_minus_ball", C["sources"]),
            ("Band $-$ ball", "band_minus_ball", C["band"]),
            ("Band $-$ sources", "band_minus_sources", C["band"])]
    for i, (lab, key, col) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, field, m in ((0.14, "cpm", "o"), (-0.14, "at_1", "D")):
            e, lo, hi = D.sources(key, field)
            ci_bar(b, y + dy, e, lo, hi, col, marker=m, ms=3.6, lw=1.1, cap=0.09)
    forest_axis(b, [r[0] for r in rows])
    b.set_xlim(-1.5, 7.2)
    b.set_xlabel("Paired difference (points), seed 1", labelpad=1.5)
    h = [Line2D([], [], color="#444444", marker="o", ls="", ms=3.6, label="CPM"),
         Line2D([], [], color="#444444", marker="D", ls="", ms=3.4, label="$S_1$")]
    b.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=2, fontsize=6.5,
             handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 5
SPARSE_ROWS = [("pooled", "Compact head (initial fit)"), ("compact", "Compact head (refit)"),
               ("blind48", "Attention, blind, width 48"), ("spatial48", "Attention, directional, 48"),
               ("blind96", "Attention, blind, width 96"), ("spatial96", "Attention, directional, 96"),
               ("band", "Ball + band (image)")]
CTRL_ROWS = [("representation_at_logit", "Endpoints $-$ one scalar"),
             ("fusion_at_endpoint", "Embedding $-$ score input"),
             ("logit_pair_vs_sham", "True $-$ shuffled pairs"),
             ("logit_pair_vs_unpaired", "True pairs $-$ unpaired"),
             ("increment_over_morphology", "Full + pairs $-$ full + morph.")]


def sparse(path):
    fig = plt.figure(figsize=(style.COL, 3.25))
    a = fig.add_axes((0.44, 0.535, 0.54, 0.415))
    b = fig.add_axes((0.44, 0.08, 0.54, 0.255))
    n = len(SPARSE_ROWS)
    flagged = {s: set(D.degenerate_heads(s)) for s in D.SEEDS}
    for i, (key, lab) in enumerate(SPARSE_ROWS):
        y = n - 1 - i
        for j, seed in enumerate(D.SEEDS):
            m, fill = SEED_MARK[seed]
            yy = y + (0.17 if j == 0 else -0.17)
            if key in flagged[seed]:
                a.text(0.15, yy, "= ball-only (first checkpoint)", fontsize=5.9, color="#888888",
                       va="center", style="italic")
                continue
            con = "band_minus_ball" if key == "band" else f"{key}_minus_ball"
            e, lo, hi = D.sparse(con, seed)
            col = C["band"] if key == "band" else (C["spatial"] if "spatial" in key else
                                                   C["blind"] if "blind" in key else C["compact"])
            ci_bar(a, yy, e, lo, hi, col, marker=m, filled=fill, ms=3.6, lw=1.05, cap=0.1)
    forest_axis(a, [r[1] for r in SPARSE_ROWS])
    a.set_xlim(-1.8, 6.6)
    a.set_xlabel("CPM gain over the ball-only ranker (points)", labelpad=1.5)

    panel(a, "a", "Sparse readouts versus the band", x=-0.80, y=1.03)
    h = seed_handles("#444444")
    a.legend(handles=h, loc="upper right", ncol=1, fontsize=6.3, handletextpad=0.2,
             borderaxespad=0.2, labelspacing=0.25)

    n = len(CTRL_ROWS)
    for i, (key, lab) in enumerate(CTRL_ROWS):
        y = n - 1 - i
        for j, seed in enumerate(D.SEEDS):
            m, fill = SEED_MARK[seed]
            e, lo, hi = D.original_contrast(key, seed)
            ci_bar(b, y + (0.16 if j == 0 else -0.16), e, lo, hi, C["compact"], marker=m, filled=fill,
                   ms=3.4, lw=1.0, cap=0.1)
    forest_axis(b, [r[1] for r in CTRL_ROWS])
    b.vlines(1, -0.45, 0.45, color="#1a1a1a", lw=0.8, ls=(0, (1, 1.2)))
    b.text(1.06, 0.12, "+1 target", fontsize=6.0, color="#1a1a1a", va="bottom")
    b.set_xlim(-1.8, 2.0)
    b.set_xlabel("Paired CPM difference (points)", labelpad=1.5)
    panel(b, "b", "Compact-head controls", x=-0.80, y=1.05)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 8
def anatomy(path):
    """One-column Fig. 7 as one row of three panels."""
    fig = plt.figure(figsize=(style.COL, 1.95))
    r = fig.add_axes((0.105, 0.30, 0.235, 0.47))
    axes = [fig.add_axes((0.475, 0.30, 0.25, 0.47)), fig.add_axes((0.745, 0.30, 0.25, 0.47))]
    # (a) how often the band reaches lesion tissue, by reference diameter
    rows = D.reach_by_diameter()
    for i, (lab, n, k) in enumerate(rows):
        r.bar(i, 100, color="none", width=0.66, edgecolor="#bbbbbb", lw=0.6, ls=(0, (2, 1.5)))
        r.bar(i, 100 * k / n, color=C["overlap"], width=0.66)
        r.text(i, 100 * k / n + 3, f"{k}/{n}", ha="center", va="bottom", fontsize=5.8)
    r.set_xticks(range(len(rows)), ["<6", "6\u201310", "\u226510"], fontsize=6.4)
    r.tick_params(axis="y", labelsize=6.4)
    r.set_ylim(0, 112)
    r.set_ylabel("Lesions extending\ninto band (%)", fontsize=6.6, labelpad=1, linespacing=1.0)
    r.set_xlabel("Diameter (mm)", fontsize=6.6, labelpad=1)
    r.grid(axis="y")
    r.set_axisbelow(True)
    r.text(-0.52, 1.04, "(a) By size", transform=r.transAxes, fontsize=7.4, fontweight="bold", va="bottom")
    groups = (("never_overlap", C["never"], "o"), ("any_overlap", C["overlap"], "s"))
    for ax, seed, letter in zip(axes, D.SEEDS, "bc"):
        for k, (g, col, m) in enumerate(groups):
            xs = np.array(D.RATES) * (0.93 if k == 0 else 1.075)
            for x, (e, lo, hi) in zip(xs, D.anatomy(seed, g)):
                ax.vlines(x, lo, hi, color=col, lw=0.8)
                ax.hlines([lo, hi], x / 1.05, x * 1.05, color=col, lw=0.7)
            ax.plot(xs, [q[0] for q in D.anatomy(seed, g)], marker=m, ms=2.6, color=col, lw=0.8)
        rate_axis(ax, label=False)
        ax.set_xticklabels(["1/8", "", "", "1", "", "", "8"])
        ax.tick_params(labelsize=6.4)
        ax.set_xlabel("FP per scan", fontsize=6.6, labelpad=1)
        zero_line(ax, vertical=False)
        ax.axvline(1, color="#bbbbbb", lw=0.6, ls=":")
        ax.grid(axis="y")
        ax.set_ylim(-8, 43)
        ax.text(0.02, 1.04, f"({letter}) {SEED_NAME[seed]}", transform=ax.transAxes, fontsize=7.4,
                fontweight="bold", va="bottom")
    axes[0].set_ylabel("Band $-$ ball (points)", fontsize=6.6, labelpad=1)
    axes[1].set_yticklabels([])
    h = [Line2D([], [], color=col, marker=m, ms=3.0, lw=0.9,
                label=(f"Lesion extends into band ($n={D.anatomy_n(g)}$)" if g == "any_overlap"
                       else f"Lesion does not reach band ($n={D.anatomy_n(g)}$)")) for g, col, m in groups]
    h = h[::-1]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.54, -0.015), ncol=2, fontsize=6.3,
               handlelength=1.5, columnspacing=1.0, handletextpad=0.4, frameon=False)
    save(fig, path)


if __name__ == "__main__":
    restoration(OUT / "restoration.pdf")
    coverage(OUT / "coverage.pdf")
    sparse(OUT / "sparse.pdf")
    anatomy(OUT / "anatomy.pdf")


# ----------------------------------------------------------------------------- Fig. 7
def robustness(path):
    fig = plt.figure(figsize=(style.COL, 3.0))
    a = fig.add_axes((0.31, 0.40, 0.67, 0.38))
    b1 = fig.add_axes((0.31, 0.085, 0.31, 0.135))
    b2 = fig.add_axes((0.67, 0.085, 0.31, 0.135))
    comps = [("3691", "Band $-$ ball, seed 1"), ("9143", "Band $-$ ball, seed 2"),
             ("sources", "Band $-$ sources, seed 1"), ("single", "Band $-$ ball,\nsingle-stream")]
    styles = [("pooled_capped", "All 353 scans", "#1a1a1a", "o", True),
              ("confirmation_capped", f"Confirmation half ({D.scope_scans('confirmation_capped')})", "#666666", "^", True),
              ("replication_capped", f"Replication half ({D.scope_scans('replication_capped')})", "#666666", "v", False),
              ("pooled_uncapped", "No 100-mark cap", "#9a9a9a", "D", True)]
    offs = [0.30, 0.10, -0.10, -0.30]
    for i, (comp, lab) in enumerate(comps):
        y = len(comps) - 1 - i
        for (scope, _, col, m, fill), dy in zip(styles, offs):
            e, lo, hi = D.robust(comp, scope)
            ci_bar(a, y + dy, e, lo, hi, col, marker=m, filled=fill, ms=3.1, lw=0.9, cap=0.06)
    forest_axis(a, [c[1] for c in comps], ylim_pad=0.5)
    a.set_xlim(-1.5, 9.2)
    a.set_xlabel("Paired CPM difference (points)", labelpad=1.5)
    panel(a, "a", "Pool halves and uncapped evaluation", x=-0.42, y=1.31)
    h = [Line2D([], [], color=col, marker=m, ls="", ms=3.4, mfc=col if fill else "white", label=lab)
         for _, lab, col, m, fill in styles]
    h = [h[0], h[2], h[1], h[3]]
    a.legend(handles=h, loc="lower center", bbox_to_anchor=(0.32, 1.0), ncol=2, fontsize=6.3,
             handletextpad=0.1, labelspacing=0.15, columnspacing=0.8, borderaxespad=0.2)

    for j, seed in enumerate(D.SEEDS):
        m, fill = SEED_MARK[seed]
        sens, fp = D.transfer(seed)
        ci_bar(b1, 1 - j, *sens, C["band"], marker=m, filled=fill, ms=3.4, lw=1.0, cap=0.12)
        ci_bar(b2, 1 - j, *fp, C["full"], marker=m, filled=fill, ms=3.4, lw=1.0, cap=0.12)
    for ax in (b1, b2):
        forest_axis(ax, [SEED_NAME[s] for s in D.SEEDS], ylim_pad=0.5)
    b2.set_yticklabels([])
    b1.set_xlabel("$\\Delta$ sensitivity (points)", labelpad=1)
    b2.set_xlabel("$\\Delta$ FP per scan ($\\leftarrow$ fewer)", labelpad=1)
    b1.set_xlim(-1.5, 7.5)
    b2.set_xlim(-0.3, 0.3)
    panel(b1, "b", "Thresholds fixed on calibration scans", x=-0.86, y=1.16)
    save(fig, path)


if __name__ == "__main__":
    robustness(OUT / "robustness.pdf")
