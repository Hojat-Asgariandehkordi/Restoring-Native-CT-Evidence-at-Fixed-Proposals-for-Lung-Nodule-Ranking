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
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

import style
from style import C, plt, save, ci_bar, zero_line, panel, SEED_MARK, SEED_NAME
import data as D

OUT = D.ROOT / "figures"


def rate_axis(ax, label=True):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(D.RATES))
    ax.set_xticklabels(D.RATE_LABELS)
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
    fig = plt.figure(figsize=(style.TEXT, 2.05))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.78, 0.78, 1.15, 1.15], wspace=0.16,
                          left=0.115, right=0.995, bottom=0.30, top=0.86)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    b = fig.add_subplot(gs[2])
    c = fig.add_subplot(gs[3])
    gs.update(wspace=0.16)
    # shift b, c to make room for b's y label
    for ax, dx in ((b, 0.055), (c, 0.085)):
        p = ax.get_position()
        ax.set_position((p.x0 + dx, p.y0, p.width - 0.03, p.height))

    rows = [("Two-scale 3D, fit 1", lambda f: D.restore("3691", "cpm_adjusted" if f == "cpm" else f)),
            ("Two-scale 3D, fit 2", lambda f: D.restore("9143", f)),
            ("Single-stream 3D", lambda f: D.single_stream(f)),
            ("Three-view 2D", lambda f: D.three_view(f))]
    for ax, field, title in ((a1, "cpm", "CPM"), (a2, "at_1", "$S_1$ (at 1 FP/scan)")):
        for i, (name, fn) in enumerate(rows):
            e, lo, hi = fn(field)
            col = C["band"] if hi > 0 and lo > 0 else C["null"]
            ci_bar(ax, len(rows) - 1 - i, e, lo, hi, col)
            ax.text(e, len(rows) - 1 - i + 0.2, f"{e:+.1f}", ha="center", va="bottom", fontsize=6.6,
                    color=col)
        forest_axis(ax, [r[0] for r in rows])
        ax.set_xlim(-3.2, 7.4)
        ax.set_xticks([-2, 0, 2, 4, 6])
        ax.set_title(title, fontsize=7.8, pad=2, fontweight="normal")
        ax.set_xlabel("Band $-$ ball (points)")
    a2.set_yticklabels([])
    panel(a1, "a", "Gain from the band", x=-0.02, y=1.12)

    rates, _ = D.froc("3691")
    for seed, ls in zip(D.SEEDS, ("-", (0, (3, 1.6)))):
        r, curves = D.froc(seed)
        for k in ("ball", "full", "band"):
            b.plot(r, curves[k], color=C[k], lw=1.25 if k == "band" else 1.0, ls=ls,
                   zorder=3 if k == "band" else 2)
    rate_axis(b)
    b.axvline(1, color="#bbbbbb", lw=0.6, ls=":")
    b.set_ylim(64, 98)
    b.set_ylabel("Sensitivity (%)")
    b.grid(axis="y")
    panel(b, "b", "Matched FROC", x=-0.02, y=1.12)
    hb = [Line2D([], [], color=C[k], lw=1.4, label=l) for k, l in
          (("ball", "Ball"), ("band", "Ball + band"), ("full", "Full"))]
    hb += [Line2D([], [], color="#444444", lw=1.0, ls=ls, label=SEED_NAME[s])
           for s, ls in zip(D.SEEDS, ("-", (0, (3, 1.6))))]
    b.legend(handles=hb, loc="lower right", fontsize=6.5, handlelength=1.8, labelspacing=0.2,
             borderaxespad=0.1)

    for j, seed in enumerate(D.SEEDS):
        m, fill = SEED_MARK[seed]
        xs = np.array(D.RATES) * (0.93 if j == 0 else 1.075)
        for x, (e, lo, hi) in zip(xs, D.restore_rates(seed)):
            c.vlines(x, lo, hi, color=C["band"], lw=1.0)
            c.hlines([lo, hi], x / 1.045, x * 1.045, color=C["band"], lw=0.9)
            c.plot(x, e, marker=m, ms=3.6, mec=C["band"], mfc=C["band"] if fill else "white", mew=0.9,
                   zorder=3)
    rate_axis(c)
    zero_line(c, vertical=False)
    c.set_ylabel("Band $-$ ball sensitivity (points)")
    c.set_ylim(-1.5, 13.5)
    c.grid(axis="y")
    panel(c, "c", "Gain per operating point", x=-0.02, y=1.12)
    c.legend(handles=seed_handles(C["band"]), loc="upper right", ncol=2, fontsize=6.5,
             handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 4
def coverage(path):
    fig = plt.figure(figsize=(style.COL, 2.25))
    a = fig.add_axes((0.30, 0.70, 0.66, 0.17))
    b = fig.add_axes((0.30, 0.13, 0.66, 0.33))
    q, band = D.native_counts()
    for y, n, col in ((1, q, C["sources"]), (0, band, C["band"])):
        a.barh(y, n / 1000, color=col, height=0.62)
        a.text(n / 1000 + 0.3, y, f"{n:,.0f}", va="center", fontsize=7, color=col)
    a.set_yticks([1, 0], ["Native sources", "Ball + band"])
    a.set_xlim(0, 20)
    a.set_xlabel("Exterior native voxels reaching the crop, per proposal ($\\times 10^3$)", labelpad=1.5)
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    a.set_ylim(-0.6, 1.6)
    panel(a, "a", "How much native CT each arm sees", x=-0.43, y=1.15)

    rows = [("Sources $-$ ball", "sources_minus_ball", C["sources"]),
            ("Band $-$ ball", "band_minus_ball", C["band"]),
            ("Band $-$ sources", "band_minus_sources", C["band"])]
    for i, (lab, key, col) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, field, filled in ((0.14, "cpm", True), (-0.14, "at_1", False)):
            e, lo, hi = D.sources(key, field)
            ci_bar(b, y + dy, e, lo, hi, col, marker="o" if field == "cpm" else "D", filled=filled, ms=3.6,
                   lw=1.1, cap=0.09)
    forest_axis(b, [r[0] for r in rows])
    b.set_xlim(-1.5, 7.2)
    b.set_xlabel("Paired difference (points), same two-scale encoder", labelpad=1.5)
    panel(b, "b", "Effect on ranking", x=-0.43, y=1.04)
    h = [Line2D([], [], color="#444444", marker="o", ls="", ms=3.6, label="CPM"),
         Line2D([], [], color="#444444", marker="D", ls="", ms=3.4, mfc="white", label="$S_1$")]
    b.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=6.5,
             handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 5
SPARSE_ROWS = [("pooled", "Compact (original fit)"), ("compact", "Compact refit"),
               ("blind48", "Attention, blind, 48"), ("spatial48", "Attention, directional, 48"),
               ("blind96", "Attention, blind, 96"), ("spatial96", "Attention, directional, 96"),
               ("band", "Dense band (reference)")]
CTRL_ROWS = [("representation_at_logit", "Endpoints vs. chord scalar"),
             ("fusion_at_endpoint", "Embedding vs. logit fusion"),
             ("logit_pair_vs_sham", "Paired vs. sham pairing"),
             ("logit_pair_vs_unpaired", "Paired vs. unpaired"),
             ("increment_over_morphology", "Full + pair vs. full + morph.")]


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
                a.plot(0, yy, marker="x", ms=4, color="#888888", mew=1.0, zorder=3)
                continue
            con = "band_minus_ball" if key == "band" else f"{key}_minus_ball"
            e, lo, hi = D.sparse(con, seed)
            col = C["band"] if key == "band" else (C["spatial"] if "spatial" in key else
                                                   C["blind"] if "blind" in key else C["compact"])
            ci_bar(a, yy, e, lo, hi, col, marker=m, filled=fill, ms=3.6, lw=1.05, cap=0.1)
    forest_axis(a, [r[1] for r in SPARSE_ROWS])
    a.set_xlim(-1.8, 6.6)
    a.set_xlabel("CPM gain over the frozen ball scorer (points)", labelpad=1.5)
    panel(a, "a", "Readouts of the same 384 values", x=-0.80, y=1.03)
    h = seed_handles("#444444", [Line2D([], [], color="#888888", marker="x", ls="", ms=4,
                                        label="no learned residual")])
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
    b.axvline(1, color=C["band"], lw=0.7, ls=(0, (1, 1.5)))
    b.text(1.06, -0.45, "+1 target", fontsize=6.2, color=C["band"], va="bottom")
    b.set_xlim(-1.8, 2.0)
    b.set_xlabel("Paired CPM difference (points)", labelpad=1.5)
    panel(b, "b", "Compact-head controls", x=-0.80, y=1.05)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 8
def anatomy(path):
    fig = plt.figure(figsize=(style.TEXT, 1.95))
    r = fig.add_axes((0.06, 0.30, 0.20, 0.56))
    axes = [fig.add_axes((0.375, 0.30, 0.295, 0.56)), fig.add_axes((0.695, 0.30, 0.295, 0.56))]
    # (a) how often the band reaches lesion tissue, by reference diameter
    rows = D.reach_by_diameter()
    for i, (lab, n, k) in enumerate(rows):
        r.bar(i, 100, color="#eeeeee", width=0.62, edgecolor="#cccccc", lw=0.5)
        r.bar(i, 100 * k / n, color=C["overlap"], width=0.62)
        r.text(i, 100 * k / n + 3, f"{k}/{n}", ha="center", va="bottom", fontsize=6.8)
    r.set_xticks(range(len(rows)), [lab.replace("\u2013<", "\u2013") for lab, _, _ in rows])
    r.set_ylim(0, 105)
    r.set_ylabel("Lesions reached (%)")
    r.set_xlabel("Reference diameter")
    r.grid(axis="y")
    r.set_axisbelow(True)
    panel(r, "a", "Which lesions the band reaches", x=-0.25, y=1.05)
    groups = (("never_overlap", C["never"], "o"), ("any_overlap", C["overlap"], "s"))
    for ax, seed, letter in zip(axes, D.SEEDS, "bc"):
        for k, (g, col, m) in enumerate(groups):
            xs = np.array(D.RATES) * (0.93 if k == 0 else 1.075)
            for x, (e, lo, hi) in zip(xs, D.anatomy(seed, g)):
                ax.vlines(x, lo, hi, color=col, lw=0.95)
                ax.hlines([lo, hi], x / 1.05, x * 1.05, color=col, lw=0.85)
            ax.plot(xs, [q[0] for q in D.anatomy(seed, g)], marker=m, ms=3.2, color=col, lw=0.9)
        rate_axis(ax)
        zero_line(ax, vertical=False)
        ax.axvline(1, color="#bbbbbb", lw=0.6, ls=":")
        ax.grid(axis="y")
        ax.set_ylim(-8, 43)
        panel(ax, letter, f"Gain by group, two-scale 3D {SEED_NAME[seed].lower()}", x=0.0, y=1.05)
    axes[0].set_ylabel("Band $-$ ball sensitivity (points)")
    axes[1].set_yticklabels([])
    h = [Line2D([], [], color=col, marker=m, ms=3.4, lw=0.9,
                label=f"{'Band reaches lesion' if g == 'any_overlap' else 'Band never reaches lesion'} "
                      f"($n={D.anatomy_n(g)}$)") for g, col, m in groups]
    axes[1].legend(handles=h, loc="upper right", fontsize=6.5, handlelength=1.6, labelspacing=0.2,
                   borderaxespad=0.1)
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
    comps = [("3691", "Band $-$ ball, fit 1"), ("9143", "Band $-$ ball, fit 2"),
             ("sources", "Band $-$ sources"), ("single", "Single-stream")]
    styles = [("pooled_capped", "All 353 scans", "#222222", "o", True),
              ("confirmation_capped", f"Confirmation half ({D.scope_scans('confirmation_capped')})", "#4c78a8", "^", True),
              ("replication_capped", f"Replication half ({D.scope_scans('replication_capped')})", "#72b7b2", "v", True),
              ("pooled_uncapped", "No 100-mark cap", "#e45756", "D", False)]
    offs = [0.27, 0.09, -0.09, -0.27]
    for i, (comp, lab) in enumerate(comps):
        y = len(comps) - 1 - i
        for (scope, _, col, m, fill), dy in zip(styles, offs):
            e, lo, hi = D.robust(comp, scope)
            ci_bar(a, y + dy, e, lo, hi, col, marker=m, filled=fill, ms=3.1, lw=0.9, cap=0.06)
    forest_axis(a, [c[1] for c in comps], ylim_pad=0.5)
    a.set_xlim(-1.5, 9.2)
    a.set_xlabel("Paired CPM difference (points)", labelpad=1.5)
    panel(a, "a", "Same effects on subsets and without the cap", x=-0.42, y=1.31)
    h = [Line2D([], [], color=col, marker=m, ls="", ms=3.4, mfc=col if fill else "white", label=lab)
         for _, lab, col, m, fill in styles]
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
    b2.set_xlabel("$\\Delta$ FP per scan", labelpad=1)
    b1.set_xlim(-1.5, 7.5)
    b2.set_xlim(-0.3, 0.3)
    panel(b1, "b", "Thresholds fixed on calibration scans", x=-0.86, y=1.16)
    save(fig, path)


if __name__ == "__main__":
    robustness(OUT / "robustness.pdf")
