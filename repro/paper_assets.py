"""Regenerate the paper's vector figures and tables from shareable aggregates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.ticker import FixedLocator, NullLocator

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": "#b6bdc5", "axes.labelcolor": "#253449", "text.color": "#172a3a",
    "xtick.color": "#425263", "ytick.color": "#425263", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .65, "grid.color": "#dce2e8", "grid.linewidth": .58, "pdf.fonttype": 42,
})

C = {"ball": "#27558a", "band": "#b94717", "full": "#465565", "sources": "#147a6f",
     "blind": "#426c7e", "spatial": "#7252a2", "compact": "#677481", "text": "#172a3a"}
LIGHT = {"ball": "#d7e4f4", "band": "#f4dccb", "full": "#e8edf1", "sources": "#a3cfc7"}
SEEDS = ("3691", "9143")
METHODS = ("ball", "band", "full")
SPARSE = {"pooled": "Archived compact", "compact": "Compact refit",
          "blind48": "Blind 48", "spatial48": "Directional 48",
          "blind96": "Blind 96", "spatial96": "Directional 96"}
RATES = [.125, .25, .5, 1, 2, 4, 8]
RATE_LABELS = ["0.125", "0.25", "0.5", "1", "2", "4", "8"]
ANATOMY = {"never_overlap": "Never overlaps", "any_overlap": "Any overlap",
           "linked_without_proposal": "Linked, no proposal", "unlinked": "Unlinked"}


def save(fig, path):
    fig.savefig(path, metadata={"Creator": "Aggregate paper renderer", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def section(ax, radius, **kwargs):
    ax.add_patch(Circle((0, 0), radius, **kwargs))


def geometry(path):
    """Draw a planar analogue with distances given in physical millimeters."""
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.97))
    fig.subplots_adjust(left=.016, right=.984, bottom=.405, top=.845, wspace=.12)
    titles = ["Ball", "Sparse readings", "Native sources", "Dense band", "Full crop"]
    details = ["10-mm radius", "2 radii, 96 axes", "Support of queries", "Complete shell", "Original field"]
    diag = np.linalg.norm([.9, .9, .8])
    near, far = 11.9, 13.4
    lo, hi = near - diag, far + diag
    theta = np.arange(12) * np.pi / 6 + .12
    unit = np.column_stack((np.cos(theta), np.sin(theta)))
    queries = np.r_[near * unit, far * unit]
    base = np.array([.23, .31])
    lattice = .9
    cells = (np.floor((queries - base) / lattice).astype(int)[:, None, :] +
             np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    cells = np.unique(cells.reshape(-1, 2), axis=0) * lattice + base
    for i, (ax, title, detail) in enumerate(zip(axes, titles, details)):
        ax.set_aspect("equal")
        ax.set_xlim(-17.5, 17.5)
        ax.set_ylim(-17.5, 17.5)
        ax.set_axis_off()
        ax.add_patch(Rectangle((-16.2, -16.2), 32.4, 32.4, facecolor=LIGHT["full"] if i == 4 else "#fafbfc",
                               edgecolor="#99a6b4", lw=.7))
        if i != 4:
            section(ax, 10, facecolor=LIGHT["ball"], edgecolor=C["ball"], lw=.95)
        else:
            section(ax, 10, fill=False, edgecolor="#738194", lw=.65, ls="--")
        if i == 1:
            for q in unit[:len(unit) // 2]:
                points = np.outer([-far, far], q)
                ax.plot(*points.T, color="#adb9c7", lw=.48, zorder=1)
            ax.scatter(*(near * unit).T, s=8, color=C["spatial"], lw=.0, zorder=3)
            ax.scatter(*(far * unit).T, s=8, facecolors="white", edgecolors=C["spatial"], lw=.8, zorder=3)
        if i == 2:
            for x, y in cells:
                ax.add_patch(Rectangle((x - lattice / 2, y - lattice / 2), lattice, lattice,
                                       facecolor=LIGHT["sources"], edgecolor=C["sources"], lw=.3))
        if i == 3:
            ax.add_patch(Wedge((0, 0), hi, 0, 360, width=hi - lo,
                               facecolor=LIGHT["band"], edgecolor=C["band"], lw=.8))
            for rad in (near, far):
                section(ax, rad, fill=False, edgecolor=C["band"], lw=.48, ls=":")
        ax.plot(0, 0, "+", color=C["text"], mew=.8, ms=5, zorder=4)
        ax.set_title(title, y=1.01, pad=3, fontsize=9.1)
        ax.text(.5, -.03, detail, transform=ax.transAxes, ha="center", va="top", fontsize=7.4)
    axes[0].plot([-14.5, -9.5], [-13.7, -13.7], color=C["text"], lw=1.15)
    axes[0].text(-12, -12.7, "5 mm", ha="center", va="bottom", fontsize=6.1)
    fig.text(.02, .966, "A   Visibility at one proposal", size=10, weight="bold", va="top")
    fig.text(.98, .96, "Axial schematic   •   same crop field", size=8, color="#526377", ha="right", va="top")
    inset = fig.add_axes((.02, .041, .96, .282))
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.set_axis_off()
    inset.axhline(.985, color="#dce2e8", lw=.7)
    inset.text(0, .81, "B   Resampling across the ball boundary", weight="bold", fontsize=9.3, va="center")
    inset.text(.997, .81, "Illustrative 1D interpolation", fontsize=7.7, color="#526377", va="center", ha="right")
    inset.hlines(.32, .02, .40, color="#99a6b4", lw=.9)
    inset.vlines(.246, .12, .55, color="#99a6b4", ls=":", lw=.8)
    inset.scatter([.078, .359], [.32, .32], marker="s", s=22, color=[C["ball"], C["band"]])
    query_x = .078 + .375 * (.359 - .078)
    inset.scatter([query_x], [.32], marker="D", s=16, facecolors="white", edgecolors=C["text"], lw=.8)
    inset.text(.078, .51, "−500 HU", color=C["ball"], ha="center", fontsize=7.8)
    inset.text(.359, .51, "+100 HU", color=C["band"], ha="center", fontsize=7.8)
    inset.text(query_x, .082, "inside query", ha="center", fontsize=7.4)
    inset.text(.359, .082, "outside source", ha="center", fontsize=7.4)
    inset.text(.455, .49, "Interpolate first", fontsize=8)
    inset.text(.455, .16, "Mask first (outside → −1000 HU)", fontsize=8)
    inset.text(.982, .49, "−275 HU  •  outside signal enters", fontsize=7.9, color=C["band"], ha="right")
    inset.text(.982, .16, "−687.5 HU  •  fixed air enters", fontsize=7.9, color=C["ball"], ha="right")
    save(fig, path)


def whisker(ax, row, y, color, marker="o", fill=True):
    x, low, high = (100 * row[k] for k in ("estimate", "lo", "hi"))
    ax.hlines(y, low, high, color=color, lw=1.2, zorder=2)
    ax.vlines([low, high], y - .06, y + .06, color=color, lw=.9, zorder=2)
    ax.plot(x, y, marker=marker, ms=4.8, mec=color, mfc=color if fill else "white",
            mew=1.1, color=color, zorder=3)


def forest_style(ax, xlabel):
    ax.grid(axis="x")
    ax.axvline(0, color="#81909e", ls=(0, (3, 2)), lw=.9, zorder=1)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines["left"].set_visible(False)


def rate_axis(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(RATES))
    ax.set_xticklabels(RATE_LABELS)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(.114, 8.75)
    ax.set_xlabel("False positives per scan")
    ax.grid(axis="y")


def restoration(data, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.69), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=.159, right=.975, bottom=.235, top=.847, wspace=.45)
    left.set_title("A   Paired restoration gains", loc="left")
    for i, seed in enumerate(SEEDS):
        effect = data["original"][seed]["pooled_capped"]["contrasts"]["restore"]
        for j, field in enumerate(("cpm_adjusted" if i == 0 else "cpm", "at_1")):
            row = effect[field]
            whisker(left, row, (3.2, 2.5, 1.1, .4)[2 * j + i], C["band"], "o" if i == 0 else "s", i == 0)
            left.text(100 * row["estimate"] + .12, (3.2, 2.5, 1.1, .4)[2 * j + i] + .20,
                      f"{100 * row['estimate']:+.2f}", fontsize=7.2, color=C["band"], va="bottom")
    left.set_yticks([3.2, 2.5, 1.1, .4], ["CPM · 3691", "CPM · 9143", "At 1 FP · 3691", "At 1 FP · 9143"])
    left.set_ylim(-.05, 3.85)
    left.set_xlim(-.4, 7.1)
    left.set_xticks([0, 2, 4, 6])
    forest_style(left, "Band minus ball (points)")
    right.set_title("B   Matched FROC", loc="left")
    for seed, ls in zip(SEEDS, ("-", (0, (3, 2)))):
        curve = data["dense_froc"][seed]
        for name in ("full", "ball", "band"):
            right.plot(curve["rates"], 100 * np.asarray(curve[name]), color=C[name], lw=1.5, ls=ls,
                       zorder=3 if name == "band" else 2)
    rate_axis(right)
    right.axvline(1, color="#a2adba", lw=.75, ls=":")
    right.set_ylabel("Sensitivity (%)")
    right.set_ylim(65, 100)
    right.set_yticks([70, 80, 90, 100])
    handles = [Line2D([], [], color=C[k], lw=2, label=label)
               for k, label in zip(METHODS, ("Ball", "Ball + band", "Full"))]
    handles.extend(Line2D([], [], color=C["full"], ls=ls, lw=1.2, label=seed)
                   for seed, ls in zip(SEEDS, ("-", (0, (3, 2)))))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.53, .005),
               columnspacing=1.6, handlelength=2.4)
    save(fig, path)


def seed_legend(fig):
    handles = [Line2D([], [], color=C["full"], marker=m, linestyle="", label=seed,
                      markerfacecolor=C["full"] if i == 0 else "white", markersize=4.8)
               for i, (seed, m) in enumerate(zip(SEEDS, ("o", "s")))]
    fig.legend(handles=handles, loc="lower center", title="Training seed", title_fontsize=7.8,
               frameon=False, ncol=2, bbox_to_anchor=(.54, .006))


def extent(ax, rows, padding=.42):
    low = min(0, *[100 * r["lo"] for r in rows])
    high = max(0, *[100 * r["hi"] for r in rows])
    ax.set_xlim(low - padding, high + padding)


def sparse_figure(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.03), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.188, right=.976, top=.88, bottom=.255, wspace=.69)
    left, right = axes
    specs = [("A   Gain over ball scorer",
              {**{f"{k}_minus_ball": v for k, v in SPARSE.items()}, "band_minus_ball": "Dense band"}),
             ("B   Geometry and width",
              {"geometry_48": "Geometry · 48", "geometry_96": "Geometry · 96",
               "capacity_blind": "Width · blind", "capacity_spatial": "Width · directional"})]
    for ax, (title, labels) in zip(axes, specs):
        ax.set_title(title, loc="left", fontsize=9.3)
        y = np.arange(len(labels) - 1, -1, -1)
        rows = []
        for i, seed in enumerate(SEEDS):
            effects = data["new"][f"spatial_{seed}_pooled_capped"]["contrasts"]
            for pos, key in zip(y, labels):
                row = effects[key]["cpm"]
                color = (C["band"] if key == "band_minus_ball" else C["spatial"] if
                         key.startswith(("geometry", "spatial")) or key == "capacity_spatial" else
                         C["blind"] if key.startswith("blind") or key == "capacity_blind" else C["compact"])
                whisker(ax, row, pos + (.145 if i == 0 else -.145), color, "o" if i == 0 else "s", i == 0)
                rows.append(row)
        ax.set_yticks(y, list(labels.values()))
        ax.set_ylim(-.52, len(labels) - .48)
        extent(ax, rows)
        forest_style(ax, "Paired CPM difference (points)")
    seed_legend(fig)
    save(fig, path)


def coverage(data, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.54), gridspec_kw={"width_ratios": [.93, 1.15]})
    fig.subplots_adjust(left=.155, right=.977, bottom=.217, top=.82, wspace=.7)
    counts = data["native_counts"]
    left.set_title("A   Exterior voxels reaching the crop", loc="left", fontsize=9.4)
    for name, y in (("band", 0), ("sources", 1)):
        n = counts["band_reached" if name == "band" else "query_reached"]
        left.barh(y, n / 1000, color=C[name], height=.44)
        left.text(n / 1000 + .33, y, f"{n:,.0f}", va="center", fontsize=8.1, color=C[name])
    left.set_yticks([1, 0], ["Query sources", "Dense band"])
    left.set_xlabel("Mean native voxels per proposal (×1,000)")
    left.set_xlim(0, 20.8)
    left.set_xticks([0, 5, 10, 15, 20])
    left.set_ylim(-.6, 1.6)
    left.grid(axis="x")
    left.set_axisbelow(True)
    left.spines["left"].set_visible(False)
    left.tick_params(axis="y", length=0)
    right.set_title("B   Same two-scale 3D encoder", loc="left", fontsize=9.4)
    effects = data["new"]["sources_3691_pooled_capped"]["contrasts"]
    series = [("sources_minus_ball", "Sources − ball", "sources"), ("band_minus_ball", "Band − ball", "band"),
              ("band_minus_sources", "Band − sources", "band")]
    rows = []
    for y, (k, _, color) in zip((2, 1, 0), series):
        row = effects[k]["cpm"]
        whisker(right, row, y, C[color])
        rows.append(row)
    extent(right, rows)
    right.set_yticks([2, 1, 0], [v[1] for v in series])
    right.set_ylim(-.57, 2.57)
    forest_style(right, "Paired CPM difference (points)")
    save(fig, path)


def anatomy(data, path):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(7.0, 2.80))
    fig.subplots_adjust(left=.094, right=.982, bottom=.254, top=.84, wspace=.12)
    groups = data["anatomy"]["groups"]
    items = (("never_overlap", C["ball"], "o"), ("any_overlap", C["band"], "s"))
    for i, (ax, seed) in enumerate(zip(axes, SEEDS)):
        result = data["anatomy"]["by_seed"][seed]["capped"]
        ax.set_title(f"{chr(ord('A') + i)}   Two-scale 3D · seed {seed}", loc="left")
        ax.axhline(0, color="#81909e", ls=(0, (3, 2)), lw=.9)
        ax.axvline(1, color="#a2adba", lw=.75, ls=":")
        for group, color, marker in items:
            s = result[group]["difference"]
            y, lo, hi = (100 * np.asarray([row[k] for row in s]) for k in ("estimate", "lo", "hi"))
            ax.errorbar(RATES, y, yerr=np.asarray([y - lo, hi - y]), marker=marker, ms=3.7, lw=1.15,
                        color=color, capsize=2, elinewidth=.9, zorder=3)
        rate_axis(ax)
        ax.set_ylim(-9, 40)
        ax.set_yticks([-5, 0, 10, 20, 30, 40])
    axes[0].set_ylabel("Band − ball sensitivity (points)")
    axes[1].tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=color, marker=m, ms=4.2, label=f"{ANATOMY[k]}  ·  {groups[k]['lesions']} lesions")
               for k, color, m in items]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.54, .018))
    save(fig, path)


def number(value, places=2, signed=False):
    if isinstance(value, dict):
        value = value["estimate"]
    return f"{100 * value:{'+' if signed else ''}.{places}f}"


def ci(row, places=2, sign=True):
    return (f"${number(row, places, sign)}\,[{number(row['lo'], places)},\,{number(row['hi'], places)}]$")


def rate_cell(row):
    estimate = number(row, 1, signed=True)
    lo, hi = (number(row[key], 1) for key in ("lo", "hi"))
    return rf"\raisebox{{-.4\height}}{{\shortstack{{${estimate}$\\$[{lo},\,{hi}]$}}}}"


def table(folder, name, label, caption, headers, cols, rows, wide=False, compact=False, position="t"):
    if name not in ("roles", "visibility", "anatomy_one"):
        position = "H"
    env = "table*" if wide and position != "H" else "table"
    style = r"\footnotesize\setlength{\tabcolsep}{4pt}" if compact else r"\small"
    body = [f"\\begin{{{env}}}[{position}]", r"\centering", style,
            f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}",
            f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
            " & ".join(headers) + r" \\", r"\midrule"]
    for row in rows:
        body.append(r"\midrule" if row is None else " & ".join(str(v) for v in row) + r" \\")
    body.extend((r"\bottomrule", r"\end{tabular}", f"\\end{{{env}}}", ""))
    (folder / f"{name}.tex").write_text("\n".join(body))


def original(data, seed):
    return data["original"][seed]["pooled_capped"]


def images(fit, seed):
    p = "tri_" if seed.startswith("tri_") else ""
    return [fit["methods"][f"{p}{mode}{seed[-4:]}_raw"] for mode in ("interior", "annulus", "full")]


def new(data, key, scope="3691"):
    return data["new"][f"{key}_{scope}_pooled_capped"]


def main_tables(data, folder):
    rows = []
    for seed, label in (("3691", "Two-scale 3D (3691)"), ("9143", "Two-scale 3D (9143)"),
                        ("single", "Single-stream 3D (3691)"), ("tri_9143", "Three-view 2D (9143)")):
        if seed == "single":
            fit = new(data, "architecture")
            methods = [fit["methods"][k] for k in METHODS]
            effect = fit["contrasts"]["band_minus_ball"]
        else:
            fit = original(data, seed)
            methods = images(fit, seed)
            effect = fit["contrasts"]["restore"]
        fraction = (methods[1]["cpm"]["estimate"] - methods[0]["cpm"]["estimate"]) / (
            methods[2]["cpm"]["estimate"] - methods[0]["cpm"]["estimate"])
        a = ci(effect.get("cpm_adjusted", effect["cpm"]))
        a += r"$^\dagger$" if "cpm_adjusted" in effect else ""
        rows.append([label, *[number(x["cpm"]) for x in methods], a,
                     ci(effect["at_1"]), f"{100 * fraction:.0f}\\%"])
    table(folder, "visibility", "visibility",
          r"Visibility on 353 reused scans. Absolute CPM is in percent. Paired band-minus-ball effects are in points. "
          r"$^\dagger$ denotes the primary CPM interval adjusted across three locked contrasts. Other intervals are "
          r"pointwise 95\%. Recovery is the band gain divided by the observed full-minus-ball CPM gap for each fit.",
          ["Fitted architecture", "Ball", "Ball + band", "Full", r"$\Delta$ CPM", r"$\Delta S_1$", "Recovery"],
          "lrrrrrr", rows, wide=True)

    rows = []
    for seed in SEEDS:
        result = data["anatomy"]["by_seed"][seed]["capped"]
        for name in ANATOMY:
            r = result[name]
            rows.append([seed, ANATOMY[name], data["anatomy"]["groups"][name]["lesions"],
                         number(r["ball"][3]), number(r["band"][3]), ci(r["difference"][3])])
    table(folder, "anatomy_one", "anatomy_one",
          r"Lesion sensitivity at global 1 FP/scan, with a separate FROC curve for each model. "
          r"All 461 lesions and all evaluated false marks contribute to the global curves. "
          r"Effects are paired, with pointwise 95\% intervals.", ["Seed", "Group", "$n$", "Ball", "Band", r"$\Delta S_1$"],
          "llrrrr", rows, wide=True)


def numbers(data, folder):
    a, b, t = (original(data, seed) for seed in (*SEEDS, "tri_9143"))
    ar = a["contrasts"]["restore"]
    br = b["contrasts"]["restore"]
    single = new(data, "architecture")
    source = new(data, "sources")
    sc = source["contrasts"]
    groups = data["anatomy"]["groups"]
    an = data["anatomy"]["by_seed"]
    development = data["new_development"]["architecture"]["development"]["3691"]
    macros = {
        "EvalScans": a["scans"], "EvalMarks": f"{a['marks']:,}", "EvalLesions": a["nodules"],
        "ProposedLesions": a["proposed_nodules"], "BallCpm": number(data["anchors"]["3691"]["ball"]),
        "BandCpm": number(data["anchors"]["3691"]["band"]), "FullCpm": number(data["anchors"]["3691"]["full"]),
        "RestorePrimary": ci(ar["cpm_adjusted"]), "RestorePrimaryPointwise": ci(ar["cpm"]),
        "RestoreRepeat": ci(br["cpm"]), "RestoreOne": ci(ar["at_1"]), "RepeatOne": ci(br["at_1"]),
        "PrimaryOneGain": number(ar["at_1"]), "RepeatOneGain": number(br["at_1"]),
        "RestoreGain": number(ar["cpm"], signed=True), "RepeatGain": number(br["cpm"], signed=True),
        "PrimaryRecovery": number(a["recovered_context_fraction"], 0),
        "RepeatRecovery": number(b["recovered_context_fraction"], 0),
        "TriGain": ci(t["contrasts"]["restore"]["cpm"]), "TriOne": ci(t["contrasts"]["restore"]["at_1"]),
        "SingleGain": ci(single["contrasts"]["band_minus_ball"]["cpm"]),
        "SingleEstimate": number(single["contrasts"]["band_minus_ball"]["cpm"]),
        "SingleOne": ci(single["contrasts"]["band_minus_ball"]["at_1"]),
        "SingleCheck": number(development["checkpoint"]["full"]["cpm"]),
        "SingleDev": number(development["fusion"]["full"]["cpm"]),
        "QuerySources": f"{data['native_counts']['query_reached']:,.0f}",
        "BandSources": f"{data['native_counts']['band_reached']:,.0f}",
        "SourcesGain": ci(sc["sources_minus_ball"]["cpm"]), "SourcesOne": ci(sc["sources_minus_ball"]["at_1"]),
        "CoverageGap": ci(sc["band_minus_sources"]["cpm"]), "CoverageOne": ci(sc["band_minus_sources"]["at_1"]),
        "AnyOverlap": groups["any_overlap"]["lesions"], "NeverOverlap": groups["never_overlap"]["lesions"],
        "AnyPrimary": ci(an["3691"]["capped"]["any_overlap"]["difference"][3]),
        "AnyRepeat": ci(an["9143"]["capped"]["any_overlap"]["difference"][3]),
        "NeverPrimary": ci(an["3691"]["capped"]["never_overlap"]["difference"][3]),
        "NeverRepeat": ci(an["9143"]["capped"]["never_overlap"]["difference"][3]),
        "HeteroPrimary": ci(an["3691"]["capped"]["heterogeneity_any_minus_never"][3]),
        "HeteroRepeat": ci(an["9143"]["capped"]["heterogeneity_any_minus_never"][3]),
        "RepresentPrimary": ci(a["contrasts"]["representation_at_logit"]["cpm_adjusted"]),
        "RepresentRepeat": ci(b["contrasts"]["representation_at_logit"]["cpm"]),
        "FusionPrimary": ci(a["contrasts"]["fusion_at_endpoint"]["cpm_adjusted"]),
        "PairShamPrimary": ci(a["contrasts"]["logit_pair_vs_sham"]["cpm"]),
        "PairShamRepeat": ci(b["contrasts"]["logit_pair_vs_sham"]["cpm"]),
        "PairUnpairedPrimary": ci(a["contrasts"]["logit_pair_vs_unpaired"]["cpm"]),
        "PairUnpairedRepeat": ci(b["contrasts"]["logit_pair_vs_unpaired"]["cpm"]),
        "PracticalPrimary": ci(a["contrasts"]["increment_over_morphology"]["cpm"]),
        "PracticalRepeat": ci(b["contrasts"]["increment_over_morphology"]["cpm"]),
        "PracticalOne": ci(a["contrasts"]["increment_over_morphology"]["at_1"]),
        "PracticalRepeatOne": ci(b["contrasts"]["increment_over_morphology"]["at_1"]),
        "TransferredPrimary": ci(ar["transfer"]["sensitivity"]), "TransferredRepeat": ci(br["transfer"]["sensitivity"]),
        "PrimaryUncapped": ci(data["original"]["3691"]["pooled_uncapped"]["contrasts"]["restore"]["cpm"]),
        "RepeatUncapped": ci(data["original"]["9143"]["pooled_uncapped"]["contrasts"]["restore"]["cpm"]),
    }
    for seed, suffix in zip(SEEDS, ("Primary", "Repeat")):
        s = new(data, "spatial", seed)
        for key, name in (("geometry_48", "GeometrySmall"), ("geometry_96", "GeometryLarge"),
                          ("capacity_blind", "CapacityBlind"), ("capacity_spatial", "CapacitySpatial"),
                          ("spatial48_minus_ball", "DirectionalSmall"), ("spatial96_minus_ball", "DirectionalLarge"),
                          ("compact_minus_pooled", "Refit"), ("band_minus_compact", "CompactGap"),
                          ("band_minus_spatial48", "ResidualSmall"),
                          ("band_minus_spatial96", "ResidualLarge")):
            macros[name + suffix] = ci(s["contrasts"][key]["cpm"])
        for key, name in (("geometry_48", "GeometrySmallOne"), ("geometry_96", "GeometryLargeOne"),
                          ("band_minus_spatial48", "ResidualSmallOne"), ("band_minus_spatial96", "ResidualLargeOne")):
            macros[name + suffix] = ci(s["contrasts"][key]["at_1"])
    (folder / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))


LOCKED_EFFECTS = {
    "embedding_readout_gain": "Embedding-only readout minus ball",
    "scalar_logit_gain": "Scalar gain with logit",
    "scalar_embedding_gain": "Scalar gain with embedding",
    "endpoint_logit_gain": "Endpoint gain with logit",
    "endpoint_embedding_gain": "Endpoint gain with embedding",
    "representation_at_logit": "Endpoint minus scalar at logit",
    "representation_at_embedding": "Endpoint minus scalar at embedding",
    "fusion_at_scalar": "Embedding minus logit gain for scalar",
    "fusion_at_endpoint": "Embedding minus logit gain for endpoints",
    "representation_by_fusion": "Representation-by-conditioning interaction",
    "dense_vs_selected_sparse_gain": "Dense restoration minus selected sparse gain",
    "logit_pair_vs_sham": "Pair minus learned sham (logit)",
    "logit_pair_vs_unpaired": "Pair minus unpaired (logit)",
    "logit_pair_vs_graymax": "Pair minus gray maximum (logit)",
    "logit_pair_vs_graymean": "Pair minus gray mean (logit)",
    "embed_pair_vs_sham": "Pair minus learned sham (embedding)",
    "embed_pair_vs_unpaired": "Pair minus unpaired (embedding)",
    "embed_pair_vs_graymax": "Pair minus gray maximum (embedding)",
    "embed_pair_vs_graymean": "Pair minus gray mean (embedding)",
    "full_endpoint_gain": "Endpoint gain over full crop",
    "full_morphology_gain": "Selected morphology gain over full crop",
    "full_pair_vs_morphology": "Endpoint minus selected morphology",
    "full_pair_vs_unpaired": "Endpoint minus unpaired",
    "full_pair_vs_sham": "Endpoint minus learned sham",
    "increment_over_morphology": "Endpoint increment over selected morphology",
    "increment_vs_sham": "Conditional endpoint minus conditional sham",
}


def roles_table(data, folder):
    labels = {"train": "Weights", "checkpoint": "Epoch selection", "fusion": "Readout development",
              "calibration": "Cutoff calibration", "confirmation": "Reuse subsets 6--7",
              "replication": "Reuse subsets 8--9", "quarantine": "Quarantined"}
    rows = [[label, v["scans"], f"{v['v2_marks']:,}"] for role, label in labels.items()
            for v in [data["fitting"]["roles"][role]]]
    table(folder, "roles", "roles", r"Patient-disjoint roles. The two quarantined scans share a patient.",
          ["Role", "Scans", "Released marks"], "lrr", rows)


def locked_tables(data, folder):
    labels = list(LOCKED_EFFECTS.items())
    for suffix, block in (("factorial", labels[:11]), ("compact", labels[11:19]), ("practical", labels[19:])):
        rows = []
        for key, label in block:
            row = [label]
            for seed in SEEDS:
                x = original(data, seed)["contrasts"][key]
                score = ci(x.get("cpm_adjusted", x["cpm"]))
                if "cpm_adjusted" in x:
                    score += r"$^\dagger$"
                row += [score, ci(x["at_1"])]
            rows.append(row)
        descriptions = {
            "factorial": r"Archived sparse factorial with each pathway's image-only baseline. "
                         r"$^\dagger$ marks a 98.33\% interval from the original three-contrast Bonferroni family.",
            "compact": r"Archived paired endpoint against the separately trained sham, unpaired set and gray shell heads. "
                       r"Each sham reference is the mean metric over eight evaluation permutations of one fitted sham.",
            "practical": r"Archived full-context comparisons. Gray mean is the development-selected morphology term. "
                         r"The conditional sham uses that same base. The upper CPM interval endpoint for the tested "
                         r"increment is below $+1$ point in both fits.",
        }
        table(folder, f"locked_{suffix}", f"locked_{suffix}",
              descriptions[suffix] + r" All other intervals are paired pointwise 95\%. Effects are in points.",
              ["Contrast", r"3691 $\Delta$ CPM", r"3691 $\Delta S_1$", r"9143 $\Delta$ CPM", r"9143 $\Delta S_1$"],
              "lrrrr", rows, wide=True, compact=True)
    rows = []
    for seed in SEEDS:
        for key, s in original(data, seed)["sham_seed_spread"].items():
            method = {"logit_pair_vs_sham": "Ball logit", "embed_pair_vs_sham": "Ball embedding",
                      "full_pair_vs_sham": "Full logit", "increment_vs_sham": "Full + morphology"}[key]
            rows.append([seed, method, number(s["cpm"]["sd"]), number(s["at_1"]["sd"]),
                         f"{number(s['cpm']['min'])}--{number(s['cpm']['max'])}"])
    table(folder, "sham_spread", "sham_spread",
          r"Variation over eight draws of each fitted sham. SD is in points and CPM range is in percent.",
          ["Seed", "Sham path", "CPM SD", "$S_1$ SD", "CPM range"], "llrrr", rows, wide=True)
    historical = data["historical"]
    hlabels = {"add_vs_raw": "Fixed chords minus raw Hessian", "add_vs_mean_shuffle": "Fixed chords minus shuffle",
               "tuned_vs_mean_shuffle": "Tuned chords minus shuffle",
               "tuned_vs_selected_shell": "Tuned chords minus gray maximum"}
    rows = [[label, ci(historical["hessian_contrasts"][k]["cpm"]), ci(historical["hessian_contrasts"][k]["at_1"])]
            for k, label in hlabels.items()]
    rows.append(["Archived visibility-by-chord interaction", ci(historical["visibility_interaction"]["cpm"]),
                 ci(historical["visibility_interaction"]["at_1"])])
    table(folder, "historical", "historical",
          r"Earlier correspondence and visibility comparisons on the same benchmark. Pointwise 95\% patient intervals.",
          ["Contrast", r"$\Delta$ CPM", r"$\Delta S_1$"], "lrr", rows, wide=True)


def visibility_tables(data, folder):
    configs = [("Two-scale 3691", original(data, "3691"), "3691"),
               ("Two-scale 9143", original(data, "9143"), "9143"),
               ("Three-view 9143", original(data, "tri_9143"), "tri_9143"),
               ("Single-stream 3691", new(data, "architecture"), None),
               ("Native sources 3691", new(data, "sources"), None)]
    rows = []
    for label, result, seed in configs:
        modes = ("Ball", "Band", "Full") if label != "Native sources 3691" else ("Query sources",)
        values = images(result, seed) if seed is not None else [result["methods"][k] for k in METHODS] if len(modes) == 3 else [result["methods"]["sources"]]
        for m, x in zip(modes, values):
            rows.append([label, m, number(x["cpm"]), *[number(z) for z in x["rates"]]])
        rows.append(None)
    table(folder, "absolute", "absolute",
          r"Absolute CPM and sensitivity in percent at the seven matched global FP/scan targets. "
          r"The query-source arm uses the original two-scale architecture and seed 3691.",
          ["Fit", "Input", "CPM", *RATE_LABELS], "llrrrrrrrr", rows[:-1], wide=True, compact=True)
    paired = []
    for label, result, seed in configs:
        keys = ("restore", "full_context", "outside_band") if seed is not None else (
            ("band_minus_ball", "full_minus_ball", "full_minus_band") if label.startswith("Single") else
            ("sources_minus_ball", "band_minus_sources"))
        names = ("Band $-$ ball", "Full $-$ ball", "Full $-$ band") if len(keys) == 3 else (
            "Sources $-$ ball", "Band $-$ sources")
        for key, name in zip(keys, names):
            paired.append([f"{label} / {name}", *[rate_cell(r) for r in result["contrasts"][key]["rates"]]])
        paired.append(None)
    table(folder, "paired_rates", "paired_rates",
          r"Paired matched sensitivity differences in points. Each cell shows the estimate above the two endpoints "
          r"of its pointwise 95\% patient interval.",
          ["Contrast", *RATE_LABELS], "lrrrrrrr", paired[:-1], wide=True, compact=True)
    rows = []
    for seed, label in (("3691", "Two-scale 3691"), ("9143", "Two-scale 9143"),
                        ("tri_9143", "Three-view 9143"), ("single", "Single-stream 3691"),
                        ("sources", "Native sources 3691")):
        datasets = (data["original"][seed] if seed in data["original"] else
                    {k.split("_3691_", 1)[1]: v for k, v in data["new"].items()
                     if k.startswith(("architecture" if seed == "single" else "sources") + "_3691_")})
        key = ("band_minus_ball" if seed == "single" else "sources_minus_ball") if seed not in data["original"] else "restore"
        for scope, fit in datasets.items():
            x = fit["contrasts"][key]
            desc = {"pooled_capped": "Pooled", "pooled_uncapped": "Pooled, uncapped",
                    "confirmation_capped": "Subsets 6--7", "replication_capped": "Subsets 8--9"}[scope]
            rows.append([label, desc, fit["scans"], ci(x["cpm"]), ci(x["at_1"])])
        rows.append(None)
    table(folder, "consistency", "consistency",
          r"Within-benchmark and cap checks. Band minus ball for dense visibility fits, and native query sources "
          r"minus ball for the source control. All intervals are pointwise 95\%.",
          ["Fit", "Scope", "Scans", r"$\Delta$ CPM", r"$\Delta S_1$"], "llrrr", rows[:-1], wide=True)
    rows = []
    for label, result, seed in configs:
        if seed:
            methods = images(result, seed)
        elif label.startswith("Single"):
            methods = [result["methods"][k] for k in METHODS]
        else:
            methods = [result["methods"][k] for k in ("ball", "sources", "band")]
        before, after = methods[:2]
        effkey = "restore" if seed else "band_minus_ball" if label.startswith("Single") else "sources_minus_ball"
        fx = result["contrasts"][effkey]["transfer"]
        v1, v2 = (m["transfer"] for m in (before, after))
        rows.append([label, f"{number(v1['sensitivity'])} / {number(v2['sensitivity'])}",
                     f"{v1['fp_per_scan']['estimate']:.3f} / {v2['fp_per_scan']['estimate']:.3f}",
                     ci(fx["sensitivity"]), f"${fx['fp_per_scan']['estimate']:+.3f}"
                     f"\,[{fx['fp_per_scan']['lo']:.3f},\,{fx['fp_per_scan']['hi']:.3f}]$"])
    table(folder, "transfer", "transfer",
          r"Transfer of strict cutoffs calibrated on a separate 88-scan role. Entries give before/after values "
          r"for ball and band, or ball and query sources. Sensitivity is in percent and differences are in points. "
          r"FP differences are in marks per scan. Paired intervals are pointwise 95\%.",
          ["Fit", "Sensitivity", "Achieved FP/scan", r"$\Delta$ Sens.", r"$\Delta$ FP/scan"], "lrrrr", rows, wide=True)


def spatial_tables(data, folder):
    effects = {"geometry_48": "Geometry at width 48", "geometry_96": "Geometry at width 96",
               "capacity_blind": "Width 96 minus 48, blind",
               "capacity_spatial": "Width 96 minus 48, directional", "compact_minus_pooled": "Compact refit minus archive"}
    for k, label in SPARSE.items():
        if k not in ("pooled", "compact"):
            effects[k + "_minus_compact"] = label + " minus compact refit"
            effects[k + "_minus_pooled"] = label + " minus archive"
    rows = []
    for key, label in effects.items():
        values = [new(data, "spatial", seed)["contrasts"][key] for seed in SEEDS]
        rows.append([label, *[ci(v[m]) for v in values for m in ("cpm", "at_1")]])
    table(folder, "attention_contrasts", "attention_contrasts",
          r"Matched angular and width contrasts and refit comparisons. Values are points with paired pointwise 95\% intervals.",
          ["Contrast", r"3691 $\Delta$ CPM", r"3691 $\Delta S_1$", r"9143 $\Delta$ CPM", r"9143 $\Delta S_1$"],
          "lrrrr", rows, wide=True, compact=True)
    rows, absolute = [], []
    for seed in SEEDS:
        result = new(data, "spatial", seed)
        for key, label in SPARSE.items():
            a, b = (result["contrasts"][n] for n in (key + "_minus_ball", "band_minus_" + key))
            rows.append([seed, label, *[ci(v[m]) for v in (a, b) for m in ("cpm", "at_1")]])
        rows.append(None)
        for key, label in {"ball": "Ball", **SPARSE, "band": "Band", "full": "Full"}.items():
            m = result["methods"][key]
            absolute.append([seed, label, number(m["cpm"]), *[number(v) for v in m["rates"]]])
        absolute.append(None)
    table(folder, "attention_gains", "attention_gains",
          r"Sparse gains over their exact image-only baseline and residual differences from the dense band. "
          r"Each effect is paired across patients. Intervals are pointwise 95\%.",
          ["Seed", "Sparse head", "Head $-$ ball CPM", "Head $-$ ball $S_1$", "Band $-$ head CPM", "Band $-$ head $S_1$"],
          "llrrrr", rows[:-1], wide=True, compact=True)
    table(folder, "attention_absolute", "attention_absolute",
          r"Absolute CPM and matched sensitivities in percent for all frozen sparse arms.",
          ["Seed", "Model", "CPM", *RATE_LABELS], "llrrrrrrrr", absolute[:-1], wide=True, compact=True)

    rows = []
    for fam in ("architecture", "sources", "spatial"):
        dev = data["new_development"][fam]["development"]
        order = (*METHODS, "sources") if fam == "sources" else (*METHODS, *SPARSE) if fam == "spatial" else METHODS
        for method in order:
            label = {**SPARSE, "ball": "Ball", "band": "Band", "full": "Full", "sources": "Query sources"}[method]
            prefix = {"architecture": "Single-stream", "sources": "Native sources", "spatial": "Sparse"}[fam]
            values = []
            for seed in SEEDS:
                for role in ("checkpoint", "fusion"):
                    m = dev[seed][role][method] if seed in dev else None
                    values.append(f"{number(m['cpm'])} / {number(m['at_1'])}" if m is not None else "--")
            rows.append([prefix, label, *values])
        rows.append(None)
    table(folder, "new_dev", "new_dev",
          r"New development-only comparisons. Each score cell gives CPM / $S_1$ in percent.",
          ["Family", "Arm", "3691 checkpoint", "3691 readout dev.", "9143 checkpoint", "9143 readout dev."],
          "llrrrr", rows[:-1], wide=True, compact=True)


def anatomy_tables(data, folder):
    groups = data["anatomy"]["groups"]
    rows = [[label, groups[k]["lesions"], groups[k]["with_proposal"], groups[k]["reference_proposal_pairs"],
             f"{groups[k]['median_reference_diameter_mm']:.2f}"] for k, label in ANATOMY.items()]
    table(folder, "anatomy_counts", "anatomy_counts",
          r"All-proposal geometry groups defined before score linkage. Proposal counts are strict mark--lesion "
          r"pairs, so one lesion can contribute more than one pair.",
          ["Group", "Lesions", "With proposal", "Pairs", "Median diameter (mm)"], "lrrrr", rows, wide=True)
    for tag in ("capped", "uncapped"):
        rows = []
        for seed in SEEDS:
            fit = data["anatomy"]["by_seed"][seed][tag]
            for k, name in ANATOMY.items():
                rows.append([seed, name, *[rate_cell(v) for v in fit[k]["difference"]]])
            rows.append([seed, "Any minus never", *[rate_cell(v) for v in fit["heterogeneity_any_minus_never"]]])
            rows.append(None)
        table(folder, f"anatomy_{tag}", f"anatomy_{tag}",
              rf"Band-minus-ball sensitivity by anatomy group at each global FP/scan target ({tag}). "
              r"Any minus never is the paired difference of subgroup gains. "
              r"Each cell shows the effect in points above the two endpoints of its 95\% pointwise patient interval.",
              ["Seed", "Group", *RATE_LABELS], "llrrrrrrr", rows[:-1], wide=True, compact=True)
    rows = []
    for seed in SEEDS:
        for mode in ("empirical", "transfer"):
            x = data["decision_audit"][f"{seed}_restore_{mode}"]["nodule_transitions"]
            for group in ("retained", "gained", "lost", "missed"):
                s = x[group]
                rows.append([seed, mode.capitalize(), group.capitalize(), s["lesions"],
                             s["before"]["visible_band"], s["after"]["visible_band"]])
            rows.append(None)
    table(folder, "decision", "decision",
          r"Descriptive overlap of selected decision marks at discrete empirical and transferred cutoffs. "
          r"The last two columns count the lesions whose respective before or after decision mark touches "
          r"the majority-reader contour in its visible band.",
          ["Seed", "Cutoff", "Status", "Lesions", "Before overlap", "After overlap"],
          "lllrrr", rows[:-1], wide=True, compact=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper", type=Path, default=Path("."))
    p.add_argument("--aggregate", type=Path)
    args = p.parse_args()
    data_path = args.aggregate or args.paper / "repro" / "aggregate.json"
    data = json.loads(data_path.read_text())
    assert data["schema"] == 1
    figures, tables = (args.paper / f for f in ("figures", "tables"))
    for folder in (figures, tables):
        folder.mkdir(parents=True, exist_ok=True)
    geometry(figures / "visibility.pdf")
    restoration(data, figures / "restoration.pdf")
    sparse_figure(data, figures / "spatial.pdf")
    coverage(data, figures / "coverage.pdf")
    anatomy(data, figures / "anatomy.pdf")
    main_tables(data, tables)
    numbers(data, tables)
    roles_table(data, tables)
    locked_tables(data, tables)
    visibility_tables(data, tables)
    spatial_tables(data, tables)
    anatomy_tables(data, tables)


if __name__ == "__main__":
    main()
