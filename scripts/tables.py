"""One-column main-paper table (absolute CPM per visibility condition).
Writes tables/visibility_main.tex. Paired effects and their intervals are in Fig. 3a;
the full wide table (with intervals) stays in the supplement as tables/visibility.tex."""
import data as D

ROWS = [("3691", "Two-scale 3D, seed 1"), ("9143", "Two-scale 3D, seed 2"),
        ("single", "Single-stream 3D"), ("tri", "Three-view 2D")]

lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\caption{Absolute CPM (\%) on the 353 evaluation scans for each visibility condition. "
         r"Recovery is the share of the ball-to-full gap closed by the band. Paired effects with "
         r"95\% intervals are in Fig.~\ref{fig:restoration}a.}",
         r"\label{tab:visibility}",
         r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Architecture & Ball & Ball+band & Full & Recovery \\", r"\midrule"]
for key, name in ROWS:
    b, a, f = D.arch_absolute(key)
    lines.append(f"{name} & {b:.2f} & {a:.2f} & {f:.2f} & {100 * (a - b) / (f - b):.0f}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(D.ROOT / "tables" / "visibility_main.tex").write_text("\n".join(lines))
print("\n".join(lines))

# ---- Absolute performance of every sparse readout (one column) -> tables/sparse_main.tex
# Source: new/spatial_<seed>_pooled_capped/methods/<m>/{cpm,at_1}
SP = [("ball", "Ball only (frozen)"), ("pooled", "Compact head, initial fit"), ("compact", "Compact head, refit"),
      ("blind48", "Attention, blind, 48"), ("spatial48", "Attention, directional, 48"),
      ("blind96", "Attention, blind, 96"), ("spatial96", "Attention, directional, 96"),
      ("band", "Dense band"), ("full", "Full field")]
deg = {s: set(D.degenerate_heads(s)) for s in D.SEEDS}
L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
     r"\caption{Absolute CPM and $S_1$ (\%) of every readout of the 384 sparse samples, with the image "
     r"conditions for reference. $^\ast$Selected the first checkpoint; scores identical to the ball-only ranker.}",
     r"\label{tab:sparse}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{Seed 1} & \multicolumn{2}{c}{Seed 2} \\",
     r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r"Readout & CPM & $S_1$ & CPM & $S_1$ \\", r"\midrule"]
for m, name in SP:
    cells = []
    for s in D.SEEDS:
        meth = D.new("spatial", s)["methods"][m]
        star = r"$^\ast$" if m in deg[s] else ""
        cells += [f"{100 * meth['cpm']['estimate']:.2f}{star}", f"{100 * meth['at_1']['estimate']:.2f}"]
    if m == "band":
        L.append(r"\midrule")
    L.append(f"{name} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(D.ROOT / "tables" / "sparse_main.tex").write_text("\n".join(L))

# ---- Absolute sensitivity at 1 FP/scan by anatomy group -> tables/anatomy_main.tex
# Source: anatomy/by_seed/<seed>/capped/<group>/{ball,band}[3]  (index 3 = 1 FP/scan)
G = [("any_overlap", "Extends into band"), ("never_overlap", "Does not reach band"),
     ("unlinked", "No contours"), ("linked_without_proposal", "No proposal")]
L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
     r"\caption{Sensitivity (\%) at a global 1 FP/scan by anatomy group, for the ball and band rankers. "
     r"Of the 461 reference lesions, 455 have usable contours and all remain in the global FROC. Diam.: median reference "
     r"diameter (mm). Between-group difference in the band$-$ball gain: $+13.16\,[3.68,\,22.85]$ (seed~1) and "
     r"$+15.61\,[4.69,\,24.92]$ (seed~2).}",
     r"\label{tab:anatomy}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
     r" & & & \multicolumn{2}{c}{Seed 1} & \multicolumn{2}{c}{Seed 2} \\",
     r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
     r"Group & $n$ & Diam. & Ball & Band & Ball & Band \\", r"\midrule"]
for g, name in G:
    cells = []
    for s in D.SEEDS:
        cells += [f"{D.anatomy(s, g, 'ball')[3][0]:.1f}", f"{D.anatomy(s, g, 'band')[3][0]:.1f}"]
    L.append(f"{name} & {D.anatomy_n(g)} & {D.anatomy_diameter(g):.1f} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(D.ROOT / "tables" / "anatomy_main.tex").write_text("\n".join(L))
