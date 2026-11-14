"""One-column main-paper table (absolute CPM per visibility condition).
Writes tables/visibility_main.tex. Paired effects and their intervals are in Fig. 3a;
the full wide table (with intervals) stays in the supplement as tables/visibility.tex."""
import data as D

ROWS = [("3691", "Two-scale 3D, seed 1"), ("9143", "Two-scale 3D, seed 2"),
        ("single", "Single-stream 3D"), ("tri", "Three-view 2D")]

lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\caption{Absolute CPM (\%) per visibility condition; bold marks the better of ball and band, and the full field "
         r"is the reference. Recovery $\rho$ is Eq.~\eqref{eq:recovery}.}",
         r"\label{tab:visibility}",
         r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Ranker & Ball & Ball+band & Full (ref.) & $\rho$ \\", r"\midrule"]
for key, name in ROWS:
    b, a, f = D.arch_absolute(key)
    bb = f"\\textbf{{{b:.2f}}}" if b > a else f"{b:.2f}"
    aa = f"\\textbf{{{a:.2f}}}" if a >= b else f"{a:.2f}"
    lines.append(f"{name} & {bb} & {aa} & {f:.2f} & {100 * (a - b) / (f - b):.0f}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(D.ROOT / "tables" / "visibility_main.tex").write_text("\n".join(lines))
print("\n".join(lines))

# ---- Absolute performance of every sparse readout (one column) -> tables/sparse_main.tex
# Source: new/spatial_<seed>_pooled_capped/methods/<m>/{cpm,at_1}
SP = [("ball", "Ball only (frozen)"), ("pooled", "Compact head, initial fit"), ("compact", "Compact head, refit"),
      ("blind48", "Attention, blind, 48"), ("spatial48", "Attention, directional, 48"),
      ("blind96", "Attention, blind, 96"), ("spatial96", "Attention, directional, 96"),
      ("band", "Dense band (reference)"), ("full", "Full field (reference)")]
deg = {s: set(D.degenerate_heads(s)) for s in D.SEEDS}
L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
     r"\caption{Absolute CPM and $S_1$ (\%) of every readout of the 384 samples; bold marks the best sparse readout. "
     r"$^\ast$First checkpoint, identical to the ball-only ranker.}",
     r"\label{tab:sparse}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{Seed 1} & \multicolumn{2}{c}{Seed 2} \\",
     r"\cmidrule(lr){2-3}\cmidrule(l){4-5}", r"Readout & CPM & $S_1$ & CPM & $S_1$ \\", r"\midrule"]
comparable = [m for m, _ in SP if m not in ("band", "full")]
best = {}
for s_ in D.SEEDS:
    for fld in ("cpm", "at_1"):
        best[(s_, fld)] = max(D.new("spatial", s_)["methods"][m][fld]["estimate"] for m in comparable)
for m, name in SP:
    cells = []
    for s_ in D.SEEDS:
        meth = D.new("spatial", s_)["methods"][m]
        star = r"$^\ast$" if m in deg[s_] else ""
        for fld in ("cpm", "at_1"):
            v = 100 * meth[fld]["estimate"]
            txt = f"{v:.2f}"
            if m in comparable and abs(meth[fld]["estimate"] - best[(s_, fld)]) < 1e-12:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt + (star if fld == "cpm" else ""))
    if m == "band":
        L.append(r"\midrule")
    L.append(f"{name} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(D.ROOT / "tables" / "sparse_main.tex").write_text("\n".join(L))

# ---- Absolute sensitivity at 1 FP/scan by anatomy group -> tables/anatomy_main.tex
# Source: anatomy/by_seed/<seed>/capped/<group>/{ball,band}[3]  (index 3 = 1 FP/scan)
G = [("any_overlap", "Reaches lesion"), ("never_overlap", "Never reaches"),
     ("unlinked", "No contours"), ("linked_without_proposal", "No proposal")]
L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
     r"\caption{Sensitivity (\%) at 1 FP/scan by lesion group (global FP axis). Diam.: median diameter (mm).}",
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
