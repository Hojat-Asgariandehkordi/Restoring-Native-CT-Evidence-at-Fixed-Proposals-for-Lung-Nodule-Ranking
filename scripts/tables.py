"""One-column main-paper table (absolute CPM per visibility condition).
Writes tables/visibility_main.tex. Paired effects and their intervals are in Fig. 3a;
the full wide table (with intervals) stays in the supplement as tables/visibility.tex."""
import data as D

ROWS = [("3691", "Two-scale 3D, fit 1"), ("9143", "Two-scale 3D, fit 2"),
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
