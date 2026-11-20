"""Relabel the supplement tables written by repro/paper_assets.py in the main paper's terminology.

Run after repro/paper_assets.py (see make_all.sh). Only labels, headers and captions change;
every number, row and the order of the tables (and hence their S-numbers) stay as generated.
"""
import re
from pathlib import Path

T = Path(__file__).resolve().parent.parent / "tables"

CAPTIONS = {
    "roles": "Patient-disjoint data roles. The two excluded scans belong to one repeated patient.",
    "visibility": r"Visibility results on the 353 evaluation scans. Absolute CPM is in percent; band-minus-ball effects are in points. $^\dagger$: interval adjusted for the three comparisons specified before evaluation; all other intervals are pointwise 95\%. Recovery is the band gain divided by the full-minus-ball CPM gap.",
    "anatomy_one": r"Sensitivity (\%) at 1~FP/scan by anatomy group, computed from one FROC curve per model over all 461 lesions and all false marks. Effects are paired band-minus-ball differences in points with pointwise 95\% intervals.",
    "absolute": r"Absolute CPM and sensitivity (\%) at the seven FP/scan rates of the CPM. The native-source condition uses the two-scale architecture and seed~1.",
    "paired_rates": r"Paired sensitivity differences (points) at each FP/scan rate. Each cell gives the estimate above its pointwise 95\% patient-level interval.",
    "consistency": r"Consistency checks: band minus ball for the dense visibility conditions (native sources minus ball for the source condition) on each half of the evaluation pool and without the 100-mark cap. Intervals are pointwise 95\%.",
    "transfer": r"Thresholds transferred from the separate 88-scan calibration role. Entries give ball/band (or ball/native sources); sensitivity is in percent, differences in points, FP burden in marks per scan. Intervals are pointwise 95\%.",
    "locked_factorial": r"Compact-head factorial (initial fits), each pathway against its own image-only baseline. Effects are in points. $^\dagger$: 98.33\% interval of the three-comparison family specified before evaluation; all other intervals are pointwise 95\%.",
    "locked_compact": r"Paired compact head (initial fit) against shuffled pairs, unpaired half-axes and grey-level morphology heads. Shuffled-pair values average eight evaluation permutations of one fitted head. Effects in points with pointwise 95\% intervals.",
    "locked_practical": r"Comparisons on top of the full-field ranker. The grey-level mean is the morphology score selected on the development role, and the conditional shuffled-pair head uses the same base. For the paired-head increment, the upper CPM bound is below $+1$ point for both seeds. Effects in points with pointwise 95\% intervals.",
    "sham_spread": r"Variation over the eight shuffled-pair permutations of each fitted head. SD in points, CPM range in percent.",
    "new_dev": r"Development-role results of the follow-up comparisons. Each cell gives CPM / $S_1$ in percent at the selected checkpoint and on the readout-development role.",
    "attention_contrasts": r"Direction and width contrasts of the attention heads and comparisons with the compact head. Effects in points with pointwise 95\% intervals. Seed-2 contrasts involving the three heads that tie the ball-only ranker (Sec.~\ref{supp:new}) are inconclusive.",
    "attention_gains": r"Gain of each sparse head over its ball-only ranker and remaining gap to the dense band. Paired effects in points with pointwise 95\% intervals.",
    "attention_absolute": r"Absolute CPM and sensitivity (\%) at the seven FP/scan rates for every sparse readout head.",
    "anatomy_counts": r"Anatomy groups over all matching proposals, defined before scores were linked. Pairs count strict mark--lesion matches, so one lesion can contribute several pairs.",
    "anatomy_capped": r"Band-minus-ball sensitivity (points) by anatomy group at each FP/scan rate, with the 100-mark cap. The last row of each seed is the difference of the two group gains. Each cell gives the estimate above its pointwise 95\% interval.",
    "anatomy_uncapped": r"As Tab.~\ref{tab:anatomy_capped}, without the 100-mark cap.",
    "decision": r"Per-lesion decisions of seed 1 and seed 2 at the empirical and transferred cutoffs (descriptive). The last two columns count lesions whose decision mark before or after restoring the band has a band that reaches the majority-reader contour.",
    "historical": r"Earlier comparisons on the same benchmark: classical half-axis readouts with fixed or tuned settings, and visibility. Pointwise 95\% patient-level intervals.",
}

PHRASES = [
    ("Never overlaps", "Does not reach band"), ("Never overlap", "Does not reach band"),
    ("Any overlap", "Extends into band"), ("Any minus never", "Extends $-$ does not reach"),
    ("Linked, no proposal", "No matching proposal"), ("Unlinked", "No contour link"),
    ("Archived compact", "Compact (initial)"), ("Compact refit minus archive", "Compact refit $-$ initial"),
    ("minus compact refit", "$-$ compact refit"), ("minus archive", "$-$ compact initial"),
    ("Blind 48", "Direction-blind 48"), ("Blind 96", "Direction-blind 96"),
    ("Directional 48", "Direction-aware 48"), ("Directional 96", "Direction-aware 96"),
    ("Width 96 minus 48, blind", "Width 96 $-$ 48, direction-blind"),
    ("Width 96 minus 48, directional", "Width 96 $-$ 48, direction-aware"),
    ("Geometry at width", "Aware $-$ blind, width"),
    ("Query sources", "Native sources"),
    ("Pair minus learned sham", "Paired $-$ shuffled pairs"), ("Pair minus unpaired", "Paired $-$ unpaired"),
    ("Pair minus gray maximum", "Paired $-$ grey max."), ("Pair minus gray mean", "Paired $-$ grey mean"),
    ("Endpoint minus learned sham", "Paired $-$ shuffled pairs"), ("Endpoint minus unpaired", "Paired $-$ unpaired"),
    ("Conditional endpoint minus conditional sham", "Conditional paired $-$ shuffled"),
    ("Endpoint minus selected morphology", "Paired $-$ selected morphology"),
    ("Endpoint increment over selected morphology", "Paired increment over morphology"),
    ("Endpoint gain over full crop", "Paired gain over full field"),
    ("Selected morphology gain over full crop", "Morphology gain over full field"),
    ("Embedding-only readout minus ball", "Embedding-only readout $-$ ball"),
    ("Dense restoration minus selected sparse gain", "Band $-$ selected sparse gain"),
    ("Representation-by-conditioning interaction", "Representation $\\times$ input interaction"),
    ("Embedding minus logit gain for endpoints", "Embedding $-$ logit, endpoints"),
    ("Embedding minus logit gain for scalar", "Embedding $-$ logit, scalar"),
    ("Endpoint minus scalar at embedding", "Endpoints $-$ scalar, embedding"),
    ("Endpoint minus scalar at logit", "Endpoints $-$ scalar, logit"),
    ("Endpoint gain with", "Endpoint gain,"), ("Scalar gain with", "Scalar gain,"),
    ("Fixed chords minus raw Hessian", "Fixed $-$ raw Hessian"),
    ("Fixed chords minus shuffle", "Fixed $-$ shuffled"),
    ("Tuned chords minus gray maximum", "Tuned $-$ grey max."),
    ("Tuned chords minus shuffle", "Tuned $-$ shuffled"),
    ("Archived visibility-by-chord interaction", "Visibility $\\times$ half-axis"),
    ("Sham path", "Shuffled-pair head"), ("Quarantined", "Excluded (repeated patient)"),
    ("Readout development", "Readout development"), ("Fitted architecture", "Architecture"),
    ("& Arm &", "& Condition &"), ("readout dev.", "readout dev."),
    ("Reuse subsets", "Evaluation subsets"), ("Released marks", "Proposals"),
    ("Two-scale", "Two-scale"),
]


def seeds(s):
    s = re.sub(r"(Two-scale|Single-stream|Three-view|Native sources) (3691|9143)", lambda m: m.group(1) + ", seed " + ("1" if m.group(2) == "3691" else "2"), s)
    s = re.sub(r"\(3691\)", "(seed 1)", s)
    s = re.sub(r"\(9143\)", "(seed 2)", s)
    s = re.sub(r"(^|& )3691 (?=\$|checkpoint|readout)", r"\1Seed 1 ", s, flags=re.M)
    s = re.sub(r"(^|& )9143 (?=\$|checkpoint|readout)", r"\1Seed 2 ", s, flags=re.M)
    s = re.sub(r"^3691 &", "1 &", s, flags=re.M)
    s = re.sub(r"^9143 &", "2 &", s, flags=re.M)
    s = re.sub(r"\b3691\b", "seed 1", s)
    s = re.sub(r"\b9143\b", "seed 2", s)
    return s


COLUMN = {"roles", "decision", "historical", "size_displacement"}  # fit one column (237 pt)


def two_column(name, s):
    """Column-width tables as table, wider ones as full-width table*; both float to page tops."""
    env = "table" if name in COLUMN else "table*"
    s = re.sub(r"\\begin\{table\*?\}(\[[^\]]*\])?", lambda m: "\\begin{" + env + "}[!t]", s)
    s = re.sub(r"\\end\{table\*?\}", lambda m: "\\end{" + env + "}", s)
    if name in COLUMN and "footnotesize" not in s:
        s = s.replace("\\centering", "\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{" + ("2pt" if name == "historical" else "4pt") + "}", 1)
    return s


def main():
    for name, cap in CAPTIONS.items():
        p = T / f"{name}.tex"
        s = p.read_text()
        s = re.sub(r"\\caption\{.*?\}(?=\s*\\label)", lambda m: "\\caption{" + cap + "}", s, count=1, flags=re.S)
        assert cap in s, name
        for a, b in PHRASES:
            s = s.replace(a, b)
        s = seeds(s)
        p.write_text(two_column(name, s))
    p = T / "size_displacement.tex"
    p.write_text(two_column("size_displacement", p.read_text()))


if __name__ == "__main__":
    main()
