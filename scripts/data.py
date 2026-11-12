"""Single source of plotted numbers.

Every value comes from ``repro/aggregate.json`` (the authors' aggregate export);
the JSON path is given next to each accessor. Fractions are converted to
percentage points here so figure scripts never rescale.
Nothing in this file is typed in by hand except the geometry constants of the
schematic figures, which are the method's own definitions (Sec. 3).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGG = json.loads((ROOT / "repro" / "aggregate.json").read_text())

SEEDS = ("3691", "9143")
RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]
RATE_LABELS = ["1/8", "1/4", "1/2", "1", "2", "4", "8"]


def pts(row):
    """{'estimate','lo','hi'} fraction -> (est, lo, hi) in points."""
    return tuple(100 * row[k] for k in ("estimate", "lo", "hi"))


# --- Dense restoration (original lock) ------------------------------------
def restore(seed, field="cpm"):
    """original/<seed>/pooled_capped/contrasts/restore/<field>; field in cpm|at_1|cpm_adjusted."""
    c = AGG["original"][seed]["pooled_capped"]["contrasts"]["restore"]
    return pts(c[field])


def restore_rates(seed):
    """original/<seed>/pooled_capped/contrasts/restore/rates  (7 matched operating points)."""
    return [pts(r) for r in AGG["original"][seed]["pooled_capped"]["contrasts"]["restore"]["rates"]]


def absolute_cpm(seed, mode):
    """original/<seed>/pooled_capped/methods/<mode><seed>_raw/cpm ; mode in interior|annulus|full."""
    return pts(AGG["original"][seed]["pooled_capped"]["methods"][f"{mode}{seed}_raw"]["cpm"])


def froc(seed):
    """dense_froc/<seed>  (matched FROC curves, 125 rates)."""
    f = AGG["dense_froc"][seed]
    return f["rates"], {k: [100 * v for v in f[k]] for k in ("ball", "band", "full")}


def recovery(seed):
    """original/<seed>/pooled_capped/recovered_context_fraction."""
    return pts(AGG["original"][seed]["pooled_capped"]["recovered_context_fraction"])


# --- Post-hoc matched comparisons ------------------------------------------
def new(key, seed="3691"):
    return AGG["new"][f"{key}_{seed}_pooled_capped"]


def single_stream(field="cpm"):
    """new/architecture_3691_pooled_capped/contrasts/band_minus_ball/<field>."""
    return pts(new("architecture")["contrasts"]["band_minus_ball"][field])


def sources(contrast, field="cpm"):
    """new/sources_3691_pooled_capped/contrasts/<contrast>/<field>."""
    return pts(new("sources")["contrasts"][contrast][field])


def native_counts():
    """native_counts/{query_reached,band_reached}: mean exterior native voxels reaching the crop."""
    n = AGG["native_counts"]
    return n["query_reached"], n["band_reached"]


def sparse(contrast, seed, field="cpm"):
    """new/spatial_<seed>_pooled_capped/contrasts/<contrast>/<field>."""
    return pts(new("spatial", seed)["contrasts"][contrast][field])


def sparse_cpm(method, seed):
    """new/spatial_<seed>_pooled_capped/methods/<method>/cpm (absolute)."""
    return pts(new("spatial", seed)["methods"][method]["cpm"])


def sparse_epoch(method, seed):
    """new_fitting/spatial<seed>/epoch/<method>: selected checkpoint epoch."""
    return AGG["new_fitting"][f"spatial{seed}"]["epoch"][method]


def degenerate_heads(seed):
    """Sparse heads whose evaluated CPM equals the frozen ball scorer to 1e-9 (no learned residual)."""
    ball = new("spatial", seed)["methods"]["ball"]["cpm"]["estimate"]
    out = []
    for m in ("pooled", "compact", "blind48", "spatial48", "blind96", "spatial96"):
        if abs(new("spatial", seed)["methods"][m]["cpm"]["estimate"] - ball) < 1e-9:
            out.append(m)
    return out


# --- Three-view (original lock) -------------------------------------------
def three_view(field="cpm"):
    """original/tri_9143/pooled_capped/contrasts/restore/<field>."""
    return pts(AGG["original"]["tri_9143"]["pooled_capped"]["contrasts"]["restore"][field])


def arch_absolute(arch):
    """Absolute CPM (ball, band, full) per architecture.
    two-scale: original/<seed>/pooled_capped/methods/{interior,annulus,full}<seed>_raw
    single:    new/architecture_3691_pooled_capped/methods/{ball,band,full}
    three-view: original/tri_9143/pooled_capped/methods/tri_{interior,annulus,full}9143_raw"""
    if arch in SEEDS:
        m = AGG["original"][arch]["pooled_capped"]["methods"]
        return [100 * m[f"{k}{arch}_raw"]["cpm"]["estimate"] for k in ("interior", "annulus", "full")]
    if arch == "single":
        m = new("architecture")["methods"]
        return [100 * m[k]["cpm"]["estimate"] for k in ("ball", "band", "full")]
    m = AGG["original"]["tri_9143"]["pooled_capped"]["methods"]
    return [100 * m[f"tri_{k}9143_raw"]["cpm"]["estimate"] for k in ("interior", "annulus", "full")]


# --- Compact controls (original lock, both seeds) -------------------------
def original_contrast(name, seed, field="cpm"):
    """original/<seed>/pooled_capped/contrasts/<name>/<field>."""
    return pts(AGG["original"][seed]["pooled_capped"]["contrasts"][name][field])


# --- Anatomy ------------------------------------------------------------------
def anatomy(seed, group, what="difference"):
    """anatomy/by_seed/<seed>/capped/<group>/<what> -> list over 7 global FP rates."""
    return [pts(r) for r in AGG["anatomy"]["by_seed"][seed]["capped"][group][what]]


def anatomy_n(group):
    """anatomy/groups/<group>/lesions."""
    return AGG["anatomy"]["groups"][group]["lesions"]


def anatomy_diameter(group):
    return AGG["anatomy"]["groups"][group]["median_reference_diameter_mm"]


# --- Robustness: subset halves, uncapped, threshold transfer --------------------
SCOPES = ("pooled_capped", "confirmation_capped", "replication_capped", "pooled_uncapped")


def robust(comparison, scope):
    """CPM effect for one comparison under one evaluation scope.
    restore fit k: original/<seed>/<scope>/contrasts/restore/cpm
    sources:       new/sources_3691_<scope>/contrasts/band_minus_sources/cpm
    single:        new/architecture_3691_<scope>/contrasts/band_minus_ball/cpm"""
    if comparison in SEEDS:
        return pts(AGG["original"][comparison][scope]["contrasts"]["restore"]["cpm"])
    key, con = {"sources": ("sources", "band_minus_sources"),
                "single": ("architecture", "band_minus_ball")}[comparison]
    return pts(AGG["new"][f"{key}_3691_{scope}"]["contrasts"][con]["cpm"])


def scope_scans(scope):
    """original/3691/<scope>/scans."""
    return AGG["original"]["3691"][scope]["scans"]


def transfer(seed):
    """original/<seed>/pooled_capped/contrasts/restore/transfer/{sensitivity,fp_per_scan}
    (thresholds fixed on the 88 calibration scans). Sensitivity in points, FP/scan as is."""
    t = AGG["original"][seed]["pooled_capped"]["contrasts"]["restore"]["transfer"]
    fp = t["fp_per_scan"]
    return pts(t["sensitivity"]), (fp["estimate"], fp["lo"], fp["hi"])


# --- Band reach by reference diameter -------------------------------------------
def reach_by_diameter():
    """geometry/diameter_stratum/<stratum>/{lesions, native_band/lesions}: linked lesions with a
    matching proposal, and how many of them the band of some matching proposal reaches."""
    g = AGG["geometry"]["diameter_stratum"]
    order = ["<6 mm", "6–<10 mm", "≥10 mm"]
    return [(k, g[k]["lesions"], g[k]["native_band"]["lesions"]) for k in order]


def fixture():
    """fixture/*: implementation checks against literal full-volume masking."""
    return AGG["fixture"]
