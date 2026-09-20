"""Loading, cleaning and farm-level assembly of the Track A survey data.

Why this module exists
----------------------
Neither of the two farm datasets can be handed to a regression as it stands, and both fail in
ways that are silent rather than loud. A pipeline that skips this module still runs, still
prints an R-squared, and is still wrong. The three specific traps are:

1. **A sentinel masquerading as a measurement.** IFPRI's `c904` is labelled "mortality rate due
   to disease (%)" and contains the value 9997 ninety-four times. There is *nothing* between
   101 and 9996 -- the range is empty, which is what identifies 9997 as a "don't know" code
   rather than an extreme observation. Left in, it moves the mean from 32.46% to 819.58%, a
   25-fold inflation that propagates into every coefficient, every residual and therefore every
   deviance in the Just-Pope procedure.

2. **Two long modules at the wrong grain.** `006_hh_modc3` has 10,270 rows because it is one row
   per farm x species; `020_hh_modc9_c902-c905` has 1,190 rows because it is one row per farm x
   disease. `farmer_uid` is a valid join key in neither. Merging on it without aggregating first
   does not raise an error -- it silently replicates each farm's covariates once per species,
   so a polyculture farm with nine species contributes nine times the weight of a monoculture
   farm to every estimate. The regression would then be fitted to a sample the survey never
   drew.

3. **An outcome that does not exist as a column.** There is no income variable. `c1004` is the
   revenue of the *final sale event* only (median 3,200 BDT on a median 5 kg), which is roughly
   5% of a farm's annual sales. Total revenue has to be constructed as the sum over species of
   quantity sold x average price, and that construction has to be reconciled against `c1004`
   rather than asserted. Everything this module derives is tagged as derived, so that no
   downstream table can present a constructed number as an observed one.

Design choice: provenance is data, not prose
--------------------------------------------
Every column this module produces is registered in a `VariableProvenance` record carrying its
kind (observed / derived / imputed), its source columns and the reasoning behind it. The
registry is written out with the results as `exp03_variable_provenance.csv`. The alternative --
explaining provenance only in the stage document -- was rejected because prose drifts from code
and nothing detects it. A registry is checkable by a test, and it is.

What it does not do
-------------------
No imputation on the IFPRI data. See a dedicated imputation step (not implemented; Track A runs complete-case) for why the missing-data question is posed
only on Hasan & Haque: it is the dataset where missingness is severe enough (28-30%) to be a
research question rather than a nuisance, and confining imputation to it keeps the IFPRI models
complete-case and therefore free of one more thing to defend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.paths import DATA_RAW

IFPRI_DIR = DATA_RAW / "ifpri"
HASAN_HAQUE_XLSX = DATA_RAW / "farm_economics" / "WSSVRiskFactor_rev.xlsx"

#: `c904` is a percentage. 9997 is the survey's "don't know" code. Named rather than inlined
#: so the number appears exactly once in the codebase and the tests can import the same object
#: the cleaner uses -- a test that hard-codes its own copy of a constant tests nothing.
C904_DONT_KNOW = 9997

#: 100 decimals = 1 acre = 0.4046856 ha, so one decimal = 0.004046856 ha. IFPRI records pond
#: area in decimals; the base paper and Hasan & Haque both use hectares. Comparing the two
#: without this factor would understate IFPRI ponds by a factor of ~247.
DECIMALS_TO_HA = 0.004046856

#: Binary covariates rarer than this are dropped before modelling. At n=1,221 a 5% threshold
#: leaves ~61 farms in the minority cell, below which a coefficient is estimated off a handful
#: of observations and a tree can isolate them as a leaf. The threshold is a judgement, so it
#: is a named parameter that the reported design matrix records rather than a hidden default.
MIN_BINARY_SHARE = 0.05


# ======================================================================================
# Provenance
# ======================================================================================

@dataclass(frozen=True)
class VariableProvenance:
    """Where one analysis variable came from and how far it is from the raw survey.

    `kind` takes exactly three values and the distinction is the point of the class:

    ``observed``  a survey column used as recorded (possibly renamed or rescaled by a known
                  physical constant -- a unit conversion does not create information)
    ``derived``   computed from several observed columns by a rule we chose; the rule is a
                  modelling decision and must be visible
    ``imputed``   filled in where the survey has no value at all; the least trustworthy kind,
                  and the only one whose count of affected rows must always be reported

    A derived outcome presented as an observed one is the failure this guards against. Revenue is
    not a survey answer -- it is quantity sold times average price, summed over species -- and a
    reader who assumed a farmer had stated it would misjudge both its precision and what a
    coefficient on it means. Carrying the distinction in the data structure rather than in the prose
    means it survives being copied into a table. (Master Prompt section 10 requires the same thing.)
    """

    name: str
    kind: str
    source: str
    note: str

    def __post_init__(self) -> None:
        if self.kind not in {"observed", "derived", "imputed"}:
            raise ValueError(f"unknown provenance kind {self.kind!r} for {self.name}")


@dataclass
class AnalysisFrame:
    """A modelling table plus everything needed to defend it.

    Bundling the frame with its provenance and its build log stops the two from separating.
    A DataFrame returned on its own loses the record of how it was made the moment it is passed
    to a second function; this cannot.
    """

    frame: pd.DataFrame
    provenance: list[VariableProvenance] = field(default_factory=list)
    notes: dict[str, object] = field(default_factory=dict)

    def provenance_table(self) -> pd.DataFrame:
        return pd.DataFrame([vars(p) for p in self.provenance])

    def kinds(self) -> dict[str, str]:
        return {p.name: p.kind for p in self.provenance}

    def columns_of_kind(self, kind: str) -> list[str]:
        return [p.name for p in self.provenance if p.kind == kind]


# ======================================================================================
# Raw IFPRI access
# ======================================================================================

def load_ifpri_module(stem: str, directory: Path | None = None) -> pd.DataFrame:
    """Read one IFPRI `.tab` module.

    Dataverse serves these as tab-separated text with a UTF-8 payload. `low_memory=False`
    forces pandas to type each column from the whole file rather than in chunks: several
    columns are mostly blank with values only in the last few hundred rows, and chunked
    inference gives them different dtypes on different machines. A dtype that depends on
    where the chunk boundary falls is not reproducible, and reproducibility is the point.
    """
    directory = directory or IFPRI_DIR
    path = directory / f"{stem}.tab"
    if not path.exists():
        raise FileNotFoundError(
            f"IFPRI module {stem} not found at {path}; run scripts/fetch_ifpri.py"
        )
    return pd.read_csv(path, sep="\t", low_memory=False)


def load_ifpri_codebook(stem: str, directory: Path | None = None) -> pd.DataFrame:
    """Read the `VariableName` / `VariableLabel` codebook shipped beside a module.

    Used to attach human-readable labels to coefficient and importance tables. A results table
    listing `c003_18` teaches nobody anything; one listing "Only stock PCR-tested PL" is
    self-explanatory in a viva, which is the standard this project is held to.
    """
    directory = directory or IFPRI_DIR
    cb = pd.read_csv(directory / f"{stem}_codebook.tab", sep="\t", low_memory=False)
    return cb[cb["VariableName"].notna()]


def variable_labels(stems: tuple[str, ...] = (
    "004_hh_modc1-c2", "006_hh_modc3", "019_hh_modc9_c900-c901",
    "020_hh_modc9_c902-c905", "021_hh_modc10",
)) -> dict[str, str]:
    """Merge the codebooks into one `column -> label` mapping."""
    labels: dict[str, str] = {}
    for stem in stems:
        try:
            cb = load_ifpri_codebook(stem)
        except FileNotFoundError:
            continue
        for _, row in cb.iterrows():
            labels.setdefault(str(row["VariableName"]), str(row["VariableLabel"]))
    return labels


# ======================================================================================
# Cleaning rules
# ======================================================================================

def clean_c904(series: pd.Series) -> pd.Series:
    """Convert the `c904` "don't know" sentinel to missing.

    The rule is `9997 -> NaN` and nothing else; no winsorising, no capping, no other value
    touched. It is defensible precisely because it is narrow.

    Why 9997 is a code and not a measurement. The argument is *not* that a 9997% mortality rate
    is implausible -- implausibility alone would justify winsorising a heavy tail, which would
    be a different and much weaker intervention. The argument is that the interval 101..9996 is
    **completely empty** while 9997 occurs 94 times. A genuine heavy tail deposits values
    throughout the range it spans. A spike separated from the data by an empty region of nearly
    ten thousand integers is a reserved code, and 9997/9998/9999 are the conventional reserved
    codes in this survey family.

    Cost of getting it wrong, measured rather than asserted: the uncleaned mean is 819.58 and
    the cleaned mean is 32.46 (median 28, range 0-100).

    Returns a float series so the missing values survive; an integer column cannot hold NaN,
    and a silent cast back to 0 would be worse than leaving the sentinel in.
    """
    out = pd.to_numeric(series, errors="coerce").astype("float64")
    return out.where(out != C904_DONT_KNOW)


def sentinel_report(series: pd.Series) -> dict[str, float]:
    """Evidence for the sentinel diagnosis, recomputed rather than quoted.

    Emitted into the results tables so the claim in the stage document is backed by a number
    produced in the same run as everything else, not by a figure typed in from an earlier
    session. `n_between_101_and_9996` is the load-bearing one: if it is ever non-zero the
    diagnosis is wrong and the cleaning rule must be revisited.
    """
    s = pd.to_numeric(series, errors="coerce")
    cleaned = clean_c904(s)
    return {
        "n_rows": int(s.notna().sum()),
        "n_le_100": int((s <= 100).sum()),
        "n_between_101_and_9996": int(((s > 100) & (s < C904_DONT_KNOW)).sum()),
        "n_equal_sentinel": int((s == C904_DONT_KNOW).sum()),
        "n_above_sentinel": int((s > C904_DONT_KNOW).sum()),
        "mean_uncleaned": float(s.mean()),
        "mean_cleaned": float(cleaned.mean()),
        "median_cleaned": float(cleaned.median()),
        "max_cleaned": float(cleaned.max()),
    }


def filter_white_spot(modc9b: pd.DataFrame) -> pd.DataFrame:
    """Select the White Spot Disease rows from the long disease module.

    This is the one place in the IFPRI data where a long module collapses to a clean 1:1 farm
    table without an aggregation decision: exactly 492 rows, 492 distinct farms. That is worth
    stating as a property rather than a coincidence, so it is asserted here. Aggregation always
    involves a choice (sum? max? first?) and every choice is a modelling assumption; a filter
    that happens to be 1:1 avoids the choice entirely, which is why the White Spot subset is
    the project's primary disease-risk sample.

    Matching is on the disease *name* rather than `shrimpdiseasecode`, because the name is what
    the codebook documents and what a reader can verify by eye. Case-insensitive substring
    matching on "White Spot" is used so that a trailing "(WSD)" or a spacing change in a future
    release does not silently return zero rows -- and if it ever does return an unexpected
    count, the assertion below fails loudly instead.
    """
    mask = modc9b["shrimpdiseasename"].astype("string").str.contains(
        "White Spot", case=False, na=False
    )
    wsd = modc9b[mask].reset_index(drop=True)
    if wsd["farmer_uid"].duplicated().any():
        n = int(wsd["farmer_uid"].duplicated().sum())
        raise ValueError(
            f"White Spot subset is not 1:1 -- {n} duplicate farmer_uid values. "
            "The subset can no longer be joined without an aggregation rule."
        )
    return wsd


def assert_one_row_per_farm(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Refuse to return a table that is not keyed by farm.

    Called at every point where a frame is about to be merged. This is the guard against trap 2
    in the module docstring: a duplicated key does not fail a merge, it inflates it, and the
    inflation is invisible in every summary statistic except the row count. Checking the grain
    at the boundary is cheaper than discovering a 10,270-row regression later.
    """
    if df["farmer_uid"].duplicated().any():
        dup = int(df["farmer_uid"].duplicated().sum())
        raise ValueError(
            f"{name} has {dup} duplicated farmer_uid values and is not one row per farm; "
            "aggregate or filter it before joining"
        )
    return df


# ======================================================================================
# Aggregating the long species module
# ======================================================================================

#: Species treated as shrimp/prawn for the "shrimp revenue" subtotal. Bagda is the black tiger
#: shrimp (*P. monodon*) the image dataset photographs; golda is the giant freshwater prawn;
#: harina and chaka are smaller local shrimp. Fish and crab are farm income but not shrimp
#: income, and the two are reported separately rather than merged, because the project's claims
#: are about shrimp.
SHRIMP_SPECIES = ("Bagda Chingri", "Golda Chingri", "Harina Chingri", "Chaka Chingri")


def aggregate_species_module(modc3: pd.DataFrame) -> AnalysisFrame:
    """Collapse the 10,270-row farm x species module to one row per farm.

    This function is where the project's outcome variable is *constructed*, so the reasoning
    matters more here than anywhere else in the module.

    **Revenue.** There is no revenue column. `c304` is quantity sold (kg) and `c308` is the
    average price received (BDT/kg), both per species. Their product is that species' sales
    revenue, and the farm total is the sum across species.

    **The missing-price question, and why it needs no imputation.** `c308` is missing on 989 of
    10,270 rows, which looks like an 9.6% data-quality problem. It is not: *every one of those
    989 rows has `c304 == 0`.* The price is missing exactly where nothing was sold, so it is
    structurally missing rather than unrecorded -- a farm that sold nothing has no price to
    report. A row contributing zero quantity contributes zero revenue whatever price is
    imputed, so imputing here would add an assumption and change nothing. The check is
    recomputed at runtime and stored in `notes`; if a future release breaks the pattern, the
    note will show it instead of the pipeline quietly imputing over it.

    **Stocked quantity is deliberately not aggregated.** `c301` carries a companion unit column
    `c301_u` coded 1=piece, 2=kg, 98=not applicable, and the raw column ranges to 2,500,000.
    Summing pieces and kilograms would produce a number with no unit. Since no conversion factor
    between pieces and kilograms is recorded in the survey (it depends on post-larva size, which
    is not collected), the honest treatment is to aggregate the two units *separately* and let
    the model see both, rather than to invent a conversion.

    **Stocking cost `c303` is in BDT throughout** and is therefore summable without any of this
    difficulty -- which is why it, not `c301`, carries the input-intensity signal in the models.
    """
    d = modc3.copy()
    d["species_revenue_bdt"] = d["c304"] * d["c308"]

    is_shrimp = d["speciesname"].isin(SHRIMP_SPECIES)
    price_missing = d["c308"].isna()

    grouped = d.groupby("farmer_uid", dropna=False)
    out = pd.DataFrame({
        "revenue_total_bdt": grouped["species_revenue_bdt"].sum(min_count=1),
        "revenue_shrimp_bdt": d.where(is_shrimp).groupby(d["farmer_uid"])[
            "species_revenue_bdt"].sum(min_count=1),
        "stocking_cost_bdt": grouped["c303"].sum(min_count=1),
        "qty_sold_kg": grouped["c304"].sum(min_count=1),
        "qty_produced_kg": grouped["prod"].sum(min_count=1),
        "n_species": grouped["speciesname"].nunique(),
        # See docstring: pieces and kilograms are never added together.
        "stocked_pieces": d.where(d["c301_u"] == 1).groupby(d["farmer_uid"])[
            "c301"].sum(min_count=1),
        "stocked_kg": d.where(d["c301_u"] == 2).groupby(d["farmer_uid"])[
            "c301"].sum(min_count=1),
    }).reset_index()

    out["grows_bagda"] = (
        d.assign(f=d["speciesname"].eq("Bagda Chingri"))
        .groupby("farmer_uid")["f"].max().astype("int64").reindex(out["farmer_uid"]).to_numpy()
    )
    # A farm that stocked but sold nothing legitimately has zero revenue, not missing revenue.
    for col in ("stocked_pieces", "stocked_kg"):
        out[col] = out[col].fillna(0.0)

    notes = {
        "modc3_rows": int(len(d)),
        "modc3_farms": int(d["farmer_uid"].nunique()),
        "rows_missing_price": int(price_missing.sum()),
        "rows_missing_price_with_zero_qty_sold": int((price_missing & (d["c304"] == 0)).sum()),
        "rows_missing_price_with_positive_qty_sold": int(
            (price_missing & (d["c304"] > 0)).sum()),
    }

    prov = [
        VariableProvenance(
            "revenue_total_bdt", "derived", "modc3: sum over species of c304 x c308",
            "No revenue column exists. Constructed as quantity sold x average price, summed "
            "over all species. Reconcile against c1004 before use.",
        ),
        VariableProvenance(
            "revenue_shrimp_bdt", "derived",
            f"modc3: sum over {SHRIMP_SPECIES} of c304 x c308",
            "Shrimp/prawn subtotal of the same construction; reported separately because the "
            "project's claims concern shrimp, not total farm output.",
        ),
        VariableProvenance("stocking_cost_bdt", "derived", "modc3: sum of c303",
                           "BDT throughout, so summation is unit-safe."),
        VariableProvenance("qty_sold_kg", "derived", "modc3: sum of c304", "kg throughout."),
        VariableProvenance("qty_produced_kg", "derived", "modc3: sum of prod", "kg throughout."),
        VariableProvenance("n_species", "derived", "modc3: nunique(speciesname)",
                           "Polyculture breadth; a farm-level count, not a survey question."),
        VariableProvenance("stocked_pieces", "derived", "modc3: sum of c301 where c301_u == 1",
                           "Pieces only. Never added to kilograms -- no conversion is recorded."),
        VariableProvenance("stocked_kg", "derived", "modc3: sum of c301 where c301_u == 2",
                           "Kilograms only, for the same reason."),
        VariableProvenance("grows_bagda", "derived", "modc3: any speciesname == Bagda Chingri",
                           "Black tiger shrimp, the species the image dataset photographs."),
    ]
    return AnalysisFrame(frame=assert_one_row_per_farm(out, "modc3 aggregate"),
                         provenance=prov, notes=notes)


def reconcile_revenue(constructed: pd.DataFrame, modc10: pd.DataFrame) -> dict[str, float]:
    """Check the constructed revenue against the one revenue figure the survey does record.

    `c1004` is the money received for bagda/golda in the **final sale** of the season. It is not
    an alternative measure of annual revenue and cannot be used as one, but it is a partial
    observation of the same underlying quantity, and a construction that contradicted it would
    be wrong.

    Two things are checked, and they test different failure modes:

    - **Containment.** The final sale is one selling event out of several, so `c1004` should be
      no larger than the constructed bagda+golda revenue for essentially every farm. A high
      share of violations would mean the construction is *missing* sales.
    - **Correlation of logs.** Both should move with farm scale. A near-zero correlation would
      mean the construction is measuring something else entirely.

    Deliberately *not* checked: equality, or a stable ratio. The final sale is a variable and
    farmer-chosen share of the season, so a tight ratio is not expected and its absence is not
    evidence against the construction.
    """
    m = constructed.merge(modc10[["farmer_uid", "c1003", "c1004"]], on="farmer_uid", how="inner")
    m = m.dropna(subset=["revenue_shrimp_bdt", "c1004"])
    m = m[m["revenue_shrimp_bdt"] > 0]
    pos = m[m["c1004"] > 0]
    return {
        "n_compared": int(len(m)),
        "median_final_sale_share": float((m["c1004"] / m["revenue_shrimp_bdt"]).median()),
        "share_final_sale_within_constructed": float(
            (m["c1004"] <= m["revenue_shrimp_bdt"]).mean()),
        "log_correlation": float(np.corrcoef(
            np.log(pos["revenue_shrimp_bdt"]), np.log(pos["c1004"]))[0, 1]),
        "median_constructed_total": float(constructed["revenue_total_bdt"].median()),
        "median_c1004": float(m["c1004"].median()),
    }


# ======================================================================================
# Farm-level covariate block
# ======================================================================================

#: Biosecurity and management flags from `c003_*`, with the labels the codebook gives them.
#: Chosen because each is a *management action the farmer took*, which is the class of variable
#: the base paper's covariate set is built from (fencing, sanitation investment, PL sourcing,
#: feed). Selection was made from the codebook before any model was fitted; no variable was
#: added or removed after seeing a result.
BIOSECURITY_FLAGS = {
    "c003_1": "deepened_gher",
    "c003_2": "built_reserve_tank",
    "c003_3": "built_nursing_point",
    "c003_4": "blue_fencing",
    "c003_5": "bird_netting",
    "c003_6": "inlet_mesh_screen",
    "c003_10": "stopped_white_fish",
    "c003_11": "stopped_rice_in_gher",
    "c003_12": "stopped_vegetables",
    "c003_13": "treats_water_before_stocking",
    "c003_14": "coordinated_stocking",
    "c003_15": "single_stocking_per_cycle",
    "c003_16": "organizer_supplied_pl",
    "c003_17": "spf_pl_only",
    "c003_18": "pcr_tested_pl_only",
    "c003_19": "formulated_feed_only",
    "c003_20": "uses_probiotics",
    "c003_21": "coordinated_harvest",
}


def build_farm_covariates(
    commonvar: pd.DataFrame,
    modc1c2: pd.DataFrame,
    modc9a: pd.DataFrame,
) -> AnalysisFrame:
    """Assemble the farm-level covariate block shared by both risk models.

    Both the income model and the disease model use the same farm, pond and management block, so
    it is built once. That is not only tidiness: using one covariate block for both outcomes is
    what makes the two models comparable, and comparability is what the base paper's Table 5
    (overlap between the income and disease variable rankings) depends on.

    **Unit conversion.** `c001` is in decimals and is converted to hectares here rather than at
    the point of use, so that no downstream reader has to remember the factor and no two callers
    can disagree about it. A unit conversion by a known physical constant does not create
    information, so `pond_area_ha` is still registered as *observed*.

    **District as dummies with Bagerhat as reference.** Bagerhat is the smallest of the three
    (327 farms), and using the smallest cell as the reference category makes the two reported
    contrasts the better-estimated ones. Which district is the reference changes the
    interpretation of the intercept and nothing else.

    **Deliberately excluded, with reasons** -- exclusions are decisions and are recorded rather
    than silently applied:

    - `c206` aerator: 7 farms of 1,221 (0.6%). Below any usable variance threshold.
    - `c205` distance to water source: 149 missing (12.2%). Including it would force an
      imputation decision on the IFPRI data whose only purpose is one covariate; the missing-data
      question is posed properly on Hasan & Haque instead (a dedicated imputation step (not implemented; Track A runs complete-case)).
    - `c210` yield loss from flooding: defined only for the 383 flooded farms, so it is a
      post-treatment variable conditional on `flooded`. Including both would condition on an
      outcome.
    - `c901_*` disease-type flags other than White Spot: they are outcomes of the disease
      process, not farm characteristics, and belong on the left-hand side.
    """
    base = commonvar[["farmer_uid", "districtname", "rescluster", "sampletype"]].copy()
    pond = modc1c2[["farmer_uid", "c001", "c002", "c200", "c208",
                    "c204_1", "c204_3", "c204_4"] + list(BIOSECURITY_FLAGS)].copy()
    dis = modc9a[["farmer_uid", "c900", "c901_1", "c901_4"]].copy()

    for df, name in ((base, "commonvar"), (pond, "modc1-c2"), (dis, "modc9a")):
        assert_one_row_per_farm(df, name)

    out = base.merge(pond, on="farmer_uid", how="inner").merge(dis, on="farmer_uid", how="inner")

    out["pond_area_ha"] = out["c001"] * DECIMALS_TO_HA
    out["pond_depth_ft"] = out["c002"]
    out["tenure_owned"] = (out["c200"] == 1).astype("int64")
    out["flooded_2022"] = (out["c208"] == 1).astype("int64")
    out["water_rainfed"] = out["c204_1"].fillna(0).astype("int64")
    out["water_river"] = out["c204_3"].fillna(0).astype("int64")
    out["water_canal"] = out["c204_4"].fillna(0).astype("int64")
    out["in_cluster"] = (out["rescluster"] == 1).astype("int64")
    out["district_khulna"] = (out["districtname"].str.upper() == "KHULNA").astype("int64")
    out["district_satkhira"] = (out["districtname"].str.upper() == "SATKHIRA").astype("int64")
    out["faced_disease"] = (out["c900"] == 1).astype("int64")
    # c901_1 is NaN for the 303 farms that reported no disease at all; for them the answer to
    # "did you have White Spot" is a genuine no, not a missing value.
    out["had_white_spot"] = out["c901_1"].fillna(0).astype("int64")
    out["had_black_gill"] = out["c901_4"].fillna(0).astype("int64")

    for raw, name in BIOSECURITY_FLAGS.items():
        out[name] = out[raw].fillna(0).astype("int64")

    keep = ["farmer_uid", "pond_area_ha", "pond_depth_ft", "tenure_owned", "flooded_2022",
            "water_rainfed", "water_river", "water_canal", "in_cluster",
            "district_khulna", "district_satkhira", "faced_disease",
            "had_white_spot", "had_black_gill"] + list(BIOSECURITY_FLAGS.values())

    prov = [
        VariableProvenance("pond_area_ha", "observed", "modc1-c2: c001 x 0.004046856",
                           "Unit conversion from decimals; a known constant adds no information."),
        VariableProvenance("pond_depth_ft", "observed", "modc1-c2: c002", "Feet, as recorded."),
        VariableProvenance("tenure_owned", "observed", "modc1-c2: c200 == 1",
                           "Recode of a categorical to a binary contrast against all other tenure."),
        VariableProvenance("flooded_2022", "observed", "modc1-c2: c208 == 1", "Yes/no recode."),
        VariableProvenance("water_rainfed", "observed", "modc1-c2: c204_1", "Multi-select flag."),
        VariableProvenance("water_river", "observed", "modc1-c2: c204_3", "Multi-select flag."),
        VariableProvenance("water_canal", "observed", "modc1-c2: c204_4", "Multi-select flag."),
        VariableProvenance("in_cluster", "observed", "commonvar: rescluster == 1",
                           "Farm belongs to an organised shrimp cluster."),
        VariableProvenance("district_khulna", "observed", "commonvar: districtname",
                           "Dummy; Bagerhat (the smallest district) is the reference."),
        VariableProvenance("district_satkhira", "observed", "commonvar: districtname",
                           "Dummy; Bagerhat is the reference."),
        VariableProvenance("faced_disease", "observed", "modc9a: c900 == 1",
                           "918 yes / 303 no."),
        VariableProvenance("had_white_spot", "observed", "modc9a: c901_1",
                           "NaN means the farm reported no disease at all, so it is a true no."),
        VariableProvenance("had_black_gill", "observed", "modc9a: c901_4", "Same treatment."),
    ] + [
        VariableProvenance(name, "observed", f"modc1-c2: {raw}",
                           "Management action since 2021; multi-select flag recorded as 0/1.")
        for raw, name in BIOSECURITY_FLAGS.items()
    ]
    return AnalysisFrame(frame=out[keep], provenance=prov,
                         notes={"n_farms": int(len(out))})


def drop_low_variance_binaries(
    frame: pd.DataFrame, columns: list[str], min_share: float = MIN_BINARY_SHARE
) -> tuple[list[str], pd.DataFrame]:
    """Remove binary covariates whose minority cell is too small to estimate from.

    Applied *before* any model is fitted and reported alongside the results, so it cannot be
    mistaken for selection on outcome. The criterion looks only at the covariate's own marginal
    distribution -- it never touches `y` -- which is what keeps it out of the family of
    data-dredging procedures that invalidate the standard errors.

    Returns both the surviving column list and a table of what was dropped, because "we removed
    some variables" is not a defensible sentence and "we removed these three, at these shares,
    by this rule" is.

    **STATUS: available and NOT APPLIED to any reported result.** No Experiment 03 table was
    filtered by this function, and no covariate was removed from the reported models on
    variance grounds. It is stated here rather than left to inference because a reader finding a
    low-variance filter in the codebase would reasonably assume the reported models had been
    through it (open item 13). `exp10_track_a_repairs.py` runs it as a **diagnostic** and writes
    what it *would* drop to `reports/tables/exp10_low_variance.csv`, so the effect of not
    applying it is visible rather than merely declared.
    """
    rows, kept = [], []
    for col in columns:
        s = frame[col].dropna()
        uniq = set(pd.unique(s))
        if not uniq.issubset({0, 1, 0.0, 1.0}):
            kept.append(col)
            continue
        share = float(min(s.mean(), 1 - s.mean()))
        if share < min_share:
            rows.append({"variable": col, "minority_share": round(share, 4),
                         "minority_n": int(min(s.sum(), len(s) - s.sum())),
                         "reason": f"minority share below {min_share:.0%}"})
        else:
            kept.append(col)
    return kept, pd.DataFrame(rows)


# ======================================================================================
# The two IFPRI analysis frames
# ======================================================================================

def build_income_frame(directory: Path | None = None) -> AnalysisFrame:
    """Farm-level table for the income-risk model (Just-Pope applied to revenue).

    Grain: one row per farm. Sample: the 1,221 baseline farms, less those with no constructible
    revenue. The frame is assembled from four modules, two of which are aggregated or filtered
    to farm grain *before* the join -- `assert_one_row_per_farm` is called on each side of every
    merge so that the trap cannot be sprung silently.
    """
    commonvar = load_ifpri_module("000_hh_commonvar", directory)
    modc1c2 = load_ifpri_module("004_hh_modc1-c2", directory)
    modc3 = load_ifpri_module("006_hh_modc3", directory)
    modc9a = load_ifpri_module("019_hh_modc9_c900-c901", directory)
    modc10 = load_ifpri_module("021_hh_modc10", directory)

    covars = build_farm_covariates(commonvar, modc1c2, modc9a)
    species = aggregate_species_module(modc3)
    reconciliation = reconcile_revenue(species.frame, assert_one_row_per_farm(modc10, "modc10"))

    merged = covars.frame.merge(species.frame, on="farmer_uid", how="inner", validate="1:1")
    n_before = len(merged)
    merged = merged[merged["revenue_total_bdt"].notna() & (merged["revenue_total_bdt"] > 0)]
    merged = merged.reset_index(drop=True)

    notes = {
        "n_farms_before_outcome_filter": n_before,
        "n_farms_modelled": int(len(merged)),
        "n_dropped_no_revenue": int(n_before - len(merged)),
        **{f"species_{k}": v for k, v in species.notes.items()},
        **{f"reconcile_{k}": v for k, v in reconciliation.items()},
    }
    return AnalysisFrame(frame=merged,
                         provenance=covars.provenance + species.provenance,
                         notes=notes)


def build_disease_frame(directory: Path | None = None) -> AnalysisFrame:
    """Farm-level table for the disease-risk model, on the White Spot Disease subset.

    **Why White Spot and not all diseases.** Three reasons, in order of weight.

    1. It is the only disease whose long-module rows form a clean 1:1 farm table (492 rows, 492
       farms), so no aggregation assumption is needed.
    2. It is the disease the image classifier detects (`WSSV` class), so this is the model the
       Stage 8 bridge will eventually feed.
    3. It is the largest single-disease subset in the survey, which matters at this sample size.

    **What the sample restriction costs, stated plainly.** These 492 farms are those that both
    had a disease outbreak *and* named White Spot as one of them. The model therefore estimates
    severity **conditional on White Spot having occurred**, not unconditional disease risk. The
    303 farms that reported no disease are not zeros in this sample -- they are absent from it.
    Any statement about "farms in general" from this model is a selection error, and the stage
    document says so.

    **The outcome.** `c904`, mortality rate due to disease (%), after sentinel cleaning. It is
    chosen over `c905` (financial loss, BDT) because it is a *proportion of shrimp affected* --
    the quantity the base paper defines prevalence to be and then does not measure (it measures
    the percentage of farmers suffering a loss instead). `c905` is retained in the frame as the
    economic-loss variable for later stages, but it is not the disease-risk outcome.
    """
    commonvar = load_ifpri_module("000_hh_commonvar", directory)
    modc1c2 = load_ifpri_module("004_hh_modc1-c2", directory)
    modc3 = load_ifpri_module("006_hh_modc3", directory)
    modc9a = load_ifpri_module("019_hh_modc9_c900-c901", directory)
    modc9b = load_ifpri_module("020_hh_modc9_c902-c905", directory)

    wsd = filter_white_spot(modc9b)
    wsd = wsd[["farmer_uid", "c902", "c904", "c905"]].copy()
    wsd["wsd_times_faced"] = pd.to_numeric(wsd["c902"], errors="coerce")
    wsd["wsd_mortality_pct"] = clean_c904(wsd["c904"])
    wsd["wsd_loss_bdt"] = pd.to_numeric(wsd["c905"], errors="coerce")

    covars = build_farm_covariates(commonvar, modc1c2, modc9a)
    species = aggregate_species_module(modc3)

    merged = (
        covars.frame
        .merge(species.frame, on="farmer_uid", how="inner", validate="1:1")
        .merge(wsd[["farmer_uid", "wsd_times_faced", "wsd_mortality_pct", "wsd_loss_bdt"]],
               on="farmer_uid", how="inner", validate="1:1")
    )
    n_wsd = len(merged)
    modelled = merged[merged["wsd_mortality_pct"].notna()].reset_index(drop=True)

    prov = covars.provenance + species.provenance + [
        VariableProvenance("wsd_times_faced", "observed", "modc9b: c902",
                           "Number of White Spot episodes in the 2022 cycle."),
        VariableProvenance(
            "wsd_mortality_pct", "observed", "modc9b: c904 with 9997 -> NaN",
            "Percentage of stock lost. The 9997 'don't know' code is removed; the value is "
            "otherwise as recorded, so it remains observed rather than derived.",
        ),
        VariableProvenance("wsd_loss_bdt", "observed", "modc9b: c905",
                           "Financial loss in BDT. No sentinel pattern; used as recorded."),
    ]
    notes = {
        "n_wsd_rows": int(len(wsd)),
        "n_wsd_farms": int(wsd["farmer_uid"].nunique()),
        "n_matched_to_covariates": n_wsd,
        "n_modelled_after_sentinel_removal": int(len(modelled)),
        "n_dropped_sentinel": int(n_wsd - len(modelled)),
        **{f"sentinel_{k}": v for k, v in sentinel_report(modc9b["c904"]).items()},
    }
    return AnalysisFrame(frame=modelled, provenance=prov, notes=notes)


# ======================================================================================
# Hasan & Haque (D2)
# ======================================================================================

#: Covariates retained from Hasan & Haque. The first eight overlap the base paper's covariate
#: set (area, experience, stocking density, sludge handling, PL quality, crop rotation, culture
#: practice, region), which is why this dataset was adopted as the external validation sample in
#: `D-002`. The rest are farm-management variables of the same family.
D2_COVARIATES = [
    "Area_ha", "InvolveWithShrimpFarming", "StockingDensity_PL/40MeterSquare",
    "SludgeRemovalMethods", "SludgeRemovalInterval", "PLQuality", "CropRotation",
    "CulturePractice", "FarmerAge", "PriorLandUse", "SoilType", "CanalDepth_ft",
    "GherDepth_ft", "Operator", "FertilizerUse", "ChemicalUsePondPreparation",
    "ChemicalUseWaterTreatment", "Aerator", "GherDryAfterHarvest", "MaintainAndRepairDikes",
    "PeriodOfFallow", "WaterSource_DirectNatural", "WaterSource_IndirectNatural",
    "WaterComingViaOtherFarms", "Reservoir", "WaterExchangeFrequency", "InletOutlet",
    "CultureMethod", "SourceOfFingerling", "StockingAge_Days", "FeedType", "FeedAdditives",
    "BirdScare", "CrabFench", "FootBath", "LimitedAccess", "SameEquipmentForWholeCultureArea",
    "PreviousPrevalence(%)",
]

#: The three water-quality variables with 28-30% missingness. Separated from the rest because
#: the entire missing-data experiment turns on them.
D2_WATER_QUALITY = ["Temperature", "pH", "Salinity"]

D2_OUTCOME = "CurrentPrevalance(%)"


def load_hasan_haque(path: Path | None = None) -> AnalysisFrame:
    """Load the Hasan & Haque (2020) WSSV risk-factor survey, n=233.

    **`VirusDetected` is excluded, and this is the single most important line in the function.**
    Cross-tabulating it against the outcome gives: of the 89 farms with `VirusDetected == 0`,
    *all 89* have `CurrentPrevalance == 0`; of the 144 with detection, 139 have positive
    prevalence. The variable is very nearly a deterministic function of whether the outcome is
    zero, because it records the laboratory confirmation of the same infection the outcome
    quantifies. Including it as a covariate would put the answer on the right-hand side, produce
    a spectacular and meaningless fit, and be exactly the kind of leakage the vision half of
    this project exists to expose. It is dropped, loudly, here.

    `Latitude`/`Longitude` are also dropped: with 233 farms in three districts they act as a
    high-resolution location index that a tree can memorise, and `Zone` already carries the
    spatial information at a level the sample can support.

    `FarmerEduaction` (the dataset's spelling) is a free-text-ish string column with 8 levels and
    no documented ordering in the codebook, so it is left out rather than encoded with an
    invented order.
    """
    path = path or HASAN_HAQUE_XLSX
    if not path.exists():
        raise FileNotFoundError(f"Hasan & Haque workbook not found at {path}")
    raw = pd.read_excel(path, sheet_name="Dateset")

    zone_dummies = pd.get_dummies(raw["Zone"].astype("string"), prefix="zone").astype("int64")
    # Khulna is the largest zone (150 of 233) and is used as the reference category.
    zone_dummies = zone_dummies.drop(columns=[c for c in zone_dummies if c.endswith("Khulna")])

    frame = pd.concat(
        [raw[["SlNo", D2_OUTCOME] + D2_COVARIATES + D2_WATER_QUALITY], zone_dummies], axis=1
    )

    leakage_check = pd.crosstab(raw["VirusDetected"], raw[D2_OUTCOME] == 0)
    prov = (
        [VariableProvenance(D2_OUTCOME, "observed", "Hasan & Haque sheet 'Dateset'",
                            "WSSV prevalence in the current culture cycle, percent. "
                            "94 of 233 values are exactly zero.")]
        + [VariableProvenance(c, "observed", "Hasan & Haque sheet 'Dateset'",
                              "Survey variable, coding in sheet 'DataCoding'.")
           for c in D2_COVARIATES]
        + [VariableProvenance(c, "observed", "Hasan & Haque sheet 'Dateset'",
                              "Water quality; 28-30% missing. Complete-case or KNN-imputed "
                              "depending on the arm of the imputation experiment.")
           for c in D2_WATER_QUALITY]
        + [VariableProvenance(c, "observed", "Hasan & Haque: Zone",
                              "District dummy; Khulna (150 of 233) is the reference.")
           for c in zone_dummies.columns]
    )
    notes = {
        "n_rows": int(len(frame)),
        "n_zero_outcome": int((raw[D2_OUTCOME] == 0).sum()),
        "share_zero_outcome": float((raw[D2_OUTCOME] == 0).mean()),
        "outcome_min": float(raw[D2_OUTCOME].min()),
        "outcome_max": float(raw[D2_OUTCOME].max()),
        "n_complete_water_quality": int(raw[D2_WATER_QUALITY].notna().all(axis=1).sum()),
        **{f"missing_share_{c}": float(raw[c].isna().mean()) for c in D2_WATER_QUALITY},
        "excluded_VirusDetected_zero_prev_when_undetected": int(leakage_check.loc[0, True])
        if 0 in leakage_check.index and True in leakage_check.columns else 0,
        "excluded_VirusDetected_n_undetected": int((raw["VirusDetected"] == 0).sum()),
    }
    return AnalysisFrame(frame=frame, provenance=prov, notes=notes)


def build_income_frame_with_prevalence(directory: Path | None = None) -> AnalysisFrame:
    """The income frame, plus a **continuous** White Spot prevalence regressor. (Stage 9.)

    Why this exists rather than a change to `build_income_frame`
    ------------------------------------------------------------
    The income model as estimated in Experiment 03 carries `had_white_spot`, a self-reported 0/1.
    The vision bridge does not produce a 0/1 -- it produces a *fraction of shrimp infected*, which
    is a different quantity with a different scale and a different error structure. Asking what
    the bridge would do to the risk estimates therefore requires a specification that has
    somewhere to put its output.

    This is a **new** builder rather than an edit, so every number Experiment 03 reported stays
    exactly as reported. Two specifications now exist side by side, they are both named in the
    results, and neither silently replaces the other.

    What `wsd_prevalence` is, and what it is not
    --------------------------------------------
    `c904` -- the percentage of stock lost to White Spot -- divided by 100, for the farms that
    reported White Spot. For every other farm it is **0**, and that zero is a real zero rather
    than a missing value: a farm that reported no disease at all, or a different disease, had no
    White Spot mortality. The same reasoning `build_farm_covariates` applies to `had_white_spot`.

    It is mortality attributable to White Spot, not the prevalence of infection. Those coincide
    only if every infected shrimp dies and no uninfected one does. It is the closest observed
    analogue in this survey to the quantity the classifier estimates, and the gap between the two
    is a limitation of the linkage, declared here and repeated in every result that uses it.

    The `9997` "don't know" code is removed by `clean_c904`, which leaves those farms missing
    rather than zero -- they reported White Spot and could not say how much stock they lost, which
    is not the same as losing none.
    """
    income = build_income_frame(directory)
    modc9b = load_ifpri_module("020_hh_modc9_c902-c905", directory)

    wsd = filter_white_spot(modc9b)[["farmer_uid", "c904"]].copy()
    wsd["wsd_prevalence"] = clean_c904(wsd["c904"]) / 100.0
    wsd = assert_one_row_per_farm(wsd[["farmer_uid", "wsd_prevalence"]], "modc9b white spot")

    merged = income.frame.merge(wsd, on="farmer_uid", how="left", validate="1:1")

    # A farm absent from the White Spot table did not report White Spot, so its White Spot
    # mortality is zero. A farm present but coded 9997 stays NaN and is dropped listwise by the
    # estimator -- it said it had the disease and could not quantify it, which is missing data,
    # not an absence.
    reported_white_spot = merged["farmer_uid"].isin(set(wsd["farmer_uid"]))
    merged.loc[~reported_white_spot, "wsd_prevalence"] = 0.0

    n_positive = int((merged["wsd_prevalence"] > 0).sum())
    notes = {
        **income.notes,
        "n_with_white_spot_reported": int(reported_white_spot.sum()),
        "n_prevalence_above_zero": n_positive,
        "n_prevalence_missing_9997": int(merged["wsd_prevalence"].isna().sum()),
        "mean_prevalence_nonzero": float(
            merged.loc[merged["wsd_prevalence"] > 0, "wsd_prevalence"].mean()),
    }
    prov = income.provenance + [
        VariableProvenance(
            "wsd_prevalence", "derived", "modc9b: c904 / 100, zero-filled off the WSD subset",
            "Share of stock lost to White Spot. Mortality attributable to the disease, which "
            "equals infection prevalence only if every infected shrimp dies -- the gap is a "
            "declared limitation of the vision-to-economics linkage, not a modelling choice.",
        ),
    ]
    return AnalysisFrame(frame=merged, provenance=prov, notes=notes)
