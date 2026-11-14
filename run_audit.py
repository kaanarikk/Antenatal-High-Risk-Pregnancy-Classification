import hashlib
import re
import numpy as np
import pandas as pd
from scipy import stats

RANDOM_SEED = 42
DATA_PATH = "user_uploaded_files/ttl=63d/user_uploaded/Book2.xlsx"

def parse_height_cm(val: str) -> float:
    s = str(val).strip().replace("''", "").replace('"', "")
    m = re.match(r"^(\d+)\.(\d+)$", s)
    if m:
        feet = int(m.group(1))
        inches = int(m.group(2))
        return round((feet * 12 + inches) * 2.54, 2)
    return np.nan

def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None, keep_default_na=False)
    header_idx = None
    for i in range(len(raw)):
        if str(raw.iloc[i, 0]).strip() == "Name":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Header row starting with 'Name' not found.")
    df = raw.iloc[header_idx + 1:].copy().reset_index(drop=True)
    df.columns = [str(c).strip() for c in raw.iloc[header_idx].tolist()]

    col_map = {
        "Name": "Name_raw",
        "Age": "age_raw",
        "Gravida": "gravida",
        "TiTi Tika": "tt_dose",
        "গর্ভকাল": "ga_raw",
        "ওজন": "weight_raw",
        "উচ্চতা": "height_raw",
        "রক্ত চাপ": "bp_raw",
        "রক্তস্বল্পতা": "anemia",
        "জন্ডিস": "jaundice",
        "গর্ভস্হ শিশু অবস্থান": "fetal_position",
        "গর্ভস্হ শিশু নাড়াচাড়া": "fetal_movement",
        "গর্ভস্হ শিশু হৃৎস্পন্দন": "fhr_raw",
        "প্রসাব পরিক্ষা এলবুমিন": "urine_albumin",
        "প্রসাব পরিক্ষা সুগার": "urine_sugar",
        "VDRL": "vdrl",
        "HRsAG": "hbsag",
        "ঝুকিপূর্ণ গর্ভ": "high_risk_raw",
    }
    df = df.rename(columns=col_map)

    # Strip whitespace across all string columns
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # De-identified group_id = SHA256 hash of lowercased stripped Name (12 hex chars)
    df["name_norm"] = df["Name_raw"].str.lower().str.strip()
    df["group_id"] = df["name_norm"].apply(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:12]
    )

    # Parse numerics
    df["age"] = pd.to_numeric(df["age_raw"], errors="coerce")
    df["ga_weeks"] = pd.to_numeric(df["ga_raw"].str.extract(r"(\d+)")[0], errors="coerce")
    df["weight_kg"] = pd.to_numeric(df["weight_raw"].str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    df["height_cm"] = df["height_raw"].apply(parse_height_cm)
    df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100.0) ** 2)

    bp_split = df["bp_raw"].str.extract(r"(\d+)\s*/\s*(\d+)")
    df["sbp"] = pd.to_numeric(bp_split[0], errors="coerce")
    df["dbp"] = pd.to_numeric(bp_split[1], errors="coerce")
    df["fhr_bpm"] = pd.to_numeric(df["fhr_raw"].str.extract(r"(\d+)")[0], errors="coerce")

    df["target"] = (df["high_risk_raw"] == "Yes").astype(int)
    df["row_idx"] = np.arange(len(df))

    return df

if __name__ == "__main__":
    df = load_and_clean()
    print("Total rows:", len(df))
    print("Missing values per parsed column:")
    check_cols = ["age", "gravida", "tt_dose", "ga_weeks", "weight_kg", "height_cm", "sbp", "dbp",
                  "anemia", "jaundice", "fetal_position", "fetal_movement", "fhr_bpm",
                  "urine_albumin", "urine_sugar", "vdrl", "hbsag", "target"]
    for c in check_cols:
        n_miss = df[c].isna().sum() if df[c].dtype != object else (df[c] == "").sum()
        print(f"  {c}: {n_miss}")

    # Duplicates
    feat_cols = ["age", "gravida", "tt_dose", "ga_weeks", "weight_kg", "height_cm", "sbp", "dbp",
                 "anemia", "jaundice", "fetal_position", "fetal_movement", "fhr_bpm",
                 "urine_albumin", "urine_sugar", "vdrl", "hbsag", "target"]
    full_dups_raw = df.duplicated(subset=["Name_raw"] + feat_cols, keep=False).sum()
    full_dups_norm = df.duplicated(subset=["name_norm"] + feat_cols, keep=False).sum()
    ign_name_dups = df.duplicated(subset=feat_cols, keep=False).sum()
    ign_name_unique_extra = df.duplicated(subset=feat_cols, keep="first").sum()
    ign_name_target_dups = df.duplicated(subset=feat_cols[:-1], keep=False).sum()

    # Check conflicting labels for identical feature rows
    grp_conf = df.groupby(feat_cols[:-1])["target"].nunique()
    n_conflicting = (grp_conf > 1).sum()

    print(f"\nDuplicates:")
    print(f"  Full-row duplicates (exact Name_raw + all features + target, involved rows): {full_dups_raw} (extra rows: {df.duplicated(subset=['Name_raw']+feat_cols).sum()})")
    print(f"  Full-row duplicates (norm name + all features + target, involved rows): {full_dups_norm} (extra rows: {df.duplicated(subset=['name_norm']+feat_cols).sum()})")
    print(f"  Duplicates ignoring Name (all features + target, involved rows): {ign_name_dups} (extra rows: {ign_name_unique_extra})")
    print(f"  Duplicates ignoring Name and target (involved rows): {ign_name_target_dups}")
    print(f"  Identical feature rows with conflicting target: {n_conflicting}")

    # Name repetition
    name_counts_raw = df["Name_raw"].value_counts()
    name_counts_norm = df["name_norm"].value_counts()
    name_age_nunique = df.groupby("name_norm")["age"].nunique()
    print(f"\nName repetition:")
    print(f"  Unique raw names: {len(name_counts_raw)}, Unique normalized names: {len(name_counts_norm)}")
    print(f"  Max repeats (raw): {name_counts_raw.max()}, Max repeats (norm): {name_counts_norm.max()}")
    print(f"  Names appearing >1 time (norm): {(name_counts_norm > 1).sum()} ({(name_counts_norm > 1).mean()*100:.1f}%)")
    print(f"  Repeated names with >1 distinct age: {(name_age_nunique[name_counts_norm > 1] > 1).sum()} / {(name_counts_norm > 1).sum()}")

    # Value distributions & implausible prevalences
    print("\nDistributions:")
    for c in ["gravida", "tt_dose", "anemia", "jaundice", "fetal_position", "fetal_movement",
              "urine_albumin", "urine_sugar", "vdrl", "hbsag", "high_risk_raw"]:
        vc = df[c].value_counts()
        pct = df[c].value_counts(normalize=True) * 100
        s = ", ".join([f"{k}: {v} ({pct[k]:.2f}%)" for k, v in vc.items()])
        print(f"  {c}: {s}")

    for c in ["age", "ga_weeks", "weight_kg", "height_cm", "bmi", "sbp", "dbp", "fhr_bpm"]:
        print(f"  {c}: min={df[c].min():.2f}, max={df[c].max():.2f}, mean={df[c].mean():.2f}, sd={df[c].std():.2f}, median={df[c].median():.2f}, IQR=[{df[c].quantile(0.25):.2f}, {df[c].quantile(0.75):.2f}]")

    # HBsAg vs target
    print("\nHBsAg vs Target:")
    print(pd.crosstab(df["hbsag"], df["high_risk_raw"], margins=True))

    # Row-order leakage / periodicity check
    print("\nRow-order / periodicity check:")
    t = df["target"].values
    for lag in range(1, 10):
        corr = np.corrcoef(t[:-lag], t[lag:])[0, 1]
        print(f"  Target autocorr lag {lag}: {corr:.4f}")
    print("  Target pattern mod 3:", pd.crosstab(df["row_idx"] % 3, df["target"]))
    print("  VDRL pattern mod 2:", pd.crosstab(df["row_idx"] % 2, df["vdrl"]))
    # Runs test on target
    runs = 1 + np.sum(t[1:] != t[:-1])
    n1 = np.sum(t == 1)
    n0 = np.sum(t == 0)
    exp_runs = 1 + 2 * n1 * n0 / (n1 + n0)
    var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
    z_runs = (runs - exp_runs) / np.sqrt(var_runs)
    p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
    print(f"  Wald-Wolfowitz runs test on target: runs={runs}, expected={exp_runs:.1f}, z={z_runs:.3f}, p={p_runs:.4e}")
    # Check row_idx correlation with target
    r_idx, p_idx = stats.pointbiserialr(df["target"], df["row_idx"])
    print(f"  Point-biserial corr(row_idx, target): r={r_idx:.4f}, p={p_idx:.4f}")
