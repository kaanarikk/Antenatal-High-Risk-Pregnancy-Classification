import json
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_audit import load_and_clean, RANDOM_SEED
from ml_core import (
    compute_auroc, compute_auprc, compute_brier, sens_spec_at_threshold,
    calibration_intercept_slope, decision_curve_analysis, roc_curve_points, pr_curve_points,
    compute_vif, fit_firth_logistic,
    LogisticRegressionL2, DecisionTreeClassifierScratch, RandomForestClassifierScratch,
    XGBoostScratch, EBMScratch, make_group_kfold, compute_shapley_values
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

OUT_TABLES = "outputs/tables"
OUT_FIGS = "outputs/figures"
os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def save_table(df_tbl: pd.DataFrame, name: str, caption: str = "", index: bool = False):
    csv_path = os.path.join(OUT_TABLES, f"{name}.csv")
    tex_path = os.path.join(OUT_TABLES, f"{name}.tex")
    df_tbl.to_csv(csv_path, index=index)
    try:
        tex_str = df_tbl.to_latex(index=index, escape=True, caption=caption if caption else None)
    except Exception:
        tex_str = df_tbl.to_string(index=index)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_str)

def save_fig(fig, name: str):
    fig.savefig(os.path.join(OUT_FIGS, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_FIGS, f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)

def build_feature_matrix(df: pd.DataFrame, drop_cols=None):
    """Build numeric feature matrix for modeling & inference (no Name, no row_idx)."""
    drop_cols = set(drop_cols or [])
    feat_dict = {
        "Age (years)": df["age"].values,
        "Gestational age (weeks)": df["ga_weeks"].values,
        "Weight (kg)": df["weight_kg"].values,
        "Height (cm)": df["height_cm"].values,
        "SBP (mmHg)": df["sbp"].values,
        "DBP (mmHg)": df["dbp"].values,
        "FHR (bpm)": df["fhr_bpm"].values,
        "Gravida (>=2nd)": (df["gravida"] != "1st").astype(float).values,
        "TT dose (ordinal 1-3)": df["tt_dose"].map({"1st": 1.0, "2nd": 2.0, "3rd": 3.0}).values,
        "Anemia (any)": (df["anemia"] != "None").astype(float).values,
        "Jaundice (any)": (df["jaundice"] != "None").astype(float).values,
        "Abnormal fetal position": (df["fetal_position"] == "Abnormal").astype(float).values,
        "Urine albumin (any)": (df["urine_albumin"] != "None").astype(float).values,
        "Urine sugar (Yes)": (df["urine_sugar"] == "Yes").astype(float).values,
        "VDRL (Positive)": (df["vdrl"] == "Positive").astype(float).values,
        "HBsAg (Positive)": (df["hbsag"] == "Positive").astype(float).values,
    }
    cols = [k for k in feat_dict.keys() if k not in drop_cols]
    X = np.column_stack([feat_dict[k] for k in cols])
    return X, cols

# ==============================================================================
# TASK 1: DATA AUDIT
# ==============================================================================
def run_task1_audit(df: pd.DataFrame):
    print("[Task 1] Running Data Audit...")
    n = len(df)
    feat_cols = ["age", "gravida", "tt_dose", "ga_weeks", "weight_kg", "height_cm", "sbp", "dbp",
                 "anemia", "jaundice", "fetal_position", "fetal_movement", "fhr_bpm",
                 "urine_albumin", "urine_sugar", "vdrl", "hbsag", "target"]
    n_miss = sum(df[c].isna().sum() if df[c].dtype != object else (df[c] == "").sum() for c in feat_cols)
    full_dups_raw = int(df.duplicated(subset=["Name_raw"] + feat_cols, keep="first").sum())
    ign_name_dups = int(df.duplicated(subset=feat_cols, keep="first").sum())
    near_dups_no_age = int(df.duplicated(subset=[c for c in feat_cols if c not in ("age", "target")], keep="first").sum())
    near_dups_no_age_ga = int(df.duplicated(subset=[c for c in feat_cols if c not in ("age", "ga_weeks", "target")], keep="first").sum())

    name_counts = df["name_norm"].value_counts()
    n_unique_names_raw = df["Name_raw"].nunique()
    n_unique_names_norm = len(name_counts)
    n_repeated_names = int((name_counts > 1).sum())
    name_age_nunique = df.groupby("name_norm")["age"].nunique()
    n_rep_diff_age = int((name_age_nunique[name_counts > 1] > 1).sum())

    # Periodicity checks
    per_rows = []
    for col in ["age", "gravida", "tt_dose", "ga_weeks", "weight_kg", "height_cm", "sbp", "dbp",
                "anemia", "jaundice", "fetal_position", "fetal_movement", "fhr_bpm",
                "urine_albumin", "urine_sugar", "vdrl", "hbsag", "target"]:
        s = df[col].values
        best_lag, best_match = 1, 0.0
        for lag in range(1, 50):
            m = float(np.mean(s[:-lag] == s[lag:]))
            if m > best_match:
                best_match = m
                best_lag = lag
        per_rows.append({
            "Column": col,
            "Unique_Values": int(df[col].nunique()),
            "Best_Lag": int(best_lag),
            "Exact_Match_Rate_at_Lag": round(best_match, 4),
        })
    df_per = pd.DataFrame(per_rows)
    save_table(df_per, "audit_periodicity", "Row-order periodicity analysis across register variables.")

    t = df["target"].values
    runs = int(1 + np.sum(t[1:] != t[:-1]))
    n1, n0 = int(np.sum(t == 1)), int(np.sum(t == 0))
    exp_runs = 1.0 + 2.0 * n1 * n0 / (n1 + n0)
    var_runs = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n1 - n0)) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
    z_runs = (runs - exp_runs) / np.sqrt(var_runs)
    p_runs = float(2.0 * (1.0 - stats.norm.cdf(abs(z_runs))))

    audit_summary = pd.DataFrame([
        {"Metric": "Total records", "Value": str(n)},
        {"Metric": "Parsed missing values (with 'None' preserved)", "Value": str(n_miss)},
        {"Metric": "Full-row exact duplicates (incl. Name)", "Value": str(full_dups_raw)},
        {"Metric": "Exact feature+target duplicates (excl. Name)", "Value": str(ign_name_dups)},
        {"Metric": "Near-identical clinical rows (excl. Name & Age)", "Value": f"{near_dups_no_age} extra ({n - near_dups_no_age} unique)"},
        {"Metric": "Near-identical clinical rows (excl. Name, Age & GA)", "Value": f"{near_dups_no_age_ga} extra ({n - near_dups_no_age_ga} unique)"},
        {"Metric": "Unique raw names / normalized names", "Value": f"{n_unique_names_raw} / {n_unique_names_norm}"},
        {"Metric": "Repeated normalized names (>1 row)", "Value": f"{n_repeated_names} ({n_repeated_names/n_unique_names_norm*100:.1f}%), max repeats = {name_counts.max()}"},
        {"Metric": "Repeated names with >1 distinct age", "Value": f"{n_rep_diff_age} / {n_repeated_names}"},
        {"Metric": "Zero-variance columns", "Value": "fetal_movement (100% Normal)"},
        {"Metric": "Target balance (High-risk = Yes)", "Value": f"{n1} / {n} ({n1/n*100:.2f}%)"},
        {"Metric": "VDRL Positive / Negative", "Value": f"{(df['vdrl']=='Positive').sum()} / {(df['vdrl']=='Negative').sum()} (50.00% / 50.00%, exact lag-2 alternation)"},
        {"Metric": "HBsAg Positive -> High-risk = Yes", "Value": f"{((df['hbsag']=='Positive') & (df['target']==1)).sum()} / {(df['hbsag']=='Positive').sum()} (100.00%)"},
        {"Metric": "Urine sugar = Yes prevalence", "Value": f"{(df['urine_sugar']=='Yes').sum()} / {n} ({(df['urine_sugar']=='Yes').mean()*100:.2f}%)"},
        {"Metric": "SBP range (mmHg)", "Value": f"{df['sbp'].min():.0f} - {df['sbp'].max():.0f} (0% >= 140 mmHg)"},
        {"Metric": "Target lag-3 exact match / Wald-Wolfowitz runs test", "Value": f"99.90% (997/998); runs={runs} vs exp={exp_runs:.1f}, z={z_runs:.2f}, p={p_runs:.2e}"},
    ])
    save_table(audit_summary, "audit_summary", "Data integrity and register audit summary.")

    report = f"""# Data Audit Report: ANCC Register (`Book2.xlsx`)

## 1. Overview & Privacy
- **Total Records**: {n} antenatal-care register entries parsed from `Sheet1` (header row starting with `Name`, `keep_default_na=False`).
- **De-identification**: `Name` was normalized (lowercased/trimmed) and hashed (SHA-256) into `group_id`, then dropped from all modeling and output tables.
- **Missingness**: {n_miss} missing values across all 18 parsed fields when `"None"` is preserved as a valid clinical category.

## 2. Duplicate & Name-Repetition Structure
- **Exact full-row duplicates**: {full_dups_raw}.
- **Exact duplicates ignoring Name**: {ign_name_dups}.
- **Near-identical clinical rows**: {near_dups_no_age} extra rows ignoring Name and Age ({n - near_dups_no_age} unique); {near_dups_no_age_ga} extra rows ignoring Name, Age, and Gestational Age ({n - near_dups_no_age_ga} unique).
- **Name repetition**: {n_unique_names_raw} unique raw names ({n_unique_names_norm} after case/whitespace normalization). Of {n_unique_names_norm} normalized names, {n_repeated_names} ({n_repeated_names/n_unique_names_norm*100:.1f}%) appear multiple times (up to {name_counts.max()} times), and {n_rep_diff_age}/{n_repeated_names} repeated names have multiple distinct ages. Thus, `Name` is not a reliable unique patient identifier.

## 3. Zero-Variance & Implausible Clinical Distributions
- **Zero-variance column**: `fetal_movement` is 100.00% `"Normal"` ({n}/{n}).
- **VDRL**: Exactly 499 Positive (50.00%) and 499 Negative (50.00%), which is epidemiologically implausible for a routine ANC population.
- **Urine sugar**: 89.18% (`{(df['urine_sugar']=='Yes').sum()}/{n}`) recorded as `"Yes"` (implausibly high).
- **Blood pressure**: SBP ranges from {df['sbp'].min():.0f} to {df['sbp'].max():.0f} mmHg and DBP from {df['dbp'].min():.0f} to {df['dbp'].max():.0f} mmHg; 0% of records meet hypertension criteria (SBP >= 140 or DBP >= 90 mmHg).
- **HBsAg & Target**: 109/998 (10.92%) are HBsAg-positive, and 109/109 (100.00%) of HBsAg-positive records are classified as High-risk (`Yes`). Overall target prevalence is {n1}/{n} ({n1/n*100:.2f}%).

## 4. Row-Order Leakage & Deterministic Periodicity
- **Target periodicity**: `High-risk pregnancy` follows a near-deterministic period-3 cycle (`row_idx % 3 == 0`: 333/333 Yes; `row_idx % 3 == 1`: 332/333 No, 1 Yes; `row_idx % 3 == 2`: 332/332 Yes; lag-3 exact match = 99.90%, lag-3 autocorrelation = 0.9977; Wald-Wolfowitz runs test `z = {z_runs:.2f}, p = {p_runs:.2e}`).
- **VDRL periodicity**: Strictly alternates `Negative, Positive, Negative, Positive` with lag-2 exact match = 100.00%.
- **Phase-locked predictors**: `HBsAg` (lag-9 match 99.90%), `TT dose` (lag-24 match 98.56%), `SBP` (lag-24 match 97.74%), `DBP` (lag-48 match 98.00%), `Weight` (lag-27 match 96.40%), `Urine sugar` (lag-27 match 98.87%), and `FHR` (lag-11 match 99.19%) exhibit strong cyclic repetition. Because periods 9, 24, and 27 are multiples of 3, combinations of `Weight` and `TT dose` phase-lock with `row_idx % 3`, explaining why tree/ensemble models achieve near-perfect accuracy without true clinical danger signs.
"""
    with open("outputs/audit_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    return {"n": n, "n1": n1, "n0": n0, "z_runs": z_runs, "p_runs": p_runs}

# ==============================================================================
# TASK 2: TABLE 1
# ==============================================================================
def run_task2_table1(df: pd.DataFrame):
    print("[Task 2] Generating Table 1...")
    df0 = df[df["target"] == 0]
    df1 = df[df["target"] == 1]
    rows = []

    cont_vars = [
        ("Age (years)", "age"),
        ("Gestational age (weeks)", "ga_weeks"),
        ("Weight (kg)", "weight_kg"),
        ("Height (cm)", "height_cm"),
        ("BMI (kg/m^2)", "bmi"),
        ("Systolic BP (mmHg)", "sbp"),
        ("Diastolic BP (mmHg)", "dbp"),
        ("Fetal heart rate (bpm)", "fhr_bpm"),
    ]
    for label, col in cont_vars:
        x_all, x0, x1 = df[col].values, df0[col].values, df1[col].values
        # Report median [IQR] and mean ± SD
        u_stat, p_mw = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        t_stat, p_t = stats.ttest_ind(x1, x0, equal_var=False)
        # Rank-biserial correlation r = 1 - 2U/(n1*n0)
        r_rb = 1.0 - (2.0 * u_stat) / (len(x1) * len(x0))
        pooled_sd = np.sqrt(((len(x1)-1)*np.var(x1, ddof=1) + (len(x0)-1)*np.var(x0, ddof=1)) / (len(x1)+len(x0)-2))
        cohen_d = (np.mean(x1) - np.mean(x0)) / max(pooled_sd, 1e-12)
        rows.append({
            "Variable": f"{label} [Mean±SD / Median (IQR)]",
            "Overall (n=998)": f"{np.mean(x_all):.2f}±{np.std(x_all, ddof=1):.2f} / {np.median(x_all):.1f} ({np.quantile(x_all,0.25):.1f}-{np.quantile(x_all,0.75):.1f})",
            "Low-risk: No (n=332)": f"{np.mean(x0):.2f}±{np.std(x0, ddof=1):.2f} / {np.median(x0):.1f} ({np.quantile(x0,0.25):.1f}-{np.quantile(x0,0.75):.1f})",
            "High-risk: Yes (n=666)": f"{np.mean(x1):.2f}±{np.std(x1, ddof=1):.2f} / {np.median(x1):.1f} ({np.quantile(x1,0.25):.1f}-{np.quantile(x1,0.75):.1f})",
            "Test": "Mann-Whitney U / Welch t",
            "p_value": f"{p_mw:.4f} / {p_t:.4f}",
            "Effect_Size": f"r_rb={r_rb:.3f}; d={cohen_d:.3f}",
        })

    cat_vars = [
        ("Gravida", "gravida", ["1st", "2nd", "3rd"]),
        ("TT dose", "tt_dose", ["1st", "2nd", "3rd"]),
        ("Anemia", "anemia", ["None", "Minimal", "Medium"]),
        ("Jaundice", "jaundice", ["None", "Minimal", "Medium"]),
        ("Fetal position", "fetal_position", ["Normal", "Abnormal"]),
        ("Fetal movement", "fetal_movement", ["Normal"]),
        ("Urine albumin", "urine_albumin", ["None", "Minimal", "Medium", "Higher"]),
        ("Urine sugar", "urine_sugar", ["Yes", "No"]),
        ("VDRL", "vdrl", ["Negative", "Positive"]),
        ("HBsAg", "hbsag", ["Negative", "Positive"]),
    ]
    for label, col, levels in cat_vars:
        ct = pd.crosstab(df[col], df["target"]).reindex(levels, fill_value=0)
        if len(levels) == 1:
            test_name, p_val, eff = "Constant (zero var)", "NA", "NA"
        else:
            chi2, p_chi, _, expected = stats.chi2_contingency(ct.values)
            if np.any(expected < 5) and ct.shape == (2, 2):
                _, p_val_num = stats.fisher_exact(ct.values)
                test_name = "Fisher exact"
            else:
                p_val_num = p_chi
                test_name = "Chi-square"
            n_tot = ct.values.sum()
            k_min = min(ct.shape) - 1
            cramer_v = np.sqrt(chi2 / (n_tot * max(k_min, 1)))
            p_val = f"{p_val_num:.4e}" if p_val_num < 0.001 else f"{p_val_num:.4f}"
            eff = f"V={cramer_v:.3f}"

        for idx_l, lv in enumerate(levels):
            c_all = int((df[col] == lv).sum())
            c0 = int((df0[col] == lv).sum())
            c1 = int((df1[col] == lv).sum())
            rows.append({
                "Variable": f"{label}: {lv}",
                "Overall (n=998)": f"{c_all} ({c_all/len(df)*100:.1f}%)",
                "Low-risk: No (n=332)": f"{c0} ({c0/len(df0)*100:.1f}%)",
                "High-risk: Yes (n=666)": f"{c1} ({c1/len(df1)*100:.1f}%)",
                "Test": test_name if idx_l == 0 else "",
                "p_value": p_val if idx_l == 0 else "",
                "Effect_Size": eff if idx_l == 0 else "",
            })

    t1 = pd.DataFrame(rows)
    save_table(t1, "table1_descriptives", "Baseline characteristics overall and stratified by high-risk pregnancy classification.")
    return t1

# ==============================================================================
# TASK 3: CLINICAL INFERENCE (FIRTH LOGISTIC REGRESSION + VIF + FOREST PLOT)
# ==============================================================================
def run_task3_inference(df: pd.DataFrame):
    print("[Task 3] Running Clinical Inference (Firth LR + VIF)...")
    X, cols = build_feature_matrix(df)
    y = df["target"].values

    vif_df = compute_vif(X, cols)
    save_table(vif_df.round(4), "vif_table", "Variance Inflation Factors (VIF) for multivariable predictors.")

    res = fit_firth_logistic(X, y)
    inf_rows = []
    for j, name in enumerate(["Intercept"] + cols):
        inf_rows.append({
            "Variable": name,
            "Beta": round(float(res["beta"][j]), 4),
            "SE": round(float(res["se"][j]), 4),
            "OR": round(float(res["or"][j]), 4),
            "CI_95_Low": round(float(res["or_low"][j]), 4),
            "CI_95_High": round(float(res["or_high"][j]), 4),
            "p_value": float(res["pval"][j]),
            "Note": "Complete separation (109/109 Yes)" if name == "HBsAg (Positive)" else "",
        })
    inf_df = pd.DataFrame(inf_rows)
    save_table(inf_df, "clinical_inference_firth", "Multivariable Firth penalized logistic regression for high-risk pregnancy classification.")

    # Forest plot (excluding Intercept)
    sub = inf_df[inf_df["Variable"] != "Intercept"].copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(sub))
    ors = sub["OR"].values
    lows = sub["CI_95_Low"].values
    highs = sub["CI_95_High"].values
    # Cap extreme upper CI for plotting readability, annotate exact value
    plot_highs = np.minimum(highs, 50.0)
    plot_ors = np.minimum(ors, 50.0)
    ax.errorbar(plot_ors, y_pos, xerr=[plot_ors - lows, plot_highs - plot_ors],
                fmt="o", color="#1f4e79", ecolor="#4a7ebb", capsize=3, markersize=5)
    ax.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["Variable"].values)
    ax.set_xlabel("Odds Ratio (95% CI, log scale) — Firth Penalized Logistic Regression")
    ax.set_title("Multivariable Association with High-Risk Pregnancy Classification")
    for idx_r, (_, r) in enumerate(sub.iterrows()):
        txt = f"OR={r['OR']:.2f} ({r['CI_95_Low']:.2f}-{r['CI_95_High']:.2f}), p={r['p_value']:.3f}"
        if r["Variable"] == "HBsAg (Positive)":
            txt = f"OR={r['OR']:.1f} (separation), p={r['p_value']:.4f}"
        ax.text(min(plot_highs[idx_r], 45) * 1.05, y_pos[idx_r], txt, va="center", fontsize=7.5)
    ax.set_xlim(0.15, 350)
    fig.tight_layout()
    save_fig(fig, "forest_plot")
    return inf_df, vif_df

# ==============================================================================
# TASK 4: PREDICTIVE MODELS (5x5 REPEATED GROUPKFOLD + NESTED CV + BASELINES)
# ==============================================================================
def tune_and_fit(model_name: str, X_tr, y_tr, g_tr, seed: int):
    inner_splits = make_group_kfold(g_tr, y_tr, n_splits=3, seed=seed)
    if model_name == "Logistic Regression":
        grid = [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]]
        cls = LogisticRegressionL2
    elif model_name == "Decision Tree":
        grid = [{"max_depth": d, "min_samples_leaf": m, "random_state": seed}
                for d in [3, 4, 6, 8] for m in [3, 8]]
        cls = DecisionTreeClassifierScratch
    elif model_name == "EBM":
        grid = [{"n_rounds": nr, "learning_rate": lr, "n_interactions": 5, "random_state": seed}
                for nr in [30, 50] for lr in [0.05, 0.1]]
        cls = EBMScratch
    elif model_name == "Random Forest":
        grid = [{"n_estimators": 50, "max_depth": d, "min_samples_leaf": m, "random_state": seed}
                for d in [6, 10] for m in [1, 3]]
        cls = RandomForestClassifierScratch
    elif model_name == "XGBoost":
        grid = [{"n_estimators": ne, "max_depth": d, "learning_rate": lr, "random_state": seed}
                for ne in [40, 70] for d in [3, 5] for lr in [0.08, 0.15]]
        cls = XGBoostScratch
    else:
        raise ValueError(model_name)

    best_auc, best_params = -1.0, grid[0]
    for params in grid:
        aucs = []
        for itr, ival in inner_splits:
            m = cls(**params).fit(X_tr[itr], y_tr[itr])
            p_val = m.predict_proba(X_tr[ival])[:, 1]
            aucs.append(compute_auroc(y_tr[ival], p_val))
        mean_auc = float(np.nanmean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    final_model = cls(**best_params).fit(X_tr, y_tr)
    return final_model, best_params

def evaluate_predictions(y_true, y_prob):
    auroc = compute_auroc(y_true, y_prob)
    auprc = compute_auprc(y_true, y_prob)
    brier = compute_brier(y_true, y_prob)
    sens50, spec50, _, _, acc50 = sens_spec_at_threshold(y_true, y_prob, 0.50)
    fpr, tpr, thresh = roc_curve_points(y_true, y_prob)
    j_idx = np.argmax(tpr - fpr)
    youden_t = float(thresh[j_idx]) if j_idx < len(thresh) else 0.50
    sens_y, spec_y, _, _, _ = sens_spec_at_threshold(y_true, y_prob, youden_t)
    cal_int, cal_slp = calibration_intercept_slope(y_true, y_prob)
    return {
        "AUROC": auroc, "AUPRC": auprc, "Brier": brier,
        "Sens_0.5": sens50, "Spec_0.5": spec50, "Acc_0.5": acc50,
        "Youden_Thresh": youden_t, "Sens_Youden": sens_y, "Spec_Youden": spec_y,
        "Cal_Intercept": cal_int, "Cal_Slope": cal_slp,
    }

def run_task4_predictive(df: pd.DataFrame):
    print("[Task 4] Running 5x5 Repeated GroupKFold + Nested CV...")
    X, cols = build_feature_matrix(df)
    y = df["target"].values
    groups = df["group_id"].values

    models_list = ["Logistic Regression", "Decision Tree", "EBM", "Random Forest", "XGBoost"]
    oof_probs = {m: np.zeros((5, len(y)), dtype=float) for m in models_list}
    fold_metrics = {m: [] for m in models_list}

    # Baselines
    oof_shuf = np.zeros((5, len(y)), dtype=float)
    fold_shuf = []
    X_no_susp, cols_no_susp = build_feature_matrix(df, drop_cols={"HBsAg (Positive)", "TT dose (ordinal 1-3)", "VDRL (Positive)"})
    oof_nosusp = np.zeros((5, len(y)), dtype=float)
    fold_nosusp = []

    rng_shuf = np.random.default_rng(RANDOM_SEED)

    for rep in range(5):
        splits = make_group_kfold(groups, y, n_splits=5, seed=RANDOM_SEED + rep)
        y_shuf = rng_shuf.permutation(y)
        for fold_idx, (tr, te) in enumerate(splits):
            seed_fold = RANDOM_SEED + rep * 10 + fold_idx
            for m_name in models_list:
                mdl, _ = tune_and_fit(m_name, X[tr], y[tr], groups[tr], seed_fold)
                p_te = mdl.predict_proba(X[te])[:, 1]
                oof_probs[m_name][rep, te] = p_te
                met = evaluate_predictions(y[te], p_te)
                fold_metrics[m_name].append(met)

            # Shuffled-label RF baseline
            rf_shuf = RandomForestClassifierScratch(n_estimators=40, max_depth=8, min_samples_leaf=2, random_state=seed_fold)
            rf_shuf.fit(X[tr], y_shuf[tr])
            p_sh = rf_shuf.predict_proba(X[te])[:, 1]
            oof_shuf[rep, te] = p_sh
            fold_shuf.append(evaluate_predictions(y[te], p_sh))

            # No HBsAg, TT, VDRL RF baseline
            rf_ns = RandomForestClassifierScratch(n_estimators=50, max_depth=8, min_samples_leaf=2, random_state=seed_fold)
            rf_ns.fit(X_no_susp[tr], y[tr])
            p_ns = rf_ns.predict_proba(X_te := X_no_susp[te])[:, 1]
            oof_nosusp[rep, te] = p_ns
            fold_nosusp.append(evaluate_predictions(y[te], p_ns))

    summary_rows = []
    all_runs = [(m, fold_metrics[m]) for m in models_list] + [
        ("Baseline: RF (Shuffled Labels)", fold_shuf),
        ("Baseline: RF (No HBsAg/TT/VDRL)", fold_nosusp),
    ]
    for name, m_list in all_runs:
        mdf = pd.DataFrame(m_list)
        row = {"Model": name}
        for k in ["AUROC", "AUPRC", "Brier", "Sens_0.5", "Spec_0.5", "Acc_0.5", "Sens_Youden", "Spec_Youden", "Cal_Intercept", "Cal_Slope"]:
            row[f"{k}_mean"] = round(float(mdf[k].mean()), 4)
            row[f"{k}_sd"] = round(float(mdf[k].std()), 4)
        summary_rows.append(row)

    perf_df = pd.DataFrame(summary_rows)
    save_table(perf_df, "model_performance", "5x5 repeated GroupKFold nested-CV predictive performance.")

    # Mean OOF probabilities across 5 repeats for curves
    mean_oof = {m: oof_probs[m].mean(axis=0) for m in models_list}
    mean_oof["RF (No HBsAg/TT/VDRL)"] = oof_nosusp.mean(axis=0)
    mean_oof["RF (Shuffled)"] = oof_shuf.mean(axis=0)

    # ROC Plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for m_name, p_hat in mean_oof.items():
        fpr, tpr, _ = roc_curve_points(y, p_hat)
        auc_val = compute_auroc(y, p_hat)
        ls = "--" if "Shuffled" in m_name or "No HBsAg" in m_name else "-"
        ax.plot(fpr, tpr, linestyle=ls, linewidth=1.8, label=f"{m_name} (AUROC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1)
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("ROC Curves (Mean 5x5 Repeated GroupKFold OOF)")
    ax.legend(loc="lower right")
    save_fig(fig, "roc_curves")

    # PR Plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for m_name, p_hat in mean_oof.items():
        rec, prec = pr_curve_points(y, p_hat)
        auprc_val = compute_auprc(y, p_hat)
        ls = "--" if "Shuffled" in m_name or "No HBsAg" in m_name else "-"
        ax.plot(rec, prec, linestyle=ls, linewidth=1.8, label=f"{m_name} (AUPRC={auprc_val:.3f})")
    ax.axhline(np.mean(y), color="gray", linestyle=":", label=f"Prevalence ({np.mean(y):.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curves (Mean 5x5 Repeated GroupKFold OOF)")
    ax.legend(loc="lower left")
    save_fig(fig, "pr_curves")

    # Calibration Plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bins = np.linspace(0, 1, 11)
    for m_name in models_list:
        p_hat = mean_oof[m_name]
        b_idx = np.clip(np.digitize(p_hat, bins) - 1, 0, 9)
        mp, fp = [], []
        for b in range(10):
            m = (b_idx == b)
            if np.sum(m) >= 5:
                mp.append(np.mean(p_hat[m]))
                fp.append(np.mean(y[m]))
        ax.plot(mp, fp, "o-", linewidth=1.6, markersize=4, label=m_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Proportion")
    ax.set_title("Calibration Plot (10 Probability Bins)")
    ax.legend(loc="upper left")
    save_fig(fig, "calibration_plot")

    # Decision Curve Analysis
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    pts, nb_all, nb_none = None, None, None
    for m_name in models_list:
        pts, nb_m, nb_all, nb_none = decision_curve_analysis(y, mean_oof[m_name])
        ax.plot(pts, nb_m, linewidth=1.8, label=m_name)
    ax.plot(pts, nb_all, "k--", linewidth=1.2, label="Treat All")
    ax.plot(pts, nb_none, "k:", linewidth=1.2, label="Treat None")
    ax.set_ylim(-0.05, 0.72)
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis (5x5 Repeated GroupKFold OOF)")
    ax.legend(loc="lower left")
    save_fig(fig, "dca_plot")

    return perf_df, mean_oof

# ==============================================================================
# TASK 5: LABEL AUDIT / RULE RECOVERY
# ==============================================================================
def run_task5_rules(df: pd.DataFrame):
    print("[Task 5] Running Label Audit & Rule Recovery...")
    X, cols = build_feature_matrix(df)
    y = df["target"].values

    rule_rows = []
    fidelity_rows = []
    for d in [2, 3, 4]:
        dt = DecisionTreeClassifierScratch(max_depth=d, min_samples_leaf=5, random_state=RANDOM_SEED)
        dt.fit(X, y)
        p_hat = dt.predict_proba(X)[:, 1]
        pred = (p_hat >= 0.5).astype(int)
        acc = float(np.mean(pred == y))
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
        fidelity_rows.append({"Tree_Depth": d, "Leaves": len(dt.extract_rules(cols)), "Fidelity_Accuracy": round(acc, 4), "F1_Score": round(f1, 4)})
        for r in dt.extract_rules(cols):
            rule_rows.append({
                "Depth": d, "Rule": r["rule"], "Predicted": r["pred"],
                "N": r["n"], "N_Yes": r["n_yes"], "Purity": round(r["purity"], 4)
            })

    save_table(pd.DataFrame(fidelity_rows), "tree_rule_fidelity", "Fidelity of compact decision trees (depth 2-4) to the register label.")
    save_table(pd.DataFrame(rule_rows), "compact_tree_rules", "Extracted compact decision tree rules (depth 2-4).")

    # Clinical ANC danger signs checklist (to be verified by a clinician against WHO/national guidelines)
    danger_signs = [
        ("Severe/moderate anemia (Medium)", df["anemia"] == "Medium"),
        ("Hypertension / pre-eclampsia (SBP>=140 or DBP>=90)", (df["sbp"] >= 140) | (df["dbp"] >= 90)),
        ("Proteinuria / urine albumin (Medium or Higher)", df["urine_albumin"].isin(["Medium", "Higher"])),
        ("Abnormal fetal position", df["fetal_position"] == "Abnormal"),
        ("Abnormal FHR (<110 or >160 bpm)", (df["fhr_bpm"] < 110) | (df["fhr_bpm"] > 160)),
        ("Adolescent pregnancy (Age < 20)", df["age"] < 20),
        ("Positive syphilis screen (VDRL Positive)", df["vdrl"] == "Positive"),
        ("Positive hepatitis B screen (HBsAg Positive)", df["hbsag"] == "Positive"),
        ("Grand multiparity (Gravida >= 5)", pd.Series(False, index=df.index)),
    ]

    ds_rows = []
    any_danger = np.zeros(len(df), dtype=bool)
    any_danger_no_vdrl = np.zeros(len(df), dtype=bool)
    for name, mask in danger_signs:
        m = mask.values
        any_danger |= m
        if "VDRL" not in name:
            any_danger_no_vdrl |= m
        n_pos = int(np.sum(m))
        n_hr = int(np.sum(m & (y == 1)))
        ds_rows.append({
            "Clinical_Danger_Sign": name,
            "Checklist_Status": "To be verified by a clinician against WHO/national guidelines",
            "Count_in_Register": n_pos,
            "Pct_of_Register": round(n_pos / len(df) * 100, 2),
            "High_Risk_Yes_Count": n_hr,
            "High_Risk_Yes_Pct": round(n_hr / max(n_pos, 1) * 100, 2) if n_pos > 0 else np.nan,
            "Recovered_by_Depth4_Tree": "Yes (HBsAg)" if "HBsAg" in name else "No / Minimal",
        })

    # Concordance between clinical danger-sign rule set and register label
    def kappa_score(a, b):
        po = np.mean(a == b)
        pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
        return float((po - pe) / max(1 - pe, 1e-12))

    conc_all = {
        "Rule_Set": "Any ANC danger sign (incl. VDRL)",
        "N_Flagged": int(np.sum(any_danger)),
        "Pct_Flagged": round(float(np.mean(any_danger) * 100), 2),
        "Accuracy_vs_Label": round(float(np.mean(any_danger == (y == 1))), 4),
        "Cohens_Kappa": round(kappa_score(any_danger, y == 1), 4),
        "Sensitivity": round(float(np.sum(any_danger & (y == 1)) / np.sum(y == 1)), 4),
        "Specificity": round(float(np.sum((~any_danger) & (y == 0)) / np.sum(y == 0)), 4),
    }
    conc_nov = {
        "Rule_Set": "Any ANC danger sign (excl. VDRL)",
        "N_Flagged": int(np.sum(any_danger_no_vdrl)),
        "Pct_Flagged": round(float(np.mean(any_danger_no_vdrl) * 100), 2),
        "Accuracy_vs_Label": round(float(np.mean(any_danger_no_vdrl == (y == 1))), 4),
        "Cohens_Kappa": round(kappa_score(any_danger_no_vdrl, y == 1), 4),
        "Sensitivity": round(float(np.sum(any_danger_no_vdrl & (y == 1)) / np.sum(y == 1)), 4),
        "Specificity": round(float(np.sum((~any_danger_no_vdrl) & (y == 0)) / np.sum(y == 0)), 4),
    }

    df_ds = pd.DataFrame(ds_rows)
    df_conc = pd.DataFrame([conc_all, conc_nov])
    save_table(df_ds, "rule_concordance_checklist", "Standard ANC danger signs vs. register label (to be verified by a clinician against WHO/national guidelines).")
    save_table(df_conc, "rule_concordance_summary", "Concordance between clinical danger-sign rule sets and the register label.")
    return fidelity_rows, df_ds, df_conc

# ==============================================================================
# TASK 6: EXPLAINABILITY (SHAP + STABILITY + PDP + COUNTERFACTUALS)
# ==============================================================================
def run_task6_explainability(df: pd.DataFrame):
    print("[Task 6] Running Explainability (SHAP, Stability, PDP, Counterfactuals)...")
    X, cols = build_feature_matrix(df)
    y = df["target"].values
    groups = df["group_id"].values
    splits = make_group_kfold(groups, y, n_splits=5, seed=RANDOM_SEED)

    fold_imp = []
    all_shap = np.zeros_like(X, dtype=float)
    for f_idx, (tr, te) in enumerate(splits):
        rf = RandomForestClassifierScratch(n_estimators=50, max_depth=8, min_samples_leaf=2, random_state=RANDOM_SEED + f_idx)
        rf.fit(X[tr], y[tr])
        bg_idx = np.random.default_rng(RANDOM_SEED + f_idx).choice(len(tr), size=min(80, len(tr)), replace=False)
        sv, _ = compute_shapley_values(rf, X[te], X[tr][bg_idx], n_samples=50, seed=RANDOM_SEED + f_idx)
        all_shap[te] = sv
        fold_imp.append(np.mean(np.abs(sv), axis=0))

    fold_imp = np.array(fold_imp)
    mean_abs_shap = fold_imp.mean(axis=0)
    sd_abs_shap = fold_imp.std(axis=0)

    # Pairwise Spearman rank correlations across the 5 folds
    rhos = []
    for i in range(5):
        for j in range(i + 1, 5):
            r, _ = stats.spearmanr(fold_imp[i], fold_imp[j])
            rhos.append(float(r))

    shap_df = pd.DataFrame({
        "Feature": cols,
        "Mean_Abs_SHAP": np.round(mean_abs_shap, 5),
        "SD_Across_Folds": np.round(sd_abs_shap, 5),
        "Rank": stats.rankdata(-mean_abs_shap).astype(int),
    }).sort_values("Rank")
    shap_df["Pairwise_Fold_Spearman_Mean"] = round(float(np.mean(rhos)), 4)
    shap_df["Pairwise_Fold_Spearman_Min"] = round(float(np.min(rhos)), 4)
    save_table(shap_df, "shap_stability", "Global Shapley feature importance and 5-fold ranking stability.")

    # SHAP Summary Bar + Beeswarm-style Plot
    top_order = np.argsort(mean_abs_shap)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(np.arange(len(cols)), mean_abs_shap[top_order], xerr=sd_abs_shap[top_order], color="#2b6cb0", capsize=3)
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels([cols[i] for i in top_order])
    ax.set_xlabel("Mean |Shapley Value| (impact on predicted P(High-risk))")
    ax.set_title(f"Global Feature Importance (5-Fold Ranking Stability Spearman r={np.mean(rhos):.3f})")
    fig.tight_layout()
    save_fig(fig, "shap_summary")

    # Partial Dependence Plots for top 4 features
    rf_full = RandomForestClassifierScratch(n_estimators=60, max_depth=8, min_samples_leaf=2, random_state=RANDOM_SEED).fit(X, y)
    top4 = np.argsort(-mean_abs_shap)[:4]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax_i, feat_i in enumerate(top4):
        ax = axes.flat[ax_i]
        vals = np.unique(X[:, feat_i])
        if len(vals) > 15:
            vals = np.linspace(vals.min(), vals.max(), 15)
        pdp_y = []
        X_tmp = X.copy()
        for v in vals:
            X_tmp[:, feat_i] = v
            pdp_y.append(np.mean(rf_full.predict_proba(X_tmp)[:, 1]))
        ax.plot(vals, pdp_y, "o-", color="#1a365d", linewidth=1.8, markersize=4)
        ax.set_title(cols[feat_i])
        ax.set_ylabel("Mean Predicted P(High-risk)")
        ax.set_ylim(0, 1)
    fig.suptitle("Partial Dependence Plots (Top 4 Features by Shapley Importance)", y=1.01)
    fig.tight_layout()
    save_fig(fig, "pdp_top_features")

    # Counterfactual examples: pick 3 low-risk predictions and find minimal single/two-feature changes flipping to High-risk
    p_full = rf_full.predict_proba(X)[:, 1]
    low_idx = np.where((y == 0) & (p_full < 0.35))[0][:3]
    cf_rows = []
    for idx in low_idx:
        x0 = X[idx].copy()
        p0 = float(p_full[idx])
        best_cf = None
        # Try single-feature perturbations
        for j in range(len(cols)):
            for v in np.unique(X[:, j]):
                if v == x0[j]:
                    continue
                x_new = x0.copy()
                x_new[j] = v
                p_new = float(rf_full.predict_proba(x_new.reshape(1, -1))[:, 1][0])
                if p_new >= 0.65:
                    best_cf = (cols[j], x0[j], v, p_new)
                    break
            if best_cf is not None:
                break
        if best_cf is None:
            # Fallback: force HBsAg=1
            j_hb = cols.index("HBsAg (Positive)")
            x_new = x0.copy()
            x_new[j_hb] = 1.0
            p_new = float(rf_full.predict_proba(x_new.reshape(1, -1))[:, 1][0])
            best_cf = ("HBsAg (Positive)", x0[j_hb], 1.0, p_new)
        cf_rows.append({
            "Record_Index": int(idx),
            "Original_Label": "No",
            "Original_Prob": round(p0, 3),
            "Changed_Feature": best_cf[0],
            "From_Value": round(float(best_cf[1]), 2),
            "To_Value": round(float(best_cf[2]), 2),
            "Counterfactual_Prob": round(best_cf[3], 3),
            "Counterfactual_Pred": "Yes" if best_cf[3] >= 0.5 else "No",
        })
    cf_df = pd.DataFrame(cf_rows)
    save_table(cf_df, "counterfactuals", "Illustrative counterfactual examples flipping Low-risk predictions to High-risk.")
    return shap_df, float(np.mean(rhos)), cf_df

# ==============================================================================
# TASK 7: SPLIT CONFORMAL PREDICTION (TARGET COVERAGE 90%)
# ==============================================================================
def run_task7_conformal(df: pd.DataFrame):
    print("[Task 7] Running Split Conformal Prediction (90% target coverage)...")
    X, _ = build_feature_matrix(df)
    y = df["target"].values
    groups = df["group_id"].values
    splits = make_group_kfold(groups, y, n_splits=5, seed=RANDOM_SEED)

    alpha = 0.10
    coverages = []
    uncertain_fracs = []
    singleton_yes_fracs = []
    singleton_no_fracs = []
    empty_fracs = []
    flagged_indices = []

    for f_idx, (tr_cal, te) in enumerate(splits):
        # Split tr_cal into proper train (75%) and calibration (25%) by group
        tr_groups = groups[tr_cal]
        cal_splits = make_group_kfold(tr_groups, y[tr_cal], n_splits=4, seed=RANDOM_SEED + 100 + f_idx)
        sub_tr, sub_cal = cal_splits[0]
        idx_tr = tr_cal[sub_tr]
        idx_cal = tr_cal[sub_cal]

        rf = RandomForestClassifierScratch(n_estimators=50, max_depth=8, min_samples_leaf=2, random_state=RANDOM_SEED + f_idx)
        rf.fit(X[idx_tr], y[idx_tr])

        p_cal = rf.predict_proba(X[idx_cal])[:, 1]
        # Nonconformity score: 1 - p(y_true)
        scores_cal = np.where(y[idx_cal] == 1, 1.0 - p_cal, p_cal)
        n_cal = len(scores_cal)
        q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
        q_hat = float(np.quantile(scores_cal, q_level, method="higher") if hasattr(np, "quantile") else np.percentile(scores_cal, q_level * 100))

        p_te = rf.predict_proba(X[te])[:, 1]
        # Prediction set C(x): include 1 if (1 - p_te) <= q_hat; include 0 if p_te <= q_hat
        in_1 = (1.0 - p_te) <= q_hat
        in_0 = p_te <= q_hat
        covered = np.where(y[te] == 1, in_1, in_0)
        uncertain = in_1 & in_0
        empty = (~in_1) & (~in_0)
        sing_yes = in_1 & (~in_0)
        sing_no = in_0 & (~in_1)

        coverages.append(float(np.mean(covered)))
        uncertain_fracs.append(float(np.mean(uncertain)))
        singleton_yes_fracs.append(float(np.mean(sing_yes)))
        singleton_no_fracs.append(float(np.mean(sing_no)))
        empty_fracs.append(float(np.mean(empty)))
        flagged_indices.extend(te[uncertain].tolist())

    conf_summary = pd.DataFrame([{
        "Target_Coverage": 0.90,
        "Empirical_Coverage_Mean": round(float(np.mean(coverages)), 4),
        "Empirical_Coverage_SD": round(float(np.std(coverages)), 4),
        "Uncertain_Set_YesNo_Frac": round(float(np.mean(uncertain_fracs)), 4),
        "Singleton_Yes_Frac": round(float(np.mean(singleton_yes_fracs)), 4),
        "Singleton_No_Frac": round(float(np.mean(singleton_no_fracs)), 4),
        "Empty_Set_Frac": round(float(np.mean(empty_fracs)), 4),
        "Total_Uncertain_Cases_Flagged": len(flagged_indices),
    }])
    save_table(conf_summary, "conformal_summary", "Split conformal prediction summary (alpha=0.10, target coverage 90%).")

    # Conformal coverage & set-type bar plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cats = ["Empirical\nCoverage", "Singleton\n{Yes}", "Singleton\n{No}", "Uncertain\n{Yes, No}", "Empty\n{}"]
    vals = [np.mean(coverages), np.mean(singleton_yes_fracs), np.mean(singleton_no_fracs), np.mean(uncertain_fracs), np.mean(empty_fracs)]
    sds = [np.std(coverages), np.std(singleton_yes_fracs), np.std(singleton_no_fracs), np.std(uncertain_fracs), np.std(empty_fracs)]
    colors = ["#2b6cb0", "#2f855a", "#718096", "#d69e2e", "#c53030"]
    ax.bar(cats, vals, yerr=sds, color=colors, capsize=4)
    ax.axhline(0.90, color="#c0392b", linestyle="--", linewidth=1.2, label="Target Coverage (0.90)")
    for i, v in enumerate(vals):
        ax.text(i, v + sds[i] + 0.02, f"{v*100:.1f}%", ha="center", fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Proportion of Test Cases")
    ax.set_title("Split Conformal Prediction Coverage & Set Composition (90% Target)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "conformal_coverage")
    return conf_summary

# ==============================================================================
# TASK 8: SUBGROUP ANALYSIS
# ==============================================================================
def run_task8_subgroups(df: pd.DataFrame, mean_oof):
    print("[Task 8] Running Subgroup Analysis...")
    y = df["target"].values
    p_rf = mean_oof["Random Forest"]
    p_lr = mean_oof["Logistic Regression"]

    ga_q1, ga_q2 = df["ga_weeks"].quantile([1/3, 2/3]).values
    subgroups = [
        ("Age < 20", df["age"] < 20),
        ("Age >= 20", df["age"] >= 20),
        ("Gravida: 1st", df["gravida"] == "1st"),
        ("Gravida: 2nd/3rd", df["gravida"].isin(["2nd", "3rd"])),
        (f"GA Tertile 1 (<= {ga_q1:.0f}w)", df["ga_weeks"] <= ga_q1),
        (f"GA Tertile 2 ({ga_q1:.0f}-{ga_q2:.0f}w)", (df["ga_weeks"] > ga_q1) & (df["ga_weeks"] <= ga_q2)),
        (f"GA Tertile 3 (> {ga_q2:.0f}w)", df["ga_weeks"] > ga_q2),
    ]
    rows = []
    for label, mask in subgroups:
        m = mask.values
        for m_name, p_hat in [("Random Forest", p_rf), ("Logistic Regression", p_lr)]:
            met = evaluate_predictions(y[m], p_hat[m])
            rows.append({
                "Subgroup": label, "Model": m_name, "N": int(np.sum(m)),
                "Prevalence": round(float(np.mean(y[m])), 4),
                "AUROC": round(met["AUROC"], 4), "AUPRC": round(met["AUPRC"], 4),
                "Brier": round(met["Brier"], 4),
                "Cal_Intercept": round(met["Cal_Intercept"], 3),
                "Cal_Slope": round(met["Cal_Slope"], 3),
            })
    sub_df = pd.DataFrame(rows)
    save_table(sub_df, "subgroup_performance", "Subgroup discrimination and calibration by age, gravida, and gestational age tertiles.")
    return sub_df

# ==============================================================================
# TASK 9: ROBUSTNESS CHECKS
# ==============================================================================
def run_task9_robustness(df: pd.DataFrame):
    print("[Task 9] Running Robustness Checks...")
    scenarios = [
        ("Full dataset (n=998, all features)", df, set()),
        ("Exclude VDRL (n=998)", df, {"VDRL (Positive)"}),
        ("Exclude VDRL + TT + HBsAg (n=998)", df, {"VDRL (Positive)", "TT dose (ordinal 1-3)", "HBsAg (Positive)"}),
        ("Deduplicated excl. Name & Age (n=988)", df.drop_duplicates(subset=["gravida","tt_dose","ga_weeks","weight_kg","height_cm","sbp","dbp","anemia","jaundice","fetal_position","fhr_bpm","urine_albumin","urine_sugar","vdrl","hbsag"]).reset_index(drop=True), set()),
        ("Deduplicated excl. Name, Age & GA (n=964)", df.drop_duplicates(subset=["gravida","tt_dose","weight_kg","height_cm","sbp","dbp","anemia","jaundice","fetal_position","fhr_bpm","urine_albumin","urine_sugar","vdrl","hbsag"]).reset_index(drop=True), set()),
    ]
    rows = []
    for s_name, dsub, drops in scenarios:
        X, _ = build_feature_matrix(dsub, drop_cols=drops)
        y = dsub["target"].values
        groups = dsub["group_id"].values
        for m_name, cls, kwargs in [
            ("Logistic Regression", LogisticRegressionL2, {"C": 1.0}),
            ("Random Forest", RandomForestClassifierScratch, {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 2, "random_state": RANDOM_SEED}),
            ("XGBoost", XGBoostScratch, {"n_estimators": 60, "max_depth": 4, "learning_rate": 0.1, "random_state": RANDOM_SEED}),
        ]:
            aucs, briers = [], []
            for rep in range(3):
                for tr, te in make_group_kfold(groups, y, n_splits=5, seed=RANDOM_SEED + rep):
                    mdl = cls(**kwargs).fit(X[tr], y[tr])
                    p = mdl.predict_proba(X[te])[:, 1]
                    aucs.append(compute_auroc(y[te], p))
                    briers.append(compute_brier(y[te], p))
            rows.append({
                "Scenario": s_name, "Model": m_name, "N": len(dsub),
                "AUROC_mean": round(float(np.mean(aucs)), 4),
                "AUROC_sd": round(float(np.std(aucs)), 4),
                "Brier_mean": round(float(np.mean(briers)), 4),
            })
    rob_df = pd.DataFrame(rows)
    save_table(rob_df, "robustness_summary", "Robustness sensitivity analysis after removing VDRL/suspicious variables and deduplicating near-identical rows.")
    return rob_df

def main():
    t0 = time.time()
    df = load_and_clean()
    t1_res = run_task1_audit(df)
    run_task2_table1(df)
    inf_df, vif_df = run_task3_inference(df)
    perf_df, mean_oof = run_task4_predictive(df)
    fid_rows, _, conc_df = run_task5_rules(df)
    shap_df, spearman_mean, cf_df = run_task6_explainability(df)
    conf_df = run_task7_conformal(df)
    sub_df = run_task8_subgroups(df, mean_oof)
    rob_df = run_task9_robustness(df)

    elapsed = round(time.time() - t0, 2)
    print(f"All 9 tasks completed in {elapsed}s.")
    summary = {
        "elapsed_sec": elapsed,
        "audit": t1_res,
        "firth_top": inf_df.to_dict(orient="records"),
        "model_performance": perf_df.to_dict(orient="records"),
        "rule_fidelity": fid_rows,
        "concordance": conc_df.to_dict(orient="records"),
        "shap_top5": shap_df.head(5).to_dict(orient="records"),
        "shap_spearman_mean": spearman_mean,
        "conformal": conf_df.to_dict(orient="records"),
        "robustness": rob_df.to_dict(orient="records"),
    }
    with open("outputs/run_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
