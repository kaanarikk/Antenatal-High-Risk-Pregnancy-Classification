import os
import re
import hashlib
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIRS = [
    "/Users/kaanarik/Desktop/Maternal Health/manuscript_source/outputs/figures",
    "/Users/kaanarik/Desktop/Maternal Health/prism_source",
    "/Users/kaanarik/Desktop/Maternal Health/bmc_submission",
]

def read_xlsx_stdlib(path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path, "r") as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root_ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root_ss.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared.append("".join(texts))
        root_sh = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows_data = []
        for row in root_sh.findall(".//a:sheetData/a:row", ns):
            row_dict = {}
            for c in row.findall("a:c", ns):
                ref = c.attrib.get("r", "")
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = 0
                for ch in col_letters:
                    col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
                col_idx -= 1
                ctype = c.attrib.get("t", "")
                v_el = c.find("a:v", ns)
                val = ""
                if v_el is not None and v_el.text is not None:
                    if ctype == "s":
                        val = shared[int(v_el.text)]
                    else:
                        val = v_el.text
                row_dict[col_idx] = val
            if row_dict:
                max_c = max(row_dict.keys())
                rows_data.append([row_dict.get(i, "") for i in range(max_c + 1)])
    max_cols = max(len(r) for r in rows_data)
    padded = [r + [""] * (max_cols - len(r)) for r in rows_data]
    return pd.DataFrame(padded)

def parse_height_cm(val: str) -> float:
    s = str(val).strip().replace("''", "").replace('"', "")
    m = re.match(r"^(\d+)\.(\d+)$", s)
    if m:
        return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 2)
    return np.nan

def load_and_clean_stdlib(path: str) -> pd.DataFrame:
    raw = read_xlsx_stdlib(path)
    header_idx = None
    for i in range(len(raw)):
        if str(raw.iloc[i, 0]).strip() == "Name":
            header_idx = i
            break
    df = raw.iloc[header_idx + 1:, :18].copy().reset_index(drop=True)
    df.columns = [str(c).strip() for c in raw.iloc[header_idx, :18].tolist()]
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
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df["name_norm"] = df["Name_raw"].str.lower().str.strip()
    df["group_id"] = df["name_norm"].apply(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:12])
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

def save_all(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_fig1_periodicity(df):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.18, 1.0]})

    ax = axes[0]
    lags = np.arange(1, 37)
    vars_to_plot = [
        ("target", "Risk label (period 3)", "#005493", "-", "o", 2.0),
        ("hbsag", "HBsAg (period 9)", "#D55E00", "--", "s", 1.5),
        ("tt_dose", "TT dose (period 24)", "#009E73", "-.", "^", 1.5),
        ("weight_kg", "Weight (period 27)", "#CC79A7", ":", "d", 1.5),
        ("vdrl", "VDRL (period 2 alternation)", "#7F7F7F", "-", None, 1.1),
    ]
    for col, label, color, ls, marker, lw in vars_to_plot:
        vals = df[col].values
        matches = [np.mean(vals[:-k] == vals[k:]) for k in lags]
        ax.plot(
            lags, matches, label=label, color=color, linestyle=ls,
            marker=marker, markersize=3.8 if marker else 0, linewidth=lw, alpha=0.92
        )

    for m3 in range(3, 37, 3):
        ax.axvline(m3, color="#005493", alpha=0.10, linewidth=0.9, zorder=0)

    ax.axhline(0.556, color="#005493", linestyle=":", linewidth=0.9, alpha=0.6, label="Label chance match (0.556)")
    ax.set_xlabel("Row-Order Lag $k$ (spreadsheet rows)")
    ax.set_ylabel("Exact Lag-$k$ Match Rate $M(X, k)$")
    ax.set_title("(a) Row-Order Periodicity Across Lags $k = 1, \\dots, 36$", fontweight="bold")
    ax.set_xlim(1, 36)
    ax.set_ylim(0.0, 1.06)
    ax.set_xticks(range(3, 37, 3))
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, ncol=1)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    ax2 = axes[1]
    name_grp = df.groupby("name_norm")["age"].agg(["count", "min", "max", "nunique"]).reset_index()
    rep = name_grp[name_grp["count"] > 1].sort_values(["count", "max"], ascending=[False, False]).head(15)
    rep = rep.iloc[::-1].reset_index(drop=True)

    y_pos = np.arange(len(rep))
    for idx, row in rep.iterrows():
        ages = sorted(df.loc[df["name_norm"] == row["name_norm"], "age"].unique())
        ax2.plot([row["min"], row["max"]], [idx, idx], color="#B22222", linewidth=2.2, zorder=2)
        ax2.scatter(ages, [idx] * len(ages), color="#005493", s=26, zorder=3, edgecolors="white", linewidth=0.5)

    ylabels = [f"Hash #{15-i} (n={int(r['count'])})" for i, r in rep.iterrows()]
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ylabels, fontsize=8)
    ax2.set_ylim(-1.7, 14.6)
    ax2.set_xlabel("Recorded Maternal Age Across Visits (years)")
    ax2.set_title("(b) Within-Name Age Discordance (Top 15 Repeated Names)", fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.45)
    ax2.annotate(
        "191 / 199 (96.0%) repeated names have discordant maternal ages",
        xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E7", edgecolor="#D55E00", alpha=0.95)
    )

    fig.tight_layout()
    save_all(fig, "periodicity_audit")

def make_fig2_baseline(df):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    effects = [
        ("Weight (kg)", 0.475, True),
        ("TT dose (1st/2nd/3rd)", 0.454, True),
        ("BMI (kg/m²)", 0.401, True),
        ("Systolic BP (mmHg)", 0.262, True),
        ("HBsAg (Positive)", 0.244, True),
        ("Diastolic BP (mmHg)", 0.151, True),
        ("Height (cm)", 0.019, False),
        ("Jaundice (any)", 0.014, False),
        ("Urine albumin (any)", 0.012, False),
        ("Gravida (1st/2nd/3rd)", 0.011, False),
        ("Anemia (any)", 0.004, False),
        ("Urine sugar (Yes)", 0.003, False),
        ("Age (years)", 0.003, False),
        ("Fetal heart rate (bpm)", 0.001, False),
        ("Gestational age (weeks)", 0.000, False),
        ("Fetal position (Abnormal)", 0.000, False),
        ("VDRL (Positive)", 0.000, False),
    ][::-1]

    labels = [x[0] for x in effects]
    vals = [x[1] for x in effects]
    colors = ["#005493" if x[2] else "#8C8C8C" for x in effects]

    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, height=0.68, alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.008, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center", fontsize=7.8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlim(0, 0.56)
    ax.set_xlabel("Univariable Effect Size ($|d|$ for continuous; Cramér's $V$ for categorical)")
    ax.set_title("(a) Standardized Group Differences by Period-3 Alignment", fontweight="bold")
    ax.axvline(0.10, color="#B22222", linestyle="--", linewidth=0.9, alpha=0.7)
    p1 = mpatches.Patch(color="#005493", label="Row-order period is multiple of 3 ($p < 0.05$)")
    p2 = mpatches.Patch(color="#8C8C8C", label="Clinical/demographic variable ($p \\geq 0.77$)")
    ax.legend(handles=[p1, p2], loc="lower right", frameon=True, facecolor="white")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax2 = axes[1]
    low_w = df.loc[df["target"] == 0, "weight_kg"].values
    high_w = df.loc[df["target"] == 1, "weight_kg"].values

    parts = ax2.violinplot([low_w, high_w], positions=[1, 2], widths=0.55, showmeans=False, showmedians=False, showextrema=False)
    for pc, col in zip(parts["bodies"], ["#4C9F70", "#D55E00"]):
        pc.set_facecolor(col)
        pc.set_edgecolor("black")
        pc.set_alpha(0.45)

    ax2.boxplot(
        [low_w, high_w], positions=[1, 2], widths=0.22, patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.1),
        medianprops=dict(color="#B22222", linewidth=1.8),
        whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0)
    )

    ax2.axhline(59.0, color="#005493", linestyle="--", linewidth=1.1, label="CART depth-2 split (59.0 kg)")
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Low-Risk Label\n($n = 332$, median 60 kg)", "High-Risk Label\n($n = 666$, median 55 kg)"])
    ax2.set_ylabel("Maternal Weight (kg)")
    ax2.set_title("(b) Maternal Weight Distribution by Register Risk Label", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    save_all(fig, "baseline_distributions")

def make_fig6_concordance():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.22, 1.0]})

    ax = axes[0]
    items = [
        ("HBsAg Positive (n=109)", 109, 109),
        ("Moderate Anemia (n=61)", 41, 61),
        ("VDRL Positive (n=499)", 333, 499),
        ("Abnormal Fetal Pos. (n=6)", 4, 6),
        ("Proteinuria Med/High (n=92)", 61, 92),
        ("Adolescent <20 y (n=164)", 107, 164),
        ("Any Danger Sign incl. VDRL (n=718)", 488, 718),
        ("Any Danger Sign excl. VDRL (n=371)", 275, 371),
    ][::-1]

    y = np.arange(len(items))
    props = []
    ci_low, ci_high = [], []
    z = 1.96
    for label, k, n in items:
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
        props.append(p * 100)
        ci_low.append(max(0, (center - half) * 100))
        ci_high.append(min(100, (center + half) * 100))

    props = np.array(props)
    err_low = np.maximum(0.0, props - np.array(ci_low))
    err_high = np.maximum(0.0, np.array(ci_high) - props)

    colors = ["#D55E00" if "HBsAg" in x[0] else "#005493" for x in items]
    ax.axvline(66.73, color="#B22222", linestyle="--", linewidth=1.3, label="Overall register high-risk prevalence (66.73%)")
    ax.barh(y, props, color=colors, height=0.62, alpha=0.85)
    ax.errorbar(props, y, xerr=[err_low, err_high], fmt="none", ecolor="black", elinewidth=1.1, capsize=3)

    for yi, p in zip(y, props):
        ax.text(28.0, yi, f"{p:.1f}%", va="center", ha="center", fontsize=8.0, fontweight="bold", color="white")

    ax.set_yticks(y)
    ax.set_yticklabels([x[0] for x in items], fontsize=8.3)
    ax.set_xlim(0, 106)
    ax.set_xlabel("Proportion Labeled High-Risk in Register (%, with 95% CI)")
    ax.set_title("(a) High-Risk Rate Across WHO Clinical Danger Signs", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=7.8)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax2 = axes[1]
    rules = [
        "WHO\n(no VDRL)",
        "WHO\n(+VDRL)",
        "CART\nDepth 2",
        "CART\nDepth 3",
        "CART\nDepth 4",
    ]
    acc = [51.20, 59.12, 83.37, 87.58, 90.78]
    sens = [41.29, 73.27, 92.04, 94.59, 95.20]
    spec = [71.08, 30.72, 65.96, 73.49, 81.93]

    x = np.arange(len(rules))
    w = 0.25
    b1 = ax2.bar(x - w, acc, width=w, label="Accuracy (%)", color="#005493", alpha=0.9)
    b2 = ax2.bar(x, sens, width=w, label="Sensitivity (%)", color="#009E73", alpha=0.85)
    b3 = ax2.bar(x + w, spec, width=w, label="Specificity (%)", color="#E69F00", alpha=0.85)

    for b in b1:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.3, f"{b.get_height():.1f}", ha="center", fontsize=7.2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(rules, fontsize=8.2)
    ax2.set_ylim(0, 114)
    ax2.set_ylabel("Agreement Metric (%)")
    ax2.set_title("(b) Clinical Checklist vs. Tree Rule Replication", fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", ncol=3, fontsize=7.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    save_all(fig, "danger_sign_concordance")

def make_fig9_subgroup_sensitivity():
    sub = pd.read_csv("/Users/kaanarik/Desktop/Maternal Health/manuscript_source/outputs/tables/subgroup_performance.csv")
    rob = pd.read_csv("/Users/kaanarik/Desktop/Maternal Health/manuscript_source/outputs/tables/robustness_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.05, 1.15]})

    ax = axes[0]
    strata = [
        "Age <20\n(n=164)",
        "Age >=20\n(n=834)",
        "Gravida\n1st (n=837)",
        "Gravida\n>=2nd (n=161)",
        "GA <=25w\n(n=336)",
        "GA 26-30w\n(n=363)",
        "GA >30w\n(n=299)",
    ]
    rf_sub = sub[sub["Model"] == "Random Forest"]["AUROC"].values
    lr_sub = sub[sub["Model"] == "Logistic Regression"]["AUROC"].values
    prev_sub = sub[sub["Model"] == "Random Forest"]["Prevalence"].values

    x = np.arange(len(strata))
    ax.plot(x, rf_sub, marker="o", linewidth=2.0, color="#005493", label="Random Forest (AUROC)")
    ax.plot(x, lr_sub, marker="s", linewidth=1.8, linestyle="--", color="#D55E00", label="L2 Logistic Regression (AUROC)")
    ax.plot(x, prev_sub, marker="^", linewidth=1.5, linestyle=":", color="#009E73", label="Subgroup High-Risk Prevalence")

    for xi, yi in zip(x, rf_sub):
        ax.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", fontsize=7.2, color="#005493")
    for xi, yi in zip(x, lr_sub):
        ax.text(xi, yi - 0.032, f"{yi:.3f}", ha="center", fontsize=7.2, color="#D55E00")

    ax.set_xticks(x)
    ax.set_xticklabels(strata, fontsize=7.5)
    ax.set_ylim(0.55, 1.05)
    ax.set_ylabel("Metric Value (Proportion / AUROC)")
    ax.set_title("(a) Subgroup Discrimination and Prevalence Invariance", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=7.8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    ax2 = axes[1]
    scen_labels = [
        "S1: Full\n(n=998)",
        "S2: No VDRL\n(n=998)",
        "S3: No VDRL,\nTT, HBsAg",
        "S4: Dedup\n(n=988)",
        "S5: Dedup+GA\n(n=964)",
    ]
    models = [
        ("XGBoost", "#005493", -0.24),
        ("Random Forest", "#009E73", 0.0),
        ("Logistic Regression", "#D55E00", 0.24),
    ]
    x2 = np.arange(len(scen_labels))
    w = 0.23
    for mname, col, offset in models:
        sub_m = rob[rob["Model"] == mname]
        means = sub_m["AUROC_mean"].values
        sds = sub_m["AUROC_sd"].values
        bars = ax2.bar(x2 + offset, means, width=w, label=mname, color=col, alpha=0.88, yerr=sds, capsize=2.5, error_kw=dict(lw=0.9))
        for b, val, sd in zip(bars, means, sds):
            ax2.text(b.get_x() + b.get_width()/2, val + sd + 0.010, f"{val:.2f}", ha="center", fontsize=6.9, rotation=0)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(scen_labels, fontsize=8.0)
    ax2.set_ylim(0.50, 1.08)
    ax2.set_ylabel("Out-of-Fold AUROC (mean ± SD, 3×5 GroupKFold)")
    ax2.set_title("(b) Sensitivity to Feature Exclusion and Deduplication", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.0)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    save_all(fig, "subgroup_sensitivity")

if __name__ == "__main__":
    df = load_and_clean_stdlib("/Users/kaanarik/Desktop/Maternal Health/Book2.xlsx")
    print(f"Loaded {len(df)} records.")
    make_fig1_periodicity(df)
    make_fig2_baseline(df)
    make_fig6_concordance()
    make_fig9_subgroup_sensitivity()
    print("Successfully generated 4 new figures.")
