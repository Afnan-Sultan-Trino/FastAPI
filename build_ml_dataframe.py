# build_ml_dataframe.py
#
# Glue between data_importer.py / deficiency.py (which produce nested,
# per-student dicts) and ml_validation_pipeline.py (which expects one flat
# DataFrame with specific column names: Fatigue, HairLoss, ..., SleepHours,
# ScreenTime, BMI, Living, Diet, {Nutrient}Def, {Nutrient}Risk).
#
# WITHOUT this script, `pd.DataFrame(records)` (as DeepSeek's plan proposed)
# produces a DataFrame with columns like 'food_intake_days_per_week' and
# 'symptoms' -- neither of which run_ml_pipeline() can use -- and it crashes
# with a KeyError the moment it tries to select feature_cols.
#
# READ BEFORE TRUSTING RESULTS -- two things this script cannot get from
# the survey as currently designed:
#
#   1. DIET (vegetarian / non-veg): known_groups_validity() needs a 'Diet'
#      column with exactly 2 groups, but the survey never asked this
#      directly. build_diet_proxy() infers it from whether the respondent
#      reported ANY weekly intake of fish/meat/egg. This is a PROXY, not a
#      self-reported variable -- say so explicitly if you use Diet-based
#      validation in your Results section, or add a real diet question to
#      future survey rounds.
#
#   2. GENDER = "অন্যান্য" (Other): calculate_deficiency_percentage() only
#      accepts gender="male"/"female" because the RDA tables are binary.
#      Rows where gender is "অন্যান্য" are NOT silently folded into
#      "female". Their deficiency % is instead computed against the
#      AVERAGE of the male and female RDA for each nutrient, and the row
#      is flagged in the `gender_note` column so you can inspect or
#      exclude them explicitly rather than have them silently miscounted.

import numpy as np
import pandas as pd

from data_importer import import_survey_csv, SYMPTOM_LABEL_TO_ONTOLOGY
from deficiency import calculate_deficiency_percentage, RDA_VALUES

# ---------------------------------------------------------------------------
# 1. Survey Bangla text -> short ML feature labels
#    (Keys here MUST match the exact category strings Google Forms exports --
#     verified against survey_data.csv column values, not guessed.)
# ---------------------------------------------------------------------------

GENDER_MAP = {"পুরুষ": "male", "নারী": "female"}  # "অন্যান্য" handled separately, not defaulted

SLEEP_MAP = {
    "৪ ঘণ্টার কম": "<4",
    "৪–৬ ঘণ্টা": "4-6",
    "৬–৮ ঘণ্টা": "6-8",
    "৮ ঘণ্টার বেশি": "8+",
}

SCREEN_MAP = {
    "৩০ মিনিটের কম": "<30min",
    "৩০ মিনিট–১ ঘণ্টা": "30-60min",
    "১–২ ঘণ্টা": "1-2hr",
    "২ ঘণ্টার বেশি": "2hr+",
}

LIVING_MAP = {
    "মেস/হোস্টেলে": "Mess/Hostel",
    "পরিবারের সাথে (বাসায়)": "Home",
    "বন্ধুদের সাথে শেয়ার্ড বাসায়": "Shared",
}

# The 11 binary symptom flags run_ml_pipeline() expects (ontology names)
SYMPTOM_FLAG_COLUMNS = sorted(set(SYMPTOM_LABEL_TO_ONTOLOGY.values()))

# Nutrients scored by both deficiency.py's RDA table and ml_validation_pipeline.py
NUTRIENTS = ["Iron", "Protein", "VitaminC", "VitaminD", "VitaminA", "Calcium", "Zinc"]

# Ontology food names counted as "non-veg" for the Diet proxy (see caveat #1 above)
NONVEG_FOODS = {"Ilish", "Rui", "GorurMangsho", "MurgirMangsho", "Dim"}

# Raw-CSV column POSITIONS for fields data_importer.py doesn't extract yet.
# Positions verified directly against survey_data.csv -- update if the form changes.
RAW_COL_INDEX = {
    "height": 10,  # 'আপনার উচ্চতা (সে.মি.)'
    "weight": 11,  # 'আপনার ওজন (কেজি)'
    "sleep": 7,    # 'গড়ে প্রতিদিন কত ঘণ্টা ঘুমান?'
    "screen": 8,   # 'ঘুমানোর আগে গড়ে কত সময় মোবাইল/স্ক্রিনে কাটান?'
    "living": 60,  # '  আপনি কোথায় থাকেন?  '
}


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

def _average_rda(nutrient):
    """For 'অন্যান্য' gender rows: average of male & female weekly RDA."""
    rda_row = RDA_VALUES.get(nutrient, {})
    vals = [v for k, v in rda_row.items() if k in ("male", "female") and v]
    return (sum(vals) / len(vals)) if vals else rda_row.get("male", 0)


def build_diet_proxy(food_intake_days: dict) -> str:
    """PROXY ONLY (see module docstring, caveat #1)."""
    nonveg_days = sum(food_intake_days.get(f, 0.0) for f in NONVEG_FOODS)
    return "Non-veg" if nonveg_days > 0 else "Vegetarian"


def _recompute_with_averaged_rda(deficiency_result: dict) -> dict:
    """Re-scores every nutrient's deficit % against the averaged male/female
    RDA, for respondents whose gender doesn't fit the binary RDA table."""
    out = {}
    for nutrient in NUTRIENTS:
        avg_rda_weekly = _average_rda(nutrient) * 7
        intake = deficiency_result.get(nutrient, {}).get("estimated_weekly_intake", 0.0)
        if avg_rda_weekly > 0:
            deficit_pct = max(0.0, ((avg_rda_weekly - intake) / avg_rda_weekly) * 100)
        else:
            deficit_pct = 0.0
        out[nutrient] = {
            "estimated_weekly_intake": round(intake, 2),
            "rda_weekly": round(avg_rda_weekly, 2),
            "deficit_percentage": round(deficit_pct, 1),
            "status": "Deficient" if deficit_pct > 20 else "Adequate",
        }
    return out


# ---------------------------------------------------------------------------
# 3. Main builder
# ---------------------------------------------------------------------------

def build_ml_dataframe(csv_path: str) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)
    records = import_survey_csv(csv_path)

    rows = []
    for record in records:
        idx = record["row_index"]
        raw_row = raw_df.iloc[idx]

        height_cm = pd.to_numeric(raw_row.iloc[RAW_COL_INDEX["height"]], errors="coerce")
        weight_kg = pd.to_numeric(raw_row.iloc[RAW_COL_INDEX["weight"]], errors="coerce")
        sleep_raw = raw_row.iloc[RAW_COL_INDEX["sleep"]]
        screen_raw = raw_row.iloc[RAW_COL_INDEX["screen"]]
        living_raw = raw_row.iloc[RAW_COL_INDEX["living"]]

        # ---- gender handling: explicit, never silently defaulted ----
        gender_raw = record["gender"]
        gender_note = ""
        if gender_raw in GENDER_MAP:
            gender_for_rda = GENDER_MAP[gender_raw]
        else:
            gender_for_rda = "male"  # placeholder only; overwritten below
            gender_note = f"non-binary gender '{gender_raw}' -- used averaged male/female RDA"

        deficiency_result = calculate_deficiency_percentage(
            record["food_intake_days_per_week"], gender=gender_for_rda
        )
        if gender_note:
            deficiency_result = _recompute_with_averaged_rda(deficiency_result)

        # ---- symptom one-hot flags ----
        symptom_flags = {s: int(s in record["symptoms"]) for s in SYMPTOM_FLAG_COLUMNS}

        # ---- BMI ----
        bmi = np.nan
        if pd.notna(height_cm) and pd.notna(weight_kg) and height_cm > 0:
            bmi = weight_kg / ((height_cm / 100) ** 2)

        row = {
            "row_index": idx,
            "age": record["age"],
            "gender_raw": gender_raw,
            "gender_note": gender_note,
            "religion": record["religion"],
            "BMI": round(bmi, 1) if pd.notna(bmi) else np.nan,
            "SleepHours": SLEEP_MAP.get(sleep_raw, np.nan),
            "ScreenTime": SCREEN_MAP.get(screen_raw, np.nan),
            "Living": LIVING_MAP.get(living_raw, np.nan),
            "Diet": build_diet_proxy(record["food_intake_days_per_week"]),
            **symptom_flags,
        }

        for nutrient in NUTRIENTS:
            info = deficiency_result.get(nutrient, {})
            row[f"{nutrient}Risk"] = info.get("deficit_percentage", np.nan)
            row[f"{nutrient}Def"] = 1 if info.get("status") == "Deficient" else 0

        rows.append(row)

    return pd.DataFrame(rows).set_index("row_index")


if __name__ == "__main__":
    df = build_ml_dataframe("survey_data.csv")
    print(f"Built ML-ready DataFrame: {df.shape}")
    print(df[["age", "gender_raw", "BMI", "SleepHours", "ScreenTime", "Living", "Diet",
              "IronRisk", "IronDef"]].head(10).to_string())

    n_flagged = (df["gender_note"] != "").sum()
    if n_flagged:
        print(f"\n[gender flag] {n_flagged} respondent(s) used averaged RDA -- inspect gender_note column:")
        print(df.loc[df["gender_note"] != "", ["gender_raw", "gender_note"]].to_string())

    missing = df[["BMI", "SleepHours", "ScreenTime", "Living"]].isna().sum()
    print(f"\nMissing values per column:\n{missing}")

    print("\nClass balance per nutrient (Def=1 means Deficient):")
    for n in NUTRIENTS:
        print(f"  {n:10s}: {df[f'{n}Def'].value_counts().to_dict()}")
