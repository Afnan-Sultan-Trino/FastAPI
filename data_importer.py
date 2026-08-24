# data_importer.py
#
# Cleans and imports the Google Forms CSV export into a list of per-student
# records ready for deficiency.py / ml_pipeline_dummy.py.
#
# Fixes vs the earlier draft:
#   1. Accepted a DataFrame directly at the test call site but the function
#      only accepted a csv path -- crashed immediately (TypeError).
#   2. Guessed grid-question column names as "{food} [সপ্তাহে কতদিন?]", but
#      Google Forms actually exports grid columns as
#      "{full question text} [{row label}]" -- the guessed pattern never
#      matches, so every food would silently show frequency=0.
#   3. Portion-size columns were never actually extracted (dead code with
#      a TODO comment) -- portion data was silently dropped entirely.
#   4. Checkbox (multi-select symptom) questions export as ONE column with
#      a comma-separated string of selected options, not one column per
#      symptom -- the per-symptom-column lookup would never find anything.
#   5. Every field used `.values[0]`, so only the FIRST survey response was
#      ever processed -- all other respondents were silently discarded.
#   6. NEW: height, weight, and the 4 lifestyle fields (sleep, screen time,
#      sun exposure, living arrangement) were never extracted at all -- the
#      importer silently dropped them even though they're needed for BMI
#      and for the budget-aware / lifestyle-aware recommendation logic in
#      main.py. Added below.
#   7. NEW: the raw survey's living-arrangement option is "মেস/হোস্টেলে",
#      but main.py / the frontend / the cleaned xlsx all use "Mess/Hostel"
#      (translated at some point during earlier cleaning). Normalized here
#      so downstream code sees one consistent value instead of two strings
#      for the same thing.

import pandas as pd
import numpy as np

# ---- 1. Frequency mapping ---------------------------------------------------
FREQ_MAP = {
    "একদমই না": 0.0,
    "সপ্তাহে ১–২ বার": 1.5,
    "সপ্তাহে ৩–৪ বার": 3.5,
    "সপ্তাহে ৫–৬ বার": 5.5,
    "প্রতিদিন": 7.0,
}

# ---- 2. Portion-size multiplier mapping -------------------------------------
PORTION_MAP = {
    "খাই না": 0.0,
    "ছোট (স্বাভাবিকের চেয়ে কম)": 0.5,
    "স্বাভাবিক পরিমাণ": 1.0,
    "বড় (স্বাভাবিকের চেয়ে বেশি)": 1.5,
}

# ---- 2a. Living-arrangement normalization -----------------------------------
# Raw survey text -> canonical value used by main.py / VALID_LIVING_ARRANGEMENTS.
LIVING_ARRANGEMENT_NORMALIZE = {
    "মেস/হোস্টেলে": "Mess/Hostel",
    "পরিবারের সাথে (বাসায়)": "পরিবারের সাথে (বাসায়)",
    "বন্ধুদের সাথে শেয়ার্ড বাসায়": "বন্ধুদের সাথে শেয়ার্ড বাসায়",
}

# ---- 2b. Gender / religion normalization ------------------------------------
# Raw survey answers are in Bangla; main.py / RESTRICTED_MAP (religion) expect
# the English keys used elsewhere in the codebase. Normalized here so this is
# done in exactly one place instead of ad hoc at each call site.
GENDER_NORMALIZE = {"পুরুষ": "male", "নারী": "female", "অন্যান্য": "other"}
RELIGION_NORMALIZE = {
    "ইসলাম": "Islam", "হিন্দু": "Hinduism", "বৌদ্ধ": "Buddhism",
    "খ্রিস্টান": "Christianity", "অন্যান্য": "Other",
}


# ---- 3. The 11 symptom labels, exactly as they appear inside the single
#         comma-separated checkbox-question cell -----------------------------
SYMPTOM_LABELS = [
    "সহজেই ক্লান্ত হয়ে পড়া / সবসময় দুর্বল লাগা (Fatigue)",
    "অতিরিক্ত চুল পড়া (Hair Loss)",
    "মাড়ি দিয়ে রক্ত পড়া (Gum Bleeding)",
    "মাড়ি ফোলা/ব্যথা/দাঁতের গোড়া দুর্বল হওয়া (Periodontitis)",
    "রাতে/আবছা আলোতে ঠিকমতো দেখতে সমস্যা হওয়া (Night Blindness)",
    "ত্বক শুষ্ক ও খসখসে হয়ে যাওয়া (Dry/Flaky Skin)",
    "হাড় বা জয়েন্টে ব্যথা (Bone/Joint Pain)",
    "ঘন ঘন ঠান্ডা-জ্বর/অসুস্থ হয়ে পড়া (Frequent Illness)",
    "পড়াশোনায় মনোযোগ ধরে রাখতে কষ্ট হওয়া (Poor Concentration)",
    "চোখের নিচের পাতার ভেতরের অংশ ফ্যাকাশে মনে হওয়া (Pale Conjunctiva)",
    "নখ ভঙ্গুর হয়ে যাওয়া বা চামচের মতো বেঁকে যাওয়া (Koilonychia)",
]

# Bangla label -> ontology symptom individual name (matches SYMPTOM_MAP keys)
SYMPTOM_LABEL_TO_ONTOLOGY = {
    "সহজেই ক্লান্ত হয়ে পড়া / সবসময় দুর্বল লাগা (Fatigue)": "Fatigue",
    "অতিরিক্ত চুল পড়া (Hair Loss)": "HairLoss",
    "মাড়ি দিয়ে রক্ত পড়া (Gum Bleeding)": "GumBleeding",
    "মাড়ি ফোলা/ব্যথা/দাঁতের গোড়া দুর্বল হওয়া (Periodontitis)": "Periodontitis",
    "রাতে/আবছা আলোতে ঠিকমতো দেখতে সমস্যা হওয়া (Night Blindness)": "NightBlindness",
    "ত্বক শুষ্ক ও খসখসে হয়ে যাওয়া (Dry/Flaky Skin)": "DryFlakySkin",
    "হাড় বা জয়েন্টে ব্যথা (Bone/Joint Pain)": "BoneJointPain",
    "ঘন ঘন ঠান্ডা-জ্বর/অসুস্থ হয়ে পড়া (Frequent Illness)": "FrequentIllness",
    "পড়াশোনায় মনোযোগ ধরে রাখতে কষ্ট হওয়া (Poor Concentration)": "PoorConcentration",
    "চোখের নিচের পাতার ভেতরের অংশ ফ্যাকাশে মনে হওয়া (Pale Conjunctiva)": "PaleConjunctiva",
    "নখ ভঙ্গুর হয়ে যাওয়া বা চামচের মতো বেঁকে যাওয়া (Koilonychia)": "Koilonychia",
}

# ---- 4. Bangla survey label -> ontology individual name --------------------
BANGLA_TO_ONTOLOGY = {
    "ইলিশ মাছ": "Ilish", "রুই মাছ": "Rui", "গরুর মাংস": "GorurMangsho", "মুরগির মাংস": "MurgirMangsho",
    "ডিম": "Dim", "দুধ": "Dudh", "দই": "Doi", "দুধ চা": "DudhCha",
    "লাল শাক": "LalShak", "পালং শাক": "PalongShak",
    "বেগুন": "Begun", "আলু": "Alu", "গাজর": "Gajar", "মিষ্টি কুমড়া": "MishtiKumra",
    "আম": "Aam", "কমলালেবু": "Kamalalebu", "আমলকী": "Amloki", "পেঁপে": "Pepe",
    "ছোলা": "Chola", "মসুর ডাল": "MasoorDal", "আটা/রুটি": "Atta", "ভাত": "Bhat",
}
FOOD_LABELS = list(BANGLA_TO_ONTOLOGY.keys())


def _find_grid_column(df_columns, food_label):
    """
    Google Forms exports grid questions as '{question text} [{row label}]'.
    Rather than guess the exact question text (which can drift if you edit
    the form later), search for any column ending in '[{food_label}]' --
    robust to question-text changes as long as the food label itself and
    the trailing bracket format stay the same.
    """
    suffix = f"[{food_label}]"
    matches = [c for c in df_columns if c.strip().endswith(suffix)]
    return matches[0] if matches else None


def _split_checkbox_cell(cell_value):
    """Google Forms checkbox cells are a single comma-separated string."""
    if pd.isna(cell_value) or not str(cell_value).strip():
        return []
    return [item.strip() for item in str(cell_value).split(",")]


def import_survey_dataframe(df: pd.DataFrame) -> list[dict]:
    """
    Takes an already-loaded DataFrame (one row per respondent) and returns
    a list of cleaned per-student record dicts. Processes ALL rows, not
    just the first.
    """
    # Locate the two grid questions and the one checkbox question by
    # matching on any food/symptom label found in the column names, since
    # exact question wording may change over time.
    freq_col_map = {food: _find_grid_column(df.columns, food) for food in FOOD_LABELS}
    portion_col_map = {food: _find_grid_column(df.columns, food) for food in FOOD_LABELS}
    # NOTE: since both grid questions share the SAME row labels ([ইলিশ মাছ] etc.),
    # _find_grid_column alone can't tell frequency vs portion apart if a food's
    # bracketed suffix appears in two columns. We disambiguate by requiring the
    # frequency column's own known response values (FREQ_MAP keys) to appear in
    # its data, and likewise for portion -- done per-row below as a safety check.

    freq_candidates = {food: [c for c in df.columns if c.strip().endswith(f"[{food}]")] for food in FOOD_LABELS}

    symptom_col = None
    for col in df.columns:
        if any(label in col for label in ["Fatigue", "উপসর্গ"]):
            symptom_col = col
            break

    # NEW: lifestyle + height/weight columns, found by keyword (question
    # wording may drift, but these keywords are distinctive enough).
    def _find_col(keyword):
        for col in df.columns:
            if keyword in col:
                return col
        return None

    height_col = _find_col("উচ্চতা")
    weight_col = _find_col("ওজন")
    sleep_col = _find_col("কত ঘণ্টা ঘুমান")
    screen_col = _find_col("মোবাইল/স্ক্রিনে")
    sun_col = _find_col("সূর্যালোকে")
    living_col = _find_col("কোথায় থাকেন")

    records = []
    for idx, row in df.iterrows():
        name_val = row.get("আপনার নাম (ঐচ্ছিক)")
        gender_raw = row.get("আপনার লিঙ্গ")
        religion_raw = row.get("আপনার ধর্ম")
        living_raw = row.get(living_col) if living_col else None

        record = {
            "row_index": idx,
            "name": "" if pd.isna(name_val) else str(name_val).strip(),
            "age": pd.to_numeric(row.get("আপনার বয়স (বছরে)"), errors="coerce"),
            "gender": GENDER_NORMALIZE.get(gender_raw, gender_raw),
            "religion": RELIGION_NORMALIZE.get(religion_raw, religion_raw),
            "height_cm": pd.to_numeric(row.get(height_col), errors="coerce") if height_col else np.nan,
            "weight_kg": pd.to_numeric(row.get(weight_col), errors="coerce") if weight_col else np.nan,
            "sleep_category": row.get(sleep_col) if sleep_col else None,
            "screen_before_sleep_category": row.get(screen_col) if screen_col else None,
            "sun_exposure_category": row.get(sun_col) if sun_col else None,
            "living_arrangement": LIVING_ARRANGEMENT_NORMALIZE.get(living_raw, living_raw),
        }

        # --- FFQ: frequency (days/week) x portion multiplier, per food ---
        food_intake = {}
        for food_label, ontology_name in BANGLA_TO_ONTOLOGY.items():
            candidates = freq_candidates.get(food_label, [])
            freq_days, portion_mult = 0.0, 1.0
            for col in candidates:
                val = row.get(col)
                if val in FREQ_MAP:
                    freq_days = FREQ_MAP[val]
                elif val in PORTION_MAP:
                    portion_mult = PORTION_MAP[val]
            food_intake[ontology_name] = round(freq_days * portion_mult, 2)
        record["food_intake_days_per_week"] = food_intake

        # --- Symptoms: single comma-separated checkbox cell ---
        reported_labels = _split_checkbox_cell(row.get(symptom_col)) if symptom_col else []
        reported_symptoms = [
            SYMPTOM_LABEL_TO_ONTOLOGY[label]
            for label in reported_labels
            if label in SYMPTOM_LABEL_TO_ONTOLOGY
        ]
        record["symptoms"] = reported_symptoms

        records.append(record)

    return records


def import_survey_csv(csv_path: str) -> list[dict]:
    """Reads the CSV file from disk, then delegates to import_survey_dataframe."""
    df = pd.read_csv(csv_path)
    return import_survey_dataframe(df)


if __name__ == "__main__":
    # Test against a CSV shaped the way Google Forms actually exports it
    # (see realistic_survey_sample.csv construction in the conversation),
    # with TWO respondents to prove batch processing works.
    records = import_survey_csv("realistic_survey_sample.csv")
    print(f"Processed {len(records)} respondents (should be 2, not 1):\n")
    for r in records:
        print(f"Row {r['row_index']}: name={r['name']!r}, age={r['age']}, gender={r['gender']}, religion={r['religion']}")
        print(f"  Ilish intake (days/week, portion-adjusted): {r['food_intake_days_per_week'].get('Ilish')}")
        print(f"  Bhat intake (days/week, portion-adjusted):  {r['food_intake_days_per_week'].get('Bhat')}")
        print(f"  Symptoms: {r['symptoms']}")
        print()
