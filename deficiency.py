# deficiency.py
#
# Real deficiency calculation using FFQ weekly-frequency data.
# Fixes two issues found in the earlier draft:
#   1. Nutrients with ZERO estimated intake were silently dropped from the
#      result instead of being reported as fully deficient.
#   2. Nutrient names were derived from data-property strings via chained
#      .replace() calls, which is fragile if a property is renamed later.
#      Replaced with an explicit, readable mapping.

from ontology_loader import FOOD_NUTRIENTS, RDA_VALUES, TREATMENT_MAP, SYMPTOM_TO_NUTRIENTS

# Explicit map: data property name -> nutrient individual name in the ontology
PROPERTY_TO_NUTRIENT = {
    "hasProteinContent_g_per_100g": "Protein",
    "hasIronContent_mg_per_100g": "Iron",
    "hasVitaminC_mg_per_100g": "VitaminC",
    "hasVitaminD_mcg_per_100g": "VitaminD",
    "hasVitaminA_mcg_per_100g": "VitaminA",
    "hasCalcium_mg_per_100g": "Calcium",
    "hasZincContent_mg_per_100g": "Zinc",
    "hasCarbohydrate_g_per_100g": "Carbohydrate",
    "hasFatContent_g_per_100g": "Fat",
}


def calculate_deficiency_percentage(user_foods_with_frequency, gender="male", reported_symptoms=None):
    """
    user_foods_with_frequency: dict, e.g. {"Ilish": 3.5, "Bhat": 7.0}
    (value = average number of days per week that food is eaten, already
    portion-adjusted -- see frequency_to_days() below)
    gender: "male" or "female" -- selects which RDA column to use
    reported_symptoms: optional list of ontology symptom names (e.g.
    ["Fatigue", "HairLoss"]) the person self-reported. When given, each
    result also gets a "symptom_supported" flag: True if at least one of
    the reported symptoms is one this nutrient's deficiency is known to
    cause (per SYMPTOM_TO_NUTRIENTS) -- i.e. the self-report agrees with
    the intake-based estimate. This does NOT change deficit_percentage or
    status; it's corroborating evidence only, since intake data is the
    actual measurement and symptoms are noisy/self-reported.
    Returns a dict for EVERY nutrient tracked in RDA_VALUES, even ones
    the person didn't eat at all (those correctly show as ~100% deficient).
    """
    reported_symptoms = reported_symptoms or []
    nutrients_backed_by_symptoms = set()
    for symptom in reported_symptoms:
        nutrients_backed_by_symptoms.update(SYMPTOM_TO_NUTRIENTS.get(symptom, []))

    weekly_intake = {nutrient: 0.0 for nutrient in RDA_VALUES}

    for food_name, days_per_week in user_foods_with_frequency.items():
        food_data = FOOD_NUTRIENTS.get(food_name)
        if not food_data:
            continue  # unknown food name -- consider logging this in production

        serving_size_g = food_data.get("hasServingSize_g", 100)

        for prop, nutrient_name in PROPERTY_TO_NUTRIENT.items():
            if prop not in food_data or nutrient_name not in weekly_intake:
                continue
            per_100g_value = food_data[prop]
            weekly_contribution = days_per_week * (serving_size_g / 100) * per_100g_value
            weekly_intake[nutrient_name] += weekly_contribution

    results = {}
    for nutrient, rda_data in RDA_VALUES.items():
        rda_daily = rda_data.get(gender) or rda_data.get("male") or 0
        if rda_daily == 0:
            continue  # nutrient has no RDA on record (e.g. Fat) -- can't score it
        rda_weekly = rda_daily * 7
        estimated_intake = weekly_intake[nutrient]  # always present now, defaults to 0.0

        deficit_pct = ((rda_weekly - estimated_intake) / rda_weekly) * 100
        results[nutrient] = {
            "estimated_weekly_intake": round(estimated_intake, 2),
            "rda_weekly": round(rda_weekly, 2),
            "deficit_percentage": round(max(0, deficit_pct), 1),
            "status": "Deficient" if deficit_pct > 20 else "Adequate",
            "symptom_supported": nutrient in nutrients_backed_by_symptoms,
        }

    return results


def frequency_to_days(freq: str, portion: str = "স্বাভাবিক পরিমাণ") -> float:
    """
    Convert a categorical FFQ frequency response (+ optional portion size)
    into a portion-adjusted average number of days per week.

    ⚠️ FIX: the label set below was previously the OLD "০ দিন" / "১-২ দিন"
    style, which no longer matches what the frontend/survey send. Every
    call would fall through to the .get(freq, 0.0) default and silently
    report zero intake for every food -- exactly the same class of bug the
    header comment in data_importer.py already fixed once for the CSV
    importer. Labels here are now copied verbatim from FREQ_MAP /
    PORTION_MAP in data_importer.py so all three entry points (this
    function, data_importer.py, main.py) agree.
    """
    freq_map = {
        "একদমই না": 0.0,
        "সপ্তাহে ১–২ বার": 1.5,
        "সপ্তাহে ৩–৪ বার": 3.5,
        "সপ্তাহে ৫–৬ বার": 5.5,
        "প্রতিদিন": 7.0,
    }
    portion_map = {
        "খাই না": 0.0,
        "ছোট (স্বাভাবিকের চেয়ে কম)": 0.5,
        "স্বাভাবিক পরিমাণ": 1.0,
        "বড় (স্বাভাবিকের চেয়ে বেশি)": 1.5,
    }
    freq_days = freq_map.get(freq, 0.0)
    portion_mult = portion_map.get(portion, 1.0)
    return round(freq_days * portion_mult, 2)


if __name__ == "__main__":
    # Same test case as before: a student eating only Ilish, Bhat, Dim all week
    test_data = {"Ilish": 3.5, "Bhat": 7.0, "Dim": 4.0}
    result = calculate_deficiency_percentage(test_data, gender="male", reported_symptoms=["Fatigue"])
    for nutrient, info in result.items():
        print(f"{nutrient:15s} -> {info}")

    # Sanity check the fixed frequency_to_days() against the new label set
    print("\nfrequency_to_days smoke test:")
    print("  'প্রতিদিন' + 'স্বাভাবিক পরিমাণ' ->", frequency_to_days("প্রতিদিন", "স্বাভাবিক পরিমাণ"), "(expect 7.0)")
    print("  'সপ্তাহে ৩–৪ বার' + 'বড় (স্বাভাবিকের চেয়ে বেশি)' ->",
          frequency_to_days("সপ্তাহে ৩–৪ বার", "বড় (স্বাভাবিকের চেয়ে বেশি)"), "(expect 5.25)")
    print("  'একদমই না' ->", frequency_to_days("একদমই না"), "(expect 0.0)")