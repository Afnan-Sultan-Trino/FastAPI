# deficiency.py
#
# Real deficiency calculation using FFQ weekly-frequency data.
# Fixes two issues found in the earlier draft:
#   1. Nutrients with ZERO estimated intake were silently dropped from the
#      result instead of being reported as fully deficient.
#   2. Nutrient names were derived from data-property strings via chained
#      .replace() calls, which is fragile if a property is renamed later.
#      Replaced with an explicit, readable mapping.

from ontology_loader import FOOD_NUTRIENTS, RDA_VALUES, TREATMENT_MAP

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


def calculate_deficiency_percentage(user_foods_with_frequency, gender="male"):
    """
    user_foods_with_frequency: dict, e.g. {"Ilish": 3.5, "Bhat": 7.0}
    (value = average number of days per week that food is eaten)
    gender: "male" or "female" -- selects which RDA column to use
    Returns a dict for EVERY nutrient tracked in RDA_VALUES, even ones
    the person didn't eat at all (those correctly show as ~100% deficient).
    """
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
        }

    return results


def frequency_to_days(freq: str) -> float:
    """
    Convert a categorical FFQ response to an average number of days per week.
    Uses the midpoint of each range as the numeric estimate.
    """
    mapping = {
        "০ দিন": 0.0,
        "১-২ দিন": 1.5,
        "৩-৪ দিন": 3.5,
        "৫-৬ দিন": 5.5,
        "প্রতিদিন": 7.0,
    }
    return mapping.get(freq, 0.0)


if __name__ == "__main__":
    # Same test case as before: a student eating only Ilish, Bhat, Dim all week
    test_data = {"Ilish": 3.5, "Bhat": 7.0, "Dim": 4.0}
    result = calculate_deficiency_percentage(test_data, gender="male")
    for nutrient, info in result.items():
        print(f"{nutrient:15s} -> {info}")