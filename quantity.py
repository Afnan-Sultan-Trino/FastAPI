# quantity.py
#
# Converts a weekly nutrient deficit into a friendly "eat X per day" quantity
# suggestion, using the food's per-100g nutrient content and (if available)
# a natural countable unit (e.g. "egg", "bowl") from the ontology.

import math
from ontology_loader import FOOD_NUTRIENTS, NATURAL_UNIT_LABELS
from deficiency import PROPERTY_TO_NUTRIENT

# Reverse lookup: nutrient name -> data property name
NUTRIENT_TO_PROPERTY = {v: k for k, v in PROPERTY_TO_NUTRIENT.items()}


def get_quantity_recommendation(nutrient: str, food_name: str, weekly_deficit: float) -> dict | None:
    """
    nutrient: e.g. "Iron"
    food_name: ontology individual name, e.g. "Dim"
    weekly_deficit: the weekly gap for that nutrient (same units as the
                    nutrient's RDA, e.g. mg for Iron, mcg for VitaminD)

    Returns a dict describing how much of this food to eat PER DAY to close
    the weekly gap over 7 days, or None if the food doesn't supply this
    nutrient or the deficit is zero/negative.
    """
    if weekly_deficit <= 0:
        return None

    food_data = FOOD_NUTRIENTS.get(food_name)
    if not food_data:
        return None

    prop = NUTRIENT_TO_PROPERTY.get(nutrient)
    if not prop or prop not in food_data:
        return None

    per_100g_value = food_data[prop]
    if per_100g_value <= 0:
        return None

    serving_size_g = food_data.get("hasServingSize_g", 100.0)
    # how much of the nutrient is in ONE natural serving/unit of this food
    content_per_unit = per_100g_value * (serving_size_g / 100.0)

    units_needed_for_week = weekly_deficit / content_per_unit
    units_per_day = units_needed_for_week / 7.0

    # Round to the nearest half-unit for countable foods (eggs, pieces),
    # since "2.5 pieces" reads oddly for some but is fine for others --
    # keeping it simple and consistent here.
    units_per_day_rounded = max(0.5, round(units_per_day * 2) / 2)

    label = NATURAL_UNIT_LABELS.get(food_name)

    # SANITY CAP: no single food should ever be prescribed at an unrealistic
    # quantity (e.g. "24 eggs a day") just because it's a low-density source
    # for this nutrient relative to the size of the gap. Above this cap, we
    # switch to a qualitative message instead of a precise (silly) number --
    # this is why deficiency recommendations should always be spread across
    # multiple foods (see the top-3 recommendation list) rather than relying
    # on one food to close the entire gap by itself.
    MAX_REASONABLE_UNITS_PER_DAY = 3.0
    if units_per_day_rounded > MAX_REASONABLE_UNITS_PER_DAY:
        quantity_text = f"নিয়মিত খাদ্যতালিকায় রাখুন ({food_name} একা পুরো ঘাটতি পূরণ করতে পারবে না, একাধিক খাবার একসাথে প্রয়োজন)"
        return {
            "food": food_name,
            "per_day": None,
            "unit_label": label or "গ্রাম",
            "text": quantity_text,
            "note": "capped: unrealistic single-food quantity",
        }

    if label:
        quantity_text = f"{units_per_day_rounded:g} {label}"
    else:
        # No natural unit defined for this food -- fall back to grams.
        grams_per_day = units_per_day_rounded * serving_size_g
        quantity_text = f"~{grams_per_day:.0f} গ্রাম ({food_name})"

    return {
        "food": food_name,
        "per_day": units_per_day_rounded,
        "unit_label": label or "গ্রাম",
        "text": quantity_text,
    }


if __name__ == "__main__":
    # Example: someone with a 100 mg weekly iron deficit
    for food in ["Dim", "LalShak", "MurgirMangsho", "Bhat"]:
        result = get_quantity_recommendation("Iron", food, weekly_deficit=100.0)
        print(f"{food:15s} -> {result}")