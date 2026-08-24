# main.py (Final - CORS + Validation + Local Static Images + Real Deficiency Calculation)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from typing import Dict, List, Literal
import pandas as pd
import joblib
import os

from ontology_loader import FOOD_NUTRIENTS, RDA_VALUES, SYMPTOM_MAP, TREATMENT_MAP, ENHANCEMENT_MAP, RESTRICTED_MAP
from deficiency import calculate_deficiency_percentage, frequency_to_days
from quantity import get_quantity_recommendation
from bmi_category import get_bmi_category

# ---------- ML+XAI deployment bundle (see train_ml_models.py Step 6) ----------
# Contains: 'feature_cols' (exact schema a request must be encoded into),
# 'models' (one RandomForestClassifier per viable nutrient, trained on the
# FULL 110-respondent dataset), and 'importance' (static top-5 SHAP features
# per nutrient, computed offline -- NOT recomputed per request).
#
# ⚠️ This is intentionally a STATIC/offline-trained bundle, not retrained on
# live traffic. If the underlying ontology or survey data changes materially,
# re-run train_ml_models.py and replace this file -- predictions will
# silently go stale otherwise (the bundle has no versioning/staleness check).
ML_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "ml_deployment_bundle.joblib")
try:
    ML_BUNDLE = joblib.load(ML_BUNDLE_PATH)
except FileNotFoundError:
    ML_BUNDLE = None
    print(f"⚠️  {ML_BUNDLE_PATH} not found -- /ml-insight will return 503 until it's generated "
          f"(run train_ml_models.py and copy ml_deployment_bundle.joblib next to main.py).")

# ---------- ১. ছবির লোকাল ম্যাপিং ----------
# Expanded from 9 to all 22 foods used in the survey / data_importer.py.
# Only the original 9 have real screenshots; the 13 new ones fall back to
# default.png until real photos are taken.
FOOD_IMAGE_MAP = {
    "ভাত": "/static/Local_Foods/Screenshot 2026-06-26 203045.png",
    "ইলিশ মাছ": "/static/Local_Foods/Screenshot 2026-06-26 203204.png",
    "লাল শাক": "/static/Local_Foods/Screenshot 2026-06-26 203247.png",
    "ডিম": "/static/Local_Foods/Screenshot 2026-06-26 203642.png",
    "মসুর ডাল": "/static/Local_Foods/Screenshot 2026-06-26 204037.png",
    "আমলকী": "/static/Local_Foods/Screenshot 2026-06-26 204202.png",
    "মুরগির মাংস": "/static/Local_Foods/Screenshot 2026-06-28 211255.png",
    "গরুর মাংস": "/static/Local_Foods/Screenshot 2026-06-28 210342.png",
    "দুধ চা": "/static/Local_Foods/Screenshot 2026-06-28 212225.png",
    # New additions -- no screenshot taken yet, so they use the default
    # fallback image until real photos are added.
    "রুই মাছ": "/static/Local_Foods/default.png",
    "দুধ": "/static/Local_Foods/default.png",
    "দই": "/static/Local_Foods/default.png",
    "পালং শাক": "/static/Local_Foods/default.png",
    "বেগুন": "/static/Local_Foods/default.png",
    "আলু": "/static/Local_Foods/default.png",
    "গাজর": "/static/Local_Foods/default.png",
    "মিষ্টি কুমড়া": "/static/Local_Foods/default.png",
    "আম": "/static/Local_Foods/default.png",
    "কমলালেবু": "/static/Local_Foods/default.png",
    "পেঁপে": "/static/Local_Foods/default.png",
    "ছোলা": "/static/Local_Foods/default.png",
    "আটা/রুটি": "/static/Local_Foods/default.png",
}

# ---------- ২. Bangla (frontend) name -> Ontology individual name ----------
# CRITICAL glue code: without this, FOOD_NUTRIENTS.get(food_name) silently
# fails for every food, because the ontology's individuals are named in
# English/Romanized form (Bhat, Ilish...) but the frontend sends Bangla
# script (ভাত, ইলিশ মাছ...).
#
# Copied verbatim from data_importer.py's BANGLA_TO_ONTOLOGY so the live
# /predict endpoint and the Google-Forms survey pipeline stay in sync.
# ⚠️ ASSUMPTION: this assumes ontology_loader.py's FOOD_NUTRIENTS actually
# has entries for all 22 of these individual names. If some of the 13 new
# foods aren't in the ontology yet, they'll just silently contribute 0
# intake below (same fail-open behavior as before) -- worth double
# checking FOOD_NUTRIENTS.keys() against this list.
BANGLA_TO_ONTOLOGY = {
    "ইলিশ মাছ": "Ilish", "রুই মাছ": "Rui", "গরুর মাংস": "GorurMangsho", "মুরগির মাংস": "MurgirMangsho",
    "ডিম": "Dim", "দুধ": "Dudh", "দই": "Doi", "দুধ চা": "DudhCha",
    "লাল শাক": "LalShak", "পালং শাক": "PalongShak",
    "বেগুন": "Begun", "আলু": "Alu", "গাজর": "Gajar", "মিষ্টি কুমড়া": "MishtiKumra",
    "আম": "Aam", "কমলালেবু": "Kamalalebu", "আমলকী": "Amloki", "পেঁপে": "Pepe",
    "ছোলা": "Chola", "মসুর ডাল": "MasoorDal", "আটা/রুটি": "Atta", "ভাত": "Bhat",
}
ONTOLOGY_TO_BANGLA = {v: k for k, v in BANGLA_TO_ONTOLOGY.items()}

# ---------- ২গ. Budget-tier map (interim, NOT ontology-driven) ----------
# ⚠️ The ontology has no cost/price data property at all (see DATA_PROPS in
# ontology_loader.py -- only nutrient content, RDA, symptom, treatment,
# enhancement, and religion-restriction exist). This hardcoded list is a
# stopgap so the "living with family vs. mess/hostel/shared" survey field
# actually does something, per the discussion in chat. For a defensible
# thesis claim, the real fix is adding a `hasCostTier` (or similar) data
# property to the .ttl, backed by actual local market price data -- this
# list is just two foods the user specifically called out as expensive
# (Hilsa, Beef), not a researched price survey.
HIGH_COST_FOODS = {"Ilish", "GorurMangsho"}

# ---------- ২ক. Frequency / Portion / Symptom vocab ----------
# Copied verbatim from data_importer.py so the live API and the survey
# importer accept/produce identical strings.
ALLOWED_FOODS = list(FOOD_IMAGE_MAP.keys())
VALID_FREQUENCIES = ["একদমই না", "সপ্তাহে ১–২ বার", "সপ্তাহে ৩–৪ বার", "সপ্তাহে ৫–৬ বার", "প্রতিদিন"]
VALID_PORTIONS = ["খাই না", "ছোট (স্বাভাবিকের চেয়ে কম)", "স্বাভাবিক পরিমাণ", "বড় (স্বাভাবিকের চেয়ে বেশি)"]
VALID_RELIGIONS = ["Hinduism", "Islam", "Buddhism", "Christianity", "Other"]

# ---------- ২খ. Lifestyle vocab ----------
# Copied verbatim from Nutritional_Deficiency_Cleaned_Final.xlsx's
# Living_arrangement / Sleep_category / Screen_before_sleep_category /
# Sun_exposure_category columns, so live /predict submissions land in the
# same categories as the cleaned survey data.
VALID_LIVING_ARRANGEMENTS = ["Mess/Hostel", "পরিবারের সাথে (বাসায়)", "বন্ধুদের সাথে শেয়ার্ড বাসায়"]
VALID_SLEEP_CATEGORIES = ["৪ ঘণ্টার কম", "৪–৬ ঘণ্টা", "৬–৮ ঘণ্টা", "৮ ঘণ্টার বেশি"]
VALID_SCREEN_CATEGORIES = ["৩০ মিনিটের কম", "৩০ মিনিট–১ ঘণ্টা", "১–২ ঘণ্টা", "২ ঘণ্টার বেশি"]
VALID_SUN_EXPOSURE_CATEGORIES = ["৩০ মিনিটের কম", "৩০ মিনিট–১ ঘণ্টা", "১–২ ঘণ্টা", "২ ঘণ্টার বেশি"]

# NOTE: life-stage (pregnant/lactating/child/elderly) categories were
# considered but removed -- this tool targets university students only,
# so gender-based RDA (RDA_VALUES has male/female columns) is sufficient.

# Note: the actual frequency/portion -> weekly-intake math now happens in
# deficiency.py's frequency_to_days() (imported above), not here -- see the
# predict_nutrition() body below.

# Symptom label (exactly as sent by the frontend / used in data_importer.py)
# -> ontology symptom individual name. Copied verbatim from data_importer.py.
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

# ---------- ২গ. Budget-aware recommendation tier ----------
# ⚠️ NOT ontology-driven -- there is no price/cost data property in the
# .ttl (see DATA_PROPS in ontology_loader.py). This is a hand-set heuristic
# based on typical Bangladeshi market prices, used to prefer affordable
# protein/nutrient sources when the user's living situation suggests a
# tighter food budget (mess/hostel/shared housing -- usually a fixed meal
# plan via a maid, little room to buy premium items like Hilsa/Beef).
# For full ontology-driven explainability, consider migrating this into the
# .ttl as a real data property (e.g. hasBudgetTier or
# hasApproxCostPerServing_BDT) so it shows up in the same reasoning chain
# as everything else.
EXPENSIVE_FOODS = {"Ilish", "Rui", "GorurMangsho"}
BUDGET_CONSTRAINED_LIVING = {"Mess/Hostel", "বন্ধুদের সাথে শেয়ার্ড বাসায়"}

# ---------- ৩. Pydantic মডেল (ডেটা ভ্যালিডেশন) ----------
class FoodEntry(BaseModel):
    frequency: str
    portion: str


class UserInput(BaseModel):
    name: str
    age: int
    gender: Literal["male", "female"]          # comes directly from the survey now
    religion: Literal["Hinduism", "Islam", "Buddhism", "Christianity", "Other"]
    weight: float
    height: float
    # NEW: lifestyle block, categories copied from the cleaned survey xlsx
    living_arrangement: Literal["Mess/Hostel", "পরিবারের সাথে (বাসায়)", "বন্ধুদের সাথে শেয়ার্ড বাসায়"]
    sleep_category: Literal["৪ ঘণ্টার কম", "৪–৬ ঘণ্টা", "৬–৮ ঘণ্টা", "৮ ঘণ্টার বেশি"]
    screen_before_sleep_category: Literal["৩০ মিনিটের কম", "৩০ মিনিট–১ ঘণ্টা", "১–২ ঘণ্টা", "২ ঘণ্টার বেশি"]
    sun_exposure_category: Literal["৩০ মিনিটের কম", "৩০ মিনিট–১ ঘণ্টা", "১–২ ঘণ্টা", "২ ঘণ্টার বেশি"]
    # NEW shape: {"ইলিশ মাছ": {"frequency": "সপ্তাহে ৩–৪ বার", "portion": "স্বাভাবিক পরিমাণ"}, ...}
    foods: Dict[str, FoodEntry]
    symptoms: List[str] = []                     # NEW -- optional, defaults to none reported

    @field_validator("foods")
    @classmethod
    def validate_foods(cls, v):
        for food_name, entry in v.items():
            if food_name not in ALLOWED_FOODS:
                raise ValueError(
                    f"'{food_name}' খাবারটি আমাদের তালিকায় নেই। "
                    f"দয়া করে সঠিক খাদ্য দিন: {', '.join(ALLOWED_FOODS)}"
                )
            if entry.frequency not in VALID_FREQUENCIES:
                raise ValueError(
                    f"'{entry.frequency}' একটি বৈধ frequency না। ব্যবহার করুন: {', '.join(VALID_FREQUENCIES)}"
                )
            if entry.portion not in VALID_PORTIONS:
                raise ValueError(
                    f"'{entry.portion}' একটি বৈধ portion size না। ব্যবহার করুন: {', '.join(VALID_PORTIONS)}"
                )
        return v

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, v):
        for symptom in v:
            if symptom not in SYMPTOM_LABEL_TO_ONTOLOGY:
                raise ValueError(
                    f"'{symptom}' একটি বৈধ উপসর্গ না। তালিকাভুক্ত উপসর্গগুলোর একটি ব্যবহার করুন।"
                )
        return v


# ---------- ৪. FastAPI অ্যাপ ----------
app = FastAPI(
    title="Nutrition Deficiency Detection API",
    description="Bangladeshi Food Ontology + XAI based System with Local Images",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- ৫. GET এন্ডপয়েন্ট (সার্ভার চেক) ----------
@app.get("/")
def root():
    return {"message": "✅ Server is running with CORS, Validation, Images, and Real Deficiency Calculation!"}


# ---------- ৬. POST এন্ডপয়েন্ট (মূল কাজ) ----------
@app.post("/predict")
def predict_nutrition(data: UserInput):
    print(f"🔥 Frontend থেকে ডেটা এসেছে: {data}")

    # --- Step 1: Bangla food names -> ontology names, with frequency x portion ---
    # deficiency.py's frequency_to_days() has now been fixed to accept the
    # NEW label set (and a portion argument), so we call it directly instead
    # of duplicating FREQ_MAP/PORTION_MAP locally.
    food_intake = {}
    for bangla_name, entry in data.foods.items():
        ontology_name = BANGLA_TO_ONTOLOGY.get(bangla_name)
        if ontology_name:
            food_intake[ontology_name] = frequency_to_days(entry.frequency, entry.portion)

    # --- Step 1a: Map reported symptoms to ontology individual names ---
    # Now fed into calculate_deficiency_percentage() below, which cross-checks
    # them against SYMPTOM_TO_NUTRIENTS and adds a "symptom_supported" flag
    # to each nutrient's result (corroborating evidence, doesn't change the
    # intake-based deficit_percentage/status itself).
    reported_symptom_ontology = [
        SYMPTOM_LABEL_TO_ONTOLOGY[label] for label in data.symptoms
        if label in SYMPTOM_LABEL_TO_ONTOLOGY
    ]

    # --- Step 2: Real deficiency calculation (uses actual gender, not age) ---
    deficiency_results = calculate_deficiency_percentage(
        food_intake, gender=data.gender, reported_symptoms=reported_symptom_ontology
    )

    # --- Step 3: Recommendations (direct sources + absorption enhancers) ---
    # NOTE: the full ontology has 27 foods, but the frontend only has
    # Bangla names/images for the 22 in ALLOWED_FOODS. We filter recommendations
    # down to that displayable set so the frontend never has to show an
    # English ontology name or a missing image. If you want the full food
    # variety recommended, add Bangla names + images for the remaining 5 in
    # FOOD_IMAGE_MAP and BANGLA_TO_ONTOLOGY instead of filtering here.
    recommendations = {}
    for nutrient, result in deficiency_results.items():
        if result["status"] == "Deficient":
            weekly_gap = max(result["rda_weekly"] - result["estimated_weekly_intake"], 0)

            direct_sources = TREATMENT_MAP.get(nutrient, [])
            enhancers = ENHANCEMENT_MAP.get(nutrient, [])

            def is_allowed(food):
                return data.religion not in RESTRICTED_MAP.get(food, [])

            # NEW: budget-aware ordering. If the user lives with family, we
            # assume they can more easily afford Ilish/Beef and leave the
            # ontology's own ordering alone. Otherwise (mess/hostel/shared),
            # affordable sources for the SAME nutrient are tried first --
            # expensive ones only appear if nothing affordable covers this
            # nutrient at all, so we never recommend nothing.
            is_family = data.living_arrangement == "পরিবারের সাথে (বাসায়)"
            deduped_sources = list(dict.fromkeys(direct_sources))
            if is_family:
                ordered_sources = deduped_sources
            else:
                affordable = [f for f in deduped_sources if f not in HIGH_COST_FOODS]
                costly = [f for f in deduped_sources if f in HIGH_COST_FOODS]
                ordered_sources = affordable + costly

            rec_items = []

            # Direct sources: get an actual portion recommendation
            for food in ordered_sources:
                if not is_allowed(food) or food not in ONTOLOGY_TO_BANGLA:
                    continue
                qty = get_quantity_recommendation(nutrient, food, weekly_gap)
                qty_text = qty["text"] if qty else "নিয়মিত খাদ্যতালিকায় রাখুন"
                qty_text = qty_text.replace(food, ONTOLOGY_TO_BANGLA[food])  # e.g. "Ilish" -> "ইলিশ"
                rec_items.append({
                    "food": ONTOLOGY_TO_BANGLA[food],
                    "type": "source",
                    "quantity_text": qty_text,
                    "budget_friendly": food not in HIGH_COST_FOODS,
                })
                if len(rec_items) == 3:
                    break

            # Absorption enhancers: no portion math (they don't supply the
            # nutrient themselves), just a qualitative pairing note.
            for food in list(dict.fromkeys(enhancers)):
                if len(rec_items) >= 3:
                    break
                if not is_allowed(food) or food not in ONTOLOGY_TO_BANGLA:
                    continue
                rec_items.append({
                    "food": ONTOLOGY_TO_BANGLA[food],
                    "type": "absorption_enhancer",
                    "quantity_text": f"{nutrient}-সমৃদ্ধ খাবারের সাথে খান (শোষণ বাড়াতে সাহায্য করে)",
                })

            recommendations[nutrient] = rec_items

    # --- Step 3a: Lifestyle-linked tips ---------------------------------
    # ⚠️ SCOPE: the ontology has no sun-exposure or sleep data properties at
    # all (see FOOD_NUTRIENTS/DATA_PROPS in ontology_loader.py) -- so this is
    # NOT ontology-driven reasoning, just one hand-written, well-established
    # rule (skin sun exposure is a major route of Vitamin D synthesis) applied
    # on top of the ontology's own VitaminD deficiency result. Sleep/screen
    # time are captured and returned below but don't drive any tip yet --
    # tying them to specific nutrients would need a cited rule, not a guess.
    lifestyle_tips = []
    low_sun_exposure = data.sun_exposure_category in ("৩০ মিনিটের কম", "৩০ মিনিট–১ ঘণ্টা")
    vitamin_d_result = deficiency_results.get("VitaminD")
    if vitamin_d_result and vitamin_d_result["status"] == "Deficient" and low_sun_exposure:
        lifestyle_tips.append(
            "আপনার ভিটামিন ডি-এর ঘাটতি আছে এবং রোদে থাকার সময়ও কম। "
            "খাবারের পাশাপাশি প্রতিদিন সকালে অন্তত ১৫-৩০ মিনিট সরাসরি রোদে থাকার চেষ্টা করুন -- "
            "ত্বকে সূর্যালোক লাগাই ভিটামিন ডি তৈরির প্রধান উপায়।"
        )

    # --- Step 4: Attach images for the foods the user reported eating ---
    food_list_with_images = [
        {"name": food_name, "image_url": FOOD_IMAGE_MAP.get(food_name, "/static/Local_Foods/default.png")}
        for food_name in data.foods.keys()
    ]

    # --- Step 5: BMI ---
    bmi = data.weight / ((data.height / 100) ** 2)
    bmi_info = get_bmi_category(bmi)

    # --- Step 6: Context-aware closing message ---
    # Only recommend professional consultation when there's a genuinely high
    # deficiency somewhere -- showing it unconditionally trains users to
    # ignore it, and it's misleading when someone's diet is actually fine.
    deficit_values = [r["deficit_percentage"] for r in deficiency_results.values()]
    max_deficit = max(deficit_values) if deficit_values else 0
    any_deficient = any(r["status"] == "Deficient" for r in deficiency_results.values())

    if max_deficit >= 50:
        closing_message = "আপনার কিছু পুষ্টি উপাদানে উল্লেখযোগ্য ঘাটতি দেখা যাচ্ছে। উপরের সুপারিশগুলো নিয়মিত মেনে চলুন, এবং ঘাটতি বেশি হলে একজন পুষ্টিবিদ বা ডাক্তারের সাথে পরামর্শ করা ভালো।"
    elif any_deficient:
        closing_message = "সামগ্রিকভাবে আপনার খাদ্যাভ্যাস মোটামুটি ভালো, তবে কিছু ছোটখাটো ঘাটতি আছে। উপরের সুপারিশগুলো মেনে চললে সহজেই পূরণ হয়ে যাবে।"
    else:
        closing_message = "চমৎকার! আপনার বর্তমান খাদ্যাভ্যাসে বড় কোনো পুষ্টি ঘাটতি পাওয়া যায়নি। এভাবেই ধারাবাহিকতা বজায় রাখুন।"

    return {
        "status": "success",
        "user_name": data.name,
        "bmi": round(bmi, 2),
        "bmi_category": bmi_info["category"],
        "bmi_category_bangla": bmi_info["bangla"],
        "bmi_risk": bmi_info["risk"],
        "foods_with_images": food_list_with_images,
        "deficiencies": deficiency_results,
        "recommendations": recommendations,
        "reported_symptoms": data.symptoms,          # Bangla labels, for display
        "reported_symptoms_ontology": reported_symptom_ontology,  # ontology names, for downstream use
        "lifestyle": {                                # NEW -- echoed back, for display/logging
            "living_arrangement": data.living_arrangement,
            "sleep_category": data.sleep_category,
            "screen_before_sleep_category": data.screen_before_sleep_category,
            "sun_exposure_category": data.sun_exposure_category,
        },
        "lifestyle_tips": lifestyle_tips,             # NEW -- see Step 3a above
        "message": closing_message,
    }


# ---------- ৭. ML + XAI insight endpoint (offline-trained, static explanations) ----------
def _build_ml_feature_vector(data: "UserInput", food_intake_ontology: dict, reported_symptom_ontology: list) -> pd.DataFrame:
    """
    Encodes one live request into EXACTLY the schema train_ml_models.py's
    get_feature_matrix() produced (same column names, same order -- stored
    in ML_BUNDLE['feature_cols']). Category label strings (gender, religion,
    sleep/screen/sun-exposure, living_arrangement) were already aligned to
    match data_importer.py's normalized values earlier in this project, so
    no extra translation is needed here -- just direct lookup.

    Any category value that wasn't seen during training (shouldn't happen,
    since UserInput's Literal types already restrict inputs to the trained
    categories) falls back to leaving that one-hot group all-zero, which
    RandomForest treats as "none of the known categories" rather than
    crashing.
    """
    feature_cols = ML_BUNDLE["feature_cols"]
    row = pd.Series(0.0, index=feature_cols)

    for onto_food, days in food_intake_ontology.items():
        col = f"food__{onto_food}"
        if col in row.index:
            row[col] = days

    for symptom in reported_symptom_ontology:
        col = f"symptom__{symptom}"
        if col in row.index:
            row[col] = 1

    if "age" in row.index:
        row["age"] = data.age or 0
    if "height_cm" in row.index:
        row["height_cm"] = data.height or 0
    if "weight_kg" in row.index:
        row["weight_kg"] = data.weight or 0

    categorical_values = {
        "gender": data.gender, "religion": data.religion,
        "sleep_category": data.sleep_category,
        "screen_before_sleep_category": data.screen_before_sleep_category,
        "sun_exposure_category": data.sun_exposure_category,
        "living_arrangement": data.living_arrangement,
    }
    for cat_col, value in categorical_values.items():
        dummy_col = f"{cat_col}_{value}"
        if dummy_col in row.index:
            row[dummy_col] = 1

    return pd.DataFrame([row])


@app.post("/ml-insight")
def ml_insight(data: UserInput):
    """
    Supplementary ML+XAI view alongside the ontology's rule-based /predict.
    Returns, for each nutrient with a trained model, the ML-predicted
    deficiency probability and a STATIC (precomputed offline, not recomputed
    per request) list of the top-5 features SHAP found most important for
    that nutrient in general -- i.e. "what usually drives this prediction
    across respondents," not a per-person SHAP explanation.

    ⚠️ Only 5 of 8 RDA-tracked nutrients have a model (VitaminC, VitaminA,
    Carbohydrate, Protein, Zinc) -- Iron, Calcium, and VitaminD still show
    ~100% deficiency prevalence in the training data (no class variance),
    so no classifier was trained for them. See ontology coverage notes.
    """
    if ML_BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail="ML models not available on this server (ml_deployment_bundle.joblib missing). "
                   "Run train_ml_models.py and deploy the generated bundle file.",
        )

    food_intake_ontology = {}
    for bangla_name, entry in data.foods.items():
        ontology_name = BANGLA_TO_ONTOLOGY.get(bangla_name)
        if ontology_name:
            food_intake_ontology[ontology_name] = frequency_to_days(entry.frequency, entry.portion)

    reported_symptom_ontology = [
        SYMPTOM_LABEL_TO_ONTOLOGY[label] for label in data.symptoms
        if label in SYMPTOM_LABEL_TO_ONTOLOGY
    ]

    X_live = _build_ml_feature_vector(data, food_intake_ontology, reported_symptom_ontology)

    results = {}
    for nutrient, model in ML_BUNDLE["models"].items():
        proba_deficient = float(model.predict_proba(X_live)[0][1])
        top_features = [
            {"feature": feat, "mean_abs_shap": round(float(score), 4)}
            for feat, score in ML_BUNDLE["importance"].get(nutrient, [])
        ]
        results[nutrient] = {
            "ml_predicted_risk_percentage": round(proba_deficient * 100, 1),
            "top_drivers_general": top_features,  # static, population-level -- not this person's own SHAP values
        }

    return {
        "status": "success",
        "note": "ML predictions are supplementary to the ontology-based /predict result, "
                "trained offline on 110 survey respondents. 'top_drivers_general' reflects "
                "population-level feature importance, not a per-person explanation.",
        "ml_insights": results,
    }