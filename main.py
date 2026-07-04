# main.py (Final - CORS + Validation + Local Static Images + Real Deficiency Calculation)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from typing import Dict, Literal

from ontology_loader import FOOD_NUTRIENTS, RDA_VALUES, SYMPTOM_MAP, TREATMENT_MAP, ENHANCEMENT_MAP
from deficiency import calculate_deficiency_percentage, frequency_to_days

# ---------- ১. ছবির লোকাল ম্যাপিং ----------
FOOD_IMAGE_MAP = {
    "ভাত": "/static/Local_Foods/Screenshot 2026-06-26 203045.png",
    "ইলিশ": "/static/Local_Foods/Screenshot 2026-06-26 203204.png",
    "লাল শাক": "/static/Local_Foods/Screenshot 2026-06-26 203247.png",
    "ডিম": "/static/Local_Foods/Screenshot 2026-06-26 203642.png",
    "ডাল": "/static/Local_Foods/Screenshot 2026-06-26 204037.png",
    "আমলকি": "/static/Local_Foods/Screenshot 2026-06-26 204202.png",
    "মুরগি": "/static/Local_Foods/Screenshot 2026-06-28 211255.png",
    "গরু": "/static/Local_Foods/Screenshot 2026-06-28 210342.png",
    "দুধ চা": "/static/Local_Foods/Screenshot 2026-06-28 212225.png",
}

# ---------- ২. Bangla (frontend) name -> Ontology individual name ----------
# CRITICAL glue code: without this, FOOD_NUTRIENTS.get(food_name) silently
# fails for every food, because the ontology's individuals are named in
# English/Romanized form (Bhat, Ilish...) but the frontend sends Bangla
# script (ভাত, ইলিশ...).
BANGLA_TO_ONTOLOGY = {
    "ভাত": "Bhat",
    "ইলিশ": "Ilish",
    "লাল শাক": "LalShak",
    "ডিম": "Dim",
    "ডাল": "MasoorDal",
    "আমলকি": "Amloki",
    "মুরগি": "MurgirMangsho",
    "গরু": "GorurMangsho",
    "দুধ চা": "DudhCha",
}
ONTOLOGY_TO_BANGLA = {v: k for k, v in BANGLA_TO_ONTOLOGY.items()}

ALLOWED_FOODS = list(FOOD_IMAGE_MAP.keys())
VALID_FREQUENCIES = ["০ দিন", "১-২ দিন", "৩-৪ দিন", "৫-৬ দিন", "প্রতিদিন"]

# ---------- ৩. Pydantic মডেল (ডেটা ভ্যালিডেশন) ----------
class UserInput(BaseModel):
    name: str
    age: int
    gender: Literal["male", "female"]          # comes directly from the survey now
    weight: float
    height: float
    foods: Dict[str, str]                       # {"ভাত": "প্রতিদিন", "ইলিশ": "৩-৪ দিন", ...}

    @field_validator("foods")
    @classmethod
    def validate_foods(cls, v):
        for food_name, freq in v.items():
            if food_name not in ALLOWED_FOODS:
                raise ValueError(
                    f"'{food_name}' খাবারটি আমাদের তালিকায় নেই। "
                    f"দয়া করে সঠিক খাদ্য দিন: {', '.join(ALLOWED_FOODS)}"
                )
            if freq not in VALID_FREQUENCIES:
                raise ValueError(
                    f"'{freq}' একটি বৈধ frequency না। ব্যবহার করুন: {', '.join(VALID_FREQUENCIES)}"
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

    # --- Step 1: Bangla food names -> ontology names, with frequency ---
    food_intake = {}
    for bangla_name, freq_str in data.foods.items():
        ontology_name = BANGLA_TO_ONTOLOGY.get(bangla_name)
        if ontology_name:
            food_intake[ontology_name] = frequency_to_days(freq_str)

    # --- Step 2: Real deficiency calculation (uses actual gender, not age) ---
    deficiency_results = calculate_deficiency_percentage(food_intake, gender=data.gender)

    # --- Step 3: Recommendations (direct sources + absorption enhancers) ---
    # NOTE: the full ontology has 27 foods, but the frontend only has
    # Bangla names/images for the 9 in ALLOWED_FOODS. We filter recommendations
    # down to that displayable set so the frontend never has to show an
    # English ontology name or a missing image. If you want the full food
    # variety recommended, add Bangla names + images for the rest in
    # FOOD_IMAGE_MAP and BANGLA_TO_ONTOLOGY instead of filtering here.
    recommendations = {}
    for nutrient, result in deficiency_results.items():
        if result["status"] == "Deficient":
            candidates = TREATMENT_MAP.get(nutrient, []) + ENHANCEMENT_MAP.get(nutrient, [])
            displayable = [ONTOLOGY_TO_BANGLA[c] for c in candidates if c in ONTOLOGY_TO_BANGLA]
            recommendations[nutrient] = list(dict.fromkeys(displayable))[:3]

    # --- Step 4: Attach images for the foods the user reported eating ---
    food_list_with_images = [
        {"name": food_name, "image_url": FOOD_IMAGE_MAP.get(food_name, "/static/Local_Foods/default.png")}
        for food_name in data.foods.keys()
    ]

    # --- Step 5: BMI ---
    bmi = data.weight / ((data.height / 100) ** 2)

    return {
        "status": "success",
        "user_name": data.name,
        "bmi": round(bmi, 2),
        "foods_with_images": food_list_with_images,
        "deficiencies": deficiency_results,
        "recommendations": recommendations,
        "message": "দয়া করে উচ্চ ঘাটতির জন্য একজন পুষ্টিবিদ বা ডাক্তারের সাথে পরামর্শ করুন।",
    }