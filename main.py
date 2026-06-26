# main.py (Final - CORS + Validation + Local Static Images)

# ১. প্রয়োজনীয় লাইব্রেরি ইমপোর্ট
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# ---------- ২. ছবির লোকাল ম্যাপিং (আপনার ফাইলের নাম ঠিক রাখা হলো) ----------
# আপনার static/Local_Foods/ ফোল্ডারের ভিতরে যেই নামে ছবি আছে, ঠিক সেই নাম বসানো হলো।
# শুধু পাথের শুরুটা '/static/' (ছোট হাতের) করা হয়েছে।


FOOD_IMAGE_MAP = {
    "ভাত": "/static/Local_Foods/Screenshot 2026-06-26 203045.png",
    "ইলিশ": "/static/Local_Foods/Screenshot 2026-06-26 203204.png",
    "লাল শাক": "/static/Local_Foods/Screenshot 2026-06-26 203247.png",
    "ডিম": "/static/Local_Foods/Screenshot 2026-06-26 203642.png",
    "ডাল": "/static/Local_Foods/Screenshot 2026-06-26 204037.png",
    "আমলকি": "/static/Local_Foods/Screenshot 2026-06-26 204202.png",
}

#ভাত, ইলিশ, ডাল,আমলকি, ডিম, লাল শাক


# অনুমোদিত খাবারের তালিকা (শুধু নামগুলো)
ALLOWED_FOODS = list(FOOD_IMAGE_MAP.keys())
# ---------------------------------------------------------------------

# ---------- ৩. Pydantic মডেল (ডেটা ভ্যালিডেশন) ----------
class UserInput(BaseModel):
    name: str
    age: int
    weight: float
    height: float
    foods: list[str]

    # ভ্যালিডেটর: ইউজার যে খাবার পাঠায়, সেটা আমাদের তালিকায় আছে কিনা চেক করে
    @field_validator('foods')
    @classmethod
    def validate_foods(cls, v):
        for item in v:
            if item not in ALLOWED_FOODS:
                raise ValueError(f"'{item}' খাবারটি আমাদের তালিকায় নেই। দয়া করে সঠিক খাদ্য দিন (যেমন: ভাত, মাছ, ডাল)")
        return v
# ---------------------------------------------------------

# ---------- ৪. FastAPI অ্যাপ ----------
app = FastAPI(
    title="Nutrition Deficiency Detection API",
    description="Bangladeshi Food Ontology + XAI based System with Local Images",
    version="1.0.0"
)

# ----- Static Files মাউন্ট (ছবি দেখানোর জন্য) -----
# আপনার কম্পিউটারের 'static' ফোল্ডারটি এখন ওয়েবসাইটের '/static' ঠিকানায় খোলা থাকবে।
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----- CORS মিডলওয়্যার (ফ্রন্টএন্ড থেকে ডেটা নেওয়ার অনুমতি) -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # সব ওয়েবসাইটকে অনুমতি
    allow_credentials=True,
    allow_methods=["*"],          # সব মেথড (GET, POST) অনুমোদিত
    allow_headers=["*"],          # সব হেডার অনুমোদিত
)
# ------------------------------------------------

# ---------- ৫. GET এন্ডপয়েন্ট (সার্ভার চেক) ----------
@app.get("/")
def root():
    return {"message": "✅ Server is running with CORS, Validation, and Local Images!"}

# ---------- ৬. POST এন্ডপয়েন্ট (এটাই মূল কাজ) ----------
@app.post("/predict")
def predict_nutrition(data: UserInput):
    # 🌟 ডিবাগিং: টার্মিনালে ডেটা দেখুন (ডেভেলপারের জন্য)
    print(f"🔥 Frontend থেকে ডেটা এসেছে: {data}")
    
    # ইউজারের পাঠানো খাবারের তালিকা থেকে ছবির URL বের করা
    food_list_with_images = []
    for food in data.foods:
        # FOOD_IMAGE_MAP থেকে URL খুঁজে বের করি, না পেলে ফাঁকা স্ট্রিং দেই
        image_url = FOOD_IMAGE_MAP.get(food, "/static/Local_Foods/default.png") 
        food_list_with_images.append({
            "name": food,
            "image_url": image_url
        })
    
    # BMI ক্যালকুলেশন (শুধু ডেমো)
    bmi = data.weight / ((data.height / 100) ** 2)
    
    # ক্লায়েন্টকে JSON রেসপন্স পাঠানো
    return {
        "status": "success",
        "message": f"হ্যালো {data.name}! আপনার বয়স {data.age} বছর।",
        "your_bmi": round(bmi, 2),
        "foods_with_images": food_list_with_images,  # এখানে ছবির লিংক আছে
        "recommendation": "শীঘ্রই আসল ML মডেল ও SHAP ব্যাখ্যা যোগ করা হবে!"
    }
# -------------------------------------------------