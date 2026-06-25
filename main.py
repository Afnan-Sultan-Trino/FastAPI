from fastapi import FastAPI
from pydantic import BaseModel   # নতুন যোগ হলো

# ১. ছাঁচ বানাই
class UserInput(BaseModel):
    age: int
    weight: float
    height: float
    foods: list[str]

# ২. অ্যাপ বানাই
app = FastAPI(title="Nutrition API", version="1.0")

# ৩. GET (আগের মতোই আছে)
@app.get("/")
def root():
    return {"message": "Hello World!"}

# ৪. POST (এটাই নতুন)
@app.post("/predict")
def predict_nutrition(data: UserInput):   # <-- লক্ষ্য করুন, 'data' হলো UserInput ছাঁচে ঢালাই করা ডেটা
    # এই ফাংশনের ভেতর আমরা ডেটা নিয়ে খেলতে পারি
    calculated_bmi = data.weight / ((data.height / 100) ** 2)
    
    return {
        "status": "success",
        "your_age": data.age,
        "your_bmi": round(calculated_bmi, 2),
        "received_foods": data.foods,
        "message": f"আপনার বয়স {data.age} এবং আপনি {', '.join(data.foods)} খান।"
    }