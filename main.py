from fastapi import FastAPI

app = FastAPI(title="Nutrition API", version="1.0")

@app.get("/")
def root():
    return {"message": "Hello World! "}



@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello {name}!"}