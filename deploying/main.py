# %%
from fastapi import FastAPI
import uvicorn
from dialect_recognition import Dialect_Recognition

app = FastAPI(debug=True)
dr = Dialect_Recognition()

@app.get("/")
async def home():
    return {"text": "Arabic Dialect Recognition"}

@app.get("/predict")
async def predict(text: str):
    prediction_cnb =  dr.predict_cnb(text)
    prediction_nn = dr.predict_nn(text)
    
    return {"ml_prediction": prediction_cnb,"nn_prediction": prediction_nn}

if __name__ == "__main__":
    uvicorn.run(app)