from fastapi import FastAPI, UploadFile, File
from congestion_helper import query_congestion_model

app = FastAPI()

MODEL_PATH = "model/best.pt"

@app.post("/predict")
async def predict(payload: dict):
    gates_info = payload.get("gates_info", [])
    forecast_minutes = payload.get("forecast_minutes", 5)

    results = query_congestion_model(
        model_path=MODEL_PATH,
        gates_info=gates_info,
        forecast_minutes=forecast_minutes
    )
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
