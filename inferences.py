import os
import json
from congestion_helper import query_congestion_model 

MODEL_FILE_NAME = "best.pt"

# Load model path (avoid loading YOLO multiple times in memory)
def model_fn(model_dir):
    model_path = os.path.join(model_dir, MODEL_FILE_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{MODEL_FILE_NAME} not found in {model_dir}")
    return model_path

# Parse incoming request
def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        input_data = json.loads(request_body)
        # Expecting: { "gates_info": [ {...}, {...} ], "forecast_minutes": 5 }
        return input_data
    raise ValueError(f"Unsupported content type: {content_type}")

# Run inference
def predict_fn(input_data, model_path):
    gates_info = input_data.get("gates_info", [])
    forecast_minutes = input_data.get("forecast_minutes", 5)

    results = query_congestion_model(
        model_path=model_path,
        gates_info=gates_info,
        forecast_minutes=forecast_minutes
    )
    return results

# Return JSON response
def output_fn(prediction, content_type="application/json"):
    if content_type == "application/json":
        return json.dumps(prediction), content_type
    raise ValueError(f"Unsupported content type: {content_type}")
