from ultralytics import YOLO
from datetime import datetime
import numpy as np
import os
from urllib.parse import urlparse
import requests
# import boto3


INCIDENT_MAP = {
    0: "No incidents",
    1: "Fire alarm triggered",
    2: "Gate scanning malfunction",
    3: "Large-scale power outage",
    4: "Lost and found report",
    5: "Medical assistance required",
    6: "Medical emergency requiring evacuation",
    7: "Minor crowd congestion",
    8: "Minor medical assistance",
    9: "Misplaced ticket issue",
    10: "Power outage in section",
    11: "Queue management issue",
    12: "Rowdy attendee disturbance",
    13: "Security breach at gate",
    14: "Spilled drink cleanup",
    15: "Unauthorized access attempt",
}


def get_image_path(image_source, tmp_dir="/tmp", s3_config=None):
    """
    Resolves the image source into a local path YOLO can read.
    Supports:
        - Local files
        - HTTP/HTTPS URLs
        - S3 URLs (s3://bucket/key) with boto3 credentials
    s3_config should be a dict: {"aws_access_key_id":..., "aws_secret_access_key":..., "region_name":...}
    """
    # ---------------- Local file ----------------
    if os.path.exists(image_source):
        return image_source

    # ---------------- HTTP/HTTPS ----------------
    if image_source.startswith("http"):
        filename = os.path.basename(urlparse(image_source).path)
        local_file = os.path.join(tmp_dir, filename)
        r = requests.get(image_source, stream=True)
        r.raise_for_status()
        os.makedirs(tmp_dir, exist_ok=True)
        with open(local_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_file

    # ---------------- S3 ----------------
    if image_source.startswith("s3://"):
        if s3_config is None:
            raise ValueError("s3_config must be provided for S3 URLs")
        parsed = urlparse(image_source)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")  # remove leading '/'
        local_file = os.path.join(tmp_dir, os.path.basename(key))

        os.makedirs(tmp_dir, exist_ok=True)
        s3 = boto3.client(
            "s3",
            aws_access_key_id=s3_config["aws_access_key_id"],
            aws_secret_access_key=s3_config["aws_secret_access_key"],
            region_name=s3_config.get("region_name"),
        )
        s3.download_file(bucket, key, local_file)
        return local_file

    raise FileNotFoundError(f"Image not found or unsupported source: {image_source}")


def run_multi_gate_inference(model_path, gates_data, forecast_minutes=5):
    model = YOLO(model_path)
    final_outputs = []

    def calculate_congestion_metrics(people_count, total_capacity):
        capacity_percentage = people_count / total_capacity * 100
        if capacity_percentage < 40:
            level, numeric = "Low", 0
        elif capacity_percentage < 75:
            level, numeric = "Medium", 1
        else:
            level, numeric = "High", 2
        return capacity_percentage, level, numeric

    def compute_risk_score(congestion_numeric, confidence, queue_rate=0.0):
        queue_factor = max(0, min(queue_rate / 50.0, 1.0))  # normalize to [0,1]
        risk_score = (
            0.5 * (congestion_numeric / 2) + 0.3 * confidence + 0.2 * queue_factor
        )
        if risk_score < 0.33:
            risk_level = "Low"
        elif risk_score < 0.66:
            risk_level = "Medium"
        else:
            risk_level = "High"
        return round(risk_score, 2), risk_level

    def compute_trend(current_count, historical_count=None):
        if historical_count is None:
            return "stable", "weak", current_count
        avg = historical_count
        if current_count > avg * 1.1:
            trend = "rising"
            strength = "strong" if current_count > avg * 1.3 else "moderate"
        elif current_count < avg * 0.9:
            trend = "falling"
            strength = "moderate"
        else:
            trend = "stable"
            strength = "weak"
        return trend, strength, avg

    def map_incident_probabilities(
        people_count, total_capacity, congestion_level, event_type
    ):
        capacity_pct = people_count / total_capacity
        probabilities = {i: 0.0 for i in INCIDENT_MAP.keys()}
        if congestion_level == "High":
            probabilities[7] = min(1.0, 0.5 + capacity_pct / 2)
            probabilities[11] = min(1.0, 0.3 + capacity_pct / 3)
        if capacity_pct > 0.95:
            probabilities[6] = min(1.0, 0.5 + (capacity_pct - 0.95) * 10)
        if event_type.lower() == "concert" and congestion_level == "High":
            probabilities[12] = 0.3
        if capacity_pct > 1.0:
            probabilities[3] = 0.05
            probabilities[13] = 0.1
        # Build incidents list
        incidents = [
            {
                "incident_id": k,
                "incident_name": INCIDENT_MAP[k],
                "probability": round(v, 2),
            }
            for k, v in probabilities.items()
            if v > 0
        ]

        # If none, return "No incidents"
        if not incidents:
            incidents = [
                {"incident_id": 0, "incident_name": INCIDENT_MAP[0], "probability": 1.0}
            ]

        return incidents

    # ------------------- PROCESS EACH GATE -------------------
    for gate in gates_data:
        img_path = get_image_path(
            gate["image_path"],
            tmp_dir="/tmp",
            # s3_config={
            #     "aws_access_key_id": "<YOUR_KEY>",
            #     "aws_secret_access_key": "<YOUR_SECRET>",
            #     "region_name": "us-east-1",
            # },
        )
        total_capacity = gate["total_capacity"]
        event_type = gate["event_type"]
        historical_count = gate.get("historical_count", None)

        results = model(img_path, conf=0.2)
        dets = results[0].boxes

        # ✅ safer counting
        current_count = len(dets)  # assume only 1 class = person

        conf_score = float(np.mean(dets.conf.tolist()) if len(dets) > 0 else 1.0)
        if historical_count is not None:
            queue_rate = current_count - historical_count
        else:
            queue_rate = 0

        # Debug print raw detections
        print(f"\n[DEBUG] Gate {gate['gate_id']} -> {img_path}")
        print(" Detected count:", current_count)
        print(" Raw classes:", dets.cls.tolist())
        print(" Confidences:", dets.conf.tolist())

        # Current metrics
        cp, lvl, num = calculate_congestion_metrics(current_count, total_capacity)
        risk_score, risk_level = compute_risk_score(num, conf_score, queue_rate)
        trend, strength, avg = compute_trend(current_count, historical_count)

        # Forecast metrics (next 5 minutes)
        if trend == "rising":
            forecast_count = current_count + abs(queue_rate) * forecast_minutes
        elif trend == "falling":
            forecast_count = max(0, current_count - abs(queue_rate) * forecast_minutes)
        else:
            forecast_count = current_count

        cp_f, lvl_f, num_f = calculate_congestion_metrics(
            forecast_count, total_capacity
        )
        risk_score_f, risk_level_f = compute_risk_score(num_f, conf_score, queue_rate)
        incidents_f = map_incident_probabilities(
            forecast_count, total_capacity, lvl_f, event_type
        )

        output = {
            "gate_id": gate["gate_id"],
            "zone": gate["zone"],
            "current_people_count": current_count,
            "total_capacity": total_capacity,
            "capacity_percentage": round(cp, 2),
            "predicted_congestion_level": lvl,
            "predicted_congestion_numeric": num,
            "confidence_score": conf_score,
            "is_high_confidence": conf_score > 0.7,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "trend_analysis": {
                "current_trend": trend,
                "trend_strength": strength,
                "average_congestion": avg,
            },
            "forecast_next_5_min": {
                "predicted_people_count": forecast_count,
                "predicted_congestion_level": lvl_f,
                "predicted_congestion_numeric": num_f,
                "risk_score": risk_score_f,
                "risk_level": risk_level_f,
                "possible_incidents": incidents_f,
            },
            "timestamp": datetime.now().isoformat(),
            "metadata": gate,
            "model_version": "1.0.0",
            "prediction_id": f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{gate['gate_id']}",
        }

        final_outputs.append(output)

    return final_outputs


def query_congestion_model(model_path, gates_info, forecast_minutes=5):
    return run_multi_gate_inference(
        model_path=model_path, gates_data=gates_info, forecast_minutes=forecast_minutes
    )
