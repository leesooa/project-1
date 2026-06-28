from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from database import engine
from schemas import DeviceInput


app = FastAPI(title="Used Device Price Prediction API")


# React와 연결하기 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에는 * 사용 가능, 배포 시에는 React 주소로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 1. 모델 로드
# =========================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

model_data = joblib.load(MODEL_PATH)
MODEL_NAME = model_data["model_name"]
MODEL_PIPELINE = model_data["pipeline"]


# =========================
# 2. 컬럼 및 설정값
# =========================

FEATURE_COLUMNS = [
    "device_brand",
    "os",
    "screen_size",
    "four_g",
    "five_g",
    "rear_camera_mp",
    "front_camera_mp",
    "internal_memory",
    "ram",
    "battery",
    "weight",
    "release_year",
    "days_used",
    "normalized_new_price",
]

NUMERIC_COLUMNS = [
    "screen_size",
    "rear_camera_mp",
    "front_camera_mp",
    "internal_memory",
    "ram",
    "battery",
    "weight",
    "release_year",
    "days_used",
    "normalized_new_price",
]

CATEGORICAL_COLUMNS = [
    "device_brand",
    "os",
    "four_g",
    "five_g",
]

WEIGHTS = {
    "device_brand": 2.0,
    "os": 1.5,
    "four_g": 0.5,
    "five_g": 0.5,
    "screen_size": 0.8,
    "rear_camera_mp": 0.7,
    "front_camera_mp": 0.5,
    "internal_memory": 1.2,
    "ram": 1.2,
    "battery": 0.6,
    "weight": 0.5,
    "release_year": 1.5,
    "days_used": 1.2,
    "normalized_new_price": 2.0,
}

INCH_TO_CM = 2.54


# =========================
# 3. 기본 확인 API
# =========================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "message": "screen_size 입력값은 inch 기준이며, 백엔드에서 cm로 변환합니다.",
    }


# =========================
# 4. 입력값 단위 변환
# =========================

def normalize_device_data(device: DeviceInput) -> dict:
    """
    프론트에서는 screen_size를 inch 단위로 입력받고,
    모델과 DB 데이터는 cm 단위이므로 백엔드에서 cm로 변환한다.
    예: 6.1 inch -> 15.494 cm
    """
    data = device.model_dump()

    # inch -> cm 변환
    data["screen_size"] = data["screen_size"] * INCH_TO_CM

    return data


def device_input_to_dataframe(device: DeviceInput) -> pd.DataFrame:
    data = normalize_device_data(device)
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


# =========================
# 5. 가격 예측
# =========================

def predict_price(device: DeviceInput) -> float:
    input_df = device_input_to_dataframe(device)
    predicted_price = MODEL_PIPELINE.predict(input_df)[0]
    return float(predicted_price)


# =========================
# 6. MySQL 데이터 로드
# =========================

def load_used_device_data() -> pd.DataFrame:
    query = "SELECT * FROM used_device_data"
    df = pd.read_sql(query, engine)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df


# =========================
# 7. 유사도 계산
# =========================

def calculate_numeric_similarity(row_value, input_value, std):
    if std == 0 or np.isnan(std):
        std = 1.0

    diff = abs(float(row_value) - float(input_value))

    # 차이가 작을수록 1에 가까움, 클수록 0에 가까움
    return float(np.exp(-diff / std))


def find_similar_devices(device: DeviceInput, top_n: int = 10):
    original_df = load_used_device_data()

    # screen_size가 inch -> cm로 변환된 입력값
    input_data = normalize_device_data(device)

    input_brand = str(input_data["device_brand"]).lower()
    input_os = str(input_data["os"]).lower()

    # 1차: 동일 브랜드 + 동일 OS 기준으로 후보군 제한
    df = original_df[
        (original_df["device_brand"].astype(str).str.lower() == input_brand) &
        (original_df["os"].astype(str).str.lower() == input_os)
    ].copy()

    search_scope = "same_brand_and_os"

    # 2차: 후보가 너무 적으면 동일 브랜드만 사용
    if len(df) < top_n:
        df = original_df[
            original_df["device_brand"].astype(str).str.lower() == input_brand
        ].copy()
        search_scope = "same_brand"

    # 3차: 그래도 부족하면 전체 데이터 사용
    if len(df) < top_n:
        df = original_df.copy()
        search_scope = "all_data"

    # 결측치 처리
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])

    similarity_scores = []
    total_weight = sum(WEIGHTS.values())

    for _, row in df.iterrows():
        weighted_score = 0.0

        # 수치형 변수 유사도
        for col in NUMERIC_COLUMNS:
            if col not in df.columns:
                continue

            std = df[col].std()

            feature_similarity = calculate_numeric_similarity(
                row[col],
                input_data[col],
                std
            )

            weighted_score += WEIGHTS.get(col, 1.0) * feature_similarity

        # 범주형 변수 유사도
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue

            row_value = str(row[col]).lower()
            input_value = str(input_data[col]).lower()

            if row_value == input_value:
                feature_similarity = 1.0
            else:
                feature_similarity = 0.0

            weighted_score += WEIGHTS.get(col, 1.0) * feature_similarity

        similarity_score = (weighted_score / total_weight) * 100
        similarity_scores.append(similarity_score)

    df["similarity_score"] = similarity_scores

    similar_df = df.sort_values(by="similarity_score", ascending=False).head(top_n)

    result_cols = [
        "device_brand",
        "os",
        "screen_size",
        "four_g",
        "five_g",
        "rear_camera_mp",
        "front_camera_mp",
        "internal_memory",
        "ram",
        "battery",
        "weight",
        "release_year",
        "days_used",
        "normalized_new_price",
        "normalized_used_price",
        "similarity_score",
    ]

    result_cols = [col for col in result_cols if col in similar_df.columns]

    similar_devices = []

    for rank, (_, row) in enumerate(similar_df.iterrows(), start=1):
        item = {"rank": rank}

        for col in result_cols:
            value = row[col]

            if pd.isna(value):
                item[col] = None
            elif isinstance(value, (np.integer,)):
                item[col] = int(value)
            elif isinstance(value, (np.floating,)):
                item[col] = round(float(value), 4)
            else:
                item[col] = value

        item["similarity_score"] = round(float(row["similarity_score"]), 2)

        # 프론트 표시용 inch/cm 제공
        if item.get("screen_size") is not None:
            item["screen_size_inch"] = round(float(item["screen_size"]) / INCH_TO_CM, 2)
            item["screen_size_cm"] = item["screen_size"]

        similar_devices.append(item)

    return similar_devices, search_scope

# =========================
# 8. 예측 이력 저장
# =========================

def save_prediction_history(device: DeviceInput, predicted_price: float):
    # DB에는 학습 데이터와 같은 단위인 cm 기준 screen_size를 저장
    data = normalize_device_data(device)

    insert_query = text("""
        INSERT INTO prediction_history (
            device_brand,
            os,
            screen_size,
            four_g,
            five_g,
            rear_camera_mp,
            front_camera_mp,
            internal_memory,
            ram,
            battery,
            weight,
            release_year,
            days_used,
            normalized_new_price,
            predicted_used_price,
            model_name
        )
        VALUES (
            :device_brand,
            :os,
            :screen_size,
            :four_g,
            :five_g,
            :rear_camera_mp,
            :front_camera_mp,
            :internal_memory,
            :ram,
            :battery,
            :weight,
            :release_year,
            :days_used,
            :normalized_new_price,
            :predicted_used_price,
            :model_name
        )
    """)

    data["predicted_used_price"] = predicted_price
    data["model_name"] = MODEL_NAME

    with engine.begin() as conn:
        conn.execute(insert_query, data)


# =========================
# 9. 예측 + 유사 사양 비교 API
# =========================

@app.post("/predict-with-similar")
def predict_with_similar(device: DeviceInput):
    predicted_price = predict_price(device)

    similar_devices, search_scope = find_similar_devices(device, top_n=10)

    prices = [
        item["normalized_used_price"]
        for item in similar_devices
        if item.get("normalized_used_price") is not None
    ]

    if prices:
        average_price = float(np.mean(prices))
        min_price = float(np.min(prices))
        max_price = float(np.max(prices))
        difference = predicted_price - average_price
        difference_rate = (difference / average_price) * 100 if average_price != 0 else 0
    else:
        average_price = None
        min_price = None
        max_price = None
        difference = None
        difference_rate = None

    save_prediction_history(device, predicted_price)

    converted_data = normalize_device_data(device)

    return {
    "predicted_price": round(predicted_price, 4),
    "model_name": MODEL_NAME,
    "input_unit_info": {
        "screen_size_input_inch": round(device.screen_size, 2),
        "screen_size_converted_cm": round(converted_data["screen_size"], 4),
    },
    "similar_summary": {
        "count": len(similar_devices),
        "search_scope": search_scope,
        "average_price": round(average_price, 4) if average_price is not None else None,
        "min_price": round(min_price, 4) if min_price is not None else None,
        "max_price": round(max_price, 4) if max_price is not None else None,
        "difference_from_average": round(difference, 4) if difference is not None else None,
        "difference_rate": round(difference_rate, 2) if difference_rate is not None else None,
    },
    "similar_devices": similar_devices,
}


# =========================
# 10. 모델 성능 결과 API
# =========================

@app.get("/model-results")
def get_model_results():
    query = """
        SELECT model_name, mae, rmse, r2, mape, created_at
        FROM model_results
        ORDER BY mae ASC
    """
    df = pd.read_sql(query, engine)

    return df.to_dict(orient="records")


# =========================
# 11. 예측 이력 조회 API
# =========================

@app.get("/history")
def get_history():
    query = """
        SELECT *
        FROM prediction_history
        ORDER BY created_at DESC
        LIMIT 20
    """
    df = pd.read_sql(query, engine)

    # DB에는 cm 기준으로 저장되어 있으므로, 프론트 표시용 inch 값도 추가
    if "screen_size" in df.columns:
        df["screen_size_inch"] = (df["screen_size"] / INCH_TO_CM).round(2)
        df["screen_size_cm"] = df["screen_size"].round(4)

    return df.to_dict(orient="records")

@app.get("/options")
def get_options():
    df = load_used_device_data()

    brands = sorted(df["device_brand"].dropna().unique().tolist())
    os_list = sorted(df["os"].dropna().unique().tolist())

    return {
        "device_brand": brands,
        "os": os_list,
        "four_g": ["yes", "no"],
        "five_g": ["yes", "no"],
        "numeric_ranges": {
            "screen_size_inch": {
                "min": round(float(df["screen_size"].min() / INCH_TO_CM), 2),
                "max": round(float(df["screen_size"].max() / INCH_TO_CM), 2)
            },
            "rear_camera_mp": {
                "min": float(df["rear_camera_mp"].min()),
                "max": float(df["rear_camera_mp"].max())
            },
            "front_camera_mp": {
                "min": float(df["front_camera_mp"].min()),
                "max": float(df["front_camera_mp"].max())
            },
            "internal_memory": {
                "min": float(df["internal_memory"].min()),
                "max": float(df["internal_memory"].max())
            },
            "ram": {
                "min": float(df["ram"].min()),
                "max": float(df["ram"].max())
            },
            "battery": {
                "min": float(df["battery"].min()),
                "max": float(df["battery"].max())
            },
            "weight": {
                "min": float(df["weight"].min()),
                "max": float(df["weight"].max())
            },
            "release_year": {
                "min": int(df["release_year"].min()),
                "max": int(df["release_year"].max())
            },
            "days_used": {
                "min": int(df["days_used"].min()),
                "max": int(df["days_used"].max())
            },
            "normalized_new_price": {
                "min": float(df["normalized_new_price"].min()),
                "max": float(df["normalized_new_price"].max())
            }
        }
    }
    
@app.get("/analysis/brand-price")
def get_brand_price_analysis():
    query = """
        SELECT 
            device_brand,
            COUNT(*) AS device_count,
            AVG(normalized_used_price) AS average_used_price
        FROM used_device_data
        GROUP BY device_brand
        HAVING COUNT(*) >= 5
        ORDER BY average_used_price DESC
    """
    df = pd.read_sql(query, engine)

    return df.to_dict(orient="records")

@app.get("/analysis/release-year-price")
def get_release_year_price_analysis():
    query = """
        SELECT 
            release_year,
            COUNT(*) AS device_count,
            AVG(normalized_used_price) AS average_used_price
        FROM used_device_data
        GROUP BY release_year
        ORDER BY release_year ASC
    """
    df = pd.read_sql(query, engine)

    return df.to_dict(orient="records")

@app.get("/analysis/days-used-price")
def get_days_used_price_analysis():
    df = load_used_device_data()

    bins = [0, 300, 600, 900, 1200, 1500, 2000, 3000, 5000]
    labels = [
        "0-300일",
        "301-600일",
        "601-900일",
        "901-1200일",
        "1201-1500일",
        "1501-2000일",
        "2001-3000일",
        "3001일 이상"
    ]

    df["days_used_group"] = pd.cut(
        df["days_used"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    result = (
        df.groupby("days_used_group", observed=True)
        .agg(
            device_count=("normalized_used_price", "count"),
            average_used_price=("normalized_used_price", "mean")
        )
        .reset_index()
    )

    result["average_used_price"] = result["average_used_price"].round(4)

    return result.to_dict(orient="records")