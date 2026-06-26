import os
import math
import joblib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. MySQL 연결 설정
# =========================

DB_USER = "root"
DB_PASSWORD = "tndk^^5005"   # 여기를 본인 MySQL 비밀번호로 수정
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "used_device_price_db"

DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    query={"charset": "utf8mb4"},
)

engine = create_engine(DATABASE_URL)


# =========================
# 2. 평가 함수
# =========================

def mean_absolute_percentage_error(y_true, y_pred):
    """
    MAPE 계산 함수.
    실제값이 0에 가까운 경우 나누기 오류를 피하기 위해 작은 값을 더한다.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    epsilon = 1e-8
    denominator = np.maximum(np.abs(y_true), epsilon)

    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100


def evaluate_model(model, X_test, y_test):
    """
    MAE, RMSE, R2, MAPE 계산.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


# =========================
# 3. 데이터 불러오기
# =========================

def load_data_from_mysql():
    query = "SELECT * FROM used_device_data"
    df = pd.read_sql(query, engine)

    print("MySQL 데이터 로드 완료")
    print(f"데이터 크기: {df.shape}")
    print(df.head())

    return df


# =========================
# 4. 데이터 전처리 및 학습 준비
# =========================

def prepare_data(df):
    # id 컬럼은 학습에 필요 없으므로 제거
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # 예측 대상
    target_col = "normalized_used_price"

    # 혹시 예측 대상 결측치가 있으면 제거
    df = df.dropna(subset=[target_col])

    # 입력 변수
    feature_cols = [
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

    # 실제 존재하는 컬럼만 사용
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    categorical_features = [
        "device_brand",
        "os",
        "four_g",
        "five_g",
    ]

    numeric_features = [
        col for col in feature_cols
        if col not in categorical_features
    ]

    print("\n사용 입력 변수:")
    print(feature_cols)

    print("\n범주형 변수:")
    print(categorical_features)

    print("\n수치형 변수:")
    print(numeric_features)

    return X, y, categorical_features, numeric_features


# =========================
# 5. 전처리 파이프라인 생성
# =========================

def create_preprocessor(categorical_features, numeric_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # scikit-learn 버전에 따라 sparse_output 지원 여부가 다를 수 있음
    try:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", categorical_encoder),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# =========================
# 6. 모델 학습 및 비교
# =========================

def train_and_compare_models(X, y, categorical_features, numeric_features):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = create_preprocessor(categorical_features, numeric_features)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
    }

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        print(f"\n모델 학습 중: {model_name}")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        result = {
            "model_name": model_name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "mape": metrics["mape"],
        }

        results.append(result)
        trained_pipelines[model_name] = pipeline

        print(
            f"{model_name} | "
            f"MAE: {metrics['mae']:.4f}, "
            f"RMSE: {metrics['rmse']:.4f}, "
            f"R2: {metrics['r2']:.4f}, "
            f"MAPE: {metrics['mape']:.2f}%"
        )

    results_df = pd.DataFrame(results)

    # MAE가 가장 낮은 모델을 최적 모델로 선정
    best_row = results_df.sort_values(by="mae", ascending=True).iloc[0]
    best_model_name = best_row["model_name"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\n최적 모델:")
    print(best_model_name)
    print(best_row)

    return results_df, best_model_name, best_pipeline, X_test, y_test


# =========================
# 7. 모델 결과 저장
# =========================

def save_results(results_df):
    os.makedirs("results", exist_ok=True)

    # CSV로 저장
    results_df.to_csv("results/model_results.csv", index=False, encoding="utf-8-sig")
    print("\n모델 성능 결과 CSV 저장 완료: results/model_results.csv")

    # MySQL model_results 테이블에 저장
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE model_results"))

    results_df.to_sql(
        "model_results",
        con=engine,
        if_exists="append",
        index=False
    )

    print("모델 성능 결과 MySQL 저장 완료: model_results")


def save_best_model(best_pipeline, best_model_name):
    os.makedirs("models", exist_ok=True)

    model_path = "models/best_model.joblib"
    joblib.dump({
        "model_name": best_model_name,
        "pipeline": best_pipeline,
    }, model_path)

    print(f"최적 모델 저장 완료: {model_path}")


# =========================
# 8. 실제값/예측값 저장
# =========================

def save_prediction_comparison(best_pipeline, X_test, y_test):
    os.makedirs("results", exist_ok=True)

    y_pred = best_pipeline.predict(X_test)

    comparison_df = pd.DataFrame({
        "actual_used_price": y_test.values,
        "predicted_used_price": y_pred,
        "error": y_test.values - y_pred,
        "abs_error": np.abs(y_test.values - y_pred),
    })

    comparison_df.to_csv(
        "results/prediction_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("실제값/예측값 비교 결과 저장 완료: results/prediction_comparison.csv")


# =========================
# 9. 메인 실행
# =========================

def main():
    df = load_data_from_mysql()

    X, y, categorical_features, numeric_features = prepare_data(df)

    results_df, best_model_name, best_pipeline, X_test, y_test = train_and_compare_models(
        X,
        y,
        categorical_features,
        numeric_features
    )

    save_results(results_df)
    save_best_model(best_pipeline, best_model_name)
    save_prediction_comparison(best_pipeline, X_test, y_test)

    print("\n모델 학습 전체 과정 완료")


if __name__ == "__main__":
    main()