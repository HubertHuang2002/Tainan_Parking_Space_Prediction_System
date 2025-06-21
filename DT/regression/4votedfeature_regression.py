import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.model_selection import GroupKFold, TimeSeriesSplit


selected_features = [ 'TotalSpaces', 'lat', 'lon', 'firstHourFee', 'relative_humidity_2m',  'et0_fao_evapotranspiration',
                       'day_sin', 'half_hour_sin',  'half_hour_cos',  'weekday_num__Saturday', 'weekday_num__Sunday'
]


def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["datetime"])

    df.loc[
        (df["ParkingSegmentID"] == "1192") &
        (df["half_hour_interval"] == 20) &
        (df["month_val"] == 3) &
        (df["day_val"] == 8),
        "TotalSpaces"
    ] = 63

    indices_to_drop = df[
        (df['ParkingSegmentID'] == "1131") &
        (df['year_val'] == 2025) &
        (df['month_val'] == 3) &
        (df['day_val'] == 27)
    ].index
    df.drop(indices_to_drop, inplace=True)
    df = df.reset_index(drop=True)
    df = df[df["TotalSpaces"] > 0]
    y = df["avg_available_spots"]

    df["day_off"] = (
        df["is_national_holiday"].astype(str).isin(["True", "TRUE", "1", "true"]).astype(int) |
        df["weekday_x"].isin(["Saturday", "Sunday"]).astype(int)
    )

    district_map = {"安平": 0, "中西": 1}
    df["district"] = df["district"].map(district_map)
    df["district"] = df["district"].fillna(0).astype(int)

    df['wind_direction_10m sin'] = np.sin(np.deg2rad(df['wind_direction_10m']))
    df['wind_direction_10m cos'] = np.cos(np.deg2rad(df['wind_direction_10m']))
    df['wind_direction_120m sin'] = np.sin(np.deg2rad(df['wind_direction_120m']))
    df['wind_direction_120m cos'] = np.cos(np.deg2rad(df['wind_direction_120m']))
    df['wind_direction_80m sin'] = np.sin(np.deg2rad(df['wind_direction_80m']))
    df['wind_direction_80m cos'] = np.cos(np.deg2rad(df['wind_direction_80m']))
    df.drop(columns=["wind_direction_10m","wind_direction_120m","wind_direction_80m"],inplace=True)

    drop_cols = [
        "avg_available_spots", "datetime", "datetime_hour", "time",
        "hour", "date", "weekday_y", "holiday_name", "ParkingSegmentID", "latitude", "longitude", "is_national_holiday"
    ]
    X = df.drop(columns=drop_cols, errors='ignore')

    # half_hour_interval (0–47)
    X["half_hour_sin"] = np.sin(2 * np.pi * X["half_hour_interval"] / 48)
    X["half_hour_cos"] = np.cos(2 * np.pi * X["half_hour_interval"] / 48)
    # day_val (1–31) → normalize to 0–30
    X["day_norm"] = X["day_val"] - 1
    X["day_sin"] = np.sin(2 * np.pi * X["day_norm"] / 31)
    X["day_cos"] = np.cos(2 * np.pi * X["day_norm"] / 31)
    X = X.drop(columns=["half_hour_interval", "day_val", "day_norm"])
    
    X = pd.get_dummies(
        X,
        columns=["weekday_x", "month_val"],
        prefix=["weekday_num_", "month_val"],
        drop_first=False
    )

    X["day_off"] = df["day_off"]
    
    # Elastic net features
    X = X[selected_features]
    return X, y, df["datetime"].dt.date


def train_decision_tree(X_full, y_full, date_series):
    d = date_series  
    idx_train = (d >= datetime(2024,1,23).date()) & (d <= datetime(2025,1,22).date())
    idx_valid = (d >= datetime(2025,1,23).date()) & (d <= datetime(2025,3,22).date())
    idx_test  = (d >= datetime(2025,3,23).date()) & (d <= datetime(2025,5,9 ).date())

    X_tr_full, y_tr = X_full[idx_train], y_full[idx_train]
    X_va_full, y_va = X_full[idx_valid], y_full[idx_valid]
    X_te_full, y_te = X_full[idx_test ], y_full[idx_test]

    param_grid = {
        "max_depth":         [4, 8, 12, None],
        "min_samples_split": [2, 10, 30],
        "min_samples_leaf":  [1, 5, 10],
        "criterion":         ["squared_error"],
        "ccp_alpha":         [0.0, 1e-4]   
    }

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
        verbose=1
    )
    grid.fit(X_tr_full, y_tr)

    print("best parameter:", grid.best_params_)

    best_tree = grid.best_estimator_

    importances = best_tree.feature_importances_
    feature_names = X_tr_full.columns  
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\nFeature importances:")
    for name, imp in feat_imp:
        print(f"{name:<30} {imp:.4f}")

    y_va_pred = best_tree.predict(X_va_full)
    mae_va  = mean_absolute_error(y_va, y_va_pred)
    rmse_va = mean_squared_error(y_va, y_va_pred, squared=False)
    mse = mean_squared_error(y_va, y_va_pred)
    r2_va   = r2_score(y_va, y_va_pred)
    n_va = len(y_va)
    p_va = X_va_full.shape[1]
    adj_r2_va = 1 - (1 - r2_va) * (n_va - 1) / (n_va - p_va - 1)
    print(f"\n—— VALID ——\nMAE: {mae_va:.2f}  RMSE: {rmse_va:.2f}  R²: {r2_va:.4f}  Adjusted R²: {adj_r2_va:.4f}")

    y_te_pred = best_tree.predict(X_te_full)
    mae_te  = mean_absolute_error(y_te, y_te_pred)
    rmse_te = mean_squared_error(y_te, y_te_pred, squared=False)
    mse = mean_squared_error(y_te, y_te_pred)
    r2_te   = r2_score(y_te, y_te_pred)
    n_te = len(y_te)
    p_te = X_te_full.shape[1]
    adj_r2_te = 1 - (1 - r2_te) * (n_te - 1) / (n_te - p_te - 1)
    print(f"\n—— TEST ——\nMAE: {mae_te:.2f}  RMSE: {rmse_te:.2f}  R²: {r2_te:.4f}  Adjusted R²: {adj_r2_te:.4f}")

    print(f"\ntotal number of features: {len(X_tr_full.columns)}")

    return best_tree


def main():
    csv_path = "final_dataset.csv"
    X, y, dates = load_and_prepare_data(csv_path)      
    model = train_decision_tree(X, y, dates)
    joblib.dump(model, "step7_4votedfeature_parking_tree_count.pkl")
    print("\nthe model and the optimal threshold have been saved as step7_4votedfeature_parking_tree_count.pkl")


if __name__ == "__main__":
    main()
    