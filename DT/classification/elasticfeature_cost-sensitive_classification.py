import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score


selected_features = [
    "laterHourFee", "firstHourFee", "day_off", "district", "half_hour_cos", "lat", "year_val",
    "month_val_2", "TotalSpaces", "month_val_12", "lon", "terrestrial_radiation_instant", "dew_point_2m",
    "weekday_num__Wednesday", "day_sin", "weekday_num__Tuesday", "half_hour_sin", "month_val_5", "cloud_cover_mid",
    "weekday_num__Monday", "pressure_msl", "vapour_pressure_deficit", "month_val_11", "wind_speed_10m",
    "cloud_cover", "precipitation", "surface_pressure"
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

    df["isfull"] = (
        (df["avg_available_spots"] / df["TotalSpaces"]) <= 0.05
    ).astype(int)
    y = df["isfull"]

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
        "hour", "date", "weekday_y", "holiday_name", "ParkingSegmentID", "latitude", "longitude" , "isfull", "is_national_holiday"
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

    N_neg = (y_tr == 0).sum()
    N_pos = (y_tr == 1).sum()
    pos_weight = N_neg / (N_pos + 1e-8)
    class_w = {0: 1, 1: pos_weight}
    
    param_grid = {
        "max_depth":         [4, 8, 12, None],
        "min_samples_split": [2, 10, 30],
        "min_samples_leaf":  [1, 5, 10],
        "criterion":         ["gini"],
        "ccp_alpha":         [0.0, 1e-4]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight=class_w),
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1",
        verbose=1
    )
    grid.fit(X_tr_full, y_tr)

    best_tree = grid.best_estimator_

    print("Best parameters found:", grid.best_params_)

    importances = best_tree.feature_importances_
    feature_names = X_tr_full.columns  
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\nFeature importances:")
    for name, imp in feat_imp:
        print(f"{name:<30} {imp:.4f}")

    y_va_proba = best_tree.predict_proba(X_va_full)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(y_va, (y_va_proba >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]

    y_va_pred = (y_va_proba >= best_threshold).astype(int)
    
    print("\n—— VALID ——")
    print(f"Best threshold for F1-score: {best_threshold:.4f}")
    print("Accuracy :", accuracy_score(y_va, y_va_pred))
    print("Precision:", precision_score(y_va, y_va_pred))
    print("Recall   :", recall_score(y_va, y_va_pred))
    print("F1-score :", f1_score(y_va, y_va_pred))
    print("AUC      :", roc_auc_score(y_va, y_va_proba))
    print("PRC AUC  :", average_precision_score(y_va, y_va_proba))

    y_te_proba = best_tree.predict_proba(X_te_full)[:,1]
    y_te_pred = (y_te_proba >= best_threshold).astype(int)
    print("\n—— TEST ——")
    print("Accuracy :", accuracy_score(y_te, y_te_pred))
    print("Precision:", precision_score(y_te, y_te_pred))
    print("Recall   :", recall_score(y_te, y_te_pred))
    print("F1-score :", f1_score(y_te, y_te_pred))
    print("AUC      :", roc_auc_score(y_te, y_te_proba))
    print("PRC AUC  :", average_precision_score(y_te, y_te_proba))
    print(f"\ntotal number of features: {len(X_tr_full.columns)}")

    return best_tree, best_threshold


def main():
    csv_path = "final_dataset.csv"
    X, y, dates = load_and_prepare_data(csv_path)
    model, threshold = train_decision_tree(X, y, dates)
    joblib.dump({"model": model, "threshold": threshold}, "step6_elasticfeature_cost-sensitive_parking_tree_isfull.pkl")
    print("\nthe model and the optimal threshold have been saved as step6_elasticfeature_cost-sensitive_parking_tree_isfull.pkl")


if __name__ == "__main__":
    main()
