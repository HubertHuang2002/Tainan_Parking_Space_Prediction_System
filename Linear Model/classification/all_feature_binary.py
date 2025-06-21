import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, average_precision_score, classification_report,
                           confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_paths):
    dataframes = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            dataframes.append(df)
        except Exception as e:
            print(f"Failed to load file: {e}")
            return None
    
    if not dataframes:
        print("No data loaded")
        return None
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # handle specific data issues
    mask_1192 = (
        (combined_df["ParkingSegmentID"] == "1192") &
        (combined_df["half_hour_interval"] == 20) &
        (combined_df["month_val"] == 3) &
        (combined_df["day_val"] == 8)
    )
    if mask_1192.any():
        combined_df.loc[mask_1192, "TotalSpaces"] = 63
    
    mask_1131 = (
        (combined_df['ParkingSegmentID'] == "1131") &
        (combined_df['year_val'] == 2025) &
        (combined_df['month_val'] == 3) &
        (combined_df['day_val'] == 27)
    )
    if mask_1131.any():
        combined_df = combined_df[~mask_1131].reset_index(drop=True)
    
    # ensure that avg_available_spots does not exceed TotalSpaces
    condition = combined_df["avg_available_spots"] > combined_df["TotalSpaces"]
    if condition.any():
        combined_df.loc[condition, "avg_available_spots"] = combined_df.loc[condition, "TotalSpaces"].values
    
    # handle negative available parking spot counts
    negative_spots = combined_df['avg_available_spots'] < 0
    if negative_spots.any():
        combined_df.loc[negative_spots, 'avg_available_spots'] = 0
    
    # remove records where TotalSpaces is zero or negative
    zero_total_spaces = combined_df['TotalSpaces'] <= 0
    if zero_total_spaces.any():
        combined_df = combined_df[~zero_total_spaces].reset_index(drop=True)
    
    return combined_df

def create_target_variable(df):
    df = df.copy()
    
    # calculate vacancy rate
    df['occupancy_rate'] = (df['TotalSpaces'] - df['avg_available_spots']) / df['TotalSpaces']
    df['available_rate'] = df['avg_available_spots'] / df['TotalSpaces']
    
    # create binary target variable: set to 1 (full) if vacancy rate is below 5%
    df['is_full'] = (df['available_rate'] < 0.05).astype(int)
    
    print(f"Target distribution:")
    print(f"  Not full (0): {(df['is_full'] == 0).sum()} ({(df['is_full'] == 0).mean():.2%})")
    print(f"  Full (1): {(df['is_full'] == 1).sum()} ({(df['is_full'] == 1).mean():.2%})")
    
    return df

def add_features(df):
    df_processed = df.copy()
    
    # add a day_off column: set to 1 for weekends or holidays, otherwise 0
    if 'weekday_x' in df_processed.columns:
        is_weekend = (df_processed['weekday_x'] == 5) | (df_processed['weekday_x'] == 6)
    else:
        is_weekend = pd.Series(False, index=df_processed.index)
    
    if 'is_national_holiday' in df_processed.columns:
        is_holiday = df_processed['is_national_holiday'] == 1
    elif 'holiday_name' in df_processed.columns:
        is_holiday = df_processed['holiday_name'].notna()
    else:
        is_holiday = pd.Series(False, index=df_processed.index)
    
    df_processed['day_off'] = ((is_weekend) | (is_holiday)).astype(int)
    
    # handle wind direction angle conversion using sin and cos
    wind_direction_columns = ['wind_direction_10m', 'wind_direction_120m', 'wind_direction_80m']
    for col in wind_direction_columns:
        if col in df_processed.columns:
            df_processed[f'{col}_sin'] = np.sin(np.deg2rad(df_processed[col]))
            df_processed[f'{col}_cos'] = np.cos(np.deg2rad(df_processed[col]))
    
    # delete the original wind direction column
    df_processed.drop(columns=wind_direction_columns, inplace=True, errors='ignore')
    
    # month one-hot encoding
    if 'month_val' in df_processed.columns:
        month_dummies = pd.get_dummies(df_processed['month_val'], prefix='month')
        df_processed = pd.concat([df_processed, month_dummies], axis=1)
    
    # weekday one-hot encoding
    if 'weekday_x' in df_processed.columns:
        weekday_dummies = pd.get_dummies(df_processed['weekday_x'], prefix='weekday')
        df_processed = pd.concat([df_processed, weekday_dummies], axis=1)
    
    # convert date into sin and cos components
    if 'day_val' in df_processed.columns:
        df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day_val'] / 31)
        df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day_val'] / 31)
    
    # convert half-hour time intervals into sin and cos components
    if 'half_hour_interval' in df_processed.columns:
        df_processed['half_hour_sin'] = np.sin(2 * np.pi * df_processed['half_hour_interval'] / 48)
        df_processed['half_hour_cos'] = np.cos(2 * np.pi * df_processed['half_hour_interval'] / 48)
    
    return df_processed

def split_data_by_date(df):
    # if date column is missing, build from year month day
    if 'date' not in df.columns and all(col in df.columns for col in ['year_val', 'month_val', 'day_val']):
        df['date'] = pd.to_datetime(df['year_val'].astype(str) + '-' + 
                                   df['month_val'].astype(str).str.zfill(2) + '-' + 
                                   df['day_val'].astype(str).str.zfill(2))
    elif 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    if 'date' not in df.columns:
        print("Missing date columns")
        return None, None, None
    
    # define date ranges
    train_start = pd.to_datetime('2024-01-23')
    train_end = pd.to_datetime('2025-01-22')
    valid_start = pd.to_datetime('2025-01-23')
    valid_end = pd.to_datetime('2025-03-22')
    test_start = pd.to_datetime('2025-03-23')
    test_end = pd.to_datetime('2025-05-09')
    
    # data split
    train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
    valid_mask = (df['date'] >= valid_start) & (df['date'] <= valid_end)
    test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)
    
    train_df = df[train_mask].copy()
    valid_df = df[valid_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Train set: {train_df.shape[0]} rows")
    print(f"Validation set: {valid_df.shape[0]} rows")
    print(f"Test set: {test_df.shape[0]} rows")
    
    return train_df, valid_df, test_df

def prepare_features(train_df, valid_df, test_df):
    # columns to exclude
    exclude_cols = ['date', 'ParkingSegmentID', 'datetime', 'datetime_hour', 'time', 
                   'avg_available_spots', 'occupancy_rate', 'available_rate']
    
    all_dfs = [train_df, valid_df, test_df]
    processed_dfs = []
    
    for df in all_dfs:
        if df is None or len(df) == 0:
            processed_dfs.append(pd.DataFrame())
            continue
            
        df_processed = df.copy()
        
        # remove duplicate features
        potential_duplicates = [
            ['lat', 'latitude'],
            ['lon', 'longitude'],
            ['weekday_x', 'weekday_y'],
        ]
        
        for dup_group in potential_duplicates:
            available_cols = [col for col in dup_group if col in df_processed.columns]
            if len(available_cols) > 1:
                if 'lat' in available_cols and 'latitude' in available_cols:
                    to_remove = ['latitude']
                elif 'lon' in available_cols and 'longitude' in available_cols:
                    to_remove = ['longitude']
                elif 'weekday_x' in available_cols and 'weekday_y' in available_cols:
                    to_remove = ['weekday_y']
                else:
                    to_remove = available_cols[1:]
                
                df_processed = df_processed.drop(to_remove, axis=1, errors='ignore')
        
        # remove unnecessary columns
        for col in exclude_cols:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)
        
        processed_dfs.append(df_processed)
    
    # handle categorical variables
    categorical_cols = []
    for col in processed_dfs[0].columns:
        if col == 'is_full':  # target variable
            continue
        if processed_dfs[0][col].dtype == 'object' or col in ['district', 'holiday_name']:
            categorical_cols.append(col)
    
    print(f"Categorical columns: {categorical_cols}")
    
    # fit encoders using training set
    label_encoders = {}
    for col in categorical_cols:
        if col in processed_dfs[0].columns:
            # collect all possible category values
            all_categories = set()
            for df in processed_dfs:
                if len(df) > 0 and col in df.columns:
                    all_categories.update(df[col].fillna('Unknown').astype(str).unique())
            
            le = LabelEncoder()
            le.fit(list(all_categories))
            label_encoders[col] = le
            
            # apply the same encoding to all datasets
            for df in processed_dfs:
                if len(df) > 0 and col in df.columns:
                    df[col] = df[col].fillna('Unknown')
                    df[col] = le.transform(df[col].astype(str))
    
    # handle missing values
    for df in processed_dfs:
        if len(df) > 0:
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].median())
    
    # ensure all datasets have the same columns
    if len(processed_dfs[0]) > 0:
        common_columns = set(processed_dfs[0].columns)
        for df in processed_dfs[1:]:
            if len(df) > 0:
                common_columns = common_columns.intersection(set(df.columns))
        
        for i, df in enumerate(processed_dfs):
            if len(df) > 0:
                processed_dfs[i] = df[list(common_columns)]
    
    # separate features and target variable
    X_sets = []
    y_sets = []
    
    for df in processed_dfs:
        if len(df) > 0:
            X = df.drop(['is_full'], axis=1, errors='ignore')
            y = df['is_full'] if 'is_full' in df.columns else None
        else:
            X = pd.DataFrame()
            y = None
        
        X_sets.append(X)
        y_sets.append(y)
    
    if len(X_sets[0]) > 0:
        print(f"Feature count: {X_sets[0].shape[1]}")
        print(f"Features: {X_sets[0].columns.tolist()}")
    
    return X_sets, y_sets, label_encoders

def find_optimal_threshold(y_true, y_prob):
    # use the validation set to find the threshold with the best F1 score
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    return best_threshold

def evaluate_classification(y_true, y_prob, threshold=0.5, set_name=""):
    # calculate evaluation metrics for classification
    y_pred = (y_prob >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    prc_auc = average_precision_score(y_true, y_prob)
    
    print(f"\n{set_name} Results (threshold={threshold:.3f}):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PRC-AUC:   {prc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'prc_auc': prc_auc,
        'threshold': threshold
    }

def logistic_model_training(X_train, X_valid, X_test, y_train, y_valid, y_test):
    print(f"Training set shape: {X_train.shape}")
    if X_valid is not None and len(X_valid) > 0:
        print(f"Validation set shape: {X_valid.shape}")
    if X_test is not None and len(X_test) > 0:
        print(f"Test set shape: {X_test.shape}")
    
    # standardize features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_valid_scaled = None
    X_test_scaled = None
    if X_valid is not None and len(X_valid) > 0:
        X_valid_scaled = scaler.transform(X_valid)
    if X_test is not None and len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
    
    # grid search
    print("Performing grid search...")
    param_grid = {
        'C': [0.1, 1.0, 10.0],  
        'penalty': ['l1', 'l2'], 
        'solver': ['liblinear'], 
        'max_iter': [1000]  
    }
    
    # perform grid search using 3-fold cross-validation
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # use the model with the best parameters
    model = grid_search.best_estimator_
    
    # pridict
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    
    y_valid_prob = None
    y_test_prob = None
    if X_valid_scaled is not None:
        y_valid_prob = model.predict_proba(X_valid_scaled)[:, 1]
    if X_test_scaled is not None:
        y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # use the validation set to find the optimal threshold
    optimal_threshold = 0.5
    if y_valid_prob is not None and y_valid is not None:
        optimal_threshold = find_optimal_threshold(y_valid, y_valid_prob)
    
    # evaluate each dataset
    train_metrics = evaluate_classification(y_train, y_train_prob, optimal_threshold, "Training")
    
    valid_metrics = None
    test_metrics = None
    
    if y_valid_prob is not None and y_valid is not None:
        valid_metrics = evaluate_classification(y_valid, y_valid_prob, optimal_threshold, "Validation")
    
    if y_test_prob is not None and y_test is not None:
        test_metrics = evaluate_classification(y_test, y_test_prob, optimal_threshold, "Test")
    
    results = {
        'model': model,
        'scaler': scaler,
        'optimal_threshold': optimal_threshold,
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'train_metrics': train_metrics,
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'predictions': {
            'train': (y_train, y_train_prob),
            'valid': (y_valid, y_valid_prob) if y_valid_prob is not None else None,
            'test': (y_test, y_test_prob) if y_test_prob is not None else None
        }
    }
    
    return results

def load_test_data(test_file_path):
    try:
        test_df = pd.read_csv(test_file_path, low_memory=False)
        print(f"Loaded test data: {test_df.shape}")
        return test_df
    except Exception as e:
        print(f"Failed to load test data: {e}")
        return None

def save_model(model, scaler, threshold, file_path):
    model_data = {
        'model': model,
        'scaler': scaler,
        'threshold': threshold
    }
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {file_path}")

def main():
    # load training data
    file_paths = [
        '/kaggle/input/dataset/_dataset.csv',
        '/kaggle/input/dataset2/_dataset.csv'
    ]
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_paths)
    if df is None:
        return None
    
    print("Creating target variable...")
    df = create_target_variable(df)
    
    print("Adding features...")
    df = add_features(df)
    
    print("Splitting data by date...")
    train_df, valid_df, test_df = split_data_by_date(df)
    if train_df is None:
        return None
    
    print("Preparing features...")
    X_sets, y_sets, label_encoders = prepare_features(train_df, valid_df, test_df)
    if not X_sets or not y_sets:
        return None
    
    X_train, X_valid, X_test = X_sets
    y_train, y_valid, y_test = y_sets
    
    # load external test data
    print("Loading external test data...")
    external_test_df = load_test_data('/kaggle/input/dataset3/test.csv')
    
    if external_test_df is not None:
        print("Processing external test data...")
        external_test_df = create_target_variable(external_test_df)
        external_test_df = add_features(external_test_df)
        external_test_df_list = [pd.DataFrame(), pd.DataFrame(), external_test_df]
        X_external_sets, y_external_sets, _ = prepare_features(X_train.reset_index(drop=True), 
                                                              pd.DataFrame(), 
                                                              external_test_df)
        X_external_test = X_external_sets[2]
        y_external_test = y_external_sets[2]
    else:
        X_external_test = None
        y_external_test = None
    
    # model training
    print("Training model...")
    results = logistic_model_training(X_train, X_valid, X_test, y_train, y_valid, y_test)
    if results is None:
        return None
    
    # if external test data exists, compute its metrics
    if X_external_test is not None and y_external_test is not None:
        print("Evaluating on external test data...")
        X_external_test_scaled = results['scaler'].transform(X_external_test)
        y_external_prob = results['model'].predict_proba(X_external_test_scaled)[:, 1]
        
        external_metrics = evaluate_classification(
            y_external_test, y_external_prob, 
            results['optimal_threshold'], "External Test"
        )
    
    # save model
    print("Saving model...")
    save_model(results['model'], results['scaler'], results['optimal_threshold'], 
              'parking_classification_model.pkl')
    
    return results

if __name__ == "__main__":
    main()