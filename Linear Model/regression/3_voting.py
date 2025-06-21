import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
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
    # if the date column is missing, create it using year, month, and day
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
    # specified feature list
    selected_features = [
        'TotalSpaces', 'lat', 'lon', 'firstHourFee', 'relative_humidity_2m', 
        'et0_fao_evapotranspiration', 'terrestrial_radiation_instant', 
        'day_sin', 'half_hour_sin', 'half_hour_cos', 'month_val_4', 
        'weekday_num_0', 'weekday_num_5', 'weekday_num_6', 'day_off'
    ]
    
    # columns to exclude
    exclude_cols = ['date', 'ParkingSegmentID', 'datetime', 'datetime_hour', 'time']
    
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
        if col == 'avg_available_spots':  # target variable
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
            # check if the specified features exist in the dataset
            available_features = [f for f in selected_features if f in df.columns]
            missing_features = [f for f in selected_features if f not in df.columns]
            
            if missing_features:
                print(f"Warning: Missing selected features: {missing_features}")
            
            # select only the specified features
            X = df[available_features]
            y = df['avg_available_spots'] if 'avg_available_spots' in df.columns else None
        else:
            X = pd.DataFrame()
            y = None
        
        X_sets.append(X)
        y_sets.append(y)
    
    if len(X_sets[0]) > 0:
        print(f"Feature count: {X_sets[0].shape[1]}")
        print(f"Features used: {X_sets[0].columns.tolist()}")
    
    return X_sets, y_sets, label_encoders

def calculate_adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def linear_model_training(X_train, X_valid, X_test, y_train, y_valid, y_test):
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
    
    # train linear regression model
    print("Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # predict
    y_train_pred = model.predict(X_train_scaled)
    
    y_valid_pred = None
    y_test_pred = None
    if X_valid_scaled is not None:
        y_valid_pred = model.predict(X_valid_scaled)
    if X_test_scaled is not None:
        y_test_pred = model.predict(X_test_scaled)
    
    # ensure predicted values are non-negative
    y_train_pred = np.maximum(y_train_pred, 0)
    if y_valid_pred is not None:
        y_valid_pred = np.maximum(y_valid_pred, 0)
    if y_test_pred is not None:
        y_test_pred = np.maximum(y_test_pred, 0)
    
    # calculate evaluation metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    
    valid_r2 = valid_mse = None
    test_r2 = test_mse = test_adj_r2 = None
    
    if y_valid_pred is not None and y_valid is not None:
        valid_r2 = r2_score(y_valid, y_valid_pred)
        valid_mse = mean_squared_error(y_valid, y_valid_pred)
    
    if y_test_pred is not None and y_test is not None:
        test_r2 = r2_score(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        # calculate adjusted R²
        n = len(y_test)
        p = X_test.shape[1]
        test_adj_r2 = calculate_adjusted_r2(test_r2, n, p)
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Test Adjusted R²: {test_adj_r2:.4f}")
    
    results = {
        'model': model,
        'scaler': scaler,
        'train_r2': train_r2,
        'valid_r2': valid_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'valid_mse': valid_mse,
        'test_mse': test_mse,
        'test_adj_r2': test_adj_r2,
        'predictions': {
            'train': (y_train, y_train_pred),
            'valid': (y_valid, y_valid_pred) if y_valid_pred is not None else None,
            'test': (y_test, y_test_pred) if y_test_pred is not None else None
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

def save_model(model, scaler, file_path):
    model_data = {
        'model': model,
        'scaler': scaler
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
    results = linear_model_training(X_train, X_valid, X_test, y_train, y_valid, y_test)
    if results is None:
        return None
    
    # if external test data exists, compute its metrics
    if X_external_test is not None and y_external_test is not None:
        print("Evaluating on external test data...")
        X_external_test_scaled = results['scaler'].transform(X_external_test)
        y_external_pred = results['model'].predict(X_external_test_scaled)
        y_external_pred = np.maximum(y_external_pred, 0)
        
        external_test_mse = mean_squared_error(y_external_test, y_external_pred)
        external_test_r2 = r2_score(y_external_test, y_external_pred)
        
        # calculate adjusted R²
        n_external = len(y_external_test)
        p_external = X_external_test.shape[1]
        external_test_adj_r2 = calculate_adjusted_r2(external_test_r2, n_external, p_external)
        
        print(f"External Test MSE: {external_test_mse:.4f}")
        print(f"External Test Adjusted R²: {external_test_adj_r2:.4f}")
    
    # save model
    print("Saving model...")
    save_model(results['model'], results['scaler'], 'parking_prediction_model.pkl')
    
    return results

if __name__ == "__main__":
    main()