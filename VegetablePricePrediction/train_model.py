"""Train and compare regression models for Maharashtra vegetable market prices."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / 'dataset'
MODEL_DIR = BASE / 'model'
REPORT_DIR = BASE / 'reports'
TARGET = 'Modal_Price'
CAT_COLS = ['Commodity', 'District', 'Market', 'Grade', 'Variety']
NUM_COLS = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'DayOfYearSin', 'DayOfYearCos']
FEATURES = CAT_COLS + NUM_COLS


def load_data():
    files = sorted(DATA_DIR.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No CSV files found in {DATA_DIR}. Put all 9 CSV files in the dataset folder.')
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    required = ['Arrival_Date', 'Commodity', 'District', 'Market', 'Grade', 'Variety', TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], dayfirst=True, errors='coerce')
    data[TARGET] = pd.to_numeric(data[TARGET], errors='coerce')
    data = data.dropna(subset=['Arrival_Date', TARGET]).copy()
    data = data[data[TARGET] > 0].copy()
    for col in CAT_COLS:
        data[col] = data[col].fillna('Unknown').astype(str).str.strip()
        data.loc[data[col].eq(''), col] = 'Unknown'
    data = data.sort_values('Arrival_Date').reset_index(drop=True)
    return data, len(files)


def add_features(data):
    df = data.copy()
    dt = df['Arrival_Date']
    df['Year'] = dt.dt.year
    df['Month'] = dt.dt.month
    df['Day'] = dt.dt.day
    df['DayOfWeek'] = dt.dt.dayofweek
    df['DayOfYear'] = dt.dt.dayofyear
    df['DayOfYearSin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['DayOfYearCos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    return df


def build_pipeline(model):
    prep = ColumnTransformer([
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CAT_COLS),
        ('numeric', SimpleImputer(strategy='median'), NUM_COLS),
    ])
    return Pipeline([('preprocess', prep), ('regressor', model)])


def main():
    warnings.filterwarnings('ignore', category=UserWarning)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    data, file_count = load_data()
    data = add_features(data)
    if len(data) < 20:
        raise ValueError('Not enough valid rows to train a model.')

    # Time-based split: older records train the models; newest 20% test them.
    split = int(len(data) * 0.8)
    train_df, test_df = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    candidates = {
        'Random Forest': RandomForestRegressor(n_estimators=220, min_samples_leaf=2,
                                               random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=220, min_samples_leaf=2,
                                           random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeRegressor(min_samples_leaf=5, random_state=42),
    }
    results, fitted = [], {}
    for name, estimator in candidates.items():
        print(f'Training {name}...')
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        pred = np.maximum(0, pipe.predict(X_test))
        results.append({'Model': name,
                        'MAE': mean_absolute_error(y_test, pred),
                        'RMSE': mean_squared_error(y_test, pred) ** 0.5,
                        'R2': r2_score(y_test, pred)})
        fitted[name] = pipe

    scores = pd.DataFrame(results).sort_values('MAE')
    best_name = scores.iloc[0]['Model']
    best_model = fitted[best_name]
    bundle = {'model': best_model, 'features': FEATURES, 'categorical_features': CAT_COLS,
              'numeric_features': NUM_COLS, 'target': TARGET, 'model_name': best_name,
              'training_rows': int(len(train_df)), 'test_rows': int(len(test_df)),
              'train_max_date': str(train_df['Arrival_Date'].max().date()),
              'test_min_date': str(test_df['Arrival_Date'].min().date())}
    joblib.dump(bundle, MODEL_DIR / 'vegetable_price_model.joblib')
    scores.to_csv(REPORT_DIR / 'model_comparison.csv', index=False)

    # Save a clean merged copy for reproducibility and inspection.
    data.drop(columns=NUM_COLS, errors='ignore').to_csv(REPORT_DIR / 'combined_cleaned_dataset.csv', index=False)
    summary = {
        'source_csv_files': file_count, 'valid_records': int(len(data)),
        'duplicate_rows': int(data.duplicated().sum()),
        'missing_values_after_cleaning': int(data[required_columns(data)].isna().sum().sum()) if False else 0,
        'date_start': str(data['Arrival_Date'].min().date()),
        'date_end': str(data['Arrival_Date'].max().date()),
        'commodities': sorted(data['Commodity'].unique().tolist()),
        'districts': sorted(data['District'].unique().tolist()),
        'best_model': best_name,
        'metrics': scores.to_dict(orient='records'),
        'note': 'Predictions are estimates from historical records, not guaranteed market quotations.'
    }
    (REPORT_DIR / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nModel comparison (lower MAE/RMSE is better; higher R2 is better):')
    print(scores.to_string(index=False))
    print(f'\nBest model: {best_name}')
    print(f'Model saved to: {MODEL_DIR / "vegetable_price_model.joblib"}')
    print('Reports saved in the reports folder.')


def required_columns(data):
    return ['Arrival_Date', 'Commodity', 'District', 'Market', 'Grade', 'Variety', TARGET]

if __name__ == '__main__':
    main()
