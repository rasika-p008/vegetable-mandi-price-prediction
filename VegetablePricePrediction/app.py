from pathlib import Path
from datetime import date
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'model' / 'vegetable_price_model.joblib'
DATA_DIR = BASE / 'dataset'

st.set_page_config(page_title='Vegetable Price Predictor', page_icon='🥬', layout='wide')
st.title('🥬 Vegetable Price Prediction')
st.caption('Machine learning based estimate of Maharashtra agricultural market modal prices')

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_options():
    frames = [pd.read_csv(p) for p in sorted(DATA_DIR.glob('*.csv'))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ['Commodity', 'District', 'Market', 'Grade', 'Variety']:
        if col in df:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()
    return df

if not MODEL_PATH.exists():
    st.warning('The trained model has not been created yet.')
    st.code('pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py', language='bash')
    st.stop()

bundle = load_bundle()
data = load_options()
if data.empty:
    st.error('Dataset CSV files were not found. Keep all CSV files in the dataset folder.')
    st.stop()

st.success(f"Loaded model: {bundle['model_name']}")
st.write('Choose the market details and date to estimate the modal price.')

commodities = sorted(data['Commodity'].dropna().astype(str).unique())
col1, col2 = st.columns(2)
with col1:
    commodity = st.selectbox('Vegetable', commodities)
    districts = sorted(data.loc[data['Commodity'].astype(str) == commodity, 'District'].dropna().astype(str).unique())
    district = st.selectbox('District', districts)
    market_values = sorted(data.loc[(data['Commodity'].astype(str) == commodity) &
                                    (data['District'].astype(str) == district), 'Market'].dropna().astype(str).unique())
    market = st.selectbox('Market', market_values)
with col2:
    date_value = st.date_input('Prediction date', value=date.today(), min_value=date(2021, 1, 1), max_value=date(2035, 12, 31))
    subset = data[(data['Commodity'].astype(str) == commodity) &
                  (data['District'].astype(str) == district) &
                  (data['Market'].astype(str) == market)]
    grade_values = sorted(subset['Grade'].dropna().astype(str).unique()) if 'Grade' in subset else ['Unknown']
    variety_values = sorted(subset['Variety'].dropna().astype(str).unique()) if 'Variety' in subset else ['Unknown']
    grade = st.selectbox('Grade', grade_values or ['Unknown'])
    variety = st.selectbox('Variety', variety_values or ['Unknown'])

if st.button('Predict Vegetable Price', type='primary', use_container_width=True):
    dt = pd.Timestamp(date_value)
    doy = dt.dayofyear
    row = pd.DataFrame([{
        'Commodity': commodity, 'District': district, 'Market': market,
        'Grade': grade, 'Variety': variety,
        'Year': dt.year, 'Month': dt.month, 'Day': dt.day,
        'DayOfWeek': dt.dayofweek, 'DayOfYear': doy,
        'DayOfYearSin': np.sin(2 * np.pi * doy / 365.25),
        'DayOfYearCos': np.cos(2 * np.pi * doy / 365.25)
    }])
    prediction = max(0.0, float(bundle['model'].predict(row)[0]))
    st.divider()
    st.subheader('Estimated Modal Price')
    st.metric('Predicted price', f'₹ {prediction:,.2f} / quintal')
    st.info('This is a machine-learning estimate based on historical patterns. Actual prices can differ due to supply, demand, weather, quality and market conditions.')

st.divider()
st.subheader('Explore the historical dataset')
if 'Modal_Price' in data:
    chart_data = data.copy()
    chart_data['Arrival_Date'] = pd.to_datetime(chart_data['Arrival_Date'], dayfirst=True, errors='coerce')
    chart_data['Modal_Price'] = pd.to_numeric(chart_data['Modal_Price'], errors='coerce')
    chart_data = chart_data.dropna(subset=['Arrival_Date', 'Modal_Price'])
    chosen = chart_data[chart_data['Commodity'].astype(str) == commodity]
    if not chosen.empty:
        daily = chosen.groupby('Arrival_Date')['Modal_Price'].mean().sort_index()
        st.line_chart(daily)
        st.caption(f'Average recorded modal price trend for {commodity}.')

with st.expander('About the model and dataset'):
    st.write(f"Model: {bundle['model_name']}")
    st.write(f"Training records: {bundle['training_rows']:,}; test records: {bundle['test_rows']:,}.")
    st.write('The model uses vegetable, district, market, grade, variety and date-derived features. Same-day minimum and maximum prices are intentionally excluded because they may not be known before the prediction date.')
