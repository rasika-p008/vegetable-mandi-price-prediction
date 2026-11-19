# Vegetable Price Prediction Using Machine Learning

A VS Code + Python + Streamlit project that estimates the historical market **Modal_Price** (₹/quintal) for vegetables in Maharashtra.

## Included dataset

The `dataset/` folder contains the nine CSV files supplied for this project. They have the same 11-column structure. The combined source data contains 48,097 rows (before cleaning), covering 1 January 2021 to 31 December 2025. The data includes Onion, Tomato, Potato and Coriander(Leaves), and districts Pune, Nashik and Mumbai.

## Requirements

- Python 3.10 or newer
- Visual Studio Code (recommended)
- Internet connection for the first package installation

## Run on Windows (VS Code)

1. Extract the ZIP file.
2. Open the extracted `VegetablePricePrediction` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\activate
```

5. Install packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Train and compare the models:

```powershell
python train_model.py
```

This creates `model/vegetable_price_model.joblib` and saves model metrics and a cleaned combined dataset in `reports/`.

7. Start the Streamlit website:

```powershell
streamlit run app.py
```

The app opens in your browser, normally at `http://localhost:8501`.

## Project structure

```text
VegetablePricePrediction/
├── dataset/                 # 9 supplied CSV files
├── model/                   # trained model is generated here
├── reports/                 # model comparison and cleaned data generated here
├── app.py                   # Streamlit user interface
├── train_model.py           # data preparation, model training and evaluation
├── requirements.txt
└── README.md
```

## Methodology

- Merge all CSV files and parse `Arrival_Date` using day-first date parsing.
- Remove rows with invalid dates or non-positive/missing target values.
- Predict `Modal_Price` using commodity, district, market, grade, variety and calendar features.
- Exclude `Min_Price` and `Max_Price` from model inputs because they may not be available at the time of a future prediction.
- Use a chronological 80/20 train/test split (older observations for training, newest observations for testing).
- Compare Decision Tree, Random Forest and Extra Trees regressors using MAE, RMSE and R²; select the model with the lowest test MAE.

## Important limitations

This is a baseline date-and-market-feature model, not a complete supply-demand forecasting system. It does not currently use weather, arrivals/quantity, holidays, transport costs or lagged prices. The data contains very different price scales for some commodities/records, so inspect `reports/combined_cleaned_dataset.csv` and model metrics before presenting results. Predictions are estimates, not guaranteed live market prices. Forecasting dates far beyond the training period is especially uncertain.
