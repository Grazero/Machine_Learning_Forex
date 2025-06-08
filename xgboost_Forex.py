import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import MetaTrader5 as mt5
from datetime import datetime

# === 1. Load Data ===
symbol = "XAUUSDm"
timeframe = mt5.TIMEFRAME_H1
n_bars = 50000

if not mt5.initialize():
    print("❌ MT5 initialize() failed")
    quit()

rates = mt5.copy_rates_from(symbol, timeframe, 0, n_bars)
if rates is None or len(rates) == 0:
    print("❌ Cannot fetch MT5 data.")
    mt5.shutdown()
    quit()

df = pd.DataFrame(rates)
df['Date'] = pd.to_datetime(df['time'], unit='s')
df = df[['Date', 'open', 'high', 'low', 'close', 'tick_volume']]
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df.set_index('Date', inplace=True)

# === 2. Feature Engineering ===
df['MA5'] = df['Close'].rolling(5).mean()
df['MA10'] = df['Close'].rolling(10).mean()
df['MA20'] = df['Close'].rolling(20).mean()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Bollinger Bands
df['BB_Middle'] = df['Close'].rolling(20).mean()
df['BB_Std'] = df['Close'].rolling(20).std()
df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

# MACD
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']

# Volume features
df['Volume_Change'] = df['Volume'].pct_change()
df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
df['Volume_Spike'] = df['Volume'] / df['Volume_MA10']

# Lag Features
for i in range(1, 4):
    df[f'Close_lag{i}'] = df['Close'].shift(i)

# Target
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Price Change Features
df['Price_Change'] = df['Close'] - df['Close_lag1']
df['Price_Change_Pct'] = df['Price_Change'] / df['Close_lag1']

# Extra features
df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
df['Candle_Range'] = df['High'] - df['Low']

# Signal interaction features
df['MACD_RSI_Interaction'] = df['MACD_Diff'] * df['RSI']
df['Price_MACD_Interaction'] = df['Price_Change'] * df['MACD_Diff']

# === 3. Prepare Dataset ===
df.dropna(inplace=True)
if df.empty:
    print("❌ No data left after dropna.")
    exit()

features = [
    'RSI', 'MACD', 'MACD_Diff',
    'BB_Upper', 'BB_Lower', 'BB_Width',
    'MA10', 'MA20', 'Close_lag2', 'Close_lag3',
    'Price_Change', 'Price_Change_Pct',
    'Candle_Range', 'Volume_Change', 'Volume_Spike',
    'MACD_RSI_Interaction', 'Price_MACD_Interaction'
]

X = df[features]
y = df['Target']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# === 4. Train XGBoost ===
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'scale_pos_weight': [0.9, 1.0, 1.1]
}

grid = GridSearchCV(
    estimator=xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    param_grid=param_grid,
    scoring='f1',  # ✅ ปรับ metric เป็น F1
    cv=3,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("✅ Best params:", grid.best_params_)
print("📊 Best F1 Score (CV):", grid.best_score_)

# Feature importance (จาก RF)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)
print("🔎 Feature Importance (RandomForest):\n", feature_importance)

# Predict & evaluate
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("✅ F1 Score:", f1_score(y_test, y_pred))
print("✅ ROC AUC:", roc_auc_score(y_test, y_prob))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

# Prediction ล่าสุด
latest = X_scaled[-1].reshape(1, -1)
latest_pred = best_model.predict_proba(latest)[0]
print(f"\n🔮 ทำนายล่าสุด: ขึ้น={latest_pred[1]:.2%}, ลง={latest_pred[0]:.2%}")
if latest_pred[1] > 0.6:
    print("🟢 แนะนำ: BUY")
elif latest_pred[0] > 0.6:
    print("🔴 แนะนำ: SELL")
else:
    print("⚪ ไม่ชัดเจน")

# บันทึกโมเดล
joblib.dump(best_model, 'rf_model_xgboost.pkl')
joblib.dump(scaler, 'scaler.pkl')