import MetaTrader5 as mt5
import pandas as pd
import ta
import joblib
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# เชื่อมต่อ MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    mt5.shutdown()
    sys.exit()

def get_timeframe_code(tf_str):
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return mapping.get(tf_str.upper(), mt5.TIMEFRAME_H1)

symbol = sys.argv[1] if len(sys.argv) > 1 else "XAUUSDm"
timeframe_str = sys.argv[2] if len(sys.argv) > 2 else "H1"
timeframe = get_timeframe_code(timeframe_str)

# ดึงข้อมูล 5000 แท่ง
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 5000)
if rates is None or len(rates) == 0:
    print("No data retrieved from MT5")
    mt5.shutdown()
    sys.exit()

data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')

# คำนวณ indicators ครบในรอบเดียว
data['ma20'] = data['close'].rolling(window=20).mean()
data['ma50'] = data['close'].rolling(window=50).mean()

data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
macd_indicator = ta.trend.MACD(close=data['close'])
data['macd'] = macd_indicator.macd()

stoch_indicator = ta.momentum.StochasticOscillator(close=data['close'], high=data['high'], low=data['low'])
data['stoch'] = stoch_indicator.stoch()

atr_indicator = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close'])
data['atr'] = atr_indicator.average_true_range()

data['price_change'] = data['close'] - data['open']

# Candle pattern
data['candle_range'] = data['high'] - data['low']
data['upper_shadow'] = data['high'] - data[['close', 'open']].max(axis=1)
data['lower_shadow'] = data[['close', 'open']].min(axis=1) - data['low']

# EMA trend
data['ema20'] = data['close'].ewm(span=20).mean()
data['ema50'] = data['close'].ewm(span=50).mean()
data['close_above_ema20'] = (data['close'] > data['ema20']).astype(int)
data['close_above_ema50'] = (data['close'] > data['ema50']).astype(int)

# Bollinger Bands
boll = ta.volatility.BollingerBands(close=data['close'], window=20)
data['bb_upper'] = boll.bollinger_hband()
data['bb_lower'] = boll.bollinger_lband()
data['bb_percent'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

# ADX
adx_indicator = ta.trend.ADXIndicator(high=data['high'], low=data['low'], close=data['close'])
data['adx'] = adx_indicator.adx()

# สร้าง target: กำไร 0.1% ขึ้นไปถือว่า "ขึ้น" (1), ต่ำกว่า -0.1% "ลง" (0)
future_return = (data['close'].shift(-3) - data['close']) / data['close']
threshold = 0.001  # 0.1%

data['target'] = np.where(future_return > threshold, 1, 0)
# ลบ row ที่ target ไม่แน่ชัด (ช่วง -threshold ถึง threshold) หรือ NaN
mask = (future_return.abs() >= threshold) & (~future_return.isna())
df = data.loc[mask].copy()

# เตรียม features และ target
features = [
    'ma20', 'ma50', 'rsi', 'macd', 'stoch', 'atr', 'price_change',
    'candle_range', 'upper_shadow', 'lower_shadow',
    'ema20', 'ema50', 'close_above_ema20', 'close_above_ema50',
    'bb_percent', 'adx'
]

X = df[features]
y = df['target']

# กำหนด TimeSeriesSplit สำหรับ cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# ตั้ง parameter grid สำหรับ tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

# สร้าง RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# สร้าง GridSearchCV
grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X, y)

print(f"Best params: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ประเมิน model ด้วย cross-validation แบบ manual (แสดง classification report ของ fold สุดท้าย)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print("\nClassification Report for one fold:")
    print(classification_report(y_test, y_pred))

# แสดง feature importance
feature_importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# Plot feature importance
feature_importances.plot(kind='bar', title='Feature Importances')
plt.tight_layout()
plt.show()

# บันทึกโมเดล
joblib.dump(best_model, 'rf_model_tuned.pkl')

# ปิด MT5
mt5.shutdown()
