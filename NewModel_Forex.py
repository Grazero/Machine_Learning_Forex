import pandas as pd
import numpy as np
import xgboost as xgb
import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import ta

# 1. ตั้งค่าพารามิเตอร์เบื้องต้น
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M15
TIMEZONE = pytz.timezone("Etc/UTC")
DATE_TO = datetime.now(TIMEZONE)
DATE_FROM = DATE_TO - timedelta(days=365)

# 2. เชื่อมต่อ MT5 และดึงข้อมูล
if not mt5.initialize():
    print("❌ ไม่สามารถเชื่อมต่อ MT5 ได้")
    quit()

if not mt5.symbol_select(SYMBOL, True):
    print(f"❌ เลือก symbol {SYMBOL} ไม่ได้")
    mt5.shutdown()
    quit()

rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, DATE_FROM, DATE_TO)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("❌ ดึงข้อมูลไม่สำเร็จ")
    quit()

# 3. เตรียม DataFrame
df = pd.DataFrame(rates)
df['Date'] = pd.to_datetime(df['time'], unit='s')
df = df[['Date', 'open', 'high', 'low', 'close', 'tick_volume']]
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# 4. สร้าง Indicators ด้วย ta-lib
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Hist'] = macd.macd_diff()

df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

rolling_std = df['Close'].rolling(window=20).std()
df['Bollinger_High'] = df['SMA_20'] + 2 * rolling_std
df['Bollinger_Low'] = df['SMA_20'] - 2 * rolling_std

stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
df['Stoch_K'] = stoch.stoch()
df['Stoch_D'] = stoch.stoch_signal()

df.dropna(inplace=True)

# 5. สร้างเป้าหมาย (Target) สำหรับการทำนายแนวโน้มใน 3 แท่งถัดไป
future_close = df['Close'].shift(-3)
df['Target'] = np.where(future_close > df['Close'] * 1.002, 2,  # Buy
                        np.where(future_close < df['Close'] * 0.998, 0, 1))  # Sell =0, Hold=1
df.dropna(inplace=True)

# 6. เตรียมข้อมูล features และ target
FEATURES = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'EMA_50',
            'Bollinger_High', 'Bollinger_Low', 'Stoch_K', 'Stoch_D']

X = df[FEATURES]
y = df['Target'].astype(int)

# 7. Scaling feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. แบ่งข้อมูล train-test แบบ time series (ไม่สุ่ม)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# 9. แก้ปัญหาข้อมูลไม่สมดุลด้วย SMOTE เฉพาะ train
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 10. สร้างโมเดล XGBoost และเทรนด้วย early stopping
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.02,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model.fit(
    X_train_res,
    y_train_res,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=30,
    verbose=True
)

# 11. ทำนายและประเมินผล
y_pred = model.predict(X_test)

print("📊 Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report")
print(classification_report(y_test, y_pred, digits=4))

# 12. ทำนายสัญญาณสำหรับข้อมูลทั้งหมด (เพื่อดูสัญญาณล่าสุด)
df['Prediction'] = model.predict(X_scaled)

# 13. แสดงสัญญาณล่าสุด
latest = df.iloc[-1]
signal_map = {0: "📉 Sell", 1: "⏸️ Hold", 2: "📈 Buy"}

print("\n📈 สัญญาณล่าสุด")
print(f"วันที่: {latest['Date']}")
print(f"ราคาปิด: {latest['Close']:.4f}")
print(f"สัญญาณ: {signal_map.get(latest['Prediction'], 'Unknown')}")
