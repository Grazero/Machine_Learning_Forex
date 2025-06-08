import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import MetaTrader5 as mt5 # ต้องติดตั้ง 'MetaTrader5' ด้วย pip install MetaTrader5
import json
from datetime import datetime

# --- 1. ตั้งค่าการดึงข้อมูลและเริ่มต้น MT5 ---
symbol = "XAUUSDm"
timeframe = mt5.TIMEFRAME_H1  # รายชั่วโมง
n_bars = 50000 # จำนวนแท่งเทียนที่ต้องการดึง

print(f"กำลังเชื่อมต่อกับ MT5...")
if not mt5.initialize():
    print("❌ MT5 initialize() failed. ตรวจสอบว่า MT5 Terminal เปิดอยู่และอนุญาต Algo Trading.")
    quit()
print(f"✅ เชื่อมต่อ MT5 สำเร็จ.")

# --- ดึงข้อมูล ---
print(f"กำลังดึงข้อมูล {n_bars} แท่งเทียนของ {symbol} ({timeframe})...")
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)

if rates is None or len(rates) == 0:
    print(f"❌ ไม่สามารถดึงข้อมูลจาก MT5 หรือได้ข้อมูลว่าง ({symbol}, {timeframe}).")
    mt5.shutdown()
    quit()

print(f"✅ ดึงข้อมูลได้ {len(rates)} แท่งเทียน.")

# --- แปลงข้อมูลเป็น DataFrame ---
df = pd.DataFrame(rates)
df['Date'] = pd.to_datetime(df['time'], unit='s')
df = df[['Date', 'open', 'high', 'low', 'close', 'tick_volume']]
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df.set_index('Date', inplace=True)

print(f"\n📊 ตัวอย่างข้อมูลเริ่มต้น (5 แถวแรก):\n{df.head()}")

# --- 2. Feature Engineering ---

print("\n⚙️ กำลังสร้าง Features...")

# MA (Moving Averages)
df['MA5'] = df['Close'].rolling(5).mean()
df['MA10'] = df['Close'].rolling(10).mean()
df['MA20'] = df['Close'].rolling(20).mean()

# RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0) # ต้องเป็นค่าบวกเสมอ
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

# จัดการหารด้วยศูนย์ใน RSI
with np.errstate(divide='ignore', invalid='ignore'): # ปิด warning สำหรับ division by zero
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
df['RSI'].replace([np.inf, -np.inf], np.nan, inplace=True) # แทนที่ inf ด้วย NaN

# Bollinger Bands
df['BB_Middle'] = df['Close'].rolling(20).mean()
df['BB_Std'] = df['Close'].rolling(20).std()
df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

# MACD (Moving Average Convergence Divergence)
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Diff'] = df['MACD'] - df['MACD_Signal'] # MACD - Signal (Momentum)

# Lag features (ดูย้อนหลัง) - สร้างก่อนใช้ Price Change
for i in range(1, 4):
    df[f'Close_lag{i}'] = df['Close'].shift(i)

# Price Change (absolute และ percentage)
df['Price_Change'] = df['Close'] - df['Close_lag1']
df['Price_Change_Pct'] = df['Price_Change'] / df['Close_lag1']

# Bollinger Band Width
df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

# Candle Range
df['Candle_Range'] = df['High'] - df['Low']

# Volume Spike (เปรียบเทียบกับ MA10 ของ Volume)
df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
df['Volume_Spike'] = df['Volume'] / df['Volume_MA10']
df['Volume_Change'] = df['Volume'].pct_change()

print("✅ สร้าง Features เสร็จสิ้น.")
print(f"📊 ตัวอย่างข้อมูลหลังสร้าง Features (5 แถวสุดท้าย):\n{df.tail()}")

# Target: ราคาขึ้นในแท่งถัดไป = 1, ไม่ขึ้น = 0
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
print(f"\n🎯 จำนวนคลาสใน Target:\n{df['Target'].value_counts()}")


# --- 3. เตรียมข้อมูลสำหรับโมเดล ---

print("\n🧹 กำลังเตรียมข้อมูลสำหรับโมเดล...")

print(f"📌 ขนาด DataFrame ก่อน dropna: {df.shape}")
print(f"📌 จำนวนค่าว่าง (NaN) ต่อคอลัมน์ก่อน dropna:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

df.dropna(inplace=True)  # ลบค่า NaN ทั้งหมด
print(f"✅ ขนาดข้อมูลหลัง dropna: {df.shape}")

if df.empty:
    print("❌ DataFrame เป็นค่าว่างหลังจากลบ NaN. ลองดึงข้อมูลแท่งเทียนให้มากขึ้น.")
    mt5.shutdown()
    exit()

# กำหนด Features ที่จะใช้
features = [
    'RSI', 'MACD', 'MACD_Diff',
    'BB_Upper', 'BB_Lower', 'BB_Width',
    'MA20', 'MA10', 'MA5', # เพิ่ม MA5 เข้าไป
    'Close_lag1', 'Close_lag2', 'Close_lag3', # ใช้ lag 1, 2, 3
    'Price_Change', 'Price_Change_Pct',
    'Candle_Range', 'Volume_Change', 'Volume_Spike'
]

# ตรวจสอบว่าทุก feature ที่เลือกมีอยู่ใน DataFrame
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"❌ Features ที่ขาดหายไป: {missing_features}")
    print("โปรดตรวจสอบการสร้าง features หรือชื่อคอลัมน์")
    exit()

X = df[features]
y = df['Target']

# Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features ถูก Normalize แล้ว.")

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)
print(f"✅ แบ่งข้อมูลเป็น Train ({len(X_train)} แถว) และ Test ({len(X_test)} แถว) แล้ว.")

# --- 4. เทรนโมเดล XGBoost (ปรับปรุงสำหรับ XGBoost 3.0.1) ---

print("\n🚀 กำลังเทรนโมเดล XGBoost...")

# สำหรับ XGBoost 3.0.1 เราจะกำหนด eval_metric ใน Constructor และ XGBoost จะใช้ Early Stopping อัตโนมัติ
# เมื่อใช้ predict() หรือ predict_proba()
model_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss', # กำหนด eval_metric ตรงนี้
    scale_pos_weight=1.0, # ควรปรับค่านี้หาก target class ไม่สมดุล
    n_estimators=1000,    # ตั้ง n_estimators ให้สูงพอที่จะให้ Early Stopping ทำงาน
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# *** แก้ไขตรงนี้: ลบ early_stopping_rounds และ callbacks ออก ***
# สำหรับ XGBoost 3.0.1+ เพียงแค่ส่ง eval_set ไป
# XGBoost จะจัดการ Early Stopping ภายในโดยอัตโนมัติเมื่อมีการเรียก predict() หรือ predict_proba()
print("กำลังเทรน XGBoost (ใช้ Early Stopping อัตโนมัติใน XGBoost 3.0.1+)...")
model_xgb.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], # ใช้ X_test, y_test เป็น validation set
              verbose=False # ไม่ต้องแสดงผลทุกรอบการฝึก
             )
print("✅ เทรน XGBoost เสร็จสิ้น.")


# --- 5. Hyperparameter Tuning ด้วย GridSearchCV (ปรับปรุงสำหรับ XGBoost 3.0.1) ---

print("\n🔍 กำลังค้นหา Hyperparameters ที่ดีที่สุดด้วย GridSearchCV...")

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200], # n_estimators ใน GridSearch มักจะไม่ต้องการ Early Stopping ในตัว
    'scale_pos_weight': [0.9, 1.0, 1.1]
}

# สำหรับ GridSearchCV ใน XGBoost 3.0.1+ จะจัดการ eval_metric เองจาก estimator
grid_search_xgb = GridSearchCV(
    estimator=xgb.XGBClassifier(
        objective='binary:logistic',
        # ไม่จำเป็นต้องกำหนด eval_metric ซ้ำที่นี่ ถ้ากำหนดใน estimator แล้ว
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    param_grid=param_grid,
    scoring='accuracy', # เน้น accuracy
    cv=3, # Cross-validation folds
    verbose=1, # แสดงผลระหว่าง Grid Search
    n_jobs=-1 # ใช้ทุก core CPU เพื่อความเร็ว
)

grid_search_xgb.fit(X_train, y_train)

print("✅ ค้นหา Hyperparameters เสร็จสิ้น.")
print(f"✨ Best params: {grid_search_xgb.best_params_}")
print(f"🎯 Best accuracy (จาก CV): {grid_search_xgb.best_score_:.4f}")

best_model = grid_search_xgb.best_estimator_ # โมเดลที่ดีที่สุดจาก Grid Search

# --- 6. เทรนโมเดล RandomForestClassifier (เป็นทางเลือก) ---
# คุณอาจจะเลือกใช้เพียงโมเดลเดียว (XGBoost) หรือเปรียบเทียบประสิทธิภาพ
print("\n🌳 กำลังเทรนโมเดล RandomForestClassifier (สำหรับ Feature Importance)...")
model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
model_rf.fit(X_train, y_train)
print("✅ เทรน RandomForestClassifier เสร็จสิ้น.")

# --- 7. ประเมินโมเดลและ Feature Importance ---

print("\n📈 กำลังประเมินโมเดล...")

# ประเมินผลจาก best_model (ซึ่งมาจาก XGBoost)
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1] # ใช้ best_model ในการทำนายความน่าจะเป็น

print(f"\n📊 F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"📊 ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance จาก RandomForest (เป็นตัวอย่าง)
print("\n🔍 Feature Importance (จาก RandomForest):\n")
importances_rf = model_rf.feature_importances_
feature_importance_rf = pd.Series(importances_rf, index=features).sort_values(ascending=False)
print(feature_importance_rf)

# Feature Importance จาก XGBoost (best_model)
print("\n🔍 Feature Importance (จาก XGBoost Best Model):\n")
importances_xgb = best_model.feature_importances_
feature_importance_xgb = pd.Series(importances_xgb, index=features).sort_values(ascending=False)
print(feature_importance_xgb)


# --- 8. ทำนายราคาล่าสุด ---

print("\n🔮 กำลังทำนายทิศทางราคาแท่งเทียนถัดไป...")

# เตรียมข้อมูลแท่งเทียนล่าสุด
# ต้องดึงข้อมูลล่าสุด 3-4 แท่ง เพื่อให้มีข้อมูลสำหรับ lag features
# ดึงเผื่อ MA/RSI/BB ที่ต้องใช้ข้อมูลย้อนหลัง 20 แท่ง (หรือมากกว่า)
required_bars = max(20, 14, 3) + 1 # max period + 1 for current bar
latest_rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, required_bars)

if latest_rates is None or len(latest_rates) < required_bars:
    print("❌ ไม่สามารถดึงข้อมูลล่าสุดได้เพียงพอสำหรับการทำนาย.")
    mt5.shutdown()
    exit()

latest_df = pd.DataFrame(latest_rates)
latest_df['Date'] = pd.to_datetime(latest_df['time'], unit='s')
latest_df = latest_df[['Date', 'open', 'high', 'low', 'close', 'tick_volume']]
latest_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
latest_df.set_index('Date', inplace=True)

# สร้าง Features สำหรับข้อมูลล่าสุด (ต้องเหมือนตอนเทรน)
# MA (Moving Averages)
latest_df['MA5'] = latest_df['Close'].rolling(5).mean()
latest_df['MA10'] = latest_df['Close'].rolling(10).mean()
latest_df['MA20'] = latest_df['Close'].rolling(20).mean()

# RSI (Relative Strength Index)
delta_latest = latest_df['Close'].diff()
gain_latest = delta_latest.where(delta_latest > 0, 0.0)
loss_latest = -delta_latest.where(delta_latest < 0, 0.0)
avg_gain_latest = gain_latest.rolling(window=14).mean()
avg_loss_latest = loss_latest.rolling(window=14).mean()
with np.errstate(divide='ignore', invalid='ignore'):
    rs_latest = avg_gain_latest / avg_loss_latest
    latest_df['RSI'] = 100 - (100 / (1 + rs_latest))
latest_df['RSI'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Bollinger Bands
latest_df['BB_Middle'] = latest_df['Close'].rolling(20).mean()
latest_df['BB_Std'] = latest_df['Close'].rolling(20).std()
latest_df['BB_Upper'] = latest_df['BB_Middle'] + 2 * latest_df['BB_Std']
latest_df['BB_Lower'] = latest_df['BB_Middle'] - 2 * latest_df['BB_Std']

# MACD (Moving Average Convergence Divergence)
ema12_latest = latest_df['Close'].ewm(span=12, adjust=False).mean()
ema26_latest = latest_df['Close'].ewm(span=26, adjust=False).mean()
latest_df['MACD'] = ema12_latest - ema26_latest
latest_df['MACD_Signal'] = latest_df['MACD'].ewm(span=9, adjust=False).mean()
latest_df['MACD_Diff'] = latest_df['MACD'] - latest_df['MACD_Signal']

# Lag features
for i in range(1, 4):
    latest_df[f'Close_lag{i}'] = latest_df['Close'].shift(i)

# Price Change
latest_df['Price_Change'] = latest_df['Close'] - latest_df['Close_lag1']
latest_df['Price_Change_Pct'] = latest_df['Price_Change'] / latest_df['Close_lag1']

# Bollinger Band Width
latest_df['BB_Width'] = latest_df['BB_Upper'] - latest_df['BB_Lower']
latest_df['Candle_Range'] = latest_df['High'] - latest_df['Low']

# Volume Spike
latest_df['Volume_MA10'] = latest_df['Volume'].rolling(window=10).mean()
latest_df['Volume_Spike'] = latest_df['Volume'] / latest_df['Volume_MA10']
latest_df['Volume_Change'] = latest_df['Volume'].pct_change()


# ดึงข้อมูลล่าสุด (แท่งเทียนปัจจุบัน) - ต้องเป็นแถวสุดท้ายที่ไม่มี NaN
# ก่อนอื่น dropna() แล้วค่อยเลือกแถวสุดท้าย
latest_df.dropna(inplace=True)
if latest_df.empty:
    print("❌ ไม่สามารถคำนวณ Features สำหรับข้อมูลล่าสุดได้ (หลัง dropna).")
    mt5.shutdown()
    exit()

latest_features_row = latest_df[features].iloc[-1:].copy()

# Scale ข้อมูลล่าสุดด้วย scaler ตัวเดิมที่ใช้ตอนเทรน
latest_scaled = scaler.transform(latest_features_row)

# ทำนาย
latest_pred_proba = best_model.predict_proba(latest_scaled)[0]
print(f"\n🔮 ทำนายล่าสุด: ขึ้น={latest_pred_proba[1]:.2%}, ลง={latest_pred_proba[0]:.2%}")

# เงื่อนไขการแนะนำ (ปรับ threshold ได้ตามต้องการ)
buy_threshold = 0.60
sell_threshold = 0.60

if latest_pred_proba[1] > buy_threshold:
    print("🟢 แนะนำ: BUY")
elif latest_pred_proba[0] > sell_threshold:
    print("🔴 แนะนำ: SELL")
else:
    print("⚪ ไม่ชัดเจน")

# --- ปิดการเชื่อมต่อ MT5 ---
mt5.shutdown()
print("\n✅ ปิดการเชื่อมต่อ MT5 แล้ว.")