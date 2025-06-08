import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange # นำเข้า ATR
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# STEP 1: ดึงข้อมูลจาก MT5
symbol = 'XAUUSDm'
timeframe = mt5.TIMEFRAME_M15  # Timeframe 15 นาที
n_bars = 90000  # จำนวนแท่งเทียนที่ต้องการดึง

# ตรวจสอบและเริ่มต้น MT5
if not mt5.initialize():
    print("MT5 initialization failed", mt5.last_error())
    quit()

# ดึงข้อมูลราคาจาก MT5
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
mt5.shutdown() # ปิดการเชื่อมต่อ MT5 เมื่อดึงข้อมูลเสร็จ

# แปลงข้อมูลเป็น DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s') # แปลง timestamp เป็น datetime
df.set_index('time', inplace=True) # ตั้งคอลัมน์ 'time' เป็น index

# STEP 2: สร้างฟีเจอร์อินดิเคเตอร์ (เน้นฟีเจอร์ที่ตอบสนองไวและสำคัญ)

# อินดิเคเตอร์ Momentum
df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi() # RSI
df['Stoch_K'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3).stoch() # Stochastic %K
df['Stoch_D'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3).stoch_signal() # Stochastic %D

# อินดิเคเตอร์ Trend (EMA)
df['EMA_fast'] = EMAIndicator(close=df['close'], window=5).ema_indicator() # EMA 5
df['EMA_slow'] = EMAIndicator(close=df['close'], window=20).ema_indicator() # EMA 20
df['EMA_diff'] = df['EMA_fast'] - df['EMA_slow'] # ผลต่าง EMA (บอกแนวโน้ม)

# อินดิเคเตอร์ MACD
macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_hist'] = df['MACD'] - df['MACD_signal'] # MACD Histogram (บอก Momentum การเปลี่ยนแปลง)

# อินดิเคเตอร์ Volatility
df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range() # Average True Range

# ฟีเจอร์ Price Action (การเคลื่อนไหวของแท่งเทียน)
df['Price_change'] = df['close'] - df['open'] # การเปลี่ยนแปลงราคาในแท่งเดียว
df['Body_size'] = abs(df['close'] - df['open']) # ขนาดตัวแท่งเทียน
df['Upper_shadow'] = df['high'] - df[['close','open']].max(axis=1) # ขนาดเงาบน
df['Lower_shadow'] = df[['close','open']].min(axis=1) - df['low'] # ขนาดเงาล่าง

# ฟีเจอร์อัตราส่วนเงา (อาจช่วยบอกแรงซื้อ/แรงขาย)
# ป้องกันหารด้วยศูนย์ ถ้า Body_size เป็น 0 ให้เป็น 0
df['Upper_shadow_ratio'] = np.where(df['Body_size'] != 0, df['Upper_shadow'] / df['Body_size'], 0)
df['Lower_shadow_ratio'] = np.where(df['Body_size'] != 0, df['Lower_shadow'] / df['Body_size'], 0)

# ฟีเจอร์การเปลี่ยนแปลงราคาแบบรวดเร็ว
df['return_1'] = df['close'].pct_change(1) # ผลตอบแทน 1 แท่ง

# ฟีเจอร์ Lagged (ค่าอินดิเคเตอร์ย้อนหลัง 1 แท่ง)
df['RSI_lag1'] = df['RSI'].shift(1)
df['MACD_hist_lag1'] = df['MACD_hist'].shift(1)
df['ATR_lag1'] = df['ATR'].shift(1)
df['Stoch_K_lag1'] = df['Stoch_K'].shift(1)
df['Stoch_D_lag1'] = df['Stoch_D'].shift(1)
df['EMA_fast_lag1'] = df['EMA_fast'].shift(1)
df['EMA_slow_lag1'] = df['EMA_slow'].shift(1)
df['close_lag1'] = df['close'].shift(1) # ราคาปิดแท่งที่แล้ว

# ฟีเจอร์ Volume (เน้นค่าเฉลี่ยเพื่อให้เรียบขึ้น)
df['volume_avg'] = df['tick_volume'].rolling(window=20).mean() # ค่าเฉลี่ย Volume

# ลบแถวที่มีค่า NaN ที่เกิดจากการคำนวณอินดิเคเตอร์
df.dropna(inplace=True)

# STEP 3: สร้าง Target (ขึ้นหรือลงเกิน 0.5%)
df['future_return'] = df['close'].shift(-3) / df['close'] - 1
# เปลี่ยน Target Definition กลับมาใช้ Threshold 0.005
df['target'] = np.where(df['future_return'] > 0.005, 1, # ถ้า future_return > 0.5% คือขึ้น (1)
                        np.where(df['future_return'] < -0.005, 0, np.nan)) # ถ้า future_return < -0.5% คือลง (0), ถ้าอยู่ระหว่างนั้นคือ NaN

# ลบแถวที่มีค่า NaN ที่เกิดจากการสร้าง target (ตอนนี้ข้อมูลจะน้อยลงมาก)
df.dropna(inplace=True)

# STEP 4: เตรียมชุดข้อมูลสำหรับเทรนโมเดล
# เลือกฟีเจอร์ที่ใช้ในการเทรน (ชุดใหม่ที่กระชับและเน้นประสิทธิภาพ)
features = [
    'open', 'high', 'low', 'close', 'tick_volume', # ราคาและ Volume พื้นฐาน
    'RSI', 'Stoch_K', 'Stoch_D', # Momentum indicators
    'EMA_fast', 'EMA_slow', 'EMA_diff', # Trend indicators
    'MACD', 'MACD_signal', 'MACD_hist', # MACD indicators
    'ATR', # Volatility
    'Price_change', 'Body_size', 'Upper_shadow', 'Lower_shadow', # Price Action
    'Upper_shadow_ratio', 'Lower_shadow_ratio', # Price Action Ratios
    'return_1', # Quick price change
    'RSI_lag1', 'MACD_hist_lag1', 'ATR_lag1', # Lagged key indicators
    'Stoch_K_lag1', 'Stoch_D_lag1', 'EMA_fast_lag1', 'EMA_slow_lag1', 'close_lag1', # Lagged other indicators
    'volume_avg' # Smoothed Volume
]

X = df[features]
y = df['target'].astype(int)

# แบ่งข้อมูลเป็น Train (70%), Validation (15%), Test (15%) โดยไม่สับเปลี่ยน (shuffle=False)
# เพื่อรักษาลำดับเวลาของข้อมูล
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# คำนวณอัตราส่วนของคลาสสำหรับ scale_pos_weight
# เพื่อจัดการกับ Class Imbalance (ถ้ามี)
neg_count = y_train.value_counts()[0] # จำนวน Class 0 (ลง)
pos_count = y_train.value_counts()[1] # จำนวน Class 1 (ขึ้น)

# ปรับ scale_pos_weight: เนื่องจาก Class 1 มีจำนวนมากกว่า Class 0 เล็กน้อย (ตามผลลัพธ์ที่คุณให้มา)
# เราจะใช้ neg_count / pos_count เพื่อให้น้ำหนักกับ Class 0 มากขึ้นเล็กน้อย
scale_pos_weight_value = neg_count / pos_count
print(f"Calculated Scale Pos Weight (neg_count / pos_count): {scale_pos_weight_value}")


# STEP 5: สร้างและเทรน XGBoost Classifier
model = XGBClassifier(
    n_estimators=1000,       # กลับมาใช้ 1000 ตามโค้ดเดิมของคุณ
    max_depth=5,             # กลับมาใช้ 5 ตามโค้ดเดิมของคุณ
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight_value, # ใช้ค่าที่คำนวณได้
    eval_metric='logloss',   # ใช้ logloss เป็นเกณฑ์การประเมินประสิทธิภาพบน eval_set
    random_state=42,
    use_label_encoder=False  # แก้ไข warning สำหรับ XGBoost เวอร์ชันใหม่
)

# เทรนโมเดล โดย XGBoost จะทำ Early Stopping อัตโนมัติเมื่อประสิทธิภาพบน eval_set ไม่ดีขึ้น
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True # แสดงผลลัพธ์ระหว่างการเทรน
)

# แสดงสัดส่วนของคลาสในชุดฝึก (y_train)
print("--- y_train value counts ---")
print(y_train.value_counts())
print("-" * 25)

# STEP 6: ประเมินผลโมเดลบนชุด Test
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("🎯 Accuracy:", round(acc, 4))
print(classification_report(y_test, y_pred))

# STEP 7: ทำนายแท่งถัดไป (ใช้ข้อมูลล่าสุด)
latest = df[features].iloc[-1:]
prediction = model.predict(latest)[0]
print("📈 คาดว่าแท่งถัดไปจะ", "⬆️ ขึ้น" if prediction == 1 else "⬇️ ลง")

# เพิ่ม column ดูว่าโมเดลถูกมั้ย และคำนวณผลตอบแทนถ้าทำตามสัญญาณ
df_test = df.loc[y_test.index].copy()

# เพิ่มคอลัมน์ y_pred
df_test['y_pred'] = y_pred

# คำนวณผลตอบแทนตามกลยุทธ์ (ถ้าทำนายขึ้นแล้วขึ้น ก็ได้กำไร ถ้าทำนายลงแล้วลงก็ได้กำไร)
df_test['strategy_return'] = np.where(
    df_test['y_pred'] == 1, df_test['future_return'], # ทำนายขึ้น (1), ถ้าขึ้นจริงได้ future_return
    np.where(df_test['y_pred'] == 0, -df_test['future_return'], 0) # ทำนายลง (0), ถ้าลงจริงได้ -future_return (กำไร)
)

# คำนวณผลตอบแทนสะสมของกลยุทธ์
cumulative_return = (1 + df_test['strategy_return']).cumprod()
final_return = cumulative_return.iloc[-1]

# คำนวณผลตอบแทนแบบ Buy & Hold (เทียบกับกลยุทธ์)
df_test['buy_and_hold'] = df_test['future_return']
baseline_return = (1 + df_test['buy_and_hold']).cumprod().iloc[-1]
print("📊 Buy & Hold Return:", round(baseline_return, 4))

print("💹 ผลตอบแทนรวมจากกลยุทธ์:", round(final_return, 4))
print("📈 กลยุทธ์นี้", "✅ มีกำไร" if final_return > 1 else "❌ ขาดทุน")

# Plot ผลตอบแทนสะสมเปรียบเทียบ
plt.figure(figsize=(12, 6))
(1 + df_test['strategy_return']).cumprod().plot(label='Strategy')
(1 + df_test['future_return']).cumprod().plot(label='Buy & Hold')
plt.title('Strategy vs Buy & Hold Cumulative Return')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

# STEP 8: บันทึกโมเดล
joblib.dump(model, 'xgboost_short_term_model.pkl')
print("✅ โมเดลถูกบันทึกแล้วในชื่อ 'xgboost_short_term_model.pkl'")
