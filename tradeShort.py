# tradeShort.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# STEP 1: ดึงข้อมูลจาก MT5
symbol = 'XAUUSDm'
timeframe = mt5.TIMEFRAME_M15  # M15 = 15 นาที
n_bars = 90000  # หรือมากกว่านี้ก็ได้

if not mt5.initialize():
    print("MT5 initialization failed", mt5.last_error())
    quit()

rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
mt5.shutdown()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.set_index('time', inplace=True)


# STEP 2: ฟีเจอร์อินดิเคเตอร์
df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
df['EMA_fast'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
df['EMA_slow'] = EMAIndicator(close=df['close'], window=20).ema_indicator()

macd = MACD(close=df['close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_hist'] = df['MACD'] - df['MACD_signal'] # เพิ่ม MACD Histogram

stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
df['Stoch_K'] = stoch.stoch()
df['Stoch_D'] = stoch.stoch_signal()

bb = BollingerBands(close=df['close'], window=20, window_dev=2)
df['BB_upper'] = bb.bollinger_hband()
df['BB_lower'] = bb.bollinger_lband()
df['BB_width'] = df['BB_upper'] - df['BB_lower']
df['BB_percent_b'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband()) # เพิ่ม %B

df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range() # ATR ย้ายมาตรงนี้

# แรงซื้อ-แรงขายเร็ว (Price Action Features)
df['Price_change'] = df['close'] - df['open']
df['Body_size'] = abs(df['close'] - df['open'])
df['Upper_shadow'] = df['high'] - df[['close','open']].max(axis=1)
df['Lower_shadow'] = df[['close','open']].min(axis=1) - df['low']

# การเปลี่ยนแปลงราคาช่วงสั้น
df['ROC'] = df['close'].pct_change(periods=3)

# สถานะ oversold / overbought (จาก RSI)
df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)

# Time Features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# Lagged Features (เพิ่มเติม)
df['RSI_lag1'] = df['RSI'].shift(1)
df['MACD_lag1'] = df['MACD'].shift(1)
df['EMA_fast_lag1'] = df['EMA_fast'].shift(1)
df['EMA_slow_lag1'] = df['EMA_slow'].shift(1)
df['Stoch_K_lag1'] = df['Stoch_K'].shift(1)
df['Stoch_D_lag1'] = df['Stoch_D'].shift(1)
df['ATR_lag1'] = df['ATR'].shift(1)


# New features from your code (ที่เพิ่มเข้ามาแต่ยังไม่ถูกใช้ใน 'features')
df['return_1'] = df['close'].pct_change(1)
df['return_5'] = df['close'].pct_change(5)
df['ema_ratio'] = df['EMA_fast'] / df['EMA_slow'] # สัดส่วน EMA

df['volume_change'] = df['tick_volume'].pct_change()
df['volume_avg'] = df['tick_volume'].rolling(20).mean()
df['volume_ratio'] = df['tick_volume'] / df['volume_avg'] # สัดส่วน Volume กับเฉลี่ย

df['price_slope'] = df['close'].diff(5) # ความชันราคา

df['close_lag1'] = df['close'].shift(1)
df['close_lag2'] = df['close'].shift(2)

# เพิ่ม Features ที่อาจเป็นประโยชน์เพิ่มเติม (แนะนำ)
df['EMA_diff'] = df['EMA_fast'] - df['EMA_slow'] # ผลต่าง EMA
df['Stoch_diff'] = df['Stoch_K'] - df['Stoch_D'] # ผลต่าง Stochastics
df['Upper_shadow_ratio'] = df['Upper_shadow'] / df['Body_size'] # อัตราส่วนเงาบนต่อขนาดตัวแท่ง
df['Lower_shadow_ratio'] = df['Lower_shadow'] / df['Body_size'] # อัตราส่วนเงาล่างต่อขนาดตัวแท่ง

df.dropna(inplace=True)

# STEP 3: สร้าง Target (ขึ้นหรือลงเกิน 0.3%)
df['future_return'] = df['close'].shift(-3) / df['close']-1
# df['target'] = np.where(df['future_return'] > 0.003, 1,
#                         np.where(df['future_return'] < -0.003, 0, np.nan))
#df['target'] = np.where(df['future_return'] > 0.003, 1,
#                         np.where(df['future_return'] < -0.003, 0, np.nan))

# วิธีใหม่ 1: เพิ่ม threshold
df['target'] = np.where(df['future_return'] > 0.005, 1,
               np.where(df['future_return'] < -0.005, 0, np.nan))

# Target แบบง่ายที่สุด: ขึ้น (1) หรือ ลง (0) โดยไม่สนขนาด
df['target_simple'] = np.where(df['close'].shift(-3) > df['close'], 1, 0)
# แล้วลองเทรนโมเดลด้วย 'target_simple' ดูว่า Accuracy โดยรวมดีขึ้นหรือไม่

df.dropna(inplace=True)

# STEP 4: เตรียมชุดข้อมูล
# features = [
#     'open', 'high', 'low', 'close', 'tick_volume',
#     'RSI', 'EMA_fast', 'EMA_slow',
#     'MACD', 'MACD_signal',
#     'Stoch_K', 'Stoch_D',
#     'BB_upper', 'BB_lower', 'BB_width'
# ]

# อัปเดต List ของ Features ให้รวมตัวใหม่ๆ ที่สร้างขึ้น
features = [
    'open', 'high', 'low', 'close', 'tick_volume',
    'RSI', 'EMA_fast', 'EMA_slow',
    'MACD', 'MACD_signal', 'MACD_hist', # เพิ่ม MACD_hist
    'Stoch_K', 'Stoch_D',
    'BB_upper', 'BB_lower', 'BB_width', 'BB_percent_b', # เพิ่ม BB_percent_b
    'ATR', # เพิ่ม ATR
    'Price_change', 'Body_size', 'Upper_shadow', 'Lower_shadow',
    'ROC',
    'RSI_Overbought', 'RSI_Oversold',
    'hour', 'dayofweek',
    'RSI_lag1', 'MACD_lag1', 'EMA_fast_lag1', 'EMA_slow_lag1', # เพิ่ม lagged EMA
    'Stoch_K_lag1', 'Stoch_D_lag1', 'ATR_lag1', # เพิ่ม lagged ATR
    'return_1', 'return_5', 'ema_ratio',
    'volume_change', 'volume_avg', 'volume_ratio',
    'price_slope',
    'close_lag1', 'close_lag2',
    'EMA_diff', 'Stoch_diff', # เพิ่มผลต่างของอินดิเคเตอร์
    'Upper_shadow_ratio', 'Lower_shadow_ratio' # เพิ่มอัตราส่วนเงา
]

X = df[features]
# y = df['target'].astype(int)

y = df['target'].astype(int)

#X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# ใช้ 70% เทรน, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# คำนวณอัตราส่วนของคลาส
# ... (ส่วนคำนวณ scale_pos_weight_value)
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight_value = neg_count / pos_count
print(f"Scale Pos Weight: {scale_pos_weight_value}")

# ลองปรับค่านี้ด้วยมือ
# scale_pos_weight_adjusted =300 # หรือ 0.9 หรือค่าอื่นๆ ที่ใกล้เคียง 1.0
print(f"Adjusted Scale Pos Weight: {scale_pos_weight_value}")

# STEP 5: สร้างและเทรน XGBoost
model = XGBClassifier(
    n_estimators=1000,       # ตั้งไว้สูงพอ
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight_value,
    eval_metric='logloss',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True
)


# เพิ่มโค้ดนี้เพื่อดูสัดส่วนของคลาสในชุดฝึก
print("--- y_train value counts ---")
print(y_train.value_counts())
print("-" * 25)

# (ส่วนที่คุณมีอยู่แล้ว)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("🎯 Accuracy:", round(acc, 4))
print(classification_report(y_test, y_pred))

# STEP 6: ประเมินผล
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("🎯 Accuracy:", round(acc, 4))
print(classification_report(y_test, y_pred))

# STEP 7: ทำนายแท่งถัดไป
latest = df[features].iloc[-1:]
prediction = model.predict(latest)[0]
print("📈 คาดว่าแท่งถัดไปจะ", "⬆️ ขึ้น" if prediction == 1 else "⬇️ ลง")

# เพิ่ม column ดูว่าโมเดลถูกมั้ย และ Return ถ้าทำตามสัญญาณ
df_test = df.loc[y_test.index].copy()

# เช็ก index ดูว่า match กันมั้ย
print("🧾 ตัวอย่าง index ใน df:", df.index[-5:])
print("📊 ตัวอย่าง index ใน y_test:", y_test.index[:5])

# เพิ่มคอลัมน์ y_pred
df_test['y_pred'] = y_pred

# คำนวณผลตอบแทนตามกลยุทธ์
df_test['strategy_return'] = np.where(
    df_test['y_pred'] == 1, df_test['future_return'],
    np.where(df_test['y_pred'] == 0, -df_test['future_return'], 0)
)

# คำนวณผลตอบแทนสะสม
cumulative_return = (1 + df_test['strategy_return']).cumprod()
final_return = cumulative_return.iloc[-1]

df_test['buy_and_hold'] = df_test['future_return']
baseline_return = (1 + df_test['buy_and_hold']).cumprod().iloc[-1]
print("📊 Buy & Hold:", round(baseline_return, 4))

print("💹 ผลตอบแทนรวมจากกลยุทธ์:", round(final_return, 4))
print("📈 กลยุทธ์นี้", "✅ มีกำไร" if final_return > 1 else "❌ ขาดทุน")

import matplotlib.pyplot as plt
(1 + df_test['strategy_return']).cumprod().plot(label='Strategy')
(1 + df_test['future_return']).cumprod().plot(label='Buy & Hold')
plt.legend()
plt.title('Strategy vs Buy & Hold')
plt.show()



# STEP 8: บันทึกโมเดล
joblib.dump(model, 'rf_model_tradeShort.pkl')
