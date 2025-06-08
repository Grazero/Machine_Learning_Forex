import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# STEP 1: ดึงข้อมูลจาก MT5
symbol = 'BTCUSDm'  # สัญลักษณ์ที่ต้องการดึงข้อมูล
timeframe = mt5.TIMEFRAME_H1 # Timeframe 1 ชั่วโมง (ตามที่คุณระบุในโค้ด)
n_bars = 90000  # จำนวนแท่งเทียนที่ต้องการดึง

# ตรวจสอบและเริ่มต้น MT5
if not mt5.initialize():
    print("MT5 initialization failed", mt5.last_error())
    quit()

# ดึงข้อมูลราคาจาก MT5
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)

# NEW: ดึงค่า spread จาก SymbolInfo
symbol_info = mt5.symbol_info(symbol)
current_spread = symbol_info.spread if symbol_info else 0 # Default to 0 if info not available

mt5.shutdown() # ปิดการเชื่อมต่อ MT5 เมื่อดึงข้อมูลเสร็จ

# แปลงข้อมูลเป็น DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s') # แปลง timestamp เป็น datetime
df.set_index('time', inplace=True) # ตั้งคอลัมน์ 'time' เป็น index

# NEW: เพิ่มคอลัมน์ 'spread' และ 'real_volume' (real_volume มักจะมีอยู่แล้วใน rates)
df['spread'] = current_spread
# ตรวจสอบว่า 'real_volume' มีอยู่ใน df หรือไม่ก่อนที่จะใช้
if 'real_volume' not in df.columns:
    df['real_volume'] = 0 # กำหนดค่าเริ่มต้นเป็น 0 ถ้าไม่มีข้อมูล real_volume
    print("Warning: 'real_volume' column not found in MT5 data. Setting to 0.")


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
# เพิ่ม lagged returns สำหรับแท่งที่ 2 และ 3
df['return_2'] = df['close'].pct_change(2) # ผลตอบแทน 2 แท่ง
df['return_3'] = df['close'].pct_change(3) # ผลตอบแทน 3 แท่ง

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

# --- เพิ่มฟีเจอร์ Divergence (แบบง่ายๆ) ---
# การตรวจจับ Divergence แบบง่ายๆ โดยใช้การเปรียบเทียบค่าในช่วงเวลาที่กำหนด
# lookback_period สำหรับการตรวจจับ Divergence
divergence_lookback = 5

# Bullish Divergence: ราคาทำ Lower Low, RSI ทำ Higher Low
df['bullish_rsi_divergence'] = np.where(
    (df['close'] < df['close'].shift(divergence_lookback)) &
    (df['RSI'] > df['RSI'].shift(divergence_lookback)),
    1, 0
)

# Bearish Divergence: ราคาทำ Higher High, RSI ทำ Lower High
df['bearish_rsi_divergence'] = np.where(
    (df['close'] > df['close'].shift(divergence_lookback)) &
    (df['RSI'] < df['RSI'].shift(divergence_lookback)),
    1, 0
)
# --- สิ้นสุดการเพิ่มฟีเจอร์ Divergence ---

# --- เพิ่มฟีเจอร์แนวโน้มและการกลับตัวเพิ่มเติม ---
# Bollinger Bands
bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
df['BB_upper'] = bollinger.bollinger_hband()
df['BB_lower'] = bollinger.bollinger_lband()
df['BB_middle'] = bollinger.bollinger_mavg()
df['BB_width'] = bollinger.bollinger_wband() # ความกว้างของแบนด์ (บอกความผันผวน)
df['BB_percent'] = bollinger.bollinger_pband() # ตำแหน่งราคาเทียบกับแบนด์ (บอก Overbought/Oversold)

# EMA Crossover Signal (เมื่อ EMA_fast ตัด EMA_slow)
# 1 = Bullish Cross, -1 = Bearish Cross, 0 = No Cross
df['EMA_cross_signal'] = 0
df.loc[(df['EMA_fast'].shift(1) < df['EMA_slow'].shift(1)) & (df['EMA_fast'] > df['EMA_slow']), 'EMA_cross_signal'] = 1
df.loc[(df['EMA_fast'].shift(1) > df['EMA_slow'].shift(1)) & (df['EMA_fast'] < df['EMA_slow']), 'EMA_cross_signal'] = -1

# Rate of Change (ROC) ของอินดิเคเตอร์หลัก
# เพื่อดูความเร็วในการเปลี่ยนแปลงของโมเมนตัม
df['RSI_ROC'] = df['RSI'].diff(periods=3) # ROC ของ RSI ใน 3 แท่ง
df['MACD_hist_ROC'] = df['MACD_hist'].diff(periods=3) # ROC ของ MACD Histogram ใน 3 แท่ง

# Average Directional Index (ADX)
adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
df['ADX'] = adx_indicator.adx()
df['ADX_pos'] = adx_indicator.adx_pos() # Positive Directional Indicator
df['ADX_neg'] = adx_indicator.adx_neg() # Negative Directional Indicator

# Volatility-adjusted Price Change
# ป้องกันหารด้วยศูนย์ ถ้า ATR เป็น 0 ให้เป็น 0
df['Price_change_ATR_ratio'] = np.where(df['ATR'] != 0, df['Price_change'] / df['ATR'], 0)
# --- สิ้นสุดการเพิ่มฟีเจอร์แนวโน้มและการกลับตัวเพิ่มเติม ---

# --- เพิ่มฟีเจอร์ Candlestick Patterns ---
# Bullish Engulfing
# แท่งปัจจุบันเป็นแท่งเขียว (close > open) และกลืนกินแท่งก่อนหน้า (open_curr < close_prev และ close_curr > open_prev)
# และ body ของแท่งปัจจุบันใหญ่กว่า body ของแท่งก่อนหน้า
df['bullish_engulfing'] = np.where(
    (df['close'] > df['open']) &
    (df['open'].shift(1) > df['close'].shift(1)) & # Previous candle was bearish
    (df['open'] < df['close'].shift(1)) &
    (df['close'] > df['open'].shift(1)) &
    (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))),
    1, 0
)

# Bearish Engulfing
# แท่งปัจจุบันเป็นแท่งแดง (close < open) และกลืนกินแท่งก่อนหน้า (open_curr > close_prev และ close_curr < open_prev)
# และ body ของแท่งปัจจุบันใหญ่กว่า body ของแท่งก่อนหน้า
df['bearish_engulfing'] = np.where(
    (df['close'] < df['open']) &
    (df['open'].shift(1) < df['close'].shift(1)) & # Previous candle was bullish
    (df['open'] > df['close'].shift(1)) &
    (df['close'] < df['open'].shift(1)) &
    (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))),
    1, 0
)

# Hammer (Bullish Reversal)
# ร่างกายเล็กอยู่ด้านบน, ไส้ล่างยาวอย่างน้อย 2 เท่าของร่างกาย, ไม่มีไส้บนหรือเล็กน้อย
df['hammer'] = np.where(
    (df['Body_size'] > 0) & # ป้องกันหารด้วยศูนย์
    (df['Lower_shadow'] >= 2 * df['Body_size']) &
    (df['Upper_shadow'] <= 0.2 * df['Body_size']), # ไส้บนเล็กน้อย
    1, 0
)

# Shooting Star (Bearish Reversal)
# ร่างกายเล็กอยู่ด้านล่าง, ไส้บนยาวอย่างน้อย 2 เท่าของร่างกาย, ไม่มีไส้ล่างหรือเล็กน้อย
df['shooting_star'] = np.where(
    (df['Body_size'] > 0) & # ป้องกันหารด้วยศูนย์
    (df['Upper_shadow'] >= 2 * df['Body_size']) &
    (df['Lower_shadow'] <= 0.2 * df['Body_size']), # ไส้ล่างเล็กน้อย
    1, 0
)
# --- สิ้นสุดการเพิ่มฟีเจอร์ Candlestick Patterns ---


# STEP 3: สร้าง Target
df['future_return'] = df['close'].shift(-3) / df['close'] - 1
# Target: ถ้าราคาอีก 3 แท่งถัดไปสูงกว่าปัจจุบันให้เป็น 1, ถ้าต่ำกว่าให้เป็น 0
df['target'] = np.where(df['close'].shift(-3) > df['close'], 1, 0)


# ลบแถวที่มีค่า NaN ที่เกิดจากการสร้าง target หรืออินดิเคเตอร์
df.dropna(inplace=True)

# STEP 4: เตรียมชุดข้อมูลสำหรับเทรนโมเดล
# เลือกฟีเจอร์ที่ใช้ในการเทรน (ชุดใหม่ที่กระชับและเน้นประสิทธิภาพ)
# IMPORTANT: This list must EXACTLY match the features sent by the MQL5 EA.
features = [
    'open', 'high', 'low', 'close', 'tick_volume',
    'real_volume', # Added real_volume
    'spread',      # Added spread
    'RSI', 'Stoch_K', 'Stoch_D',
    'EMA_fast', 'EMA_slow', 'EMA_diff',
    'MACD', 'MACD_signal', 'MACD_hist',
    'ATR',
    'Price_change', 'Body_size', 'Upper_shadow', 'Lower_shadow',
    'Upper_shadow_ratio', 'Lower_shadow_ratio',
    'return_1', 'return_2', 'return_3',
    'RSI_lag1', 'MACD_hist_lag1', 'ATR_lag1',
    'Stoch_K_lag1', 'Stoch_D_lag1', 'EMA_fast_lag1', 'EMA_slow_lag1', 'close_lag1',
    'volume_avg',
    'bullish_rsi_divergence', 'bearish_rsi_divergence',
    'BB_upper', # Added BB_upper
    'BB_lower', # Added BB_lower
    'BB_middle', # Added BB_middle
    'BB_width', 'BB_percent', # Bollinger Bands
    'EMA_cross_signal', # EMA Crossover
    'RSI_ROC', 'MACD_hist_ROC', # Rate of Change
    'ADX', 'ADX_pos', 'ADX_neg', # ADX indicators
    'Price_change_ATR_ratio', # Volatility-adjusted Price Change
    'bullish_engulfing',
    'bearish_engulfing',
    'hammer',
    'shooting_star'
]

# ตรวจสอบว่าฟีเจอร์ทั้งหมดมีอยู่ใน DataFrame
missing_cols_in_df = [col for col in features if col not in df.columns]
if missing_cols_in_df:
    print(f"Error: Missing expected features in DataFrame after feature engineering: {missing_cols_in_df}")
    sys.exit(1)

# X is now directly df[features]
X = df[features]
y = df['target']

# แบ่งข้อมูลเป็น Train (80%), Test (20%) โดยไม่สับเปลี่ยน (shuffle=False)
# เพื่อรักษาลำดับเวลาของข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# คำนวณอัตราส่วนของคลาสสำหรับ scale_pos_weight (สำหรับ XGBoost)
# เพื่อจัดการกับ Class Imbalance (ถ้ามี)
neg_count = y_train.value_counts()[0] # จำนวน Class 0 (ลง)
pos_count = y_train.value_counts()[1] # จำนวน Class 1 (ขึ้น)

# ปรับ scale_pos_weight: ให้ความสำคัญกับ Class 1 มากขึ้นอย่างชัดเจน
# หากต้องการปรับค่านี้เอง ให้เปลี่ยน float(neg_count) / pos_count เป็นค่าที่ต้องการ เช่น 1.5 หรือ 2.0
# การเพิ่มค่านี้จะทำให้โมเดลระมัดระวังในการทำนาย Class 0 มากขึ้น
scale_pos_weight_value = float(neg_count) / pos_count if pos_count != 0 else 1.0
print(f"Calculated Scale Pos Weight: {scale_pos_weight_value:.2f}")


# STEP 5: สร้างและเทรน XGBoost Classifier
model = XGBClassifier(
    n_estimators=3000,        # จำนวน estimators
    max_depth=7,              # max_depth
    learning_rate=0.03,       # learning_rate
    subsample=0.9,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight_value, # ใช้ค่าที่ปรับใหม่
    eval_metric='logloss',    # ใช้ logloss เป็นเกณฑ์การประเมินประสิทธิภาพบน eval_set
    random_state=42,
    min_child_weight=1,       # min_child_weight
    gamma=0.1                 # gamma
)

# เทรนโมเดล
# สำหรับ XGBoost, eval_set ใช้สำหรับ early stopping หรือการตรวจสอบระหว่างเทรน
# ถ้าไม่มี validation set แยกต่างหาก ก็สามารถเทรนโดยตรงได้
model.fit(X_train, y_train)

# กำหนดเกณฑ์การทำนายสำหรับ Class 1
# หากความน่าจะเป็นของ Class 1 สูงกว่าค่านี้ จะถูกจัดเป็น Class 1
prediction_threshold = 0.50 # <<< IMPORTANT: ปรับเกณฑ์การทำนายเป็น 0.50

# แสดงสัดส่วนของคลาสในชุดฝึก (y_train)
print("--- y_train value counts ---")
print(y_train.value_counts())
print("-" * 25)

# STEP 6: ประเมินผลโมเดลบนชุด Test
# ใช้ predict_proba เพื่อดึงค่าความน่าจะเป็น
y_pred_proba = model.predict_proba(X_test)[:, 1] # ความน่าจะเป็นของ Class 1
y_pred = (y_pred_proba > prediction_threshold).astype(int) # ทำนายตามเกณฑ์ที่กำหนด

acc = accuracy_score(y_test, y_pred)
print("🎯 Accuracy:", round(acc, 4))
print(classification_report(y_test, y_pred))

# Plot ผลตอบแทนสะสมเปรียบเทียบ (ย้ายมาที่นี่เพื่อให้โค้ดนี้เป็น Training Script เท่านั้น)
df_test = df.loc[y_test.index].copy()
df_test['y_pred'] = y_pred
df_test['strategy_return'] = np.where(
    df_test['y_pred'] == 1, df_test['future_return'], # ทำนายขึ้น (1), ถ้าขึ้นจริงได้ future_return
    np.where(df_test['y_pred'] == 0, -df_test['future_return'], 0) # ทำนายลง (0), ถ้าลงจริงได้ -future_return (กำไร)
)
cumulative_return = (1 + df_test['strategy_return']).cumprod()
final_return = cumulative_return.iloc[-1]
df_test['buy_and_hold'] = df_test['future_return']
baseline_return = (1 + df_test['buy_and_hold']).cumprod().iloc[-1]
print("📊 Buy & Hold Return:", round(baseline_return, 4))
print("💹 ผลตอบแทนรวมจากกลยุทธ์:", round(final_return, 4))
print("📈 กลยุทธ์นี้", "✅ มีกำไร" if final_return > 1 else "❌ ขาดทุน")

plt.figure(figsize=(12, 6))
(1 + df_test['strategy_return']).cumprod().plot(label='Strategy')
(1 + df_test['future_return']).cumprod().plot(label='Buy & Hold')
plt.title('Strategy vs Buy & Hold Cumulative Return')
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

# STEP 7: บันทึกโมเดล
joblib.dump(model, 'xgboost_shortBTC_term_model.pkl')
joblib.dump(features, 'features_list.pkl') # Save the exact feature list used for training

print("✅ โมเดลถูกบันทึกแล้วในชื่อ 'xgboost_shortBTC_term_model.pkl'")
print("✅ รายการฟีเจอร์ถูกบันทึกแล้วในชื่อ 'features_list.pkl'")

