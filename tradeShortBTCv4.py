# FILE NAME: BTCshortTradeV3.py
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import joblib # สำหรับบันทึก/โหลด scaler และ model
import MetaTrader5 as mt5 # สำหรับดึงข้อมูลจาก MT5
import matplotlib.pyplot as plt # สำหรับ Plot กราฟ
from xgboost import XGBClassifier # สำหรับโมเดล XGBoost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

# Import ta library สำหรับ Technical Indicators
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# --- Global Configuration สำหรับจำนวนแท่งเทียนสูงสุดที่สามารถดึงได้จาก MT5 ---
MAX_AVAILABLE_BARS = {
    mt5.TIMEFRAME_M1: 99999,
    mt5.TIMEFRAME_M5: 99999,
    mt5.TIMEFRAME_M15: 99999,
    mt5.TIMEFRAME_M30: 99999,
    mt5.TIMEFRAME_H1: 63890,
    mt5.TIMEFRAME_H4: 15989,
    mt5.TIMEFRAME_D1: 2669,
    mt5.TIMEFRAME_W1: 99999, 
    mt5.TIMEFRAME_MN1: 99999 
}

# --- 1. ฟังก์ชันช่วยในการดึงข้อมูลและสร้าง Features สำหรับแต่ละ Timeframe ---
def _get_features_for_timeframe_data(symbol, timeframe, n_bars):
    """
    ดึงข้อมูลราคาจาก MT5 สำหรับ timeframe ที่กำหนด และสร้าง features ทางเทคนิค.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    
    if rates is None or len(rates) == 0:
        print(f"❌ ไม่สามารถดึงข้อมูล rates สำหรับ {symbol} Timeframe {timeframe} จำนวน {n_bars} แท่งได้, หรือไม่มีข้อมูล. MT5 Error: {mt5.last_error()}")
        return pd.DataFrame() # คืนค่า DataFrame ว่างเปล่าหากล้มเหลว

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # เพิ่มคอลัมน์ 'real_volume' และ 'spread' หากไม่มี (จาก MT5 rates)
    if 'real_volume' not in df.columns:
        df['real_volume'] = 0
    if 'spread' not in df.columns:
        # พยายามดึงค่า spread ปัจจุบันจาก SymbolInfo
        symbol_info = mt5.symbol_info(symbol)
        df['spread'] = symbol_info.spread if symbol_info else 0
        if symbol_info is None:
            print(f"⚠️ ไม่สามารถดึงข้อมูล SymbolInfo สำหรับ {symbol} ได้. ตั้งค่า Spread เป็น 0.")

    # --- Feature Engineering: เพิ่ม Indicators ยอดนิยม ---
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
   
    df['EMA_fast'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
    df['EMA_slow'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator() # เพิ่ม EMA50
    macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_lower'] = bollinger.bollinger_lband()

    

    # --- ฟีเจอร์ Price Action และ Lagged ---
    df['Body_size'] = abs(df['close'] - df['open'])
    df['Upper_shadow'] = df['high'] - df[['close','open']].max(axis=1)
    df['Lower_shadow'] = df[['close','open']].min(axis=1) - df['low']
    
    # df['return_1'] = df['close'].pct_change(1)
    # df['return_3'] = df['close'].pct_change(3)
    # df['return_7'] = df['close'].pct_change(7) # เพิ่ม %เปลี่ยนแปลงใน 7 วัน


    # --- ฟีเจอร์ Divergence (RSI) ---
    divergence_lookback = 14 # จำนวนแท่งเทียนที่มองย้อนหลัง
    df['bullish_rsi_divergence'] = np.where(
        (df['close'] < df['close'].shift(divergence_lookback)) & (df['RSI'] > df['RSI'].shift(divergence_lookback)), 1, 0
    )
    df['bearish_rsi_divergence'] = np.where(
        (df['close'] > df['close'].shift(divergence_lookback)) & (df['RSI'] < df['RSI'].shift(divergence_lookback)), 1, 0
    )

    # --- ฟีเจอร์ที่สร้างจาก Logic การเทรด (แปลงเป็น Binary Features) ---
    # เงื่อนไขสัญญาณ RSI < 30 เข้า Buy
    df['signal_rsi_oversold'] = np.where(df['RSI'] < 30, 1, 0)
    # เงื่อนไขสัญญาณ RSI > 70 เข้า Sell
    df['signal_rsi_overbought'] = np.where(df['RSI'] > 70, 1, 0)
    # เงื่อนไขสัญญาณ MACD ตัดเส้น Signal ขึ้น Buy
    df['signal_macd_cross_up'] = np.where(
        (df['MACD'].shift(1) < df['MACD_signal'].shift(1)) & (df['MACD'] > df['MACD_signal']), 1, 0
    )
    # เงื่อนไขสัญญาณ MA สั้น ตัด MA ยาวจากล่างขึ้นบน (Golden Cross) Buy
    df['signal_golden_cross'] = np.where(
        (df['EMA_fast'].shift(1) < df['EMA_slow'].shift(1)) & (df['EMA_fast'] > df['EMA_slow']), 1, 0
    )
    # เงื่อนไขสัญญาณ ราคาแตะ Bollinger Band ล่าง + RSI ต่ำ Buy
    df['signal_bb_lower_rsi_low'] = np.where(
        (df['close'] <= df['BB_lower']) & (df['RSI'] < 40), 1, 0
    )
    # เงื่อนไขสัญญาณ Close < MA50 (ใช้ EMA50) Sell
    df['signal_close_below_ema50'] = np.where(df['close'] < df['EMA_50'], 1, 0) # ใช้ EMA_50 ที่สร้างขึ้นใหม่

    # --- ฟีเจอร์ความชันของ Moving Average ---
    df['ema_fast_slope'] = df['EMA_fast'].diff(periods=3)
    df['ema_slow_slope'] = df['EMA_slow'].diff(periods=3)

    # df['Volume_change'] = df['real_volume']
    # df['Candle_range'] = df['high'] - df['low']
    # df['trend_slope_5'] = df['close'].diff(5)



    # --- ฟีเจอร์ Candlestick Patterns ---
    # Bullish Engulfing
    df['bullish_engulfing'] = np.where(
        (df['close'] > df['open']) & # แท่งปัจจุบันเป็นแท่งเขียว
        (df['open'].shift(1) > df['close'].shift(1)) & # แท่งก่อนหน้าเป็นแท่งแดง
        (df['open'] < df['close'].shift(1)) & # แท่งปัจจุบันเปิดต่ำกว่าปิดของแท่งก่อนหน้า
        (df['close'] > df['open'].shift(1)) & # แท่งปัจจุบันปิดสูงกว่าเปิดของแท่งก่อนหน้า
        (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))), # Body ปัจจุบันใหญ่กว่า Body ก่อนหน้า
        1, 0
    )
    # Bearish Engulfing
    df['bearish_engulfing'] = np.where(
        (df['close'] < df['open']) & # แท่งปัจจุบันเป็นแท่งแดง
        (df['open'].shift(1) < df['close'].shift(1)) & # แท่งก่อนหน้าเป็นแท่งเขียว
        (df['open'] > df['close'].shift(1)) & # แท่งปัจจุบันเปิดสูงกว่าปิดของแท่งก่อนหน้า
        (df['close'] < df['open'].shift(1)) & # แท่งปัจจุบันปิดต่ำกว่าเปิดของแท่งก่อนหน้า
        (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))), # Body ปัจจุบันใหญ่กว่า Body ก่อนหน้า
        1, 0
    )
    # Hammer
    df['hammer'] = np.where(
        (df['Body_size'] > 0) & # มี Body
        (df['Lower_shadow'] >= 2 * df['Body_size']) & # ไส้ล่างยาวอย่างน้อย 2 เท่าของ Body
        (df['Upper_shadow'] <= 0.2 * df['Body_size']), # ไส้บนสั้นมาก
        1, 0
    )
    # Shooting Star
    df['shooting_star'] = np.where(
        (df['Body_size'] > 0) & # มี Body
        (df['Upper_shadow'] >= 2 * df['Body_size']) & # ไส้บนยาวอย่างน้อย 2 เท่าของ Body
        (df['Lower_shadow'] <= 0.2 * df['Body_size']), # ไส้ล่างสั้นมาก
        1, 0
    )
    # Doji
    df['doji_val'] = np.where(
        (df['Body_size'] < (df['high'] - df['low']) * 0.1) & # Body เล็กมาก (น้อยกว่า 10% ของช่วงราคา)
        ((df['high'] - df['low']) > df['ATR'] * 0.1), # แต่ช่วงราคาทั้งหมดไม่เล็กเกินไป (ไม่ใช่แท่งแบนราบ)
        1, 0
    )

    # --- Sentiment จากข่าวหรือโซเชียล (ยังไม่ได้นำมาใช้ เนื่องจากไม่มีข้อมูล) ---
    # หากมีข้อมูล Sentiment สามารถเพิ่ม Feature ตรงนี้ได้
    # df['sentiment_score'] = ...

    # ลบแถวที่มีค่า NaN ที่เกิดจากการคำนวณอินดิเคเตอร์และ Features
    df.dropna(inplace=True)
    return df

# --- 2. ฟังก์ชันหลักในการโหลดและเตรียมข้อมูล Multi-Timeframe สำหรับการเทรน ---
def load_and_preprocess_multi_timeframe_data(symbol):
    """
    โหลดข้อมูลสำหรับ H1, M15, H4, ทำ Feature Engineering,
    จัดเรียงข้อมูล และสร้างชุดข้อมูลสำหรับเทรนโมเดล XGBoost.
    """
    # ตรวจสอบและเริ่มต้น MT5
    if not mt5.initialize():
        print("❌ การเริ่มต้น MT5 ล้มเหลวใน load_and_preprocess_multi_timeframe_data", mt5.last_error())
        sys.exit(1)

    # ใช้จำนวนแท่งเทียนสูงสุดที่มี
    max_h1_bars = MAX_AVAILABLE_BARS.get(mt5.TIMEFRAME_H1, 0)
    max_m15_bars = MAX_AVAILABLE_BARS.get(mt5.TIMEFRAME_M15, 0)
    max_h4_bars = MAX_AVAILABLE_BARS.get(mt5.TIMEFRAME_H4, 0)
    
    # กำหนดจำนวนแท่งเทียนที่ต้องการดึงสำหรับ H1 (ปรับได้ตามความต้องการข้อมูล)
    # ควรดึงให้มากพอที่จะครอบคลุมช่วงเวลาที่ต้องการเทรน และมีข้อมูลเพียงพอสำหรับการคำนวณ Features
    n_bars_h1_to_fetch = min(60000, max_h1_bars) # ดึงสูงสุด 60000 แท่ง H1 หรือเท่าที่มี
    
    # คำนวณจำนวนแท่งเทียนสำหรับ Timeframe อื่นๆ โดยอ้างอิงจาก H1 และจำกัดด้วย MAX_AVAILABLE_BARS
    n_bars_m15_to_fetch = min(n_bars_h1_to_fetch * 4, max_m15_bars) # M15 มี 4 แท่งใน H1
    n_bars_h4_to_fetch = min(n_bars_h1_to_fetch // 4, max_h4_bars) # H4 มี 1 แท่งใน 4 H1

    print(f"กำลังดึงข้อมูล: {n_bars_h1_to_fetch} แท่ง H1, {n_bars_m15_to_fetch} แท่ง M15, {n_bars_h4_to_fetch} แท่ง H4.")

    # ดึงข้อมูลและสร้าง Features สำหรับแต่ละ Timeframe
    print("กำลังโหลดข้อมูล H1...")
    df_h1 = _get_features_for_timeframe_data(symbol, mt5.TIMEFRAME_H1, n_bars_h1_to_fetch).add_suffix('_H1')
    print(f"DEBUG: df_h1_features shape: {df_h1.shape}")
    if df_h1.empty:
        print("❌ DataFrame H1 ว่างเปล่าหลังจากสร้าง features. สคริปต์จะหยุดทำงาน.")
        mt5.shutdown()
        sys.exit()

    print("กำลังโหลดข้อมูล M15...")
    df_m15 = _get_features_for_timeframe_data(symbol, mt5.TIMEFRAME_M15, n_bars_m15_to_fetch).add_suffix('_M15')
    print(f"DEBUG: df_m15_features shape: {df_m15.shape}")
    if df_m15.empty:
        print("❌ DataFrame M15 ว่างเปล่าหลังจากสร้าง features. สคริปต์จะหยุดทำงาน.")
        mt5.shutdown()
        sys.exit()

    print("กำลังโหลดข้อมูล H4...")
    df_h4 = _get_features_for_timeframe_data(symbol, mt5.TIMEFRAME_H4, n_bars_h4_to_fetch).add_suffix('_H4')
    print(f"DEBUG: df_h4_features shape: {df_h4.shape}")
    if df_h4.empty:
        print("❌ DataFrame H4 ว่างเปล่าหลังจากสร้าง features. สคริปต์จะหยุดทำงาน.")
        mt5.shutdown()
        sys.exit()

    mt5.shutdown() # ปิดการเชื่อมต่อ MT5 หลังจากดึงข้อมูลทั้งหมดเสร็จสิ้น

    # จัดเรียงข้อมูลจาก Timeframe ต่างๆ ให้ตรงกับ Index ของ H1
    print("กำลังจัดเรียงข้อมูล Multi-Timeframe...")
    df_combined = pd.merge_asof(df_h1, df_m15, left_index=True, right_index=True, direction='backward')
    df_combined = pd.merge_asof(df_combined, df_h4, left_index=True, right_index=True, direction='backward')
    print(f"DEBUG: df_combined shape หลังจาก merge_asof: {df_combined.shape}")
    df_combined.dropna(inplace=True) # ลบแถวที่มีค่า NaN ที่เหลืออยู่
    print(f"DEBUG: df_combined shape หลังจาก dropna: {df_combined.shape}")

    if df_combined.empty:
        print("❌ Combined DataFrame ว่างเปล่าหลังจากรวมและลบ NaNs. สคริปต์จะหยุดทำงาน.")
        sys.exit()

    # --- สร้าง Target Variable ---
    # Target: 1 ถ้า future_return > 0.5%, 0 ถ้า future_return < -0.5%, NaN ถ้าอยู่ระหว่างนั้น
    # ใช้ future_return_H1 เพื่อให้ Target อ้างอิงกับ Timeframe หลัก
    df_combined['future_return_H1'] = df_combined['close_H1'].shift(-3) / df_combined['close_H1'] - 1
    df_combined['target'] = np.where(df_combined['future_return_H1'] > 0.005, 1,
                                     np.where(df_combined['future_return_H1'] < -0.005, 0, np.nan))
    df_combined.dropna(inplace=True) # ลบแถวที่ target เป็น NaN

    if df_combined.empty:
        print("❌ Combined DataFrame ว่างเปล่าหลังจากสร้าง Target และลบ NaNs. สคริปต์จะหยุดทำงาน.")
        sys.exit()

    # เลือก Features ทั้งหมดที่ใช้ในการเทรน
    features_list = [col for col in df_combined.columns if col not in ['future_return_H1', 'target']]
    
    X = df_combined[features_list]
    y = df_combined['target'].astype(int) # ตรวจสอบให้แน่ใจว่าเป็น int

    # Scale Features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features_list, index=X.index) # แปลงกลับเป็น DataFrame เพื่อรักษา Index

    print(f"ชุดข้อมูลพร้อมแล้ว. X shape: {X_scaled_df.shape}, y shape: {y.shape}")
    return X_scaled_df, y, scaler, features_list, df_combined # คืนค่า df_combined เพื่อใช้ในการ Plot

# --- 3. ฟังก์ชันสำหรับเทรนโมเดล XGBoost ---
def train_xgboost_model(X_train, y_train, X_valid, y_valid):
    """
    สร้างและเทรนโมเดล XGBoost Classifier.
    """
    # คำนวณอัตราส่วนของคลาสสำหรับ scale_pos_weight
    neg_count = y_train.value_counts()[0] # จำนวน Class 0 (ลง)
    pos_count = y_train.value_counts()[1] # จำนวน Class 1 (ขึ้น)
    
    # ปรับ scale_pos_weight: ให้ความสำคัญกับ Class 1 (Buy) มากขึ้นอย่างชัดเจน
    # สามารถปรับค่านี้ได้ตามผลลัพธ์การ Backtest เพื่อเพิ่ม Recall ของ Class 1
    scale_pos_weight_value = float(neg_count / pos_count) # คำนวณจากอัตราส่วน
    print(f"คำนวณ Scale Pos Weight: {scale_pos_weight_value}")

    model = XGBClassifier(
        n_estimators=10000,       # จำนวนต้นไม้ (สามารถเพิ่มได้ถ้าโมเดลยัง Underfit)
        max_depth=7,             # ความลึกสูงสุดของต้นไม้ (ควบคุมความซับซ้อน)
        learning_rate=0.03,      # อัตราการเรียนรู้
        subsample=0.9,           # สัดส่วนของ Samples ที่ใช้ในแต่ละต้นไม้
        colsample_bytree=0.8,    # สัดส่วนของ Features ที่ใช้ในแต่ละต้นไม้
        scale_pos_weight=scale_pos_weight_value, # จัดการ Class Imbalance
        eval_metric='logloss',   # Metric สำหรับการประเมินบน eval_set
        random_state=42,         # เพื่อให้ผลลัพธ์สามารถทำซ้ำได้ # แก้ไข warning สำหรับ XGBoost เวอร์ชันใหม่
        min_child_weight=1,      # น้ำหนักขั้นต่ำของ Child node
        gamma=0.1,               # ค่า Regularization
        tree_method='hist',      # ใช้ Histograms สำหรับความเร็ว
        early_stopping_rounds=100 # หยุดการเทรนก่อนกำหนดหากไม่มีการปรับปรุงใน Validation Loss
    )

    print("กำลังเทรนโมเดล XGBoost Classifier...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=True # แสดงผลลัพธ์ระหว่างการเทรน
    )
    print("การเทรนโมเดล XGBoost เสร็จสมบูรณ์.")
    return model

# --- 4. ฟังก์ชันสำหรับประเมินผลโมเดล ---
def evaluate_model(model, X_test, y_test, threshold=0.50):
    """
    ประเมินประสิทธิภาพของโมเดลบนชุดข้อมูลทดสอบ.
    """
    print("กำลังประเมินประสิทธิภาพโมเดล...")
    y_pred_proba = model.predict_proba(X_test)[:, 1] # ความน่าจะเป็นของ Class 1
    y_pred = (y_pred_proba > threshold).astype(int) # ทำนายตามเกณฑ์ที่กำหนด

    accuracy = accuracy_score(y_test, y_pred)
    recall_class_1 = recall_score(y_test, y_pred, pos_label=1) # Recall สำหรับ 'Buy' signals
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (Class 1 - Buy Signal): {recall_class_1:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return accuracy, recall_class_1

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- โหมดการเทรน XGBoost Model (BTCshortTradeV3.py) ---")
    
    symbol_to_trade = 'XAUUSDm' # สัญลักษณ์ที่จะใช้เทรนโมเดล

    # STEP 1 & 2: โหลดข้อมูลจาก MT5, ทำ Feature Engineering, และสร้าง Target
    X, y, scaler, features_list, df_combined_full = load_and_preprocess_multi_timeframe_data(symbol_to_trade)
    
    # STEP 3: เตรียมชุดข้อมูลสำหรับเทรนโมเดล
    # แบ่งข้อมูลเป็น Train (80%), Validation (10%), Test (10%) โดยไม่สับเปลี่ยน
    # เพื่อรักษาลำดับเวลาของข้อมูล
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    print(f"ขนาดข้อมูล Train: X={X_train.shape}, y={y_train.shape}")
    print(f"ขนาดข้อมูล Validation: X={X_valid.shape}, y={y_valid.shape}")
    print(f"ขนาดข้อมูล Test: X={X_test.shape}, y={y_test.shape}")

    # STEP 4: สร้างและเทรน XGBoost Classifier
    model = train_xgboost_model(X_train, y_train, X_valid, y_valid)

    # STEP 5: ประเมินผลโมเดลบนชุด Test
    # สามารถปรับ prediction_threshold เพื่อปรับสมดุลระหว่าง Precision และ Recall
    prediction_threshold = 0.40 # ค่าเริ่มต้น, สามารถปรับจูนได้
    accuracy, recall_class_1 = evaluate_model(model, X_test, y_test, threshold=prediction_threshold)

    # --- Plot ผลตอบแทนสะสมเปรียบเทียบ ---
    # ดึงข้อมูลที่เกี่ยวข้องกับ Test Set จาก df_combined_full
    # เนื่องจาก X_test, y_test ถูก split แบบ shuffle=False
    # เราสามารถใช้ index ของ X_test เพื่อดึงข้อมูลเดิมจาก df_combined_full ได้
    df_test = df_combined_full.loc[X_test.index].copy()
    
    # ทำนายผลบน Test Set อีกครั้งเพื่อใช้ในการ Plot
    y_pred_proba_for_plot = model.predict_proba(X_test)[:, 1]
    df_test['y_pred'] = (y_pred_proba_for_plot > prediction_threshold).astype(int)

    # คำนวณผลตอบแทนตามกลยุทธ์
    # ถ้าทำนายขึ้น (1) แล้วได้กำไร = future_return_H1
    # ถ้าทำนายลง (0) แล้วได้กำไร = -future_return_H1 (Short Sell)
    # ถ้าทำนายผิด หรือไม่มีสัญญาณ = 0
    df_test['strategy_return'] = np.where(
        df_test['y_pred'] == 1, df_test['future_return_H1'], 
        np.where(df_test['y_pred'] == 0, -df_test['future_return_H1'], 0) 
    )
    cumulative_strategy_return = (1 + df_test['strategy_return']).cumprod()
    final_strategy_return = cumulative_strategy_return.iloc[-1]
    
    # คำนวณผลตอบแทนแบบ Buy & Hold (Baseline)
    df_test['buy_and_hold_return'] = (1 + df_test['future_return_H1']).cumprod()
    final_buy_and_hold_return = df_test['buy_and_hold_return'].iloc[-1]
    
    print("📊 ผลตอบแทน Buy & Hold:", round(final_buy_and_hold_return, 4))
    print("💹 ผลตอบแทนรวมจากกลยุทธ์:", round(final_strategy_return, 4))
    print("📈 กลยุทธ์นี้", "✅ มีกำไร" if final_strategy_return > 1 else "❌ ขาดทุน")

    plt.figure(figsize=(12, 6))
    cumulative_strategy_return.plot(label='Strategy Cumulative Return')
    df_test['buy_and_hold_return'].plot(label='Buy & Hold Cumulative Return') 
    plt.title('Strategy vs Buy & Hold Cumulative Return (Test Set)')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

    # STEP 6: บันทึกโมเดล, Scaler, และ Features List
    joblib.dump(model, 'xgboost_model_v3.pkl')
    joblib.dump(scaler, 'scaler_v3.pkl')
    joblib.dump(features_list, 'features_list_v3.pkl')

    print("✅ โมเดล XGBoost ถูกบันทึกแล้วในชื่อ 'xgboost_model_v3.pkl'")
    print("✅ Scaler ถูกบันทึกแล้วในชื่อ 'scaler_v3.pkl'")
    print("✅ Features list ถูกบันทึกแล้วในชื่อ 'features_list_v3.pkl'")
