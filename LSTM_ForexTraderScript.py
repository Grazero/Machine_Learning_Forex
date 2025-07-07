import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ta

from ta.trend import MACD 
from ta.momentum import RSIIndicator 
from ta.trend import SMAIndicator, EMAIndicator 

import sys
import inspect
import joblib

# --- การตั้งค่า (Configuration) ---
DATA_DIR = 'DataCSV'
LOOKBACK_PERIOD = 60
FORECAST_HORIZON = 1
TRAIN_SPLIT_RATIO = 0.8

SHORT_MA_PERIOD = 10
LONG_MA_PERIOD = 30
RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

EXPORT_DIR = 'exported_models'

# --- 1. ฟังก์ชันโหลดและเตรียมข้อมูลเบื้องต้น (Data Loading and Preprocessing) ---
def load_and_preprocess_data(file_path):
    """
    โหลดข้อมูล CSV, แปลงคอลัมน์ 'Time' เป็น datetime,
    ตั้งค่า 'Time' เป็น index และจัดการค่าว่าง
    """
    try:
        df = pd.read_csv(file_path)
        if 'Time' not in df.columns:
            print(f"ข้อผิดพลาด: ไม่พบคอลัมน์ 'Time' ในไฟล์ {file_path}")
            return pd.DataFrame()
            
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.set_index('Time')
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'TickVolume', 'RealVolume', 'Spread']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"คำเตือน: ไม่พบคอลัมน์ '{col}' ในไฟล์ {file_path}")

        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดหรือประมวลผลไฟล์ {file_path}: {e}")
        return pd.DataFrame()

# --- 2. ฟังก์ชันเพิ่มตัวชี้วัดทางเทคนิค (Feature Engineering - Technical Indicators) ---
def add_technical_indicators(df):
    """
    คำนวณและเพิ่มตัวชี้วัดทางเทคนิคต่างๆ ลงใน DataFrame
    และสร้างคอลัมน์ 'Target' สำหรับการทำนายของ LSTM
    """
    df_copy = df.copy()

    sma_short_indicator = SMAIndicator(close=df_copy['Close'], window=SHORT_MA_PERIOD)
    df_copy['SMA_Short'] = sma_short_indicator.sma_indicator()
    sma_long_indicator = SMAIndicator(close=df_copy['Close'], window=LONG_MA_PERIOD)
    df_copy['SMA_Long'] = sma_long_indicator.sma_indicator()

    ema_short_indicator = EMAIndicator(close=df_copy['Close'], window=SHORT_MA_PERIOD)
    df_copy['EMA_Short'] = ema_short_indicator.ema_indicator()
    ema_long_indicator = EMAIndicator(close=df_copy['Close'], window=LONG_MA_PERIOD)
    df_copy['EMA_Long'] = ema_long_indicator.ema_indicator()

    rsi_indicator = RSIIndicator(close=df_copy['Close'], window=RSI_PERIOD)
    df_copy['RSI'] = rsi_indicator.rsi()

    macd_indicator = MACD(
        close=df_copy['Close'],
        window_fast=MACD_FAST_PERIOD,
        window_slow=MACD_SLOW_PERIOD,
        window_sign=MACD_SIGNAL_PERIOD 
    )
    df_copy['MACD'] = macd_indicator.macd()
    df_copy['MACD_Signal'] = macd_indicator.macd_signal()
    df_copy['MACD_Hist'] = macd_indicator.macd_diff()

    df_copy['Price_Change'] = df_copy['Close'].diff(FORECAST_HORIZON).shift(-FORECAST_HORIZON)
    df_copy['Target'] = (df_copy['Price_Change'] > 0).astype(int)

    df_copy.dropna(inplace=True)
    return df_copy

# --- 3. ฟังก์ชันเตรียมข้อมูลสำหรับ LSTM (Data Preparation for LSTM) ---
def create_sequences(data, lookback_period, target_column='Target'):
    """
    สร้างลำดับข้อมูล (sequences) สำหรับ LSTM
    X: Features (ข้อมูลย้อนหลัง lookback_period แท่ง)
    y: Target (สิ่งที่ต้องการทำนาย)
    """
    X, y = [], []
    features = data.drop(columns=[target_column]).values
    targets = data[target_column].values

    for i in range(len(data) - lookback_period - FORECAST_HORIZON + 1):
        X.append(features[i:(i + lookback_period)])
        y.append(targets[i + lookback_period + FORECAST_HORIZON - 1])

    return np.array(X), np.array(y)

# --- 4. ฟังก์ชันสร้างและฝึกโมเดล LSTM (LSTM Model Building and Training) ---
def build_and_train_lstm(X_train, y_train, input_shape):
    """
    สร้างและฝึกโมเดล LSTM ด้วย TensorFlow/Keras
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape), 
        Dropout(0.2), 
        LSTM(units=50, return_sequences=False), 
        Dropout(0.2),
        Dense(units=1, activation='sigmoid') 
    ])

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy']) 

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("--- เริ่มฝึกโมเดล LSTM ---")
    history = model.fit(X_train, y_train,
                        epochs=100, 
                        batch_size=32, 
                        validation_split=0.2, 
                        callbacks=[early_stopping], 
                        verbose=1) 
    print("--- ฝึกโมเดล LSTM เสร็จสิ้น ---")
    return model

# --- 5. ฟังก์ชันรัน Backtest Strategy ร่วมกับผลการทำนาย LSTM (Backtesting Strategy) ---
def run_backtest(df_test_original, lstm_model, scaler_features, lookback_period, features_for_lstm_names):
    """
    รันกลยุทธ์การเทรดบนข้อมูลทดสอบ โดยใช้เฉพาะผลการทำนายจาก LSTM
    """
    signals = pd.DataFrame(index=df_test_original.index)
    signals['Buy_Signal'] = 0 
    signals['Sell_Signal'] = 0 
    signals['LSTM_Prediction'] = 0.5 

    features_for_lstm = [col for col in features_for_lstm_names if col in df_test_original.columns]
    
    temp_df_for_lstm_scaling = df_test_original[features_for_lstm].copy()
    features_test_scaled = scaler_features.transform(temp_df_for_lstm_scaling.values)

    df_test_scaled_for_sequences = pd.DataFrame(features_test_scaled, 
                                                index=df_test_original.index, 
                                                columns=features_for_lstm)
    df_test_scaled_for_sequences['Target'] = 0 

    X_test_lstm, _ = create_sequences(df_test_scaled_for_sequences, lookback_period, target_column='Target')
    
    if len(X_test_lstm) > 0:
        raw_lstm_preds = lstm_model.predict(X_test_lstm).flatten()
        
        start_index_for_preds = lookback_period + FORECAST_HORIZON - 1
        
        expected_len_slice = len(signals) - start_index_for_preds

        print(f"DEBUG (run_backtest): Length of raw_lstm_preds: {len(raw_lstm_preds)}")
        print(f"DEBUG (run_backtest): Expected length of target slice in signals: {expected_len_slice}")

        if len(raw_lstm_preds) == expected_len_slice:
            signals.iloc[start_index_for_preds:, signals.columns.get_loc('LSTM_Prediction')] = raw_lstm_preds
        else:
            print(f"ERROR: Mismatch between number of predictions ({len(raw_lstm_preds)}) "
                  f"and expected target slice length ({expected_len_slice}). "
                  f"This indicates a deeper data alignment issue. Attempting partial assignment.")
            min_len = min(len(raw_lstm_preds), expected_len_slice)
            signals.iloc[start_index_for_preds : start_index_for_preds + min_len, signals.columns.get_loc('LSTM_Prediction')] = raw_lstm_preds[:min_len]
            print(f"WARNING: Assigned {min_len} predictions due to length mismatch.")

    else:
        print("ไม่สามารถสร้างลำดับข้อมูลทดสอบสำหรับ LSTM ได้เพียงพอ (ข้อมูลสั้นเกินไป)")
        return signals
    
    # ปรับ start_applying_signals_idx ให้ขึ้นอยู่กับ LOOKBACK_PERIOD และ FORECAST_HORIZON เท่านั้น
    start_applying_signals_idx = lookback_period + FORECAST_HORIZON - 1
    
    for i in range(start_applying_signals_idx, len(df_test_original)):
        current_data = df_test_original.iloc[i]
        lstm_pred = signals.loc[current_data.name, 'LSTM_Prediction']

        # --- เงื่อนไขการเข้าคำสั่ง BUY (ใช้เฉพาะ LSTM) ---
        lstm_predicts_up = lstm_pred > 0.5

        if lstm_predicts_up:
            signals.loc[current_data.name, 'Buy_Signal'] = 1

        # --- เงื่อนไขการเข้าคำสั่ง SELL (ใช้เฉพาะ LSTM) ---
        lstm_predicts_down = lstm_pred < 0.5

        if lstm_predicts_down:
            signals.loc[current_data.name, 'Sell_Signal'] = 1
            
    return signals

# --- 6. ฟังก์ชันสำหรับบันทึกโมเดลและพารามิเตอร์ (Export Model and Parameters) ---
def export_model_and_params(model, scaler_features, features_list, model_name="lstm_forex_model_combined"):
    """
    บันทึกโมเดล Keras, MinMaxScaler และรายการคุณสมบัติที่ใช้
    """
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
    
    model_path = os.path.join(EXPORT_DIR, f'{model_name}.keras')
    scaler_path = os.path.join(EXPORT_DIR, f'{model_name}_scaler.pkl')
    features_list_path = os.path.join(EXPORT_DIR, f'{model_name}_features_list.pkl')

    try:
        model.save(model_path)
        print(f"บันทึกโมเดล Keras ที่: {model_path}")

        joblib.dump(scaler_features, scaler_path)
        print(f"บันทึก Scaler ที่: {scaler_path}")

        joblib.dump(features_list, features_list_path)
        print(f"บันทึกรายการ Features ที่: {features_list_path}")
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")

# --- ส่วนหลักของการรันโปรแกรม (Main Execution Block) ---
def main():
    """
    ฟังก์ชันหลักที่ควบคุมขั้นตอนการทำงานทั้งหมด
    """
    print("\n--- ข้อมูลสำหรับการวินิจฉัย ---")
    print(f"Python executable ที่กำลังรันสคริปต์: {sys.executable}")
    
    try:
        print(f"เวอร์ชันของไลบรารี 'ta' ที่ติดตั้ง: {ta.__version__}")
    except AttributeError:
        print("ไม่พบ __version__ สำหรับไลบรารี 'ta' อาจจะติดตั้งไม่ถูกต้อง")

    try:
        macd_class_signature = inspect.signature(MACD.__init__)
        print(f"Signature ของ MACD Class Constructor: {macd_class_signature}")
        if 'window_sign' not in macd_class_signature.parameters:
            print("คำเตือน: 'window_sign' ไม่พบในพารามิเตอร์ของ MACD Class Constructor นี่อาจเป็นสาเหตุของปัญหา!")
            print("โปรดตรวจสอบเวอร์ชันของไลบรารี ta หรือพารามิเตอร์ที่ถูกต้อง")
        else:
            print("ยืนยัน: 'window_sign' พบในพารามิเตอร์ของ MACD Class Constructor")

    except AttributeError:
        print("ไม่สามารถตรวจสอบ MACD Class Constructor ได้ โปรดตรวจสอบว่า 'ta' ติดตั้งอย่างถูกต้อง")
    print("---------------------------------\n")

    if not os.path.exists(DATA_DIR):
        print(f"ข้อผิดพลาด: ไม่พบโฟลเดอร์ '{DATA_DIR}' โปรดสร้างโฟลเดอร์นี้และใส่ไฟล์ CSV ของคุณลงไป")
        return

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not all_files:
        print(f"ไม่พบไฟล์ CSV ในโฟลเดอร์ '{DATA_DIR}' โปรดตรวจสอบเส้นทางและชื่อไฟล์")
        return

    # --- NEW LOGIC: Combine all CSV files into one DataFrame ---
    combined_df = pd.DataFrame()
    print("--- กำลังรวมไฟล์ CSV ทั้งหมดเข้าด้วยกัน ---")
    for file_name in all_files:
        print(f"  - โหลดไฟล์: {file_name}")
        file_path = os.path.join(DATA_DIR, file_name)
        df_temp = load_and_preprocess_data(file_path)
        if not df_temp.empty:
            combined_df = pd.concat([combined_df, df_temp])
    
    if combined_df.empty:
        print("ข้อผิดพลาด: ไม่พบข้อมูลที่ถูกต้องจากไฟล์ CSV ทั้งหมด หรือไม่สามารถรวมข้อมูลได้")
        return

    # เรียงลำดับข้อมูลตามเวลา เพื่อให้แน่ใจว่าลำดับถูกต้อง
    combined_df = combined_df.sort_index()
    # --- เพิ่ม: ตรวจสอบและลบ Index ที่ซ้ำกันหลังจากรวมและเรียงลำดับ ---
    if not combined_df.index.is_unique:
        print("WARNING: Duplicate indices found in combined_df. Dropping duplicates (keeping first occurrence).")
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        print(f"combined_df after dropping duplicate indices: {len(combined_df)} rows.")
    # --- สิ้นสุดการเพิ่ม ---
    print(f"รวมข้อมูลทั้งหมดได้ {len(combined_df)} แถว")
    # --- END NEW LOGIC ---

    # ดำเนินการต่อด้วย DataFrame ที่รวมแล้ว
    df = add_technical_indicators(combined_df) 
    if df.empty:
        print(f"ไม่สามารถเพิ่มตัวชี้วัดทางเทคนิคได้ (อาจมีข้อมูลไม่พอหลังจากรวมและ Drop NaN)")
        return
    print(f"เพิ่มตัวชี้วัดทางเทคนิคและ Target แล้ว มี {len(df)} แถวหลัง Drop NaN")

    features_to_use = ['Open', 'High', 'Low', 'Close', 'TickVolume', 'RealVolume', 'Spread',
                         'SMA_Short', 'SMA_Long', 'EMA_Short', 'EMA_Long', 'RSI',
                         'MACD', 'MACD_Signal', 'MACD_Hist']
    
    available_features = [f for f in features_to_use if f in df.columns]
    if not available_features:
        print(f"ไม่พบคุณสมบัติที่เพียงพอสำหรับ LSTM จากข้อมูลที่รวมแล้ว")
        return

    train_size = int(len(df) * TRAIN_SPLIT_RATIO)
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()

    # --- เพิ่ม: ตรวจสอบและลบ Index ที่ซ้ำกันใน df_test ด้วย ---
    if not df_test.index.is_unique:
        print("WARNING: Duplicate indices found in df_test. Dropping duplicates (keeping first occurrence).")
        df_test = df_test[~df_test.index.duplicated(keep='first')]
        print(f"df_test after dropping duplicate indices: {len(df_test)} rows.")
    # --- สิ้นสุดการเพิ่ม ---

    if len(df_train) < LOOKBACK_PERIOD + FORECAST_HORIZON:
        print(f"ข้อมูลฝึกฝนไม่เพียงพอสำหรับสร้างลำดับ LSTM หลังจากรวมไฟล์")
        return
    
    scaler_features = MinMaxScaler()
    X_train_features_raw = df_train[available_features].values
    X_train_features_scaled = scaler_features.fit_transform(X_train_features_raw)

    df_train_scaled_for_sequences = pd.DataFrame(X_train_features_scaled, index=df_train.index, columns=available_features)
    df_train_scaled_for_sequences['Target'] = df_train['Target']

    X_train, y_train = create_sequences(df_train_scaled_for_sequences, LOOKBACK_PERIOD, target_column='Target')
    
    if X_train.shape[0] == 0:
        print(f"ไม่สามารถสร้างลำดับข้อมูลฝึกฝนได้เพียงพอจากข้อมูลที่รวมแล้ว")
        return

    print(f"เตรียมข้อมูลสำหรับ LSTM: X_train shape {X_train.shape}, y_train shape {y_train.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2]) 
    lstm_model = build_and_train_lstm(X_train, y_train, input_shape)

    # --- NEW LOGIC: Export only ONE set of files for the combined model ---
    export_model_and_params(lstm_model, scaler_features, available_features, model_name="lstm_forex_model_combined")
    # --- END NEW LOGIC ---

    if len(df_test) < LOOKBACK_PERIOD + FORECAST_HORIZON:
        print(f"ข้อมูลทดสอบไม่เพียงพอสำหรับ Backtest หลังจากรวมไฟล์")
        return

    print("--- กำลังเรียกใช้ Backtest Strategy (LSTM Only) ---")
    signals = run_backtest(df_test.copy(), lstm_model, scaler_features, LOOKBACK_PERIOD, available_features)
    print("--- Backtest Strategy (LSTM Only) เสร็จสิ้น ---")

    # --- ปรับปรุง Logic การจำลอง Backtest ให้สมจริงยิ่งขึ้น ---
    # ย้ายการประกาศ df_test_with_signals มาที่นี่
    df_test_with_signals = df_test.loc[signals.index].copy()
    df_test_with_signals['Buy_Signal'] = signals['Buy_Signal']
    df_test_with_signals['Sell_Signal'] = signals['Sell_Signal']

    position = 0 # 0: ไม่มีสถานะ, 1: Long (ซื้อ), -1: Short (ขาย)
    initial_balance = 10000.00
    balance = initial_balance
    trade_count = 0
    entry_price = 0.0 # ราคาที่เข้าสถานะปัจจุบัน
    
    # พารามิเตอร์สำหรับการจัดการความเสี่ยง (สามารถปรับเปลี่ยนได้)
    TRADE_AMOUNT_PER_TRADE = 100 # จำนวนเงินทุนที่ใช้ในการเทรดแต่ละครั้ง (สมมติเป็น USD)
    STOP_LOSS_PERCENT = 0.02  # 2% Stop Loss จากราคาเข้า
    TAKE_PROFIT_PERCENT = 0.03 # 3% Take Profit จากราคาเข้า

    # วนลูปผ่านข้อมูลทดสอบพร้อมสัญญาณ
    for i in range(len(df_test_with_signals)):
        current_data = df_test_with_signals.iloc[i]
        current_close = current_data['Close']
        current_high = current_data['High'] # ใช้สำหรับตรวจสอบ TP/SL
        current_low = current_data['Low']   # ใช้สำหรับตรวจสอบ TP/SL
        
        # --- จัดการสถานะที่เปิดอยู่ (ตรวจสอบ SL/TP) ---
        if position != 0:
            # ตรวจสอบ Stop Loss
            if position == 1: # สถานะ Long
                if current_low <= entry_price * (1 - STOP_LOSS_PERCENT):
                    pnl = -STOP_LOSS_PERCENT * TRADE_AMOUNT_PER_TRADE
                    balance += pnl
                    trade_count += 1
                    # print(f"SL Hit (Long) ที่ {current_low:.4f}. P/L: {pnl:.2f}. ยอดเงิน: {balance:.2f}")
                    position = 0
                    if balance <= 0: break # หยุดถ้าเงินหมด
                    continue # ไปยังแท่งเทียนถัดไป
            elif position == -1: # สถานะ Short
                if current_high >= entry_price * (1 + STOP_LOSS_PERCENT):
                    pnl = -STOP_LOSS_PERCENT * TRADE_AMOUNT_PER_TRADE
                    balance += pnl
                    trade_count += 1
                    # print(f"SL Hit (Short) ที่ {current_high:.4f}. P/L: {pnl:.2f}. ยอดเงิน: {balance:.2f}")
                    position = 0
                    if balance <= 0: break # หยุดถ้าเงินหมด
                    continue # ไปยังแท่งเทียนถัดไป

            # ตรวจสอบ Take Profit (ตรวจสอบหลังจาก SL เพื่อให้ SL มีความสำคัญกว่า)
            if position == 1: # สถานะ Long
                if current_high >= entry_price * (1 + TAKE_PROFIT_PERCENT):
                    pnl = TAKE_PROFIT_PERCENT * TRADE_AMOUNT_PER_TRADE
                    balance += pnl
                    trade_count += 1
                    # print(f"TP Hit (Long) ที่ {current_high:.4f}. P/L: {pnl:.2f}. ยอดเงิน: {balance:.2f}")
                    position = 0
                    if balance <= 0: break # หยุดถ้าเงินหมด
                    continue # ไปยังแท่งเทียนถัดไป
            elif position == -1: # สถานะ Short
                if current_low <= entry_price * (1 - TAKE_PROFIT_PERCENT):
                    pnl = TAKE_PROFIT_PERCENT * TRADE_AMOUNT_PER_TRADE
                    balance += pnl
                    trade_count += 1
                    # print(f"TP Hit (Short) ที่ {current_low:.4f}. P/L: {pnl:.2f}. ยอดเงิน: {balance:.2f}")
                    position = 0
                    if balance <= 0: break # หยุดถ้าเงินหมด
                    continue # ไปยังแท่งเทียนถัดไป

        # --- ตรวจสอบสัญญาณใหม่ (เฉพาะเมื่อไม่มีสถานะเปิดอยู่ หรือมีสัญญาณกลับทิศทาง) ---
        if position == 0: # ไม่มีสถานะเปิดอยู่
            if current_data['Buy_Signal'] == 1:
                position = 1 # เปิดสถานะ Long
                entry_price = current_close
                # print(f"เปิด Long ที่ {entry_price:.4f}. ยอดเงิน: {balance:.2f}")
            elif current_data['Sell_Signal'] == 1:
                position = -1 # เปิดสถานะ Short
                entry_price = current_close
                # print(f"เปิด Short ที่ {entry_price:.4f}. ยอดเงิน: {balance:.2f}")
        else: # มีสถานะเปิดอยู่ ตรวจสอบสัญญาณกลับทิศทาง
            if position == 1 and current_data['Sell_Signal'] == 1: # มี Long อยู่ แต่มีสัญญาณ Sell
                # ปิดสถานะ Long ก่อน
                pnl_percent = (current_close - entry_price) / entry_price
                pnl = pnl_percent * TRADE_AMOUNT_PER_TRADE
                balance += pnl
                trade_count += 1
                # print(f"ปิด Long ที่ {current_close:.4f}. P/L: {pnl:.2f}. ยอดเงิน: {balance:.2f}")
                
                if balance <= 0: break # หยุดถ้าเงินหมด

                # แล้วเปิดสถานะ Short ใหม่
                position = -1
                entry_price = current_close
                # print(f"เปิด Short ที่ {entry_price:.4f}. ยอดเงิน: {balance:.2f}")

            elif position == -1 and current_data['Buy_Signal'] == 1: # มี Short อยู่ แต่มีสัญญาณ Buy
                # ปิดสถานะ Short ก่อน
                pnl_percent = (entry_price - current_close) / entry_price
                pnl = pnl_percent * TRADE_AMOUNT_PER_TRADE
                balance += pnl
                trade_count += 1
                # print(f"ปิด Short ที่ {current_close:.4f}. P/L: {pnl:.2f}. ยอดเงิน: {balance:.2f}")
                
                if balance <= 0: break # หยุดถ้าเงินหมด

                # แล้วเปิดสถานะ Long ใหม่
                position = 1
                entry_price = current_close
                # print(f"เปิด Long ที่ {entry_price:.4f}. ยอดเงิน: {balance:.2f}")
        
    final_balance = balance
    total_return_percent = ((final_balance - initial_balance) / initial_balance) * 100

    print(f"\n--- ผลลัพธ์ Backtest สำหรับข้อมูลรวม (LSTM Only) ---")
    print(f"ยอดเงินเริ่มต้น: {initial_balance:.2f} USD")
    print(f"ยอดเงินสุดท้าย: {final_balance:.2f} USD")
    print(f"ผลตอบแทนรวม: {total_return_percent:.2f}%")
    print(f"จำนวนการเทรดทั้งหมด: {trade_count} ครั้ง")
    print("--- สิ้นสุดการประมวลผล ---")

if __name__ == "__main__":
    main()
