import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ta

from ta.trend import MACD 
from ta.momentum import RSIIndicator,StochasticOscillator 
from ta.trend import SMAIndicator, EMAIndicator 
from ta.volatility import BollingerBands, AverageTrueRange # เพิ่ม BollingerBands


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

ATR_PERIOD = 14
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3 # สำหรับ Stochastic Signal Line

# เพิ่ม Period สำหรับการตรวจจับ Divergence และ Candlestick Patterns
DIVERGENCE_LOOKBACK_PERIOD = 20 # จำนวนแท่งเทียนย้อนหลังที่ใช้หาจุด Swing High/Low สำหรับ Divergence
CANDLE_PATTERN_LOOKBACK = 5 # จำนวนแท่งเทียนย้อนหลังที่ใช้พิจารณา Candlestick Pattern (สำหรับบางรูปแบบ)

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

    # Average True Range (ATR)
    atr_indicator = AverageTrueRange(high=df_copy['High'], low=df_copy['Low'], close=df_copy['Close'], window=ATR_PERIOD)
    df_copy['ATR'] = atr_indicator.average_true_range()

    # Stochastic Oscillator
    stoch_indicator = StochasticOscillator(high=df_copy['High'], low=df_copy['Low'], close=df_copy['Close'], window=STOCH_K_PERIOD, smooth_window=STOCH_D_PERIOD)
    df_copy['Stoch_K'] = stoch_indicator.stoch()
    df_copy['Stoch_D'] = stoch_indicator.stoch_signal()

    # Bollinger Bands (เพื่อคำนวณ Band Width)
    bb_indicator = BollingerBands(close=df_copy['Close'], window=20, window_dev=2) # ตัวอย่าง period
    df_copy['BB_Upper'] = bb_indicator.bollinger_hband()
    df_copy['BB_Lower'] = bb_indicator.bollinger_lband()
    df_copy['BB_Width'] = df_copy['BB_Upper'] - df_copy['BB_Lower'] # Feature ใหม่: ความกว้างของ Bands

    # --- NEW: Candlestick Body and Shadow Ratios ---
    df_copy['Candle_Body'] = abs(df_copy['Close'] - df_copy['Open'])
    df_copy['Candle_Range'] = df_copy['High'] - df_copy['Low']

    # ป้องกันหารด้วยศูนย์
    df_copy['Candle_Body_Ratio'] = np.where(df_copy['Candle_Range'] != 0, df_copy['Candle_Body'] / df_copy['Candle_Range'], 0)
    
    df_copy['Upper_Shadow'] = df_copy['High'] - df_copy[['Open', 'Close']].max(axis=1)
    df_copy['Lower_Shadow'] = df_copy[['Open', 'Close']].min(axis=1) - df_copy['Low']

    df_copy['Upper_Shadow_Ratio'] = np.where(df_copy['Candle_Range'] != 0, df_copy['Upper_Shadow'] / df_copy['Candle_Range'], 0)
    df_copy['Lower_Shadow_Ratio'] = np.where(df_copy['Candle_Range'] != 0, df_copy['Lower_Shadow'] / df_copy['Candle_Range'], 0)

    # Simple Doji detection (body is very small relative to range)
    # ค่า Threshold 0.1 สามารถปรับได้
    df_copy['Is_Doji'] = ((df_copy['Candle_Body_Ratio'] < 0.1) & (df_copy['Candle_Range'] > df_copy['Close'].std() * 0.1)).astype(int) 
    
    # Simple Hammer/Hanging Man (long lower/upper shadow, small body, body at top/bottom of range)
    # Hammer: small body, long lower shadow, body at top of candle range
    df_copy['Is_Hammer'] = (((df_copy['Candle_Body_Ratio'] < 0.3) & 
                            (df_copy['Lower_Shadow_Ratio'] > 0.6) & 
                            (df_copy['Upper_Shadow_Ratio'] < 0.1) & # Upper shadow should be very small
                            (df_copy['Close'] > df_copy['Open'])).astype(int)) # Bullish hammer
    
    # Hanging Man: small body, long lower shadow, body at top of candle range (similar to hammer but bearish implication)
    # For simplicity, we'll use the same structure as hammer but check for bearish body
    df_copy['Is_HangingMan'] = (((df_copy['Candle_Body_Ratio'] < 0.3) & 
                                (df_copy['Lower_Shadow_Ratio'] > 0.6) & 
                                (df_copy['Upper_Shadow_Ratio'] < 0.1) & # Upper shadow should be very small
                                (df_copy['Close'] < df_copy['Open'])).astype(int)) # Bearish hanging man

    # --- NEW: Divergence Detection (Simplified) ---
    # This is a simplified approach. A more robust solution would involve
    # identifying significant peaks/troughs using a library like `scipy.signal.find_peaks`
    # and then comparing their values and positions.
    
    df_copy['RSI_Bullish_Divergence'] = 0
    df_copy['RSI_Bearish_Divergence'] = 0
    df_copy['Stoch_Bullish_Divergence'] = 0
    df_copy['Stoch_Bearish_Divergence'] = 0

    # Iterate through the DataFrame to find divergence patterns
    # We need enough history for the lookback period
    for i in range(DIVERGENCE_LOOKBACK_PERIOD, len(df_copy)):
        current_idx = df_copy.index[i]
        
        # Define the lookback window for finding previous lows/highs
        # We look back from current_idx - 1 to current_idx - DIVERGENCE_LOOKBACK_PERIOD
        lookback_slice = df_copy.iloc[i - DIVERGENCE_LOOKBACK_PERIOD : i]
        
        # --- RSI Divergence ---
        # Bullish Divergence: Price makes a lower low, RSI makes a higher low
        # Find the lowest price and corresponding RSI in the lookback_slice
        prev_low_price_idx = lookback_slice['Low'].idxmin()
        prev_low_price = lookback_slice.loc[prev_low_price_idx, 'Low']
        prev_low_rsi = lookback_slice.loc[prev_low_price_idx, 'RSI']

        if df_copy.loc[current_idx, 'Low'] < prev_low_price and \
           df_copy.loc[current_idx, 'RSI'] > prev_low_rsi:
            df_copy.loc[current_idx, 'RSI_Bullish_Divergence'] = 1

        # Bearish Divergence: Price makes a higher high, RSI makes a lower high
        # Find the highest price and corresponding RSI in the lookback_slice
        prev_high_price_idx = lookback_slice['High'].idxmax()
        prev_high_price = lookback_slice.loc[prev_high_price_idx, 'High']
        prev_high_rsi = lookback_slice.loc[prev_high_price_idx, 'RSI']

        if df_copy.loc[current_idx, 'High'] > prev_high_price and \
           df_copy.loc[current_idx, 'RSI'] < prev_high_rsi:
            df_copy.loc[current_idx, 'RSI_Bearish_Divergence'] = 1

        # --- Stochastic Divergence (using Stoch_K) ---
        # Bullish Divergence: Price makes a lower low, Stoch_K makes a higher low
        prev_low_stoch_k = lookback_slice.loc[prev_low_price_idx, 'Stoch_K'] # Use same price low index
        if df_copy.loc[current_idx, 'Low'] < prev_low_price and \
           df_copy.loc[current_idx, 'Stoch_K'] > prev_low_stoch_k:
            df_copy.loc[current_idx, 'Stoch_Bullish_Divergence'] = 1

        # Bearish Divergence: Price makes a higher high, Stoch_K makes a lower high
        prev_high_stoch_k = lookback_slice.loc[prev_high_price_idx, 'Stoch_K'] # Use same price high index
        if df_copy.loc[current_idx, 'High'] > prev_high_price and \
           df_copy.loc[current_idx, 'Stoch_K'] < prev_high_stoch_k:
            df_copy.loc[current_idx, 'Stoch_Bearish_Divergence'] = 1


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
        raw_lstm_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten() # เพิ่ม verbose=0
        
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
    
    model_path = os.path.join(EXPORT_DIR, f'{model_name}.h5')
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

    # อัปเดต features_to_use ให้รวม Features ใหม่ทั้งหมด
    features_to_use = ['Open', 'High', 'Low', 'Close', 'TickVolume', 'RealVolume', 'Spread',
                     'SMA_Short', 'SMA_Long', 'EMA_Short', 'EMA_Long', 'RSI',
                     'MACD', 'MACD_Signal', 'MACD_Hist',
                     'ATR', 'Stoch_K', 'Stoch_D', 'BB_Width',
                     'Candle_Body_Ratio', 'Upper_Shadow_Ratio', 'Lower_Shadow_Ratio',
                     'Is_Doji', 'Is_Hammer', 'Is_HangingMan',
                     'RSI_Bullish_Divergence', 'RSI_Bearish_Divergence',
                     'Stoch_Bullish_Divergence', 'Stoch_Bearish_Divergence']
    
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
    TAKE_PROFIT_PERCENT = 0.10 # 3% Take Profit จากราคาเข้า

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
