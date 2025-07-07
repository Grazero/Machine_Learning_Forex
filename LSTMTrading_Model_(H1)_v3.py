import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import joblib 
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical # สำหรับ One-Hot Encoding
from tensorflow.keras.optimizers import Adam # Import Adam optimizer

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.utils import class_weight # สำหรับจัดการ class imbalance

from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# --- Global Configurations (ตั้งค่าโมเดลและข้อมูล) ---
DATA_PATH = 'C:/ProjectLek/Machine_Learning_Forex/DataCSV/' # Path ไปยังโฟลเดอร์ที่มีไฟล์ CSV หลายไฟล์
MODEL_SAVE_PATH = './models/'
SCALER_SAVE_PATH = './scalers/'

# New: Configurations for Support/Resistance features
SWING_WINDOW = 10 # Lookback/lookforward window for identifying swing highs/lows
SR_THRESHOLD_PERCENT = 0.001 # 0.1% threshold to identify if price is "at" S/R level

# ชื่อคอลัมน์ที่คุณจะใช้เป็น Features สำหรับ LSTM (จะถูกอัปเดตแบบไดนามิกสำหรับ Symbol One-Hot Encoding)
# นี่คือชุด Features ที่ถูกลดทอนลงเหลือเพียงที่เกี่ยวข้องกับเงื่อนไขของคุณ
FEATURES_TO_USE = [
    'open_H1', 'high_H1', 'low_H1', 'close_H1', 'realvolume_H1', 
    'RSI_H1', 
    'Body_size_H1', 'Upper_shadow_H1', 'Lower_shadow_H1', # Candlestick features (normalized by ATR)
    # Support/Resistance Features
    'distance_to_nearest_swing_high_H1', 
    'distance_to_nearest_swing_low_H1',
    'is_at_swing_high_area_H1',
    'is_at_swing_low_area_H1',
    # RSI Crossing Signals
    'signal_rsi_oversold_H1', 
    'signal_rsi_overbought_H1', 
    'rsi_cross_below_30_H1',
    'rsi_cross_above_70_H1',
    # Candlestick Patterns:
    'bullish_engulfing_H1', 
    'bearish_engulfing_H1', 
    'hammer_H1', 
    'shooting_star_H1', 
    'doji_val_H1',
    # Combined Entry Signals
    'bullish_entry_signal_H1',
    'bearish_entry_signal_H1'
]

# Removed Lagged Features as per "clean code" request.
# Removed other indicators like EMA, MACD, ADX, BB, STOCH, OBV, their related signals, and slopes.

# Target Definition: 0 (Sell), 1 (Buy), 2 (Hold)
# New: Price change threshold for defining Buy/Sell signals (e.g., 0.05% for profit)
PRICE_CHANGE_THRESHOLD_PERCENT = 0.0005 # 0.05% profit/loss threshold per bar
TARGET_COLUMN = 'target' # Column name for the 3-class target

SEQUENCE_LENGTH = 60 # จำนวน Time Steps สำหรับ LSTM (ควรเท่ากับ InpSequenceLength ใน MT5 EA)
BATCH_SIZE = 32
BUFFER_SIZE = 100000 # สำหรับ tf.data.Dataset
EPOCHS = 100 # จำนวน Epochs สูงสุด
PATIENCE = 10 # จำนวน Epochs ที่จะรอหาก Validation Loss ไม่ดีขึ้น (สำหรับ EarlyStopping)

# Revert LSTM_UNITS and DROPOUT_RATE to simpler values for clean start
LSTM_UNITS = 64 
DROPOUT_RATE = 0.2 
LEARNING_RATE = 0.001 # Custom learning rate for Adam optimizer
SPLIT_RATIO = 0.8 # 80% train, 20% validation+test

# --- New Function: Calculate Swing Highs/Lows and S/R Features ---
def _calculate_swing_sr(df, window=SWING_WINDOW, sr_threshold_percent=SR_THRESHOLD_PERCENT):
    """
    Calculates swing highs, swing lows, and derived S/R features.
    
    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns.
        window (int): Lookback/lookforward window to define a swing point.
        sr_threshold_percent (float): Percentage threshold to define an S/R "area".
        
    Returns:
        pd.DataFrame: Original DataFrame with new S/R features.
    """
    df['is_swing_high'] = False
    df['is_swing_low'] = False

    # Identify Swing Highs and Swing Lows
    if len(df) > 2 * window:
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == df['high'].iloc[i - window : i + window + 1].max():
                df.loc[df.index[i], 'is_swing_high'] = True
            if df['low'].iloc[i] == df['low'].iloc[i - window : i + window + 1].min():
                df.loc[df.index[i], 'is_swing_low'] = True

    # Get the actual values of swing highs/lows
    swing_high_prices = df[df['is_swing_high']]['high']
    swing_low_prices = df[df['is_swing_low']]['low']

    # Initialize new S/R features
    df['distance_to_nearest_swing_high'] = np.nan
    df['distance_to_nearest_swing_low'] = np.nan
    df['is_at_swing_high_area'] = 0 # Binary (0 or 1)
    df['is_at_swing_low_area'] = 0 # Binary (0 or 1)

    for i in range(len(df)):
        current_close = df['close'].iloc[i]
        
        # Distance to nearest Swing High
        if not swing_high_prices.empty:
            relevant_swing_highs = swing_high_prices[swing_high_prices.index <= df.index[i]]
            if not relevant_swing_highs.empty:
                nearest_high_price = relevant_swing_highs.loc[(relevant_swing_highs - current_close).abs().idxmin()]
                df.loc[df.index[i], 'distance_to_nearest_swing_high'] = (nearest_high_price - current_close) 
                if abs(df.loc[df.index[i], 'distance_to_nearest_swing_high']) <= sr_threshold_percent * current_close:
                    df.loc[df.index[i], 'is_at_swing_high_area'] = 1

        # Distance to nearest Swing Low
        if not swing_low_prices.empty:
            relevant_swing_lows = swing_low_prices[swing_low_prices.index <= df.index[i]]
            if not relevant_swing_lows.empty:
                nearest_low_price = relevant_swing_lows.loc[(relevant_swing_lows - current_close).abs().idxmin()]
                df.loc[df.index[i], 'distance_to_nearest_swing_low'] = (current_close - nearest_low_price) 
                if abs(df.loc[df.index[i], 'distance_to_nearest_swing_low']) <= sr_threshold_percent * current_close:
                    df.loc[df.index[i], 'is_at_swing_low_area'] = 1

    # Drop temporary columns
    df.drop(columns=['is_swing_high', 'is_swing_low'], inplace=True, errors='ignore')
    return df

# --- 1. ฟังก์ชันช่วยในการโหลดข้อมูลจาก CSV และสร้าง Features ---
def _load_and_create_features_from_csv(file_path):
    """
    โหลดข้อมูลราคาจากไฟล์ CSV และสร้าง features ทางเทคนิค.
    """
    try:
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
        df.sort_index(inplace=True)
        
        # เพิ่ม: ตรวจสอบและลบ Index ที่ซ้ำกัน หากมี (ภายในไฟล์ CSV เดียว)
        if not df.index.is_unique:
            print(f"⚠️ Warning: Duplicate timestamps found in {file_path}. Dropping duplicates.")
            df = df.loc[~df.index.duplicated(keep='first')] # เก็บค่าแรกของ Timestamp ที่ซ้ำกัน
            
    except Exception as e:
        print(f"❌ ไม่สามารถอ่านไฟล์ CSV ได้: {file_path}. Error: {e}")
        return pd.DataFrame() 

    # ทำให้ชื่อคอลัมน์ทั้งหมดเป็นตัวพิมพ์เล็กเสมอ
    df.columns = df.columns.str.lower()
    
    # กำหนดคอลัมน์ที่จำเป็นที่เราคาดหวังในชื่อตัวพิมพ์เล็ก
    required_cols_expected = ['open', 'high', 'low', 'close', 'tickvolume', 'realvolume', 'spread']
    
    # สร้าง DataFrame ใหม่ที่มีเฉพาะคอลัมน์ที่เราต้องการ
    df_processed = pd.DataFrame(index=df.index)
    
    for col_name in required_cols_expected:
        if col_name in df.columns:
            df_processed[col_name] = pd.to_numeric(df[col_name], errors='coerce') # Ensure numeric
        else:
            print(f"⚠️ ไฟล์ {file_path} ไม่มีคอลัมน์ '{col_name}' หลังจากแปลงเป็นตัวพิมพ์เล็ก. จะเติมด้วย NaN.")
            if col_name in ['open', 'high', 'low', 'close']:
                df_processed[col_name] = np.nan # เติมด้วย NaN สำหรับคอลัมน์ราคา
            else:
                df_processed[col_name] = 0 # เติมด้วย 0 สำหรับคอลัมน์อื่น (volume, spread)
    
    df = df_processed.copy() # แทนที่ DataFrame เดิมด้วย DataFrame ที่ประมวลผลแล้ว
    
    df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) # Drop rows with missing price data

    if 'high' not in df.columns or df['high'].empty: # Add check for empty column after dropna
        print(f"DEBUG Critical: 'high' column is still missing or empty after processing! This indicates a problem with the CSV data.")
        return pd.DataFrame() # คืนค่า DataFrame ว่างเปล่าเพื่อหยุดการทำงาน

    # --- Feature Engineering: เพิ่ม Indicators ที่จำเป็นเท่านั้น ---
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    # คำนวณขนาดของ Body และ Shadow ก่อน Normalize
    df['Body_size_raw'] = abs(df['close'] - df['open'])
    df['Upper_shadow_raw'] = df['high'] - df[['close','open']].max(axis=1)
    df['Lower_shadow_raw'] = df[['close','open']].min(axis=1) - df['low']

    # --- Normalize Body_size และ Shadow ด้วย ATR ---
    df['Body_size'] = np.where(df['ATR'].fillna(0) > 0, df['Body_size_raw'] / df['ATR'], 0)
    df['Upper_shadow'] = np.where(df['ATR'].fillna(0) > 0, df['Upper_shadow_raw'] / df['ATR'], 0)
    df['Lower_shadow'] = np.where(df['ATR'].fillna(0) > 0, df['Lower_shadow_raw'] / df['ATR'], 0)
    
    # ลบคอลัมน์ raw ที่ไม่จำเป็นแล้ว
    df.drop(columns=['Body_size_raw', 'Upper_shadow_raw', 'Lower_shadow_raw'], errors='ignore', inplace=True)


    # --- Signal Features (เฉพาะที่เกี่ยวข้องกับเงื่อนไขของคุณ) ---
    signal_cols_to_init = [
        'signal_rsi_oversold', 'signal_rsi_overbought', 
        'rsi_cross_below_30', 'rsi_cross_above_70', # New RSI crosses
        'bullish_engulfing', 'bearish_engulfing', 
        'hammer', 'shooting_star', 'doji_val',
        'bullish_entry_signal', 'bearish_entry_signal' # New combined signals
    ]
    for col in signal_cols_to_init:
        df[col] = 0.0

    # RSI Overbought/Oversold
    df.loc[df['RSI'] < 30, 'signal_rsi_oversold'] = 1
    df.loc[df['RSI'] > 70, 'signal_rsi_overbought'] = 1

    # RSI Crosses
    df.loc[(df['RSI'].shift(1) >= 30) & (df['RSI'] < 30), 'rsi_cross_below_30'] = 1
    df.loc[(df['RSI'].shift(1) <= 70) & (df['RSI'] > 70), 'rsi_cross_above_70'] = 1

    # --- Candlestick Patterns ---
    # Bullish Engulfing
    bullish_engulfing_cond = (
        (df['close'] > df['open']) & # Current bar is bullish
        (df['open'].shift(1) > df['close'].shift(1)) & # Previous bar is bearish
        (df['open'] < df['close'].shift(1)) & # Current open is below previous close
        (df['close'] > df['open'].shift(1)) & # Current close is above previous open
        (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))) # Current body is larger than previous body
    )
    df.loc[bullish_engulfing_cond, 'bullish_engulfing'] = 1

    # Bearish Engulfing
    bearish_engulfing_cond = (
        (df['close'] < df['open']) & # Current bar is bearish
        (df['open'].shift(1) < df['close'].shift(1)) & # Previous bar is bullish (Correction from prev version, needs to be bullish to engulf)
        (df['open'] > df['close'].shift(1)) & # Current open is above previous close
        (df['close'] < df['open'].shift(1)) & # Current close is below previous open
        (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))) # Current body is larger than previous body
    )
    df.loc[bearish_engulfing_cond, 'bearish_engulfing'] = 1

    # Hammer (simplified: small body, long lower shadow, small upper shadow)
    hammer_cond = (
        (df['Body_size'] > 0) & # Must have a body
        (df['Lower_shadow'] >= 2 * df['Body_size']) & # Lower shadow at least twice the body
        (df['Upper_shadow'] <= 0.2 * df['Body_size']) & # Small or no upper shadow
        (df['ATR'].fillna(0) > 0) # Ensure ATR is not zero for proper normalization
    )
    df.loc[hammer_cond, 'hammer'] = 1

    # Shooting Star (simplified: small body, long upper shadow, small lower shadow)
    shooting_star_cond = (
        (df['Body_size'] > 0) & # Must have a body
        (df['Upper_shadow'] >= 2 * df['Body_size']) & # Upper shadow at least twice the body
        (df['Lower_shadow'] <= 0.2 * df['Body_size']) & # Small or no lower shadow
        (df['ATR'].fillna(0) > 0) # Ensure ATR is not zero for proper normalization
    )
    df.loc[shooting_star_cond, 'shooting_star'] = 1
    
    # Doji (simplified: very small body relative to total range and ATR)
    doji_cond = (
        (df['Body_size'] < (df['high'] - df['low']) * 0.1) & # Body is less than 10% of total range
        ((df['high'] - df['low']) > df['ATR'] * 0.1) & # Total range is still significant relative to ATR
        (df['ATR'].fillna(0) > 0) # Ensure ATR is not zero for proper normalization
    )
    df.loc[doji_cond, 'doji_val'] = 1

    # New: Calculate Swing S/R Features
    # Ensure 'ATR' is not NaN for normalization before passing to _calculate_swing_sr
    df_processed_sr = _calculate_swing_sr(df.copy(), window=SWING_WINDOW, sr_threshold_percent=SR_THRESHOLD_PERCENT)
    # Merge S/R features back to the main df
    df = df.merge(df_processed_sr[['distance_to_nearest_swing_high', 'distance_to_nearest_swing_low', 
                                    'is_at_swing_high_area', 'is_at_swing_low_area']],
                  left_index=True, right_index=True, how='left')


    # New: Combined Entry Signals (based on user's rules)
    # Bullish Entry Signal
    bullish_entry_cond = (
        (df['is_at_swing_low_area'] == 1) & 
        ((df['signal_rsi_oversold'] == 1) | (df['rsi_cross_below_30'] == 1)) & 
        ((df['hammer'] == 1) | (df['bullish_engulfing'] == 1) | (df['doji_val'] == 1))
    )
    df.loc[bullish_entry_cond, 'bullish_entry_signal'] = 1

    # Bearish Entry Signal
    bearish_entry_cond = (
        (df['is_at_swing_high_area'] == 1) & 
        ((df['signal_rsi_overbought'] == 1) | (df['rsi_cross_above_70'] == 1)) & 
        ((df['shooting_star'] == 1) | (df['bearish_engulfing'] == 1) | (df['doji_val'] == 1))
    )
    df.loc[bearish_entry_cond, 'bearish_entry_signal'] = 1

    df.dropna(inplace=True) # Drop rows with NaN values created by indicators or S/R calculation
    return df

# --- 2. ฟังก์ชันหลักในการโหลดและเตรียมข้อมูล Multi-Timeframe สำหรับการเทรน (ใช้ MultiIndex) ---
def load_and_preprocess_multi_timeframe_data_from_csv(data_folder=DATA_PATH):
    all_combined_dfs = []
    
    if not os.path.exists(data_folder):
        print(f"❌ ไม่พบโฟลเดอร์ข้อมูล '{data_folder}'. โปรดตรวจสอบว่าได้วางไฟล์ CSV ไว้ในโฟลเดอร์นี้.")
        sys.exit(1)
        
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ ไม่พบไฟล์ CSV ในโฟลเดอร์ '{data_folder}'. โปรดตรวจสอบว่าไฟล์ถูกวางไว้ถูกต้อง.")
        sys.exit(1)

    symbols_tfs = {} 
    for f_name in csv_files:
        parts = f_name.replace('.csv', '').split('_')
        # แก้ไขเงื่อนไขการตรวจสอบชื่อไฟล์ให้ยืดหยุ่นขึ้น (HistoricalData_SYMBOL_PERIOD_TF.csv)
        if len(parts) >= 4 and parts[0] == 'HistoricalData' and parts[2].startswith('PERIOD'): 
            symbol = parts[1] 
            tf_str = parts[3] 
            if symbol not in symbols_tfs:
                symbols_tfs[symbol] = {}
            symbols_tfs[symbol][tf_str] = os.path.join(data_folder, f_name)
        else:
            print(f"⚠️ ชื่อไฟล์ '{f_name}' ไม่ตรงตามรูปแบบที่คาดหวัง (HistoricalData_SYMBOL_PERIOD_TF.csv). จะข้ามไฟล์นี้.")
    
    if not symbols_tfs:
        print(f"❌ ไม่สามารถแยก Symbol และ Timeframe จากชื่อไฟล์ CSV ได้. รูปแบบชื่อไฟล์อาจไม่ถูกต้อง.")
        sys.exit(1)

    print(f"✅ ตรวจพบ {len(symbols_tfs)} Symbol ในโฟลเดอร์ '{data_folder}'.")
    for symbol_name, tfs_data in symbols_tfs.items():
        print(f"   Symbol: {symbol_name}, Timeframes: {list(tfs_data.keys())}")
        
        df_h1_file = tfs_data.get('H1')
        df_m15_file = tfs_data.get('M15')
        df_h4_file = tfs_data.get('H4')
        
        # NOTE: Keeping this multi-timeframe load structure from the provided immersive.
        if not df_h1_file or not df_m15_file or not df_h4_file:
            print(f"⚠️ ข้อมูลไม่ครบทุก Timeframe (H1, M15, H4) สำหรับ Symbol: {symbol_name}. จะข้าม Symbol นี้ไป.")
            continue

        print(f"กำลังโหลดและสร้าง Features สำหรับ {symbol_name}...")
        
        df_h1 = _load_and_create_features_from_csv(df_h1_file).add_suffix('_H1')
        if df_h1.empty: print(f"   ❌ DataFrame H1 ว่างเปล่าสำหรับ {symbol_name}. ข้าม Symbol นี้."); continue
        df_m15 = _load_and_create_features_from_csv(df_m15_file).add_suffix('_M15')
        if df_m15.empty: print(f"   ❌ DataFrame M15 ว่างเปล่าสำหรับ {symbol_name}. ข้าม Symbol นี้."); continue
        df_h4 = _load_and_create_features_from_csv(df_h4_file).add_suffix('_H4')
        if df_h4.empty: print(f"   ❌ DataFrame H4 ว่างเปล่าสำหรับ {symbol_name}. ข้าม Symbol นี้."); continue

        print(f"   กำลังจัดเรียงข้อมูล Multi-Timeframe สำหรับ {symbol_name}...")
        df_combined = pd.merge_asof(df_h1, df_m15, left_index=True, right_index=True, direction='backward')
        df_combined = pd.merge_asof(df_combined, df_h4, left_index=True, right_index=True, direction='backward')
        
        df_combined['Symbol'] = symbol_name 
        
        df_combined.dropna(inplace=True) 
        
        if df_combined.empty:
            print(f"   ❌ Combined DataFrame ว่างเปล่าหลังจากรวมและลบ NaNs สำหรับ {symbol_name}. ข้าม Symbol นี้.")
            continue
        
        all_combined_dfs.append(df_combined)
        print(f"   ✅ เตรียมข้อมูลสำหรับ {symbol_name} เสร็จสิ้น. Shape: {df_combined.shape}")

    if not all_combined_dfs:
        print("❌ ไม่มีข้อมูลที่พร้อมสำหรับการเทรนจาก Symbol ที่ถูกโหลดมาทั้งหมด.")
        sys.exit(1)

    print("กำลังรวมข้อมูลจากทุก Symbol เข้าด้วยกัน...")
    final_combined_df = pd.concat(all_combined_dfs, axis=0)
    final_combined_df.sort_index(inplace=True) 

    # --- ปรับปรุง: สร้าง Target Variable 3 Classes (Sell, Buy, Hold) ---
    final_combined_df['next_close_H1'] = final_combined_df.groupby('Symbol')['close_H1'].shift(-1)
    
    # Explicitly calculate future_return_H1
    final_combined_df['future_return_H1'] = (final_combined_df['next_close_H1'] - final_combined_df['close_H1']) / final_combined_df['close_H1']

    # Initialize target to 'Hold' (Class 2)
    final_combined_df[TARGET_COLUMN] = 2 

    # Class 1: Buy (price goes up by threshold)
    buy_cond = final_combined_df['future_return_H1'] >= PRICE_CHANGE_THRESHOLD_PERCENT
    final_combined_df.loc[buy_cond, TARGET_COLUMN] = 1

    # Class 0: Sell (price goes down by threshold)
    sell_cond = final_combined_df['future_return_H1'] <= -PRICE_CHANGE_THRESHOLD_PERCENT # Corrected for negative threshold for sell
    final_combined_df.loc[sell_cond, TARGET_COLUMN] = 0

    # DO NOT drop 'next_close_H1' here, it's needed for metrics calculation later
    # final_combined_df.drop(columns=['next_close_H1'], inplace=True)

    final_combined_df.dropna(subset=[TARGET_COLUMN, 'future_return_H1', 'next_close_H1'], inplace=True) # Ensure these critical columns are not NaN
    
    if final_combined_df.empty:
        print("❌ Final Combined DataFrame ว่างเปล่าหลังจากสร้าง Target และลบ NaNs. สคริปต์จะหยุดทำงาน.")
        sys.exit()

    # *** CRITICAL STEP: ตอนนี้ทำ MultiIndex (Time, Symbol) เพื่อให้แต่ละแถวมี Index ที่ไม่ซ้ำกัน ***
    # เนื่องจาก 'Time' เป็น Index อยู่แล้ว เราจะเพิ่ม 'Symbol' เข้าไปใน Index
    final_combined_df.set_index('Symbol', append=True, inplace=True) # เพิ่ม 'Symbol' เป็น Level ที่ 2 ของ Index
    final_combined_df.sort_index(inplace=True) # จัดเรียงตาม MultiIndex

    print(f"DEBUG: Final Combined DataFrame shape after MultiIndex: {final_combined_df.shape}")
    print(f"DEBUG: Final Combined DataFrame index unique after MultiIndex? {final_combined_df.index.is_unique}")
    if not final_combined_df.index.is_unique:
        print(f"DEBUG: ERROR: Final Combined DataFrame still has {final_combined_df.index.duplicated().sum()} duplicate MultiIndex entries. Removing them.")
        final_combined_df = final_combined_df.loc[~final_combined_df.index.duplicated(keep='first')]
        print(f"DEBUG: Removed duplicates. New shape: {final_combined_df.shape}")

    # --- สร้าง df_with_one_hot สำหรับ Features (X) และ Target (y) ---
    # เนื่องจาก 'Symbol' ตอนนี้เป็นส่วนหนึ่งของ MultiIndex แล้ว
    # เราต้อง reset_index ชั่วคราวเพื่อใช้ pd.get_dummies
    df_for_one_hot = final_combined_df.reset_index() # 'Time' และ 'Symbol' กลับมาเป็นคอลัมน์
    
    # ทำ One-Hot Encoding สำหรับ Symbol (เพื่อให้โมเดลเรียนรู้ความแตกต่างระหว่าง Symbol)
    df_with_one_hot = pd.get_dummies(df_for_one_hot, columns=['Symbol'], prefix='Symbol')
    
    # ตั้งค่า MultiIndex กลับไป เพื่อให้ X และ y มี MultiIndex (Time, Symbol)
    df_with_one_hot.set_index(['Time', df_for_one_hot['Symbol']], inplace=True, verify_integrity=True) 
    df_with_one_hot.index.set_names(['Time', 'Symbol'], inplace=True)
    df_with_one_hot.sort_index(inplace=True) # จัดเรียงอีกครั้ง

    # List ของ Features ที่จะใช้ในการเทรน (รวม One-Hot Encoded Symbol Features)
    features_list_for_X_with_one_hot = [
        col for col in df_with_one_hot.columns 
        if col not in ['next_close_H1', 'future_return_H1', TARGET_COLUMN] # Ensure targets are not in features
    ]
    
    # Ensure all FEATURES_TO_USE are in the dataframe and add symbol one-hot features
    final_features_list = []
    for feature in FEATURES_TO_USE:
        if feature in df_with_one_hot.columns:
            final_features_list.append(feature)
        else:
            print(f"WARNING: Feature '{feature}' not found in the processed DataFrame.")
    
    # Add one-hot encoded symbol features
    one_hot_symbol_features = [col for col in df_with_one_hot.columns if col.startswith('Symbol_')]
    for feature in one_hot_symbol_features:
        if feature not in final_features_list: # Avoid duplicates
            final_features_list.append(feature)

    X = df_with_one_hot[final_features_list]
    y = df_with_one_hot[TARGET_COLUMN].astype(int) 

    # Scale Features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_features_list, index=X.index) # Preserve MultiIndex

    print(f"ชุดข้อมูลรวมพร้อมแล้ว. X shape: {X_scaled_df.shape}, y shape: {y.shape}")
    print(f"DEBUG: X_scaled_df index unique? {X_scaled_df.index.is_unique}") # ควรจะเป็น TRUE!
    
    return X_scaled_df, y, scaler, final_features_list, final_combined_df # ส่ง final_combined_df กลับไปด้วยสำหรับ Metrics

# --- New Function: สร้าง Sequences สำหรับ LSTM ---
def create_sequences_for_lstm(X_data, y_data, timesteps, dataset_name=""):
    X_sequences = []
    y_sequences = []
    y_indices_list = [] 
    
    print(f"DEBUG {dataset_name}: Starting sequence creation. X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")
    
    # Ensure the initial X_data and y_data have MultiIndex
    if not (isinstance(X_data.index, pd.MultiIndex) and X_data.index.nlevels == 2 and
            isinstance(y_data.index, pd.MultiIndex) and y_data.index.nlevels == 2):
        print(f"CRITICAL ERROR {dataset_name}: Initial X_data or y_data does not have expected MultiIndex structure. Exiting function.")
        empty_X_shape = (0, timesteps, X_data.shape[1] if X_data.shape[0] > 0 else 0)
        return np.array([]).reshape(empty_X_shape), np.array([]), pd.MultiIndex.from_tuples([], names=['Time', 'Symbol'])


    unique_symbols = X_data.index.get_level_values('Symbol').unique()
    print(f"DEBUG {dataset_name}: Found {len(unique_symbols)} unique symbols in this split.")

    if len(unique_symbols) == 0:
        print(f"DEBUG {dataset_name}: No unique symbols found. Returning empty sequences.")
        empty_X_shape = (0, timesteps, X_data.shape[1] if X_data.shape[0] > 0 else 0)
        return np.array([]).reshape(empty_X_shape), np.array([]), pd.MultiIndex.from_tuples([], names=['Time', 'Symbol'])


    for symbol in unique_symbols:
        symbol_X = X_data.loc[(slice(None), symbol), :]
        symbol_y = y_data.loc[(slice(None), symbol)]
        
        print(f"DEBUG {dataset_name} Symbol: {symbol}: Length of symbol_X: {len(symbol_X)}, Length of symbol_y: {len(symbol_y)}")
        
        if len(symbol_X) <= timesteps:
            print(f"DEBUG {dataset_name}: Skipping symbol '{symbol}' due to insufficient data ({len(symbol_X)} bars) for {timesteps} timesteps.")
            continue

        current_symbol_sequences_count = 0 
        for i in range(len(symbol_X) - timesteps):
            target_datetime = symbol_X.index[i + timesteps]
            current_multi_index_tuple = (target_datetime, symbol) 
            y_indices_list.append(current_multi_index_tuple) 

            X_sequences.append(symbol_X.iloc[i:(i + timesteps)].values)
            y_sequences.append(symbol_y.iloc[i + timesteps]) 
            current_symbol_sequences_count += 1 

        print(f"DEBUG {dataset_name} Symbol: {symbol}: Successfully created {current_symbol_sequences_count} sequences.")
    
    print(f"DEBUG {dataset_name}: Total sequences before duplicate handling: X={len(X_sequences)}, y={len(y_sequences)}, indices={len(y_indices_list)}")

    if not y_indices_list:
        print(f"DEBUG {dataset_name}: y_indices_list is empty after all symbols processed. Returning empty arrays/index.")
        empty_X_shape = (0, timesteps, X_data.shape[1] if X_data.shape[0] > 0 else 0)
        return np.array([]).reshape(empty_X_shape), np.array([]), pd.MultiIndex.from_tuples([], names=['Time', 'Symbol'])

    combined_multiindex = pd.MultiIndex.from_tuples(y_indices_list, names=['Time', 'Symbol'])
    temp_df = pd.DataFrame({'y_seq': y_sequences}, index=combined_multiindex)
    
    initial_temp_df_len = len(temp_df)
    if not temp_df.index.is_unique:
        num_duplicates = temp_df.index.duplicated().sum()
        print(f"DEBUG {dataset_name}: Found {num_duplicates} duplicate target indices after sequence creation. Filtering to unique.")
        temp_df = temp_df.loc[~temp_df.index.duplicated(keep='first')]
    print(f"DEBUG {dataset_name}: temp_df length after duplicate filtering: {len(temp_df)} (originally {initial_temp_df_len})")

    original_index_map = {idx_tuple: i for i, idx_tuple in enumerate(y_indices_list)}
    
    filtered_X_sequences = []
    filtered_y_sequences = []
    filtered_indices = []

    for unique_idx_tuple in temp_df.index:
        original_seq_idx = original_index_map.get(unique_idx_tuple)
        if original_seq_idx is not None:
            filtered_X_sequences.append(X_sequences[original_seq_idx])
            filtered_y_sequences.append(y_sequences[original_seq_idx])
            filtered_indices.append(unique_idx_tuple)

    print(f"DEBUG {dataset_name}: Final lengths after sequence filtering: X_sequences={len(filtered_X_sequences)}, y_sequences={len(filtered_y_sequences)}, y_indices={len(filtered_indices)}")
    
    return np.array(filtered_X_sequences), np.array(filtered_y_sequences), pd.MultiIndex.from_tuples(filtered_indices, names=['Time', 'Symbol'])

# --- 3. ฟังก์ชันสำหรับสร้างและเทรนโมเดล LSTM ---
def build_and_train_lstm_model(X_train, y_train, X_valid, y_valid, timesteps, num_features, num_classes, learning_rate=LEARNING_RATE):
    """
    สร้างและเทรนโมเดล LSTM Classifier.
    """
    # ตรวจสอบค่า NaN/inf ในข้อมูล Input ของ Keras
    if np.isnan(X_train).sum() > 0 or np.isinf(X_train).sum() > 0:
        print(f"CRITICAL ERROR: NaN or Inf found in X_train. NaNs: {np.isnan(X_train).sum()}, Infs: {np.isinf(X_train).sum()}")
        raise ValueError("NaN or Inf values detected in X_train. Cannot proceed with model training.")
    if np.isnan(y_train).sum() > 0 or np.isinf(y_train).sum() > 0:
        print(f"CRITICAL ERROR: NaN or Inf found in y_train. NaNs: {np.isnan(y_train).sum()}, Infs: {np.isinf(y_train).sum()}")
        raise ValueError("NaN or Inf values detected in y_train. Cannot proceed with model training.")
    if np.isnan(X_valid).sum() > 0 or np.isinf(X_valid).sum() > 0:
        print(f"CRITICAL ERROR: NaN or Inf found in X_valid. NaNs: {np.isnan(X_valid).sum()}, Infs: {np.isinf(X_valid).sum()}")
        raise ValueError("NaN or Inf values detected in X_valid. Cannot proceed with model training.")
    if np.isnan(y_valid).sum() > 0 or np.isinf(y_valid).sum() > 0:
        print(f"CRITICAL ERROR: NaN or Inf found in y_valid. NaNs: {np.isnan(y_valid).sum()}, Infs: {np.isinf(y_valid).sum()}")
        raise ValueError("NaN or Inf values detected in y_valid. Cannot proceed with model training.")

    model = Sequential([
        LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=(timesteps, num_features)),
        Dropout(DROPOUT_RATE),
        LSTM(units=LSTM_UNITS // 2, return_sequences=False), # Simpler architecture
        Dropout(DROPOUT_RATE),
        Dense(units=num_classes, activation='softmax') # Softmax สำหรับ Multi-class Classification
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for multi-class
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_lstm_model.keras', monitor='val_loss', save_best_only=True, mode='min')

    print("กำลังเทรนโมเดล LSTM Classifier...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    print("การเทรนโมเดล LSTM เสร็จสมบูรณ์.")
    
    # โหลดโมเดลที่ดีที่สุดกลับมา
    model = tf.keras.models.load_model('best_lstm_model.keras')
    return model, history

# --- 4. ฟังก์ชันสำหรับประเมินผลโมเดล (ปรับสำหรับ 3 Classes) ---
def evaluate_model(model, X_test, y_test, df_test_for_metrics=None, deduct_spread_in_metrics=True, num_classes=3):
    """
    ประเมินประสิทธิภาพของโมเดลบนชุดข้อมูลทดสอบ และคำนวณ Metrics ทางการเทรด.
    ปรับปรุงสำหรับ Multi-class Classification (Buy, Sell, Hold).
    """
    print("กำลังประเมินประสิทธิภาพโมเดล...")
    
    y_pred_proba = model.predict(X_test) 
    y_pred = np.argmax(y_pred_proba, axis=1) # Get the class with the highest probability
    y_true = np.argmax(y_test, axis=1) # Convert one-hot encoded y_test back to class labels

    accuracy = accuracy_score(y_true, y_pred)
    
    target_names = ['Sell (0)', 'Buy (1)', 'Hold (2)'] # Define target names for report
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=target_names))

    # --- เพิ่ม: Metrics ทางการเทรด ---
    if df_test_for_metrics is not None and not df_test_for_metrics.empty:
        # ตรวจสอบว่า df_test_for_metrics มีคอลัมน์ที่จำเป็นสำหรับการคำนวณ Metrics
        required_trade_cols = ['close_H1', 'future_return_H1', 'spread_H1', 'next_close_H1'] # Added next_close_H1 here
        if not all(col in df_test_for_metrics.columns for col in required_trade_cols):
            print(f"⚠️ ไม่สามารถคำนวณ Metrics ทางการเทรดได้: คอลัมน์ที่จำเป็น ({required_trade_cols}) ไม่อยู่ใน df_test_for_metrics.")
            return accuracy, 0.0 # Return partial results if columns are missing

        trades = df_test_for_metrics.copy() 
        trades['predicted_signal'] = y_pred # Assign predicted classes (0, 1, 2)
        trades['actual_target'] = y_true # Assign actual classes (0, 1, 2)
            
        # คำนวณผลตอบแทนของการเทรดแต่ละครั้ง
        # กำหนดให้ Trade เกิดขึ้นเฉพาะเมื่อโมเดลทำนาย Buy (1) หรือ Sell (0)
        trades['strategy_return_gross'] = 0.0
        
        # If predicted Buy (1)
        trades.loc[trades['predicted_signal'] == 1, 'strategy_return_gross'] = trades['future_return_H1']
        
        # If predicted Sell (0)
        trades.loc[trades['predicted_signal'] == 0, 'strategy_return_gross'] = -trades['future_return_H1']
        
        # Only consider entries where a trade was predicted (0 or 1)
        trades_executed = trades[
            (trades['predicted_signal'] == 1) | (trades['predicted_signal'] == 0)
        ].copy()

        # Calculate net return by deducting spread
        trades_executed['trade_return_net'] = trades_executed['strategy_return_gross'].copy()
        if deduct_spread_in_metrics and 'spread_H1' in trades_executed.columns:
            trades_executed['trade_return_net'] -= trades_executed['spread_H1'] 
            print(f"\n✅ คำนวณกำไร/ขาดทุนโดยหักค่า Spread เฉลี่ยต่อแท่ง: {trades_executed['spread_H1'].mean():.6f}")
        else:
            print("\n❌ ไม่ได้หักค่า Spread ในการคำนวณ Metrics ทางการเทรด (สำหรับ Debugging)")
        
        # Use 'trade_return_net' for all profitability metrics
        current_trade_return_col = 'trade_return_net'

        total_profit = trades_executed[trades_executed[current_trade_return_col] > 0][current_trade_return_col].sum()
        total_loss = trades_executed[trades_executed[current_trade_return_col] < 0][current_trade_return_col].sum() 
        
        profit_factor = -total_profit / total_loss if total_loss < 0 else np.inf
        
        num_trades = len(trades_executed)
        num_winning_trades = len(trades_executed[trades_executed[current_trade_return_col] > 0])
        num_losing_trades = len(trades_executed[trades[current_trade_return_col] < 0])

        avg_win = trades_executed[trades_executed[current_trade_return_col] > 0][current_trade_return_col].mean()
        avg_loss = trades_executed[trades_executed[current_trade_return_col] < 0][current_trade_return_col].mean()
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf

        cumulative_returns = (1 + trades_executed[current_trade_return_col]).cumprod()
        max_drawdown = 0
        if not cumulative_returns.empty:
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
        
        print("\n--- Metrics ทางการเทรด ---")
        print(f"จำนวน Trade ทั้งหมด: {num_trades}")
        print(f"จำนวน Trade ที่กำไร: {num_winning_trades}")
        print(f"จำนวน Trade ที่ขาดทุน: {num_losing_trades}")
        print(f"Profit Factor: {profit_factor:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        print(f"Average Winning Trade: {avg_win:.4f}")
        print(f"Average Losing Trade: {avg_loss:.4f}")
        print(f"Average Win/Loss Ratio: {avg_win_loss_ratio:.4f}")

        if not trades_executed[trades_executed[current_trade_return_col] > 0].empty:
            print("\n--- ตัวอย่าง Trade ที่กำไรสุทธิ ---")
            print(trades_executed[trades_executed[current_trade_return_col] > 0][['Symbol', 'close_H1', 'next_close_H1', 'spread_H1', 'strategy_return_gross', 'trade_return_net', 'predicted_signal', 'actual_target']].head())
        else:
            print("\n--- ไม่มี Trade ที่กำไรสุทธิในชุด Test ---")
        
        if not trades_executed[trades_executed[current_trade_return_col] < 0].empty:
            print("\n--- ตัวอย่าง Trade ที่ขาดทุนสุทธิ ---")
            print(trades_executed[trades_executed[current_trade_return_col] < 0][['Symbol', 'close_H1', 'next_close_H1', 'spread_H1', 'strategy_return_gross', 'trade_return_net', 'predicted_signal', 'actual_target']].head())
        else:
            print("\n--- ไม่มี Trade ที่ขาดทุนสุทธิในชุด Test ---")

    return accuracy, 0.0 # Return recall as 0.0 if not directly relevant to 3-class eval
    
# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- โหมดการเทรน LSTM Model (lstm_trading_bot.py) ---")
    data_folder_path = 'C:/ProjectLek/Machine_Learning_Forex/DataCSV/'
    
    LSTM_TIMESTEPS = 20 
    
    # STEP 1 & 2: โหลดและเตรียมข้อมูล Multi-Timeframe พร้อม Feature Engineering
    X, y, scaler, features_list_for_X, final_combined_df = load_and_preprocess_multi_timeframe_data_from_csv(data_folder_path)
    
    # STEP 3: แบ่งข้อมูล Train, Validation, Test (ยังคงแบ่งตาม Time series เหมือนเดิม)
    total_samples = len(X)
    train_size = int(total_samples * SPLIT_RATIO)
    valid_size = int(total_samples * 0.1) # 10% for validation

    X_train_df, y_train_df = X.iloc[:train_size], y.iloc[:train_size]
    X_valid_df, y_valid_df = X.iloc[train_size : train_size + valid_size], y.iloc[train_size : train_size + valid_size]
    X_test_df, y_test_df = X.iloc[train_size + valid_size :], y.iloc[train_size + valid_size :]

    print(f"\n--- ขนาดข้อมูลก่อนสร้าง Sequence ---")
    print(f"ขนาดข้อมูล Train DF: X={X_train_df.shape}, y={y_train_df.shape}")
    print(f"ขนาดข้อมูล Validation DF: X={X_valid_df.shape}, y={y_valid_df.shape}")
    print(f"ขนาดข้อมูล Test DF: X={X_test_df.shape}, y={y_test_df.shape}")

    # STEP 4: สร้าง Sequences สำหรับ LSTM
    print(f"\n--- กำลังสร้าง Sequences สำหรับ LSTM (timesteps={LSTM_TIMESTEPS}) ---")
    X_train_seq, y_train_seq_raw, y_train_indices = create_sequences_for_lstm(X_train_df, y_train_df, LSTM_TIMESTEPS, "Train")
    X_valid_seq, y_valid_seq_raw, y_valid_indices = create_sequences_for_lstm(X_valid_df, y_valid_df, LSTM_TIMESTEPS, "Valid")
    X_test_seq, y_test_seq_raw, y_test_indices = create_sequences_for_lstm(X_test_df, y_test_df, LSTM_TIMESTEPS, "Test")

    # Convert raw target classes to one-hot encoding for Keras
    num_classes = len(np.unique(y_train_seq_raw)) # Should be 3 (0, 1, 2)
    if num_classes < 2:
        print(f"Exiting: Not enough classes in target variable (found {num_classes}). Need at least 2 for classification.")
        sys.exit(1)

    y_train_seq = to_categorical(y_train_seq_raw, num_classes=num_classes)
    y_valid_seq = to_categorical(y_valid_seq_raw, num_classes=num_classes)
    y_test_seq = to_categorical(y_test_seq_raw, num_classes=num_classes) # For evaluation

    print(f"ขนาดข้อมูล Train Sequence: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"ขนาดข้อมูล Validation Sequence: X={X_valid_seq.shape}, y={y_valid_seq.shape}")
    print(f"ขนาดข้อมูล Test Sequence: X={X_test_seq.shape}, y={y_test_seq.shape}")

    # จัดการ NaN/Inf ในข้อมูล Sequence ก่อนเทรน
    print("\n--- กำลังตรวจสอบและทำความสะอาด NaN/Inf ในข้อมูล Sequence ก่อนเทรน ---")
    X_train_seq = np.nan_to_num(X_train_seq, nan=0.0, posinf=1e10, neginf=-1e10)
    X_valid_seq = np.nan_to_num(X_valid_seq, nan=0.0, posinf=1e10, neginf=-1e10)
    X_test_seq = np.nan_to_num(X_test_seq, nan=0.0, posinf=1e10, neginf=-1e10) 
    print("--- ทำความสะอาด NaN/Inf เสร็จสิ้น ---")

    # Handle class imbalance for 3 classes
    y_train_original_classes = np.argmax(y_train_seq, axis=1) # Convert one-hot back to single labels
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_original_classes), y=y_train_original_classes)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Computed class weights: {class_weight_dict}")

    # STEP 5: สร้างและเทรน LSTM Classifier
    num_features = X_train_seq.shape[2] # จำนวน features ในแต่ละ timestep
    model, history = build_and_train_lstm_model(X_train_seq, y_train_seq, X_valid_seq, y_valid_seq, 
                                                LSTM_TIMESTEPS, num_features, num_classes)

    # STEP 6: ประเมินผลโมเดลบนชุด Test
    print(f"DEBUG: Type of final_combined_df.index: {type(final_combined_df.index)}")
    print(f"DEBUG: First 5 elements of final_combined_df.index: {final_combined_df.index[:5].tolist()}")
    print(f"DEBUG: Type of y_test_indices: {type(y_test_indices)}")
    print(f"DEBUG: First 5 elements of y_test_indices: {y_test_indices[:5].tolist()}")

    # Select columns needed for metrics from final_combined_df (including next_close_H1 now for profit calc)
    # Ensure all required columns are available
    cols_for_metrics_and_target = ['close_H1', 'future_return_H1', 'spread_H1', 'target', 'next_close_H1'] 
    
    # Use .reindex() to create df_test_for_metrics from final_combined_df using y_test_indices
    df_test_for_metrics = final_combined_df.reindex(y_test_indices)[cols_for_metrics_and_target].copy()
    
    df_test_for_metrics.dropna(inplace=True)

    filtered_test_indices = df_test_for_metrics.index
    
    original_seq_index_map = {idx: i for i, idx in enumerate(y_test_indices)}
    
    X_test_seq_final = np.array([X_test_seq[original_seq_index_map[idx]] for idx in filtered_test_indices])
    y_test_seq_final = np.array([y_test_seq[original_seq_index_map[idx]] for idx in filtered_test_indices])

    print(f"DEBUG: X_test_seq_final shape after alignment: {X_test_seq_final.shape}")
    print(f"DEBUG: y_test_seq_final shape after alignment: {y_test_seq_final.shape}")
    print(f"DEBUG: df_test_for_metrics shape after alignment: {df_test_for_metrics.shape}")

    if not (len(X_test_seq_final) == len(y_test_seq_final) == len(df_test_for_metrics)):
        print("CRITICAL ERROR: Length mismatch after final alignment for test set. Exiting.")
        sys.exit(1)
        
    if X_test_seq_final.shape[0] == 0:
        print("WARNING: X_test_seq_final is empty. Cannot perform prediction or evaluation on test set.")
        accuracy = 0.0
        print("\n--- Metrics ทางการเทรด ---")
        print("No trades to evaluate as test set is empty.")
        plt.figure(figsize=(1,1)) 
        plt.text(0.5, 0.5, "No data for plotting", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.show()
        joblib.dump(scaler, 'scaler_lstm_multi_symbol.pkl')
        joblib.dump(features_list_for_X, 'features_list_lstm_multi_symbol.pkl')
        print("✅ โมเดล LSTM ที่ดีที่สุดถูกบันทึกแล้วในชื่อ 'best_lstm_model.keras'")
        print("✅ Scaler ถูกบันทึกแล้วในชื่อ 'scaler_lstm_multi_symbol.pkl'")
        print("✅ Features list ถูกบันทึกแล้วในชื่อ 'features_list_lstm_multi_symbol.pkl'")
        sys.exit(0) 

    deduct_spread_for_eval = True 
    accuracy, _ = evaluate_model(model, X_test_seq_final, y_test_seq_final, 
                                              df_test_for_metrics=df_test_for_metrics,
                                              deduct_spread_in_metrics=deduct_spread_for_eval,
                                              num_classes=num_classes)

    # --- คำนวณผลตอบแทนสะสมของกลยุทธ์ (หลังจากการรวม y_pred เข้าไปใน df_test_for_metrics) ---
    def calculate_symbol_strategy_return(group, deduct_spread=True):
        # Ensure 'predicted_signal', 'actual_target', 'close_H1', 'next_close_H1', 'spread_H1' are present
        if not all(col in group.columns for col in ['predicted_signal', 'actual_target', 'close_H1', 'next_close_H1', 'spread_H1', 'future_return_H1']): # Added future_return_H1
            print("WARNING: Missing columns for strategy return calculation in group.")
            return pd.Series([1.0] * len(group), index=group.index) # Return neutral cumulative product

        spread_cost_per_trade = group['spread_H1'].mean() if 'spread_H1' in group.columns else 0.0

        group['trade_return_raw'] = 0.0
        
        # Calculate raw return based on predicted signal
        # If predicted Buy (1)
        buy_predicted_mask = (group['predicted_signal'] == 1)
        group.loc[buy_predicted_mask, 'trade_return_raw'] = group['future_return_H1'] # Use future_return_H1 directly
        
        # If predicted Sell (0)
        sell_predicted_mask = (group['predicted_signal'] == 0)
        group.loc[sell_predicted_mask, 'trade_return_raw'] = -group['future_return_H1'] # Use -future_return_H1 for sell
        
        # Apply spread deduction only for trades that were actually executed (predicted Buy or Sell)
        group['trade_return_net_after_spread'] = group['trade_return_raw'].copy()
        if deduct_spread:
            # Only subtract spread for actual trade signals (0 or 1)
            trade_mask = (group['predicted_signal'] == 1) | (group['predicted_signal'] == 0)
            group.loc[trade_mask, 'trade_return_net_after_spread'] -= spread_cost_per_trade 
        
        # If predicted 'Hold' (2), return is 0
        group.loc[group['predicted_signal'] == 2, 'trade_return_net_after_spread'] = 0.0

        return (1 + group['trade_return_net_after_spread']).cumprod()

    deduct_spread_for_plot = True 
    df_test_for_metrics['cumulative_strategy_return_by_symbol'] = df_test_for_metrics.groupby(level='Symbol', group_keys=False).apply(
        lambda group: calculate_symbol_strategy_return(group, deduct_spread=deduct_spread_for_plot)
    )
    df_test_for_metrics['buy_and_hold_return_by_symbol'] = df_test_for_metrics.groupby(level='Symbol', group_keys=False)['close_H1'].apply(lambda x: (x / x.iloc[0]))
    
    # Plot สำหรับแต่ละ Symbol
    print("\n--- ผลตอบแทนสะสม (Cumulative Return) แยกตาม Symbol (Test Set) ---")
    
    unique_symbols = df_test_for_metrics.index.get_level_values('Symbol').unique()
    num_symbols = len(unique_symbols)
    num_cols = 2 
    num_rows = (num_symbols + num_cols - 1) // num_cols 

    plt.figure(figsize=(num_cols * 8, num_rows * 5)) 
    
    for i, symbol in enumerate(unique_symbols):
        symbol_df = df_test_for_metrics.loc[(slice(None), symbol), :] 
        
        plt.subplot(num_rows, num_cols, i + 1)
        if not symbol_df['cumulative_strategy_return_by_symbol'].empty:
            symbol_df['cumulative_strategy_return_by_symbol'].plot(label=f'Strategy {symbol}', alpha=0.7)
        if not symbol_df['buy_and_hold_return_by_symbol'].empty:
            symbol_df['buy_and_hold_return_by_symbol'].plot(label=f'Buy & Hold {symbol}', alpha=0.7, linestyle='--')
        
        if not symbol_df.empty:
            final_strategy_return = symbol_df['cumulative_strategy_return_by_symbol'].iloc[-1] if not symbol_df['cumulative_strategy_return_by_symbol'].empty else np.nan
            final_buy_and_hold_return = symbol_df['buy_and_hold_return_by_symbol'].iloc[-1] if not symbol_df['buy_and_hold_return_by_symbol'].empty else np.nan
            plt.title(f'{symbol} (Strat: {final_strategy_return:.2f}, B&H: {final_buy_and_hold_return:.2f})')
        else:
            plt.title(f'{symbol} (No data for plotting)')
            
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.show()

    # STEP 7: บันทึกโมเดล, Scaler, และ Features List
    # โมเดล LSTM ถูกบันทึกอัตโนมัติด้วย ModelCheckpoint ในชื่อ 'best_lstm_model.keras'
    joblib.dump(scaler, 'scaler_lstm_multi_symbol.pkl')
    joblib.dump(features_list_for_X, 'features_list_lstm_multi_symbol.pkl')

    print("✅ โมเดล LSTM ที่ดีที่สุดถูกบันทึกแล้วในชื่อ 'best_lstm_model.keras'")
    print("✅ Scaler ถูกบันทึกแล้วในชื่อ 'scaler_lstm_multi_symbol.pkl'")
    print("✅ Features list ถูกบันทึกแล้วในชื่อ 'features_list_lstm_multi_symbol.pkl'")
