# โค้ดที่ได้รับการแก้ไขแล้วใน Canvas:
# (ส่วนที่แก้ไขอยู่ภายในฟังก์ชัน evaluate_model)

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import joblib 
import matplotlib.pyplot as plt 
# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# --- 1. ฟังก์ชันช่วยในการโหลดข้อมูลจาก CSV และสร้าง Features (คงเดิมจากเวอร์ชั่นล่าสุด) ---
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
    # และเติมค่า NaN หากคอลัมน์นั้นไม่มีในไฟล์ CSV
    df_processed = pd.DataFrame(index=df.index)
    
    for col_name in required_cols_expected:
        if col_name in df.columns:
            df_processed[col_name] = df[col_name]
        else:
            print(f"⚠️ ไฟล์ {file_path} ไม่มีคอลัมน์ '{col_name}' หลังจากแปลงเป็นตัวพิมพ์เล็ก. จะเติมด้วย NaN.")
            if col_name in ['open', 'high', 'low', 'close']:
                df_processed[col_name] = np.nan # เติมด้วย NaN สำหรับคอลัมน์ราคา
            else:
                df_processed[col_name] = 0 # เติมด้วย 0 สำหรับคอลัมน์อื่น (volume, spread)
    
    df = df_processed.copy() # แทนที่ DataFrame เดิมด้วย DataFrame ที่ประมวลผลแล้ว
    
    if 'high' not in df.columns:
        print(f"DEBUG Critical: 'high' column is still missing after processing! This indicates a problem with the CSV data.")
        return pd.DataFrame() # คืนค่า DataFrame ว่างเปล่าเพื่อหยุดการทำงาน

    # --- Feature Engineering: เพิ่ม Indicators ยอดนิยม ---
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['EMA_fast'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
    df['EMA_slow'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator() 
    macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_lower'] = bollinger.bollinger_lband()
    
    # คำนวณขนาดของ Body และ Shadow ก่อน Normalize
    df['Body_size_raw'] = abs(df['close'] - df['open'])
    df['Upper_shadow_raw'] = df['high'] - df[['close','open']].max(axis=1)
    df['Lower_shadow_raw'] = df[['close','open']].min(axis=1) - df['low']

    # --- เพิ่ม: Normalize Body_size และ Shadow ด้วย ATR ---
    # ป้องกันการหารด้วยศูนย์หาก ATR เป็น 0 หรือ NaN
    df['Body_size'] = np.where(df['ATR'].fillna(0) > 0, df['Body_size_raw'] / df['ATR'], 0)
    df['Upper_shadow'] = np.where(df['ATR'].fillna(0) > 0, df['Upper_shadow_raw'] / df['ATR'], 0)
    df['Lower_shadow'] = np.where(df['ATR'].fillna(0) > 0, df['Lower_shadow_raw'] / df['ATR'], 0)
    
    # ลบคอลัมน์ raw ที่ไม่จำเป็นแล้ว
    df.drop(columns=['Body_size_raw', 'Upper_shadow_raw', 'Lower_shadow_raw'], errors='ignore', inplace=True)


    divergence_lookback = 14 
    df['bullish_rsi_divergence'] = np.where(
        (df['close'] < df['close'].shift(divergence_lookback)) & (df['RSI'] > df['RSI'].shift(divergence_lookback)), 1, 0
    )
    df['bearish_rsi_divergence'] = np.where(
        (df['close'] > df['close'].shift(divergence_lookback)) & (df['RSI'] < df['RSI'].shift(divergence_lookback)), 1, 0
    )
    df['signal_rsi_oversold'] = np.where(df['RSI'] < 30, 1, 0)
    df['signal_rsi_overbought'] = np.where(df['RSI'] > 70, 1, 0)
    df['signal_macd_cross_up'] = np.where(
        (df['MACD'].shift(1) < df['MACD_signal'].shift(1)) & (df['MACD'] > df['MACD_signal']), 1, 0
    )
    df['signal_golden_cross'] = np.where(
        (df['EMA_fast'].shift(1) < df['EMA_slow'].shift(1)) & (df['EMA_fast'] > df['EMA_slow']), 1, 0
    )
    df['signal_bb_lower_rsi_low'] = np.where(
        (df['close'] <= df['BB_lower']) & (df['RSI'] < 40), 1, 0
    )
    df['signal_close_below_ema50'] = np.where(df['close'] < df['EMA_50'], 1, 0)
    df['ema_fast_slope'] = df['EMA_fast'].diff(periods=3)
    df['ema_slow_slope'] = df['EMA_slow'].diff(periods=3)
    df['bullish_engulfing'] = np.where(
        (df['close'] > df['open']) & 
        (df['open'].shift(1) > df['close'].shift(1)) & 
        (df['open'] < df['close'].shift(1)) & 
        (df['close'] > df['open'].shift(1)) & 
        (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))), 
        1, 0
    )
    df['bearish_engulfing'] = np.where(
        (df['close'] < df['open']) & 
        (df['open'].shift(1) < df['close'].shift(1)) & 
        (df['open'] > df['close'].shift(1)) & 
        (df['close'] < df['open'].shift(1)) & 
        (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))), 
        1, 0
    )
    df['hammer'] = np.where(
        (df['Body_size'] > 0) & 
        (df['Lower_shadow'] >= 2 * df['Body_size']) & 
        (df['Upper_shadow'] <= 0.2 * df['Body_size']), 
        1, 0
    )
    df['shooting_star'] = np.where(
        (df['Body_size'] > 0) & 
        (df['Upper_shadow'] >= 2 * df['Body_size']) & 
        (df['Lower_shadow'] <= 0.2 * df['Body_size']), 
        1, 0
    )
    df['doji_val'] = np.where(
        (df['Body_size'] < (df['high'] - df['low']) * 0.1) & 
        ((df['high'] - df['low']) > df['ATR'] * 0.1), 
        1, 0
    )

    df.dropna(inplace=True)
    return df

# --- 2. ฟังก์ชันหลักในการโหลดและเตรียมข้อมูล Multi-Timeframe สำหรับการเทรน (ใช้ MultiIndex) ---
def load_and_preprocess_multi_timeframe_data_from_csv(data_folder='DataCSV'):
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

    # --- ปรับปรุง: สร้าง Target Variable โดยใช้ shift(-1) เพื่อทำนายแท่งถัดไป 1 แท่ง H1 ---
    final_combined_df['future_return_H1'] = final_combined_df.groupby('Symbol')['close_H1'].shift(-1) / final_combined_df['close_H1'] - 1
    
    # ใช้ Threshold 0.007 ตามที่โค้ดเดิมระบุ
    final_combined_df['target'] = np.where(final_combined_df['future_return_H1'] > 0.005, 1,
                                            np.where(final_combined_df['future_return_H1'] < -0.005, 0, np.nan))
    final_combined_df.dropna(inplace=True) # ลบแถวที่ target เป็น NaN
    
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
    # ใช้ Symbol ดั้งเดิมจาก df_for_one_hot เป็นส่วนหนึ่งของ Index Level 2
    # NOTE: The 'Symbol' column in df_with_one_hot is now the original symbol string,
    # and the one-hot encoded columns are 'Symbol_BTCUSDm', 'Symbol_XAUUSDm'.
    # We want the index to be (Time, Original_Symbol_Name).
    # So we use df_for_one_hot['Symbol'] (which has the original string) for the second level of index.
    df_with_one_hot.set_index(['Time', df_for_one_hot['Symbol']], inplace=True, verify_integrity=True) 
    # ตั้งชื่อ Level ให้ชัดเจน
    df_with_one_hot.index.set_names(['Time', 'Symbol'], inplace=True)
    df_with_one_hot.sort_index(inplace=True) # จัดเรียงอีกครั้ง

    # List ของ Features ที่จะใช้ในการเทรน (รวม One-Hot Encoded Symbol Features)
    # เพื่อให้โมเดล LSTM สามารถเรียนรู้จากความแตกต่างของ Symbol ได้
    features_list_for_X = [
        col for col in df_with_one_hot.columns 
        if col not in ['future_return_H1', 'target'] 
    ]
    
    # X และ y ตอนนี้มี MultiIndex (Time, Symbol) แล้ว
    X = df_with_one_hot[features_list_for_X]
    y = df_with_one_hot['target'].astype(int) 

    # Scale Features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features_list_for_X, index=X.index) # Preserve MultiIndex

    print(f"ชุดข้อมูลรวมพร้อมแล้ว. X shape: {X_scaled_df.shape}, y shape: {y.shape}")
    print(f"DEBUG: X_scaled_df index unique? {X_scaled_df.index.is_unique}") # ควรจะเป็น TRUE!
    
    return X_scaled_df, y, scaler, features_list_for_X, final_combined_df # ส่ง final_combined_df กลับไปด้วยสำหรับ Metrics

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
        # When slicing by symbol, Pandas often drops the 'Symbol' level if it becomes uniform.
        # This is expected behavior. The index of symbol_X/Y will be a DatetimeIndex.
        symbol_X = X_data.loc[(slice(None), symbol), :]
        symbol_y = y_data.loc[(slice(None), symbol)]
        
        print(f"DEBUG {dataset_name} Symbol: {symbol}: Length of symbol_X: {len(symbol_X)}, Length of symbol_y: {len(symbol_y)}")
        
        # REMOVED: The problematic MultiIndex check for symbol_X and symbol_y
        # because slicing by symbol will simplify the index to DatetimeIndex, which is fine.
        
        if len(symbol_X) <= timesteps:
            print(f"DEBUG {dataset_name}: Skipping symbol '{symbol}' due to insufficient data ({len(symbol_X)} bars) for {timesteps} timesteps.")
            continue

        current_symbol_sequences_count = 0 
        for i in range(len(symbol_X) - timesteps):
            # *** START OF FIX ***
            # Ensure target_datetime is a Timestamp, not a tuple (Time, Symbol)
            if isinstance(symbol_X.index, pd.MultiIndex):
                # This case should ideally not happen if loc slicing drops the 'Symbol' level when uniform.
                # But if it does, get only the Time component.
                target_datetime = symbol_X.index.get_level_values('Time')[i + timesteps]
                # print(f"DEBUG: MultiIndex still present for symbol_X.index, extracting Time level. Sample: {target_datetime}")
            else:
                # This is the expected case: symbol_X.index is a DatetimeIndex
                target_datetime = symbol_X.index[i + timesteps]
                # print(f"DEBUG: DatetimeIndex as expected for symbol_X.index. Sample: {target_datetime}")
            # *** END OF FIX ***

            current_multi_index_tuple = (target_datetime, symbol) 
            y_indices_list.append(current_multi_index_tuple) # Use the reconstructed tuple

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

    # Rest of the filtering logic remains the same
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
def build_and_train_lstm_model(X_train, y_train, X_valid, y_valid, timesteps, num_features):
    """
    สร้างและเทรนโมเดล LSTM Classifier.
    """
    # ตรวจสอบค่า NaN/inf ในข้อมูล Input ของ Keras
    if np.isnan(X_train).sum() > 0 or np.isinf(X_train).sum() > 0:
        print(f"CRITICAL ERROR: NaN or Inf found in X_train. NaNs: {np.isnan(X_train).sum()}, Infs: {np.isinf(X_train).sum()}")
        # สามารถเพิ่มโค้ดสำหรับ Debug หรือจัดการกับค่าเหล่านี้ได้
        # เช่น X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
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
        LSTM(units=50, return_sequences=True, input_shape=(timesteps, num_features)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1, activation='sigmoid') # Sigmoid สำหรับ Binary Classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_lstm_model.keras', monitor='val_loss', save_best_only=True, mode='min')

    print("กำลังเทรนโมเดล LSTM Classifier...")

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_seq),
        y=y_train_seq
    )
    class_weights_dict = dict(enumerate(class_weights))
    
    history = model.fit(
        X_train, y_train,
        epochs=100, # สามารถปรับได้
        batch_size=32, # สามารถปรับได้
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1,
        class_weight=class_weights_dict
    )
    print("การเทรนโมเดล LSTM เสร็จสมบูรณ์.")
    
    # โหลดโมเดลที่ดีที่สุดกลับมา
    model = tf.keras.models.load_model('best_lstm_model.keras')
    return model, history

# --- 4. ฟังก์ชันสำหรับประเมินผลโมเดล (คงเดิมจากเวอร์ชั่นล่าสุด) ---
def evaluate_model(model, X_test, y_test, threshold=0.50, df_test_for_metrics=None, deduct_spread_in_metrics=True, is_lstm=False):
    """
    ประเมินประสิทธิภาพของโมเดลบนชุดข้อมูลทดสอบ และคำนวณ Metrics ทางการเทรด.
    เพิ่ม deduct_spread_in_metrics เพื่อควบคุมการหัก Spread ในการคำนวณ PnL
    เพิ่ม is_lstm เพื่อรองรับการคาดการณ์จาก LSTM (predict_proba จะให้ 1D array)
    """
    print("กำลังประเมินประสิทธิภาพโมเดล...")
    
    if is_lstm:
        # สำหรับ Keras/TensorFlow, model.predict() จะให้ค่า probability โดยตรง
        y_pred_proba = model.predict(X_test).flatten() # LSTM output is usually (samples, 1)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1] 
    
    y_pred = (y_pred_proba > threshold).astype(int) 

    accuracy = accuracy_score(y_test, y_pred)
    
    # *** START OF FIX ***
    if 1 in np.unique(y_test): # แก้ไขตรงนี้: ใช้ np.unique() แทน .unique()
        recall_class_1 = recall_score(y_test, y_pred, pos_label=1) 
    # *** END OF FIX ***
    else:
        recall_class_1 = 0.0 
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (Class 1 - Buy Signal): {recall_class_1:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- เพิ่ม: Metrics ทางการเทรด ---
    if df_test_for_metrics is not None and not df_test_for_metrics.empty:
        # ตรวจสอบว่า df_test_for_metrics มี 'y_pred' และ 'future_return_H1' หรือไม่
        if 'y_pred' in df_test_for_metrics.columns and 'future_return_H1' in df_test_for_metrics.columns:
            trades = df_test_for_metrics[
                (df_test_for_metrics['y_pred'] == 1) | (df_test_for_metrics['y_pred'] == 0)
            ].copy() # ใช้ .copy() เพื่อหลีกเลี่ยง SettingWithCopyWarning
            
            # >>>>> แก้ไขตรงนี้: reset_index() เพื่อให้ 'Symbol' เป็นคอลัมน์ธรรมดาสำหรับแสดงผล <<<<<
            trades = trades.reset_index() 
            
            # คำนวณผลตอบแทนของการเทรดแต่ละครั้ง
            trades['trade_return_gross'] = np.where(
                trades['y_pred'] == 1, trades['future_return_H1'], # Buy: ถ้าขึ้นได้กำไร, ถ้าลงขาดทุน
                np.where(trades['y_pred'] == 0, -trades['future_return_H1'], 0) # Sell: ถ้าลงได้กำไร, ถ้าลงขาดทุน
            )
            
            # หักค่า Spread ออกจากผลตอบแทน (ใช้ค่า Spread_H1 ของแต่ละแท่ง)
            trades['trade_return_net'] = trades['trade_return_gross'].copy()
            if deduct_spread_in_metrics and 'spread_H1' in trades.columns:
                trades['trade_return_net'] -= trades['spread_H1'] # หัก Spread ออกจากการเทรด
                print(f"\n✅ คำนวณกำไร/ขาดทุนโดยหักค่า Spread เฉลี่ยต่อแท่ง: {trades['spread_H1'].mean():.6f}")
            else:
                print("\n❌ ไม่ได้หักค่า Spread ในการคำนวณ Metrics ทางการเทรด (สำหรับ Debugging)")
            
            # ใช้ 'trade_return_net' สำหรับการคำนวณ Metrics ทั้งหมด
            current_trade_return_col = 'trade_return_net'

            total_profit = trades[trades[current_trade_return_col] > 0][current_trade_return_col].sum()
            total_loss = trades[trades[current_trade_return_col] < 0][current_trade_return_col].sum() # จะเป็นค่าติดลบ
            
            profit_factor = -total_profit / total_loss if total_loss < 0 else np.inf
            
            num_trades = len(trades)
            num_winning_trades = len(trades[trades[current_trade_return_col] > 0])
            num_losing_trades = len(trades[trades[current_trade_return_col] < 0])

            avg_win = trades[trades[current_trade_return_col] > 0][current_trade_return_col].mean()
            avg_loss = trades[trades[current_trade_return_col] < 0][current_trade_return_col].mean()
            avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf

            cumulative_returns = (1 + trades[current_trade_return_col]).cumprod()
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

            # --- เพิ่ม: แสดงรายละเอียด Trade ที่ถูกจัดว่า "ชนะ" แต่จริงๆ แล้วอาจจะขาดทุนหลังหัก Spread ---
            potential_wins_classified_as_losses = trades[
                (trades['y_pred'] == 1) & (trades['future_return_H1'] > 0) & (trades['trade_return_net'] <= 0)
            ]
            potential_losses_classified_as_wins = trades[
                (trades['y_pred'] == 0) & (trades['future_return_H1'] < 0) & (trades['trade_return_net'] <= 0)
            ]

            if not potential_wins_classified_as_losses.empty:
                print(f"\n--- Trade ที่ถูก 'ทำนายว่าขึ้น' และ 'ขึ้นจริง' แต่ 'ขาดทุนสุทธิ' หลังหัก Spread ({len(potential_wins_classified_as_losses)} รายการ) ---")
                print(potential_wins_classified_as_losses[['Symbol', 'close_H1', 'future_return_H1', 'spread_H1', 'trade_return_gross', 'trade_return_net']].head())
            
            if not potential_losses_classified_as_wins.empty:
                print(f"\n--- Trade ที่ถูก 'ทำนายว่าลง' และ 'ลงจริง' แต่ 'ขาดทุนสุทธิ' หลังหัก Spread ({len(potential_losses_classified_as_wins)} รายการ) ---")
                print(potential_losses_classified_as_wins[['Symbol', 'close_H1', 'future_return_H1', 'spread_H1', 'trade_return_gross', 'trade_return_net']].head())

            # แสดงรายละเอียดของ Trade ที่กำไรสุทธิ (ถ้ามี)
            if not trades[trades[current_trade_return_col] > 0].empty:
                print("\n--- ตัวอย่าง Trade ที่กำไรสุทธิ ---")
                print(trades[trades[current_trade_return_col] > 0][['Symbol', 'close_H1', 'future_return_H1', 'spread_H1', 'trade_return_gross', 'trade_return_net']].head())
            else:
                print("\n--- ไม่มี Trade ที่กำไรสุทธิในชุด Test ---")
            
            # แสดงรายละเอียดของ Trade ที่ขาดทุนสุทธิ (ถ้ามี)
            if not trades[trades[current_trade_return_col] < 0].empty:
                print("\n--- ตัวอย่าง Trade ที่ขาดทุนสุทธิ ---")
                print(trades[trades[current_trade_return_col] < 0][['Symbol', 'close_H1', 'future_return_H1', 'spread_H1', 'trade_return_gross', 'trade_return_net']].head())


        else:
            print("⚠️ ไม่สามารถคำนวณ Metrics ทางการเทรดได้: 'y_pred' หรือ 'future_return_H1' หรือ 'spread_H1' ไม่อยู่ใน df_test_for_metrics.")
    
    return accuracy, recall_class_1

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- โหมดการเทรน LSTM Model (lstm_trading_bot.py) ---")
    data_folder_path = 'DataCSV'
    
    # กำหนดจำนวนแท่งเทียนย้อนหลัง (Timesteps) ที่ LSTM จะใช้ในการเรียนรู้
    # ลองเริ่มต้นที่ 10-30 แท่งก่อน แล้วค่อยปรับ
    LSTM_TIMESTEPS = 40 
    
    # STEP 1 & 2: โหลดและเตรียมข้อมูล Multi-Timeframe พร้อม Feature Engineering
    # X, y, scaler, features_list_for_X, final_combined_df (สำหรับ Metrics และ Plotting)
    X, y, scaler, features_list_for_X, final_combined_df = load_and_preprocess_multi_timeframe_data_from_csv(data_folder_path)
    
    # STEP 3: แบ่งข้อมูล Train, Validation, Test (ยังคงแบ่งตาม Time series เหมือนเดิม)
    total_samples = len(X)
    train_size = int(total_samples * 0.8)
    valid_size = int(total_samples * 0.1)
    
    X_train_df, y_train_df = X.iloc[:train_size], y.iloc[:train_size]
    X_valid_df, y_valid_df = X.iloc[train_size : train_size + valid_size], y.iloc[train_size : train_size + valid_size]
    X_test_df, y_test_df = X.iloc[train_size + valid_size :], y.iloc[train_size + valid_size :]

    print(f"\n--- ขนาดข้อมูลก่อนสร้าง Sequence ---")
    print(f"ขนาดข้อมูล Train DF: X={X_train_df.shape}, y={y_train_df.shape}")
    print(f"ขนาดข้อมูล Validation DF: X={X_valid_df.shape}, y={y_valid_df.shape}")
    print(f"ขนาดข้อมูล Test DF: X={X_test_df.shape}, y={y_test_df.shape}")

    # STEP 4: สร้าง Sequences สำหรับ LSTM
    print(f"\n--- กำลังสร้าง Sequences สำหรับ LSTM (timesteps={LSTM_TIMESTEPS}) ---")
    X_train_seq, y_train_seq, y_train_indices = create_sequences_for_lstm(X_train_df, y_train_df, LSTM_TIMESTEPS, "Train")
    X_valid_seq, y_valid_seq, y_valid_indices = create_sequences_for_lstm(X_valid_df, y_valid_df, LSTM_TIMESTEPS, "Valid")
    X_test_seq, y_test_seq, y_test_indices = create_sequences_for_lstm(X_test_df, y_test_df, LSTM_TIMESTEPS, "Test")


    print("\n--- Target Distribution in Training Set ---")
    print(pd.Series(y_train_seq).value_counts(normalize=True))

    print("\n--- Target Distribution in Validation Set ---")
    print(pd.Series(y_valid_seq).value_counts(normalize=True))

    print(f"ขนาดข้อมูล Train Sequence: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"ขนาดข้อมูล Validation Sequence: X={X_valid_seq.shape}, y={y_valid_seq.shape}")
    print(f"ขนาดข้อมูล Test Sequence: X={X_test_seq.shape}, y={y_test_seq.shape}")

    # >>>>> เพิ่ม: จัดการ NaN/Inf ในข้อมูล Sequence ก่อนเทรน <<<<<
    # Convert any NaN or Inf values to finite numbers (e.g., 0 for NaN, large/small numbers for Inf)
    print("\n--- กำลังตรวจสอบและทำความสะอาด NaN/Inf ในข้อมูล Sequence ก่อนเทรน ---")
    X_train_seq = np.nan_to_num(X_train_seq, nan=0.0, posinf=1e10, neginf=-1e10)
    y_train_seq = np.nan_to_num(y_train_seq, nan=0.0, posinf=1.0, neginf=0.0) # Target should be 0 or 1
    X_valid_seq = np.nan_to_num(X_valid_seq, nan=0.0, posinf=1e10, neginf=-1e10)
    y_valid_seq = np.nan_to_num(y_valid_seq, nan=0.0, posinf=1.0, neginf=0.0) # Target should be 0 or 1
    X_test_seq = np.nan_to_num(X_test_seq, nan=0.0, posinf=1e10, neginf=-1e10) # Also clean test set for evaluation
    print("--- ทำความสะอาด NaN/Inf เสร็จสิ้น ---")

    # >>>>> เพิ่ม: จัดการ NaN/Inf ในข้อมูล Sequence ก่อนเทรน <<<<<
    print("\n --- ข้อมูล Sequence หลังจากทำความสะอาด NaN/Inf ---",X_train_seq)
    print("\n --- ข้อมูล Target หลังจากทำความสะอาด NaN/Inf ---",y_train_seq)
    print("\n --- ข้อมูล Sequence หลังจากทำความสะอาด NaN/Inf ---",X_valid_seq)
    print("\n --- ข้อมูล Target หลังจากทำความสะอาด NaN/Inf ---",y_valid_seq)
    print("\n --- ข้อมูล Sequence หลังจากทำความสะอาด NaN/Inf ---",X_test_seq)
    print("\n --- ข้อมูล Target หลังจากทำความสะอาด NaN/Inf ---",y_test_seq)

    # STEP 5: สร้างและเทรน LSTM Classifier
    num_features = X_train_seq.shape[2] # จำนวน features ในแต่ละ timestep
    model, history = build_and_train_lstm_model(X_train_seq, y_train_seq, X_valid_seq, y_valid_seq, LSTM_TIMESTEPS, num_features)

    # STEP 6: ประเมินผลโมเดลบนชุด Test
    prediction_threshold = 0.50 # เริ่มต้นด้วย 0.50 สำหรับ LSTM ก่อน
    
    print(f"DEBUG: Type of final_combined_df.index: {type(final_combined_df.index)}")
    print(f"DEBUG: First 5 elements of final_combined_df.index: {final_combined_df.index[:5].tolist()}")
    print(f"DEBUG: Type of y_test_indices: {type(y_test_indices)}")
    print(f"DEBUG: First 5 elements of y_test_indices: {y_test_indices[:5].tolist()}")

    # Select columns needed for metrics from final_combined_df
    cols_for_metrics_and_target = ['close_H1', 'future_return_H1', 'spread_H1', 'target']
    
    # Use .reindex() to create df_test_for_metrics from final_combined_df using y_test_indices
    # This is robust as it will fill with NaN for any missing keys instead of raising KeyError.
    df_test_for_metrics = final_combined_df.reindex(y_test_indices)[cols_for_metrics_and_target].copy()
    
    # Drop rows that became NaN due to reindexing (i.e., indices from y_test_indices not in final_combined_df)
    df_test_for_metrics.dropna(inplace=True)

    # Align X_test_seq and y_test_seq to the final df_test_for_metrics index
    # We must also re-filter X_test_seq and y_test_seq to match the (potentially smaller)
    # set of indices that survived the df_test_for_metrics.dropna() operation.
    
    # Create a mask for indices that exist in df_test_for_metrics
    # This ensures that X_test_seq_final and y_test_seq_final only contain data
    # for which we also have corresponding entries in df_test_for_metrics.
    filtered_test_indices = df_test_for_metrics.index
    
    # Recreate the original_index_map from y_test_indices to match the original
    # positions in X_test_seq and y_test_seq.
    # This map needs to store the original index of the sequence in X_test_seq/y_test_seq
    # for each MultiIndex tuple.
    original_seq_index_map = {idx: i for i, idx in enumerate(y_test_indices)}

    # Now, use this map to get the correct slice from the *original* X_test_seq and y_test_seq
    # for only those indices that made it into df_test_for_metrics
    X_test_seq_final = np.array([X_test_seq[original_seq_index_map[idx]] for idx in filtered_test_indices])
    y_test_seq_final = np.array([y_test_seq[original_seq_index_map[idx]] for idx in filtered_test_indices])

    print(f"DEBUG: X_test_seq_final shape after alignment: {X_test_seq_final.shape}")
    print(f"DEBUG: y_test_seq_final shape after alignment: {y_test_seq_final.shape}")
    print(f"DEBUG: df_test_for_metrics shape after alignment: {df_test_for_metrics.shape}")

    if not (len(X_test_seq_final) == len(y_test_seq_final) == len(df_test_for_metrics)):
        print("CRITICAL ERROR: Length mismatch after final alignment for test set. Exiting.")
        sys.exit(1)
        
    # Check if X_test_seq_final is empty before prediction
    if X_test_seq_final.shape[0] == 0:
        print("WARNING: X_test_seq_final is empty. Cannot perform prediction or evaluation on test set.")
        # Provide dummy results to avoid errors further down
        accuracy = 0.0
        recall_class_1 = 0.0
        print("\n--- Metrics ทางการเทรด ---")
        print("No trades to evaluate as test set is empty.")
        plt.figure(figsize=(1,1)) # Create a dummy figure to avoid error
        plt.text(0.5, 0.5, "No data for plotting", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.show()
        # Save dummy files
        joblib.dump(scaler, 'scaler_lstm_multi_symbol.pkl')
        joblib.dump(features_list_for_X, 'features_list_lstm_multi_symbol.pkl')
        print("✅ โมเดล LSTM ที่ดีที่สุดถูกบันทึกแล้วในชื่อ 'best_lstm_model.keras'")
        print("✅ Scaler ถูกบันทึกแล้วในชื่อ 'scaler_lstm_multi_symbol.pkl'")
        print("✅ Features list ถูกบันทึกแล้วในชื่อ 'features_list_lstm_multi_symbol.pkl'")
        sys.exit(0) # Exit gracefully if no test data

    # Make sure 'y_pred' column is added correctly
    y_pred_proba_for_metrics = model.predict(X_test_seq_final).flatten()
    
    # Assign predictions to df_test_for_metrics, aligning by index
    df_test_for_metrics['y_pred'] = pd.Series(
        (y_pred_proba_for_metrics > prediction_threshold).astype(int), 
        index=df_test_for_metrics.index # Use the *final* index of df_test_for_metrics
    )

    # Assign actual targets to df_test_for_metrics, also aligning by index
    # Note: y_test_seq_final is already aligned with X_test_seq_final and df_test_for_metrics's index
    df_test_for_metrics['target'] = pd.Series(
        y_test_seq_final, 
        index=df_test_for_metrics.index # Use the *final* index of df_test_for_metrics
    )

    # >>>>>> สำคัญ: ปรับ deduct_spread_in_metrics เป็น False เพื่อดูผลตอบแทน "ดิบ" ก่อนหัก Spread <<<<<<
    # >>>>>> หลังจากวิเคราะห์แล้ว ค่อยปรับกลับเป็น True เพื่อดูผลตอบแทนสุทธิ <<<<<<
    deduct_spread_for_eval = True # <<< เปลี่ยนตรงนี้เป็น False เพื่อ Debuggying ครั้งแรก
    accuracy, recall_class_1 = evaluate_model(model, X_test_seq_final, y_test_seq_final, 
                                              threshold=prediction_threshold, 
                                              df_test_for_metrics=df_test_for_metrics,
                                              deduct_spread_in_metrics=deduct_spread_for_eval,
                                              is_lstm=True) # ระบุว่าเป็นโมเดล LSTM

    # --- คำนวณผลตอบแทนสะสมของกลยุทธ์ (หลังจากการรวม y_pred เข้าไปใน df_test_for_metrics) ---
    def calculate_symbol_strategy_return(group, deduct_spread=True):
        spread_cost_per_trade = group['spread_H1'].mean() if 'spread_H1' in group.columns else 0.0

        group['strategy_return'] = np.where(
            group['y_pred'] == 1, group['future_return_H1'], 
            np.where(group['y_pred'] == 0, -group['future_return_H1'], 0) 
        )
        
        if deduct_spread:
            group['strategy_return'] -= spread_cost_per_trade 

        return (1 + group['strategy_return']).cumprod()

    # >>>>>> สำคัญ: ปรับ deduct_spread ในการคำนวณ Plot เป็น False เพื่อดูผลตอบแทน "ดิบ" ก่อนหัก Spread <<<<<<
    # >>>>>> หลังจากวิเคราะห์แล้ว ค่อยปรับกลับเป็น True เพื่อดูผลตอบแทนสุทธิ <<<<<<
    deduct_spread_for_plot = True # <<< เปลี่ยนตรงนี้เป็น False เพื่อ Debuggying ครั้งแรก
    df_test_for_metrics['cumulative_strategy_return_by_symbol'] = df_test_for_metrics.groupby(level='Symbol', group_keys=False).apply(
        lambda group: calculate_symbol_strategy_return(group, deduct_spread=deduct_spread_for_plot)
    )
    df_test_for_metrics['buy_and_hold_return_by_symbol'] = df_test_for_metrics.groupby(level='Symbol', group_keys=False)['future_return_H1'].apply(lambda x: (1 + x).cumprod())
    
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