import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import joblib 
import matplotlib.pyplot as plt 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

# --- 1. ฟังก์ชันช่วยในการโหลดข้อมูลจาก CSV และสร้าง Features ---
def _load_and_create_features_from_csv(file_path):
    """
    โหลดข้อมูลราคาจากไฟล์ CSV และสร้าง features ทางเทคนิคตามที่แนะนำ.
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
    
    # ตรวจสอบคอลัมน์อีกครั้งก่อนคำนวณ Indicator (เพื่อ DEBUG)
    print(f"DEBUG in _load_and_create_features_from_csv: Columns in DF before feature engineering: {df.columns.tolist()}")
    if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        print(f"DEBUG Critical: Essential price columns (high, low, close) are missing! This indicates a problem with the CSV data.")
        return pd.DataFrame() # คืนค่า DataFrame ว่างเปล่าเพื่อหยุดการทำงาน

    # --- Feature Engineering: เพิ่ม Indicators ที่แนะนำ ---
    # 1. RSI (14)
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    # 2. EMA (Fast 8, Slow 34, Trend 200)
    df['EMA_fast'] = EMAIndicator(close=df['close'], window=8).ema_indicator()
    df['EMA_slow'] = EMAIndicator(close=df['close'], window=34).ema_indicator()
    df['EMA_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator() # ใช้สำหรับ Trend Confirmation บน H4

    # 3. MACD (Fast 12, Slow 26, Signal 9)
    macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD_line'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    # MACD Histogram ไม่ได้ถูกใช้ในกลยุทธ์ EA แต่เป็นข้อมูลเสริมที่ดี
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal'] 
    
    # 4. ATR (14)
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
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
        
        # ตรวจสอบว่ามีข้อมูล H1, M15, H4 ครบถ้วนหรือไม่
        # สำหรับกลยุทธ์นี้ เราต้องการ H1 และ H4 เป็นหลัก แต่ M15 อาจมีอยู่ในข้อมูลเก่าก็ได้
        df_h1_file = tfs_data.get('H1')
        df_h4_file = tfs_data.get('H4')
        
        if not df_h1_file or not df_h4_file:
            print(f"⚠️ ข้อมูลไม่ครบ Timeframe ที่จำเป็น (H1, H4) สำหรับ Symbol: {symbol_name}. จะข้าม Symbol นี้ไป.")
            continue

        print(f"กำลังโหลดและสร้าง Features สำหรับ {symbol_name}...")
        
        df_h1 = _load_and_create_features_from_csv(df_h1_file).add_suffix('_H1')
        if df_h1.empty: print(f"   ❌ DataFrame H1 ว่างเปล่าสำหรับ {symbol_name}. ข้าม Symbol นี้."); continue
        
        # โหลด H4 โดยเฉพาะสำหรับ EMA_200_H4
        df_h4 = _load_and_create_features_from_csv(df_h4_file).add_suffix('_H4')
        if df_h4.empty: print(f"   ❌ DataFrame H4 ว่างเปล่าสำหรับ {symbol_name}. ข้าม Symbol นี้."); continue

        print(f"   กำลังจัดเรียงข้อมูล Multi-Timeframe สำหรับ {symbol_name}...")
        # ผนวกข้อมูล H4 เข้ากับ H1
        df_combined = pd.merge_asof(df_h1, df_h4, left_index=True, right_index=True, direction='backward')
        
        # ตรวจสอบและรวม M15 หากมี (หากไม่มีก็ไม่เป็นไรเพราะกลยุทธ์ไม่ใช้ M15 โดยตรง)
        df_m15_file = tfs_data.get('M15')
        if df_m15_file:
            df_m15 = _load_and_create_features_from_csv(df_m15_file).add_suffix('_M15')
            if not df_m15.empty:
                df_combined = pd.merge_asof(df_combined, df_m15, left_index=True, right_index=True, direction='backward')
                print(f"   รวมข้อมูล M15 สำหรับ {symbol_name} แล้ว.")
            else:
                print(f"   ⚠️ DataFrame M15 ว่างเปล่าสำหรับ {symbol_name}. จะไม่รวม M15.")

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

    # --- สร้าง Target Variable ก่อนที่จะทำ MultiIndex และ One-Hot Encoding ---
    # เพื่อให้การคำนวณ future_return_H1 โดย groupby('Symbol') ทำได้ง่าย
    # 'Time' เป็น Index และ 'Symbol' เป็นคอลัมน์อยู่
    final_combined_df['future_return_H1'] = final_combined_df.groupby('Symbol')['close_H1'].shift(-3) / final_combined_df['close_H1'] - 1
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
    
    # ทำ One-Hot Encoding
    df_with_one_hot = pd.get_dummies(df_for_one_hot, columns=['Symbol'], prefix='Symbol')
    
    # ตั้งค่า MultiIndex กลับไป เพื่อให้ X และ y มี MultiIndex (Time, Symbol)
    df_with_one_hot.set_index(['Time', df_for_one_hot['Symbol']], inplace=True, verify_integrity=True) # ใช้ Symbol ดั้งเดิมจาก df_for_one_hot เป็นส่วนหนึ่งของ Index Level 2
    # ตั้งชื่อ Level ให้ชัดเจน
    df_with_one_hot.index.set_names(['Time', 'Symbol'], inplace=True)
    df_with_one_hot.sort_index(inplace=True) # จัดเรียงอีกครั้ง

    # *** CRITICAL STEP: กรองคอลัมน์ที่ไม่ใช่ Feature ออกจาก Features list ที่ส่งเข้าโมเดล ***
    # คอลัมน์ที่ถูกใช้ในกลยุทธ์ (เช่น 'close_H1' สำหรับ target calculation หรือ 'open', 'high', 'low', 'close' ดั้งเดิม) 
    # จะยังคงอยู่ใน DataFrame แต่จะไม่ถูกส่งเป็น Feature เข้าโมเดล
    features_list_for_X = [
        col for col in df_with_one_hot.columns 
        if col not in ['future_return_H1', 'target', 
                       'open_H1', 'high_H1', 'low_H1', 'close_H1', 
                       'open_M15', 'high_M15', 'low_M15', 'close_M15', # หากมี M15
                       'open_H4', 'high_H4', 'low_H4', 'close_H4'] # และคอลัมน์ Symbol_XXXm ที่ถูกสร้างโดย One-Hot Encoding
        and not col.startswith('Symbol_') # แยก One-Hot encoded symbol ออก
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
    
    # ส่งคืน X_scaled_df (scaled X), y, scaler, features_list_for_X, df_with_one_hot (สำหรับ debugging/validation),
    # และ final_combined_df (สำหรับ plotting original columns และมี MultiIndex)
    return X_scaled_df, y, scaler, features_list_for_X, df_with_one_hot, final_combined_df 

# --- 3. ฟังก์ชันสำหรับเทรนโมเดล XGBoost ---
def train_xgboost_model(X_train, y_train, X_valid, y_valid):
    """
    สร้างและเทรนโมเดล XGBoost Classifier.
    """
    print(f"DEBUG: y_train value counts before scale_pos_weight: {y_train.value_counts()}")
    class_counts = y_train.value_counts()
    neg_count = class_counts.get(0, 0) 
    pos_count = class_counts.get(1, 0) 
    
    scale_pos_weight_value = 1.0 
    if pos_count > 0: 
        scale_pos_weight_value = float(neg_count / pos_count)
    
    print(f"คำนวณ Scale Pos Weight: {scale_pos_weight_value}")

    model = XGBClassifier(
        n_estimators=10000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_value,
        eval_metric='logloss',
        random_state=42,
        min_child_weight=1,
        gamma=0.1,
        tree_method='hist',
        early_stopping_rounds=100
    )

    print("กำลังเทรนโมเดล XGBoost Classifier...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=True
    )
    print("การเทรนโมเดล XGBoost เสร็จสมบูรณ์.")
    return model

# --- 4. ฟังก์ชันสำหรับประเมินผลโมเดล ---
def evaluate_model(model, X_test, y_test, threshold=0.50):
    """
    ประเมินประสิทธิภาพของโมเดลบนชุดข้อมูลทดสอบ.
    """
    print("กำลังประเมินประสิทธิภาพโมเดล...")
    y_pred_proba = model.predict_proba(X_test)[:, 1] 
    y_pred = (y_pred_proba > threshold).astype(int) 

    accuracy = accuracy_score(y_test, y_pred)
    if 1 in y_test.unique():
        recall_class_1 = recall_score(y_test, y_pred, pos_label=1) 
    else:
        recall_class_1 = 0.0 
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (Class 1 - Buy Signal): {recall_class_1:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return accuracy, recall_class_1

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- โหมดการเทรน XGBoost Model (BTCshortTradeV3_CSV_Data.py) ---")
    data_folder_path = 'DataCSV'
    
    # X, y, scaler, features_list_for_X, df_processed_for_split (df_with_one_hot), df_combined_original_for_plot (final_combined_df)
    X, y, scaler, features_list_for_X, df_processed_for_split, df_combined_original_for_plot = load_and_preprocess_multi_timeframe_data_from_csv(data_folder_path)
    
    # STEP 3: เตรียมชุดข้อมูลสำหรับเทรนโมเดล
    # X และ y ตอนนี้มี MultiIndex (Time, Symbol)
    total_samples = len(X)
    train_size = int(total_samples * 0.8)
    valid_size = int(total_samples * 0.1)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_valid, y_valid = X.iloc[train_size : train_size + valid_size], y.iloc[train_size : train_size + valid_size]
    X_test, y_test = X.iloc[train_size + valid_size :], y.iloc[train_size + valid_size :]

    print(f"ขนาดข้อมูล Train: X={X_train.shape}, y={y_train.shape}")
    print(f"ขนาดข้อมูล Validation: X={X_valid.shape}, y={y_valid.shape}")
    print(f"ขนาดข้อมูล Test: X={X_test.shape}, y={y_test.shape}")
    
    print(f"DEBUG: X_test index unique? {X_test.index.is_unique}") # ควรจะเป็น TRUE!
    print(f"DEBUG: X_test length: {len(X_test)}") 

    # STEP 4: สร้างและเทรน XGBoost Classifier
    model = train_xgboost_model(X_train, y_train, X_valid, y_valid)

    # STEP 5: ประเมินผลโมเดลบนชุด Test
    prediction_threshold = 0.40
    accuracy, recall_class_1 = evaluate_model(model, X_test, y_test, threshold=prediction_threshold)

    # --- Plot ผลตอบแทนสะสมเปรียบเทียบ ---
    # df_test จะต้องมี MultiIndex เหมือนกับ X_test เพื่อให้ loc ทำงานได้อย่างถูกต้อง
    # df_combined_original_for_plot ก็มี MultiIndex (Time, Symbol) แล้ว
    df_test = df_combined_original_for_plot.loc[X_test.index].copy()
    
    print(f"DEBUG: df_test length before y_pred assignment: {len(df_test)}") 
    print(f"DEBUG: df_test index unique? {df_test.index.is_unique}") # ควรจะเป็น TRUE!

    y_pred_proba_for_plot = model.predict_proba(X_test)[:, 1]
    
    # *** บรรทัดที่เกิดปัญหา (df_test['y_pred'] = ...) ควรจะทำงานได้ถูกต้องแล้วตอนนี้ ***
    df_test['y_pred'] = (y_pred_proba_for_plot > prediction_threshold).astype(int)

    df_test['target'] = y_test # y_test มี MultiIndex ตรงกับ X_test

    # --- คำนวณผลตอบแทนตามกลยุทธ์ (ใช้ MultiIndex 'Symbol') ---
    def calculate_symbol_strategy_return(group):
        group['strategy_return'] = np.where(
            group['y_pred'] == 1, group['future_return_H1'], 
            np.where(group['y_pred'] == 0, -group['future_return_H1'], 0) 
        )
        return (1 + group['strategy_return']).cumprod()

    # การ groupby ด้วยระดับของ MultiIndex (ระดับที่ 1 คือ 'Symbol')
    df_test['cumulative_strategy_return_by_symbol'] = df_test.groupby(level='Symbol', group_keys=False).apply(calculate_symbol_strategy_return)
    df_test['buy_and_hold_return_by_symbol'] = df_test.groupby(level='Symbol', group_keys=False)['future_return_H1'].apply(lambda x: (1 + x).cumprod())
    
    # Plot สำหรับแต่ละ Symbol
    print("\n--- ผลตอบแทนสะสม (Cumulative Return) แยกตาม Symbol (Test Set) ---")
    
    # ดึง Symbol จากระดับ Index (Level 'Symbol')
    unique_symbols = df_test.index.get_level_values('Symbol').unique()
    num_symbols = len(unique_symbols)
    num_cols = 2 
    num_rows = (num_symbols + num_cols - 1) // num_cols 

    plt.figure(figsize=(num_cols * 8, num_rows * 5)) 
    
    for i, symbol in enumerate(unique_symbols):
        # การเลือกข้อมูลจาก MultiIndex โดยใช้ .loc
        symbol_df = df_test.loc[(slice(None), symbol), :] # เลือกทุก Time ที่มี Symbol นี้
        
        plt.subplot(num_rows, num_cols, i + 1)
        # ใช้ .plot() โดยตรงบน Series ที่มี MultiIndex (Time)
        symbol_df['cumulative_strategy_return_by_symbol'].plot(label=f'Strategy {symbol}', alpha=0.7)
        symbol_df['buy_and_hold_return_by_symbol'].plot(label=f'Buy & Hold {symbol}', alpha=0.7, linestyle='--')
        
        if not symbol_df.empty:
            final_strategy_return = symbol_df['cumulative_strategy_return_by_symbol'].iloc[-1]
            final_buy_and_hold_return = symbol_df['buy_and_hold_return_by_symbol'].iloc[-1]
            plt.title(f'{symbol} (Strat: {final_strategy_return:.2f}, B&H: {final_buy_and_hold_return:.2f})')
        else:
            plt.title(f'{symbol} (No data for plotting)')
            
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.show()

    # STEP 6: บันทึกโมเดล, Scaler, และ Features List
    joblib.dump(model, 'xgboost_model_v3_multi_symbol.pkl')
    joblib.dump(scaler, 'scaler_v3_multi_symbol.pkl')
    joblib.dump(features_list_for_X, 'features_list_v3_multi_symbol.pkl')

    print("✅ โมเดล XGBoost ถูกบันทึกแล้วในชื่อ 'xgboost_model_v3_multi_symbol.pkl'")
    print("✅ Scaler ถูกบันทึกแล้วในชื่อ 'scaler_v3_multi_symbol.pkl'")
    print("✅ Features list ถูกบันทึกแล้วในชื่อ 'features_list_v3_multi_symbol.pkl'")
