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

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator # Removed AccumulationDistributionIndex

# --- Global Configurations (ตั้งค่าโมเดลและข้อมูล) ---
DATA_PATH = 'C:/ProjectLek/Machine_Learning_Forex/DataCSV/' # Path ไปยังโฟลเดอร์ที่มีไฟล์ CSV หลายไฟล์
MODEL_SAVE_PATH = './models/'
SCALER_SAVE_PATH = './scalers/'

# ชื่อคอลัมน์ที่คุณจะใช้เป็น Features สำหรับ LSTM (จะถูกอัปเดตแบบไดนามิกสำหรับ Symbol One-Hot Encoding)
FEATURES_TO_USE = [
    'Open', 'High', 'Low', 'Close', 'RealVolume', 
    'RSI', 'EMA_10', 'EMA_20', 'MACD', 'MACD_signal', 'MACD_hist',
    'ADX', 'ADX_Positive', 'ADX_Negative', 'ATR', 
    'BB_High', 'BB_Low', 'BB_Mid', # Bollinger Bands
    'STOCH_K', 'STOCH_D', # Stochastic Oscillator
    'Body_size', 'Upper_shadow', 'Lower_shadow', # Candlestick features (normalized by ATR)
    'bullish_rsi_divergence', 'bearish_rsi_divergence', 
    'signal_rsi_oversold', 'signal_rsi_overbought', 
    'signal_macd_cross_up', 'signal_golden_cross', 
    'signal_bb_lower_rsi_low', 'signal_close_below_ema50',
    'ema_fast_slope', 'ema_slow_slope',
    # Candlestick Patterns:
    'bullish_engulfing', 'bearish_engulfing', 
    'hammer', 'shooting_star', 'doji_val',
    # New Volume-based indicators
    'OBV' # Removed ADL
]

# New: Define lag periods for features
LAG_PERIODS = [5, 10, 20] 
# New: Define features to apply lagging to
FEATURES_TO_LAG = ['RSI', 'MACD', 'Close'] 

# Update FEATURES_TO_USE with lagged features
for feature in FEATURES_TO_LAG:
    for lag in LAG_PERIODS:
        FEATURES_TO_USE.append(f'{feature}_lag{lag}')


# Target Definition: 0 (Sell), 1 (Buy), 2 (Hold)
# New: Price change threshold for defining Buy/Sell signals (e.g., 0.05% for profit)
PRICE_CHANGE_THRESHOLD_PERCENT = 0.0005 # 0.05% profit/loss threshold per bar
TARGET_COLUMN = 'next_bar_direction' 

SEQUENCE_LENGTH = 60 # จำนวน Time Steps สำหรับ LSTM (ควรเท่ากับ InpSequenceLength ใน MT5 EA)
BATCH_SIZE = 32
BUFFER_SIZE = 100000 # สำหรับ tf.data.Dataset
EPOCHS = 100 # จำนวน Epochs สูงสุด
PATIENCE = 10 # จำนวน Epochs ที่จะรอหาก Validation Loss ไม่ดีขึ้น (สำหรับ EarlyStopping)
LSTM_UNITS = 128 # Increased LSTM units for more complexity
DROPOUT_RATE = 0.3 # Increased Dropout rate
LEARNING_RATE = 0.001 # New: Custom learning rate for Adam optimizer
SPLIT_RATIO = 0.8 # 80% train, 20% validation+test

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
            df = df.loc[~df.index.duplicated(keep='first')] # เก็บค่าแรกที่ซ้ำกัน

        # Ensure numeric types for calculations
        for col in ['Open', 'High', 'Low', 'Close', 'RealVolume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # Drop rows with missing price data

        # Calculate Technical Indicators
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator() # เพิ่ม EMA_50
        
        macd = MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff() # MACD Histogram
        
        df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
        
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()

        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['STOCH_K'] = stoch.stoch()
        df['STOCH_D'] = stoch.stoch_signal()

        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['ADX_Positive'] = adx.adx_pos()
        df['ADX_Negative'] = adx.adx_neg()

        # New: Volume-based indicators
        df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['RealVolume']).on_balance_volume()
        # Removed ADL calculation: df['ADL'] = AccumulationDistributionIndex(high=df['High'], low=df['Low'], close=df['Close'], volume=df['RealVolume']).accumulation_distribution_index() 

        # Candlestick Features (normalized by ATR)
        df['Body_size'] = np.abs(df['Close'] - df['Open'])
        df['Upper_shadow'] = df['High'] - np.maximum(df['Close'], df['Open'])
        df['Lower_shadow'] = np.minimum(df['Close'], df['Open']) - df['Low']
        
        # Normalize by ATR to make them comparable across different volatility regimes
        df['Body_size'] = df['Body_size'] / df['ATR'].replace(0, np.nan) # Avoid division by zero
        df['Upper_shadow'] = df['Upper_shadow'] / df['ATR'].replace(0, np.nan)
        df['Lower_shadow'] = df['Lower_shadow'] / df['ATR'].replace(0, np.nan)


        # --- Signal Features (simplified logic; for full accuracy, Python's numpy logic is needed) ---
        # Initialize signal columns to 0
        signal_cols = [
            'bullish_rsi_divergence', 'bearish_rsi_divergence', 
            'signal_rsi_oversold', 'signal_rsi_overbought', 
            'signal_macd_cross_up', 'signal_golden_cross', 
            'signal_bb_lower_rsi_low', 'signal_close_below_ema50',
            'ema_fast_slope', 'ema_slow_slope',
            'bullish_engulfing', 'bearish_engulfing', 
            'hammer', 'shooting_star', 'doji_val' # New patterns
        ]
        for col in signal_cols:
            df[col] = 0.0

        # RSI Divergence (simplified - requires looking back multiple periods)
        # For actual divergence, you'd need more complex logic comparing peaks/troughs.
        # This is a basic example: price lower, RSI higher = bullish div
        # Using shift(14) as a simplified lookback for divergence concept
        df.loc[(df['Close'] < df['Close'].shift(14)) & (df['RSI'] > df['RSI'].shift(14)), 'bullish_rsi_divergence'] = 1
        df.loc[(df['Close'] > df['Close'].shift(14)) & (df['RSI'] < df['RSI'].shift(14)), 'bearish_rsi_divergence'] = 1

        # RSI Overbought/Oversold
        df.loc[df['RSI'] < 30, 'signal_rsi_oversold'] = 1
        df.loc[df['RSI'] > 70, 'signal_rsi_overbought'] = 1

        # MACD Cross Up
        df.loc[(df['MACD'].shift(1) < df['MACD_signal'].shift(1)) & (df['MACD'] > df['MACD_signal']), 'signal_macd_cross_up'] = 1

        # Golden Cross (EMA_10 crosses above EMA_20)
        df.loc[(df['EMA_10'].shift(1) < df['EMA_20'].shift(1)) & (df['EMA_10'] > df['EMA_20']), 'signal_golden_cross'] = 1

        # BB Lower RSI Low (e.g., price hits lower band and RSI is low)
        df.loc[(df['Close'] <= df['BB_Low']) & (df['RSI'] < 40), 'signal_bb_lower_rsi_low'] = 1

        # Close below EMA50
        df.loc[df['Close'] < df['EMA_50'], 'signal_close_below_ema50'] = 1

        # EMA Slopes (simplified: current value vs. N periods ago)
        df['ema_fast_slope'] = df['EMA_10'] - df['EMA_10'].shift(3) # Slope over 3 periods
        df['ema_slow_slope'] = df['EMA_20'] - df['EMA_20'].shift(3) # Slope over 3 periods

        # --- Candlestick Patterns ---
        # Bullish Engulfing
        bullish_engulfing_cond = (
            (df['Close'] > df['Open']) & # Current bar is bullish
            (df['Close'].shift(1) < df['Open'].shift(1)) & # Previous bar is bearish
            (df['Open'] < df['Close'].shift(1)) & # Current open is below previous close
            (df['Close'] > df['Open'].shift(1)) & # Current close is above previous open
            (df['Body_size'] > df['Body_size'].shift(1)) # Current body is larger than previous body
        )
        df.loc[bullish_engulfing_cond, 'bullish_engulfing'] = 1

        # Bearish Engulfing
        bearish_engulfing_cond = (
            (df['Close'] < df['Open']) & # Current bar is bearish
            (df['Close'].shift(1) > df['Open'].shift(1)) & # Previous bar is bullish
            (df['Open'] > df['Close'].shift(1)) & # Current open is above previous close
            (df['Close'] < df['Open'].shift(1)) & # Current close is below previous open
            (df['Body_size'] > df['Body_size'].shift(1)) # Current body is larger than previous body
        )
        df.loc[bearish_engulfing_cond, 'bearish_engulfing'] = 1

        # Hammer (simplified: small body, long lower shadow, small upper shadow)
        # Using ATR normalized values
        hammer_cond = (
            (df['Body_size'] > 0) & # Must have a body
            (df['Lower_shadow'] >= 2 * df['Body_size']) & # Lower shadow at least twice the body
            (df['Upper_shadow'] <= 0.2 * df['Body_size']) # Small or no upper shadow
        )
        df.loc[hammer_cond, 'hammer'] = 1

        # Shooting Star (simplified: small body, long upper shadow, small lower shadow)
        # Using ATR normalized values
        shooting_star_cond = (
            (df['Body_size'] > 0) & # Must have a body
            (df['Upper_shadow'] >= 2 * df['Body_size']) & # Upper shadow at least twice the body
            (df['Lower_shadow'] <= 0.2 * df['Body_size']) # Small or no lower shadow
        )
        df.loc[shooting_star_cond, 'shooting_star'] = 1
        
        # Doji (simplified: very small body relative to total range and ATR)
        doji_cond = (
            (df['Body_size'] < (df['High'] - df['Low']) * 0.1) & # Body is less than 10% of total range
            ((df['High'] - df['Low']) > df['ATR'] * 0.1) # Total range is still significant relative to ATR
        )
        df.loc[doji_cond, 'doji_val'] = 1

        # New: Lagged Features
        for feature in FEATURES_TO_LAG:
            for lag in LAG_PERIODS:
                if feature in df.columns:
                    df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

        # New: Define the target variable for multi-class classification (Buy/Sell/Hold)
        df['next_close'] = df['Close'].shift(-1)
        
        # Initialize TARGET_COLUMN to 'Hold' (class 2)
        df[TARGET_COLUMN] = 2 

        # Class 1: Buy (price goes up by threshold)
        # Check if the next close price is significantly higher than the current close price
        df.loc[(df['next_close'] - df['Close']) / df['Close'] >= PRICE_CHANGE_THRESHOLD_PERCENT, TARGET_COLUMN] = 1

        # Class 0: Sell (price goes down by threshold)
        # Check if the next close price is significantly lower than the current close price
        df.loc[(df['Close'] - df['next_close']) / df['Close'] >= PRICE_CHANGE_THRESHOLD_PERCENT, TARGET_COLUMN] = 0

        # Drop the temporary 'next_close' column
        df.drop(columns=['next_close'], inplace=True)

        # Drop rows with NaN values created by indicators or shifting
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        print(f"❌ Error loading or processing file {file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- 2. ฟังก์ชันหลักสำหรับโหลดข้อมูลทั้งหมด ---
def load_all_data(path):
    all_symbols_data = []
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            print(f"Loading and processing {filename}...")
            df = _load_and_create_features_from_csv(file_path)
            if not df.empty:
                # Extract symbol from filename (assuming filename format like HistoricalData_SYMBOL_PERIOD.csv)
                symbol_name = filename.replace('HistoricalData_', '').replace('.csv', '').split('_PERIOD')[0]
                df['Symbol'] = symbol_name
                all_symbols_data.append(df)
    
    if not all_symbols_data:
        print("❌ No data loaded. Check DATA_PATH and CSV files.")
        sys.exit(1)

    combined_df = pd.concat(all_symbols_data)
    combined_df.sort_index(inplace=True) # Sort by time
    print(f"✅ Combined data shape: {combined_df.shape}")
    print(f"Unique symbols loaded: {combined_df['Symbol'].unique()}")
    return combined_df

# --- 3. ฟังก์ชันสำหรับสร้าง Sequences (X) และ Targets (y) สำหรับ LSTM ---
def create_sequences(data, sequence_length, features, target_column):
    X, y = [], []
    
    # Iterate through unique symbols within the data
    # This loop ensures that sequences are not created across different symbols
    for symbol in data['Symbol'].unique(): 
        symbol_df = data[data['Symbol'] == symbol].copy()
        
        # Ensure only selected features and target are in symbol_df
        # And ensure the order of features is consistent
        # Filter features that exist in this symbol_df (should all exist if prepared correctly)
        current_features = [f for f in features if f in symbol_df.columns]
        
        if target_column not in symbol_df.columns:
            print(f"❌ Target column '{target_column}' not found for symbol {symbol}. Skipping.")
            continue

        if not current_features:
            print(f"❌ No valid features found for symbol {symbol}. Skipping.")
            continue
            
        # Select and reorder features and target
        X_data = symbol_df[current_features].values
        y_data = symbol_df[target_column].values

        if len(symbol_df) < sequence_length:
            print(f"⚠️ Not enough data for symbol {symbol} to create sequences. Length: {len(symbol_df)}. Skipping.")
            continue

        for i in range(len(symbol_df) - sequence_length):
            X.append(X_data[i:(i + sequence_length)])
            y.append(y_data[i + sequence_length]) # Predict the target of the bar *after* the sequence

    return np.array(X), np.array(y)

# --- 4. ฟังก์ชันสำหรับสร้างและคอมไพล์โมเดล LSTM ---
# Modified: Added an extra LSTM layer and custom learning rate
def build_lstm_model(input_shape, num_classes, learning_rate=LEARNING_RATE):
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS, return_sequences=True), # Added another LSTM layer
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS // 2, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    optimizer = Adam(learning_rate=learning_rate) # Use Adam with custom learning rate
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Main Training and Evaluation Workflow ---
if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(SCALER_SAVE_PATH, exist_ok=True)

    # STEP 1: Load and prepare data
    print("STEP 1: Loading and preparing data...")
    combined_df = load_all_data(DATA_PATH)
    
    # Make sure we have enough data after dropping NaNs
    if combined_df.empty:
        print("Exiting: No sufficient data to proceed after processing.")
        sys.exit(1)

    # Get unique symbols from the combined DataFrame for one-hot encoding
    unique_symbols = combined_df['Symbol'].unique().tolist()
    print(f"Unique symbols for one-hot encoding: {unique_symbols}")

    # STEP 1.5: Perform One-Hot Encoding for Symbols
    print("STEP 1.5: Performing One-Hot Encoding for Symbols...")
    # Create dummy variables for 'Symbol' column
    symbol_dummies = pd.get_dummies(combined_df['Symbol'], prefix='Symbol', dtype=int)
    # Concatenate the new dummy columns to the combined_df
    combined_df = pd.concat([combined_df, symbol_dummies], axis=1)
    # The original 'Symbol' column is RETAINED here. Do NOT drop it.
    
    # Add new one-hot encoded symbol columns to FEATURES_TO_USE
    one_hot_symbol_features = [col for col in symbol_dummies.columns]
    # Ensure no duplicates if this script is run multiple times
    for feature in one_hot_symbol_features:
        if feature not in FEATURES_TO_USE:
            FEATURES_TO_USE.append(feature)
    print(f"Updated FEATURES_TO_USE with symbol one-hot features: {FEATURES_TO_USE}")


    # STEP 2: Normalize features
    print("STEP 2: Normalizing features...")
    # Select only the features to be scaled
    # Exclude 'Symbol' related columns as they are already 0/1 and should not be scaled by MinMaxScaler
    # unless you explicitly want to. Here, they are already prepared.
    features_to_scale = [f for f in FEATURES_TO_USE if f in combined_df.columns and not f.startswith('Symbol_')]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scaler only on training data (conceptually, we'll fit on the first part of combined_df)
    # For simplicity here, we fit on the whole dataset to ensure consistent scaling for backtesting
    # In a real-world scenario, you might want to fit on training set only and transform test set.
    combined_df[features_to_scale] = scaler.fit_transform(combined_df[features_to_scale])
    
    print(f"Scaler fitted on {len(features_to_scale)} features.")

    # STEP 3: Create sequences
    print("STEP 3: Creating sequences...")
    # Now combined_df already contains the one-hot encoded symbol features
    X, y = create_sequences(combined_df, SEQUENCE_LENGTH, FEATURES_TO_USE, TARGET_COLUMN)
    
    if X.size == 0 or y.size == 0:
        print("Exiting: No sequences created. Check data, sequence length, and target column.")
        sys.exit(1)

    # Convert target to categorical (one-hot encoding)
    num_classes = len(np.unique(y)) # This will now be 3 (0, 1, 2)
    if num_classes < 2:
        print(f"Exiting: Not enough classes in target variable (found {num_classes}). Need at least 2 for classification.")
        sys.exit(1)
    
    y_categorical = to_categorical(y, num_classes=num_classes)

    print(f"X shape: {X.shape}, y shape: {y_categorical.shape}")

    # STEP 4: Split data into training and validation sets
    print("STEP 4: Splitting data into training and validation sets...")
    split_index = int(len(X) * SPLIT_RATIO)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y_categorical[:split_index], y_categorical[split_index:]

    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")

    # Handle class imbalance (if applicable)
    y_train_original = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_original), y=y_train_original)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Computed class weights: {class_weight_dict}")

    # STEP 5: Build and train the LSTM model
    print("STEP 5: Building and training the LSTM model...")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes, learning_rate=LEARNING_RATE)
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_PATH, 'best_lstm_model.keras'), 
                                       monitor='val_loss', 
                                       save_best_only=True, 
                                       verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight_dict # Apply class weights
    )

    # STEP 6: Evaluate the model
    print("STEP 6: Evaluating the model...")
    # Load the best model saved by ModelCheckpoint
    try:
        best_model = tf.keras.models.load_model(os.path.join(MODEL_SAVE_PATH, 'best_lstm_model.keras'))
        print("Loaded best model for evaluation.")
        
        loss, accuracy = best_model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Classification Report
        y_pred_probs = best_model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis=1) # Convert probabilities to class labels
        y_true = np.argmax(y_val, axis=1) # Convert one-hot to class labels

        # New: Target names for classification report
        target_names = ['Sell (0)', 'Buy (1)', 'Hold (2)']
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names)) 
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    except Exception as e:
        print(f"❌ Error during model evaluation: {e}. Evaluation skipped.")

    # STEP 7: บันทึก Scaler และ Features List
    scaler_file_path = os.path.join(SCALER_SAVE_PATH, 'minmax_scaler.joblib')
    joblib.dump(scaler, scaler_file_path)
    print(f"7. Scaler saved to {scaler_file_path}")

    features_list_file_path = os.path.join(SCALER_SAVE_PATH, 'features_list.joblib')
    joblib.dump(FEATURES_TO_USE, features_list_file_path) # Save the actual list used
    print(f"8. Features list saved to {features_list_file_path}")

    # STEP 8: Backtesting และ Plotting (ใช้ข้อมูลทั้งหมดเพื่อ Backtest)
    # โหลดโมเดลที่ดีที่สุดอีกครั้งสำหรับการ Backtest
    try:
        best_model_for_backtest = tf.keras.models.load_model(os.path.join(MODEL_SAVE_PATH, 'best_lstm_model.keras'))
        print(f"9. Loaded best model for backtesting.")
        
        # Prepare data for backtesting: use the scaled combined_df directly
        # The combined_df already has the one-hot encoded symbol features
        
        # Re-create sequences using the combined_df and selected features for backtesting
        X_backtest, y_backtest = create_sequences(combined_df, SEQUENCE_LENGTH, FEATURES_TO_USE, TARGET_COLUMN)

        # Predict on the backtest data
        predictions = best_model_for_backtest.predict(X_backtest)
        predicted_classes = np.argmax(predictions, axis=1)

        # Align predictions with original DataFrame
        # The predictions correspond to the bar *after* each sequence.
        # So, the first prediction is for index = SEQUENCE_LENGTH
        # The DataFrame for backtesting should start from SEQUENCE_LENGTH index
        backtest_df = combined_df.iloc[SEQUENCE_LENGTH:].copy()
        backtest_df['predicted_signal'] = predicted_classes # Assign predictions

        # New: Calculate returns based on 3 classes (0:Sell, 1:Buy, 2:Hold)
        backtest_df['strategy_return'] = 0.0 # Initialize returns to 0

        # If predicted Buy (1) and actual next bar was up (target 1) -> +1 return
        backtest_df.loc[(backtest_df['predicted_signal'] == 1) & (y_backtest == 1), 'strategy_return'] = 1
        # If predicted Buy (1) and actual next bar was down/hold (target 0/2) -> -1 return
        backtest_df.loc[(backtest_df['predicted_signal'] == 1) & (y_backtest != 1), 'strategy_return'] = -1

        # If predicted Sell (0) and actual next bar was down (target 0) -> +1 return
        backtest_df.loc[(backtest_df['predicted_signal'] == 0) & (y_backtest == 0), 'strategy_return'] = 1
        # If predicted Sell (0) and actual next bar was up/hold (target 1/2) -> -1 return
        backtest_df.loc[(backtest_df['predicted_signal'] == 0) & (y_backtest != 0), 'strategy_return'] = -1

        # If predicted Hold (2) -> 0 return (no trade)
        # This is already handled by initializing strategy_return to 0.0

        # Calculate cumulative returns for strategy
        backtest_df['cumulative_strategy_return'] = backtest_df['strategy_return'].cumsum()
        
        # Calculate Buy & Hold return for comparison
        initial_price_for_bh = backtest_df['Close'].iloc[0]
        backtest_df['buy_and_hold_return'] = (backtest_df['Close'] / initial_price_for_bh - 1) * 100 # Percentage return

        # Group by symbol for plotting
        grouped_returns = backtest_df.groupby('Symbol')
        
        plt.figure(figsize=(15, 8))
        for name, group in grouped_returns:
            # Ensure indices are aligned for plotting
            group['cumulative_strategy_return_by_symbol'] = group['strategy_return'].cumsum()
            group['buy_and_hold_return_by_symbol'] = (group['Close'] / group['Close'].iloc[0] - 1) * 100
            
            group['cumulative_strategy_return_by_symbol'].plot(label=f'Strategy {name}', alpha=0.7)
            group['buy_and_hold_return_by_symbol'].plot(label=f'Buy & Hold {name}', alpha=0.7, linestyle='--')
            
            final_strategy_return = group['cumulative_strategy_return_by_symbol'].iloc[-1]
            final_buy_and_hold_return = group['buy_and_hold_return_by_symbol'].iloc[-1]
            plt.title(f'{name} (Strat: {final_strategy_return:.2f}, B&H: {final_buy_and_hold_return:.2f})')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Return (%)')
            plt.legend()
            plt.grid(True)
            plt.show() # Show plot for each symbol

    except Exception as e:
        print(f"❌ Error loading best model for backtesting: {e}. Backtesting skipped.")

    print("\n--- Training and Evaluation Process Completed Successfully ---")
    sys.exit(0)
