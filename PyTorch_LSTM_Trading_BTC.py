# FILE NAME: runModel_BTCshortTrade_InputFile.py
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import joblib
import json
import time
import os
import sys 
from datetime import datetime, timedelta
import pytz # for timezone awareness
import MetaTrader5 as mt5 # <-- แก้ไขบรรทัดนี้แล้ว

# Diagnostic print to confirm MT5 import
print("DEBUG: MetaTrader5 imported successfully at script start.")

# Import ta library for technical indicators (needed for real-time feature generation)
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# --- Global Configurations ---
# Sequence Length for CNN+LSTM (number of past bars to consider)
SEQUENCE_LENGTH = 10 
# Path to the trained model weights
MODEL_WEIGHTS_PATH = 'cnn_lstm_model_weights.pth'
# Path to the list of features (from training)
FEATURES_LIST_PATH = 'features_list.pkl'
# Path to the scaler (from training)
SCALER_PATH = 'scaler.pkl'

# Define the FIXED, HARDCODED PATH to the MT5 Tester Agent's MQL5\Files folder.
# !!! IMPORTANT: This path is dynamic and changes with each Strategy Tester run.
# You MUST update this path in the script if your MT5 Tester Agent's path changes.
FIXED_AGENT_FILES_PATH = "C:\\Users\\graze\\AppData\\Roaming\\MetaQuotes\\Tester\\53785E099C927DB68A545C249CDBCE06\\Agent-127.0.0.1-3000\\MQL5\\Files" # <-- Update this path if it changes!

INPUT_DATA_FILE_NAME = "input_data.json"
OUTPUT_RESULT_FILE_NAME = "prediction_result.txt"

# Construct full paths for input and output files
input_file_path = os.path.join(FIXED_AGENT_FILES_PATH, INPUT_DATA_FILE_NAME)
output_file_path = os.path.join(FIXED_AGENT_FILES_PATH, OUTPUT_RESULT_FILE_NAME)

LAST_INPUT_FILE_MOD_TIME = 0 # Last modification time of the input_data.json file

# --- 1. Model Definition (MUST BE IDENTICAL to training script) ---
class CNN_LSTM_Model(nn.Module): # Renamed to match training script
    def __init__(self, input_features_per_step, sequence_length, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(CNN_LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # CNN layer to extract features across the sequence
        # Input to Conv1d: (batch_size, in_channels, sequence_length)
        # Here, in_channels = input_features_per_step, sequence_length = sequence_length
        self.conv1d = nn.Conv1d(in_channels=input_features_per_step, out_channels=64, kernel_size=3, padding='same') # padding='same' to maintain sequence length
        self.relu_conv = nn.ReLU()
        self.dropout_conv = nn.Dropout(dropout_rate)
        
        # LSTM layer
        # Input to LSTM: (batch_size, sequence_length, input_size_for_lstm)
        # input_size_for_lstm will be the output_channels of Conv1d
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=dropout_rate) # num_layers is now 3
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn = nn.BatchNorm1d(hidden_size // 2) # <-- Added BatchNorm1d
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size // 2, output_size) 

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_features_per_step)
        
        # Permute x for Conv1d: (batch_size, input_features_per_step, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Apply Conv1d
        conv_out = self.conv1d(x) # (batch_size, out_channels, new_sequence_length)
        conv_out = self.relu_conv(conv_out)
        conv_out = self.dropout_conv(conv_out)
        
        # Permute back for LSTM: (batch_size, new_sequence_length, out_channels)
        conv_out = conv_out.permute(0, 2, 1)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(conv_out) # lstm_out shape: (batch_size, new_sequence_length, hidden_size)
        
        # Take the output from the last time step for sequence-to-one prediction
        out = lstm_out[:, -1, :] # out shape: (batch_size, hidden_size)
        
        # Pass through fully connected layers with BatchNorm
        out = self.fc1(out)
        out = self.bn(out) # <-- Used BatchNorm1d
        out = self.relu_fc(out)
        out = self.dropout_fc(out)
        out = self.fc2(out) # Output are now logits (NO Sigmoid here)
        return out

# --- 2. Load Model and Preprocessing Components ---
# Determine device for inference (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for inference: {device}")

try:
    # Initialize the model with the correct dimensions
    _dummy_features_list = joblib.load(FEATURES_LIST_PATH)
    NUM_FEATURES = len(_dummy_features_list)
    
    # Model parameters (MUST MATCH TRAINING SCRIPT)
    hidden_size = 128 
    num_layers = 3    
    output_size = 1
    dropout_rate = 0.3

    # Instantiate the correct model class: CNN_LSTM_Model
    model = CNN_LSTM_Model(NUM_FEATURES, SEQUENCE_LENGTH, hidden_size, num_layers, output_size, dropout_rate).to(device)
    
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    print(f"✅ Model '{MODEL_WEIGHTS_PATH}' loaded successfully.")

    features_list = _dummy_features_list
    print(f"✅ Features list '{FEATURES_LIST_PATH}' loaded successfully.")

    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Scaler '{SCALER_PATH}' loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ Error: Required file not found - {e}. Please ensure model, features list, and scaler files are in the same directory.")
    sys.exit() 
except Exception as e:
    print(f"❌ Error loading model or components: {e}")
    sys.exit() 

# --- 3. MT5 Data Fetching Setup ---
try:
    if not mt5.initialize():
        print("❌ initialize() failed, error code =", mt5.last_error())
        sys.exit() 
    print("✅ MetaTrader5 initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing MetaTrader5: {e}")
    sys.exit() 

# --- Global Configuration for Max Available Bars (from training script) ---
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

# --- 4. Feature Engineering Functions (from training notebook) ---
def _get_features_for_timeframe_data(data, prefix=''):
    if data.empty:
        return pd.DataFrame()

    df = data.copy()
    
    # Ensure numerical types
    for col in ['open', 'high', 'low', 'close', 'tick_volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add 'real_volume' and 'spread' if not present (from MT5 rates)
    if 'real_volume' not in df.columns:
        df['real_volume'] = 0
    if 'spread' not in df.columns:
        df['spread'] = 0 # Default to 0, MT5 data doesn't provide spread per bar directly

    # Feature Engineering (MUST BE IDENTICAL to training script)
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['Stoch_K'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3).stoch()
    df['Stoch_D'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3).stoch_signal()
    df['EMA_fast'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
    df['EMA_slow'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['EMA_diff'] = df['EMA_fast'] - df['EMA_slow']
    macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['Price_change'] = df['close'] - df['open']
    df['Body_size'] = abs(df['close'] - df['open'])
    df['Upper_shadow'] = df['high'] - df[['close','open']].max(axis=1)
    df['Lower_shadow'] = df[['close','open']].min(axis=1) - df['low']
    df['Upper_shadow_ratio'] = np.where(df['Body_size'] != 0, df['Upper_shadow'] / df['Body_size'], 0)
    df['Lower_shadow_ratio'] = np.where(df['Body_size'] != 0, df['Lower_shadow'] / df['Body_size'], 0)
    df['return_1'] = df['close'].pct_change(1)
    df['return_2'] = df['close'].pct_change(2)
    df['return_3'] = df['close'].pct_change(3)
    df['return_7'] = df['close'].pct_change(7) 
    df['RSI_lag1'] = df['RSI'].shift(1)
    df['MACD_hist_lag1'] = df['MACD_hist'].shift(1)
    df['ATR_lag1'] = df['ATR'].shift(1)
    df['Stoch_K_lag1'] = df['Stoch_K'].shift(1)
    df['Stoch_D_lag1'] = df['Stoch_D'].shift(1)
    df['EMA_fast_lag1'] = df['EMA_fast'].shift(1)
    df['EMA_slow_lag1'] = df['EMA_slow'].shift(1)
    df['close_lag1'] = df['close'].shift(1)
    df['volume_avg'] = df['tick_volume'].rolling(window=20).mean()
    
    divergence_lookback = 5
    df['bullish_rsi_divergence'] = np.where(
        (df['close'] < df['close'].shift(divergence_lookback)) & (df['RSI'] > df['RSI'].shift(divergence_lookback)), 1, 0
    )
    df['bearish_rsi_divergence'] = np.where(
        (df['close'] > df['close'].shift(divergence_lookback)) & (df['RSI'] < df['RSI'].shift(divergence_lookback)), 1, 0
    )

    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_width'] = bollinger.bollinger_wband()
    df['BB_percent'] = bollinger.bollinger_pband()
    df['EMA_cross_signal'] = 0
    df.loc[(df['EMA_fast'].shift(1) < df['EMA_slow'].shift(1)) & (df['EMA_fast'] > df['EMA_slow']), 'EMA_cross_signal'] = 1
    df.loc[(df['EMA_fast'].shift(1) > df['EMA_slow'].shift(1)) & (df['EMA_fast'] < df['EMA_slow']), 'EMA_cross_signal'] = -1
    df['RSI_ROC'] = df['RSI'].diff(periods=3)
    df['MACD_hist_ROC'] = df['MACD_hist'].diff(periods=3)
    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx_indicator.adx()
    df['ADX_pos'] = adx_indicator.adx_pos()
    df['ADX_neg'] = adx_indicator.adx_neg()
    df['Price_change_ATR_ratio'] = np.where(df['ATR'] != 0, df['Price_change'] / df['ATR'], 0)

    df['bullish_engulfing'] = np.where(
        (df['close'] > df['open']) & (df['open'].shift(1) > df['close'].shift(1)) & 
        (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) &
        (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))), 1, 0
    )
    df['bearish_engulfing'] = np.where(
        (df['close'] < df['open']) & (df['open'].shift(1) < df['close'].shift(1)) & 
        (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) &
        (abs(df['close'] - df['open']) > abs(df['close'].shift(1) - df['open'].shift(1))), 1, 0
    )
    df['hammer'] = np.where(
        (df['Body_size'] > 0) & (df['Lower_shadow'] >= 2 * df['Body_size']) & (df['Upper_shadow'] <= 0.2 * df['Body_size']), 1, 0
    )
    df['shooting_star'] = np.where(
        (df['Body_size'] > 0) & (df['Upper_shadow'] >= 2 * df['Body_size']) & (df['Lower_shadow'] <= 0.2 * df['Body_size']), 1, 0
    )
    df['doji_val'] = np.where(
        (df['Body_size'] < (df['high'] - df['low']) * 0.1) & 
        ((df['high'] - df['low']) > df['ATR'] * 0.1), 
        1, 0
    )

    df['rsi_oversold_signal'] = np.where(df['RSI'] < 30, 1, 0)
    df['rsi_overbought_signal'] = np.where(df['RSI'] > 70, 1, 0)
    
    df['macd_bullish_cross_signal'] = np.where(
        (df['MACD'].shift(1) < df['MACD_signal'].shift(1)) & (df['MACD'] > df['MACD_signal']), 1, 0
    )
    df['ma_golden_cross_signal'] = np.where(
        (df['EMA_fast'].shift(1) < df['EMA_slow'].shift(1)) & (df['EMA_fast'] > df['EMA_slow']), 1, 0
    )
    df['bb_lower_touch_signal'] = np.where(
        (df['close'] <= df['BB_lower']) & (df['RSI'] < 40), 1, 0
    )

    df['ema_fast_slope'] = df['EMA_fast'].diff(periods=3)
    df['ema_slow_slope'] = df['EMA_slow'].diff(periods=3)
    
    # Drop rows with NaN values created by indicators
    df.dropna(inplace=True)
    return df

def fetch_and_prepare_data(symbol, sequence_length, features_list):
    # Determine required bars for each timeframe to ensure enough history for indicators and sequence
    # A generous buffer (e.g., 50 bars) for indicator calculations and dropna
    safety_bars = 50 
    
    max_h1_bars_inf = MAX_AVAILABLE_BARS.get(mt5.TIMEFRAME_H1, 0)
    max_m15_bars_inf = MAX_AVAILABLE_BARS.get(mt5.TIMEFRAME_M15, 0)
    max_h4_bars_inf = MAX_AVAILABLE_BARS.get(mt5.TIMEFRAME_H4, 0)

    n_bars_h1_req = min(sequence_length + safety_bars, max_h1_bars_inf)
    n_bars_m15_req = min((sequence_length + safety_bars) * 4, max_m15_bars_inf)
    n_bars_h4_req = min((sequence_length + safety_bars) // 4 + 1, max_h4_bars_inf)

    if n_bars_h1_req < sequence_length or n_bars_m15_req < sequence_length or n_bars_h4_req < sequence_length:
        raise ValueError(f"Not enough historical data available for all timeframes to form sequences of length {sequence_length}.")

    print(f"Fetching {n_bars_h1_req} H1 bars, {n_bars_m15_req} M15 bars, {n_bars_h4_req} H4 bars for inference.")

    h1_raw_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_bars_h1_req)
    m15_raw_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, n_bars_m15_req)
    h4_raw_data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, n_bars_h4_req)

    if h1_raw_data is None or len(h1_raw_data) == 0:
        raise ValueError("Failed to fetch H1 data.")
    if m15_raw_data is None or len(m15_raw_data) == 0:
        raise ValueError("Failed to fetch M15 data.")
    if h4_raw_data is None or len(h4_raw_data) == 0:
        raise ValueError("Failed to fetch H4 data.")

    df_h1 = pd.DataFrame(h1_raw_data)
    df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
    df_h1.set_index('time', inplace=True)

    df_m15 = pd.DataFrame(m15_raw_data)
    df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
    df_m15.set_index('time', inplace=True)

    df_h4 = pd.DataFrame(h4_raw_data)
    df_h4['time'] = pd.to_datetime(df_h4['time'], unit='s')
    df_h4.set_index('time', inplace=True)

    # Calculate features for each timeframe
    df_h1_features = _get_features_for_timeframe_data(df_h1).add_suffix('_H1')
    df_m15_features = _get_features_for_timeframe_data(df_m15).add_suffix('_M15')
    df_h4_features = _get_features_for_timeframe_data(df_h4).add_suffix('_H4')

    # Align features to H1 index
    df_combined = pd.merge_asof(df_h1_features, df_m15_features, left_index=True, right_index=True, direction='backward')
    df_combined = pd.merge_asof(df_combined, df_h4_features, left_index=True, right_index=True, direction='backward')
    df_combined.dropna(inplace=True)

    if len(df_combined) < sequence_length:
        raise ValueError(f"Not enough combined data to form a sequence of length {sequence_length}. Available: {len(df_combined)}")

    # Get the latest sequence of features
    latest_sequence_df = df_combined[features_list].tail(sequence_length)

    if len(latest_sequence_df) < sequence_length:
        raise ValueError(f"Not enough data to form a complete sequence of length {sequence_length} after alignment and feature selection.")
        
    # Scale the sequence data
    scaled_sequence = scaler.transform(latest_sequence_df)

    # Convert to PyTorch tensor
    # Input tensor shape: (batch_size, sequence_length, num_features)
    input_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).to(device) # Move to device

    return input_tensor

# --- 5. Prediction Processing Function ---
def process_prediction(symbol):
    try:
        # Fetch and prepare the data sequence for the model
        input_tensor = fetch_and_prepare_data(symbol, SEQUENCE_LENGTH, features_list)
        
        # Make prediction
        with torch.no_grad(): # Disable gradient calculation for inference
            # Model outputs logits, apply sigmoid for probability
            logits = model(input_tensor)
            proba = torch.sigmoid(logits).cpu().item() 
        
        # Using 0.50 as the threshold (consistent with training evaluation)
        pred = int(proba > 0.50) 
        
        return {'prediction': pred, 'probability': float(proba)}
    
    except ValueError as ve:
        print(f"❌ Data preparation error: {ve}")
        return {'prediction': -1, 'probability': 0.0, 'error': str(ve)} # -1 for error
    except Exception as e:
        print(f"❌ Prediction processing error: {e}")
        return {'prediction': -1, 'probability': 0.0, 'error': str(e)} # -1 for error

# --- Main File Monitoring Loop ---
def main_file_monitor():
    global LAST_INPUT_FILE_MOD_TIME

    print(f"Python Prediction Service Started. Monitoring for '{input_file_path}'...")

    # Check if the fixed path actually exists. If not, warn the user.
    if not os.path.isdir(FIXED_AGENT_FILES_PATH):
        print(f"⚠️ Warning: The specified FIXED_AGENT_FILES_PATH '{FIXED_AGENT_FILES_PATH}' does not exist.")
        print("Please ensure this directory exists and is the correct path for your current MT5 Tester Agent.")
        print("The script will continue to monitor, but it might not find the files.")

    # MT5 connection check
    if not mt5.is_connected():
        print("❌ MT5 connection failed during initial check.")
        if not mt5.initialize():
            print("❌ Failed to initialize MT5 on startup. Please check your MT5 installation and path.")
            sys.exit() # Critical error, exit if MT5 cannot be initialized

    while True:
        try:
            # Check if input_data.json exists
            if os.path.exists(input_file_path):
                current_modified_time = os.path.getmtime(input_file_path)
                
                # Check if the file has been modified (new data written by EA)
                if current_modified_time > LAST_INPUT_FILE_MOD_TIME:
                    # Read input_data.json
                    with open(input_file_path, 'r', encoding='utf-8') as f:
                        data_str = f.read()
                    
                    LAST_INPUT_FILE_MOD_TIME = current_modified_time # Update last modified time
                    
                    print(f"📖 Received data from MT5 (via file '{INPUT_DATA_FILE_NAME}'): {data_str[:100]}...") 

                    # Parse JSON (it's just a timestamp trigger from EA)
                    json.loads(data_str) # Just parse to validate, content not used for features
                    
                    # Extract symbol (assuming it's hardcoded or known, or passed in JSON)
                    symbol = "BTCUSDm" # <-- CHANGE THIS TO YOUR TRADING SYMBOL IF DIFFERENT!
                    print(f"⚡ Processing prediction for symbol: {symbol}...")

                    # Make prediction
                    result = process_prediction(symbol)
                    
                    # Write prediction to prediction_result.txt
                    with open(output_file_path, 'w', encoding='ascii') as f:
                        json.dump(result, f)
                    print(f"✍️ Wrote prediction to '{OUTPUT_RESULT_FILE_NAME}': {result}")
                    
                    # Delete input.json to signal EA that processing is complete and it can write new data
                    try:
                        os.remove(input_file_path)
                        print(f"🗑️ Deleted '{INPUT_DATA_FILE_NAME}' to signal completion.")
                    except OSError as e:
                        print(f"❌ Error deleting input file: {e}")
            
            time.sleep(0.01) # Check every 10ms for quick response

        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON from '{INPUT_DATA_FILE_NAME}': {e}. Content: '{data_str}'")
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ Deleted corrupted '{INPUT_DATA_FILE_NAME}'.")
                except Exception as e_clean:
                    print(f"❌ Could not delete corrupted input file: {e_clean}")
        except ValueError as e:
            print(f"❌ Feature or data error: {e}")
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ Deleted processed '{INPUT_DATA_FILE_NAME}' due to data error.")
                except Exception as e_clean:
                    print(f"❌ Could not delete input file after data error: {e_clean}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ Deleted '{INPUT_DATA_FILE_NAME}' due to unexpected error.")
                except Exception as e_clean:
                    print(f"❌ Could not delete input file after unexpected error: {e_clean}")


if __name__ == '__main__':
    main_file_monitor() # Start the file monitoring immediately
