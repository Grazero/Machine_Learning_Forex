import pandas as pd
import joblib
import json
import time
import os

# --- 1. Model Setup ---
try:
    model = joblib.load('xgboost_short_term_model.pkl')
    print("✅ Model 'xgboost_short_term_model.pkl' loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'xgboost_short_term_model.pkl' not found. Please ensure the model file is in the same directory as this script.")
    exit()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Define expected features for the model
# IMPORTANT: These must exactly match the keys in the JSON string sent by the MQL5 EA.
# Based on your provided JSON example and the EA code:
expected_features = [
    'open', 'high', 'low', 'close', 'tick_volume', # ราคาและ Volume พื้นฐาน
    'RSI', 'Stoch_K', 'Stoch_D', # Momentum indicators
    'EMA_fast', 'EMA_slow', 'EMA_diff', # Trend indicators
    'MACD', 'MACD_signal', 'MACD_hist', # MACD indicators
    'ATR', # Volatility
    'Price_change', 'Body_size', 'Upper_shadow', 'Lower_shadow', # Price Action
    'Upper_shadow_ratio', 'Lower_shadow_ratio', # Price Action Ratios
    'return_1', # Quick price change
    'RSI_lag1', 'MACD_hist_lag1', 'ATR_lag1', # Lagged key indicators
    'Stoch_K_lag1', 'Stoch_D_lag1', 'EMA_fast_lag1', 'EMA_slow_lag1', 'close_lag1', # Lagged other indicators
    'volume_avg'
]

# --- 2. Prediction Processing Function ---
def process_prediction(data_dict):
    missing = [f for f in expected_features if f not in data_dict]
    if missing:
        raise ValueError(f"Missing features: {missing}. Expected: {expected_features}. Received: {list(data_dict.keys())}")

    features_df = pd.DataFrame([[data_dict[f] for f in expected_features]], columns=expected_features)
    
    # Make prediction
    proba = model.predict_proba(features_df)[:, 1][0]
    pred = int(proba > 0.4) # Using 0.4 as the threshold as seen in previous Flask code
    
    return {'prediction': pred, 'probability': float(proba)}

# --- GLOBAL VARIABLES for File Monitoring ---
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

# --- Main File Monitoring Loop ---
def main_file_monitor():
    global LAST_INPUT_FILE_MOD_TIME

    print(f"Python Prediction Service Started. Monitoring for '{input_file_path}'...")

    # Check if the fixed path actually exists. If not, warn the user.
    if not os.path.isdir(FIXED_AGENT_FILES_PATH):
        print(f"⚠️ Warning: The specified FIXED_AGENT_FILES_PATH '{FIXED_AGENT_FILES_PATH}' does not exist.")
        print("Please ensure this directory exists and is the correct path for your current MT5 Tester Agent.")
        print("The script will continue to monitor, but it might not find the files.")
        # You might want to exit here if the path is critical
        # exit() 

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

                    # Parse JSON
                    data_json = json.loads(data_str)
                    
                    # Make prediction
                    result = process_prediction(data_json)
                    
                    # Write prediction to prediction_result.json
                    with open(output_file_path, 'w', encoding='ascii') as f:
                        json.dump(result, f)
                    print(f"✍️ Wrote prediction to '{OUTPUT_RESULT_FILE_NAME}': {result}")
                    
                    # Delete input.json to signal EA that processing is complete and it can write new data
                    # os.remove(input_file_path)
                   

            time.sleep(0.01) # Check every 10ms for quick response

        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON from '{INPUT_DATA_FILE_NAME}': {e}. Content: '{data_str}'")
            # Attempt to delete corrupted file to prevent infinite loop
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ Deleted corrupted '{INPUT_DATA_FILE_NAME}'.")
                except Exception as e_clean:
                    print(f"❌ Could not delete corrupted input file: {e_clean}")
        except ValueError as e:
            print(f"❌ Feature error: {e}")
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    print(f"🗑️ Deleted processed '{INPUT_DATA_FILE_NAME}' due to feature error.")
                except Exception as e_clean:
                    print(f"❌ Could not delete input file after feature error: {e_clean}")
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
