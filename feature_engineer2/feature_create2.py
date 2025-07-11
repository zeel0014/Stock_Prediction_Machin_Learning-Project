import pandas as pd

# ========================================
# CONFIG
# ========================================
INPUT_FILE = 'new_features_add.csv'   
OUTPUT_FILE = 'output_features_v2.csv'

# ========================================
# STEP 1: Load Data
# ========================================
print("STEP 1: Loading data...")
df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ========================================
# STEP 2: Add OHLCV-derived features
# ========================================
print("STEP 2: Creating new OHLCV-derived features...")

# Body = Close - Open
df['Body'] = df['Close'] - df['Open']

# Range = High - Low
df['Range'] = df['High'] - df['Low']

# VWAP_Diff = Close - VWAP
df['VWAP_Diff'] = df['Close'] - df['VWAP']

# Volume SMA 10
df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()

# Volume Spike = Volume / Volume_SMA_10
df['Vol_Spike'] = df['Volume'] / df['Volume_SMA_10']

# ========================================
# STEP 3: Drop rows with NaN (from rolling mean)
# ========================================
print("STEP 3: Dropping NaN rows...")
df = df.dropna().reset_index(drop=True)

# ========================================
# STEP 4: Save final selected columns
# ========================================
print("STEP 4: Saving final features...")

final_columns = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
    'Return_1min', 'Return_3min', 'Return_5min',
    'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI_14',
    'Body', 'Range', 'VWAP_Diff', 'Volume_SMA_10', 'Vol_Spike',
    'Label'
]

df[final_columns].to_csv(OUTPUT_FILE, index=False)

print(f"✅ New feature file saved to: {OUTPUT_FILE}")
print(f"✅ Final shape: {df[final_columns].shape}")
