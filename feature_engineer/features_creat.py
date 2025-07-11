import pandas as pd

INPUT_FILE = r'C:\Users\SPC11\Desktop\Projects\stock_prediction\label_data\label_add.csv'
OUTPUT_FILE = 'output_features.csv'

# ================================
# STEP 1: Load labeled data
# ================================

print("STEP 1: Loading labeled data...")
df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('America/New_York')
df = df.sort_values('Date').reset_index(drop=True)

# ================================
# STEP 2: Lagged Returns
# ================================

print("STEP 2: Adding Lagged Returns...")
df['Return_1min'] = df['Close'].pct_change(1)
df['Return_3min'] = df['Close'].pct_change(3)
df['Return_5min'] = df['Close'].pct_change(5)

# ================================
# STEP 3: Moving Averages
# ================================

print("STEP 3: Adding SMA/EMA...")
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

# ================================
# STEP 4: RSI (14)
# ================================

print("STEP 4: Adding RSI...")
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))

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

# ================================
# STEP 5: Drop rows with NaN
# ================================
print("STEP 5: Dropping NaN rows...")
df = df.dropna().reset_index(drop=True)

# ================================
# STEP 6: Save
# ================================
keep_cols = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP',
    'Return_1min', 'Return_3min', 'Return_5min',
    'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI_14',
    'Body', 'Range', 'VWAP_Diff', 'Volume_SMA_10', 'Vol_Spike',
    'Label'
]

df[keep_cols].to_csv(OUTPUT_FILE, index=False)

print(f"✅ Features saved to: {OUTPUT_FILE}")
print(f"✅ Final shape: {df.shape}")
