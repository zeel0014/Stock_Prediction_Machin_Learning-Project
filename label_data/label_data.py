import pandas as pd

INPUT_FILE = r'C:\Users\SPC11\Desktop\Projects\stock_prediction\preprocessing_and_cleaning\output_cleaned.csv'   
OUTPUT_FILE = 'output_label.csv'    
THRESHOLD = 0.0002  

# ========================================
# STEP 1: Load cleaned data
# ========================================

print("\nSTEP 1: Loading Data...")

df = pd.read_csv(INPUT_FILE)
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('America/New_York')

print(f"Date column dtype AFTER parse: {df['Date'].dtype}")
print(df['Date'].head())

# ========================================
# STEP 2: Generate next candle close
# =======================================

print("\nSTEP 2: Generating Future_Close column...")
df['Future_Close'] = df['Close'].shift(-1)

# ========================================
# STEP 3: Generate Label (UP/DOWN)
# ========================================

print("STEP 3: Generating Labels...")
returns = (df['Future_Close'] - df['Close']) / df['Close']

df['Label'] = 0
df.loc[returns > THRESHOLD, 'Label'] = 1

# ========================================
# STEP 4: Remove last row of each day (no next candle)
# ========================================
print("STEP 4: Dropping last row of each day...")

df['Date_only'] = df['Date'].dt.date
last_indexes = df.groupby('Date_only').tail(1).index
df = df.drop(index=last_indexes).reset_index(drop=True)

# ========================================
# STEP 5: Save final labeled data
# ========================================
print("STEP 5: Saving labeled data...")

final_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Label']  
df[final_columns].to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Labeled data saved to: {OUTPUT_FILE}")
print(f"✅ Final shape: {df.shape}")

# ========================================
# STEP 6: Class balance check (UP/DOWN %)
# ========================================
print("\nSTEP 6: Class balance check...")

label_counts = df['Label'].value_counts()
print("Label distribution:")
print(label_counts)

total = label_counts.sum()
up_percent = (label_counts[1] / total) * 100 if 1 in label_counts else 0
down_percent = (label_counts[0] / total) * 100 if 0 in label_counts else 0

print(f"UP%: {up_percent:.2f}%")
print(f"DOWN%: {down_percent:.2f}%")

print("\n✅ Label Generation complete!")
