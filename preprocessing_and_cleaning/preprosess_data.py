import pandas as pd

INPUT_FILE = 'data/AAPL_minute_2years.csv'  
OUTPUT_FILE = 'data/output_cleaned.csv'

# ========================
# STEP 1: Load Data
# ========================

df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Ensure Date is timezone aware - America/New_York
df['Date'] = pd.to_datetime(df['Date'], utc=True)      
df['Date'] = df['Date'].dt.tz_convert('America/New_York')


all_clean_days = []
removed_days = []

# ========================
# STEP 2: Process each day
# ========================

unique_days = df['Date'].dt.date.unique()

for day in unique_days:

    day_df = df[df['Date'].dt.date == day].copy()
    day_df = day_df.set_index('Date')

    start_time = pd.Timestamp(f"{day} 09:30:00", tz='America/New_York')
    end_time = pd.Timestamp(f"{day} 15:59:00", tz='America/New_York')

    full_index = pd.date_range(start=start_time, end=end_time, freq='1min', tz='America/New_York')

    day_df = day_df.reindex(full_index)

    missing_count = day_df['Open'].isna().sum()

    if missing_count >= 10:
        print(f"Dropping day: {day} → Missing candles: {missing_count}")
        removed_days.append({'Date': day, 'Missing': missing_count})
        continue  

    day_df[['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']] = day_df[['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].ffill()

    day_df = day_df.reset_index().rename(columns={'index': 'Date'})

    all_clean_days.append(day_df)

# ========================
# STEP 3: Final DataFrame
# ========================

final_df = pd.concat(all_clean_days).reset_index(drop=True)

final_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']  
final_df = final_df[final_columns]

# ========================
# STEP 4: Save cleaned data
# ========================

final_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nCleaned data saved to: {OUTPUT_FILE}")
print(f"Final shape: {final_df.shape}")

print(f"\nRemoved days with >=10 missing candles:")
for entry in removed_days:
    print(f"{entry['Date']}: {entry['Missing']} missing")

print(f"\nTotal removed days: {len(removed_days)}")
