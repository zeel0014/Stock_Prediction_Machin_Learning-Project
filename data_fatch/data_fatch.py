import os
import requests
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import time
import sys
import pytz

# ============================== #
#        CONFIGURATION           #
# ============================== #

API_KEY = "NyENoHksPMSLaZQUv6fj8IgDGxZrFpDj"         # Replace with your actual Polygon.io API key
TICKER = "AAPL"
TIMEFRAME = "minute"
MAX_LIMIT = 50000
FINAL_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "VWAP"]
URL_PATTERN = f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/1/{TIMEFRAME}"
OUTPUT_FILE = f"data/{TICKER}_{TIMEFRAME}_2years.csv"

# ============================== #
#         UTILITY FUNCS          #
# ============================== #

def is_trading_day(date, nyse_calendar):
    schedule = nyse_calendar.schedule(start_date=date, end_date=date)
    return not schedule.empty

def format_df(df):
    """
    Format raw polygon.io response into desired columns.
    """
    return pd.DataFrame({
        "Date": pd.to_datetime(df["t"], unit="ms").dt.tz_localize('UTC').dt.tz_convert('America/New_York'),
        "Open": df["o"],
        "High": df["h"],
        "Low": df["l"],
        "Close": df["c"],
        "Volume": df["v"],
        "VWAP": df["vw"]
    })

# ============================== #
#         MAIN SCRIPT            #
# ============================== #

def main():
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now().date()

    # Default full 2-year range
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=730)
    existing_df = pd.DataFrame(columns=FINAL_COLUMNS)

    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing file: {OUTPUT_FILE}. Resuming from last date...")
        existing_df = pd.read_csv(OUTPUT_FILE, parse_dates=["Date"])
        existing_df = existing_df[FINAL_COLUMNS]

        last_date = existing_df["Date"].max().date()
        print(f"Last date in file: {last_date}")

        start_date = last_date + timedelta(days=1)

        if start_date > end_date:
            print(f"Already up to date! No new data to fetch.")
            sys.exit(0)
    else:
        print(f"No existing file. Will fetch full 2 years of data.")

    print(f"Fetching from: {start_date} to {end_date}")

    current_date = start_date
    all_data = []
    total_rows = 0

    requests_this_minute = 0
    minute_start_time = time.time()

    try:
        while current_date <= end_date:
            date_str = current_date.strftime("%d/%m/%Y")

            if not is_trading_day(current_date, nyse):
                print(f"{date_str} - Market Closed (Weekend/Holiday)")
                current_date += timedelta(days=1)
                continue

            print(f"{date_str} - Fetching Data ...")

            from_zone = pytz.timezone('America/New_York')
            day_start = from_zone.localize(datetime.combine(current_date, datetime.strptime('09:30', '%H:%M').time()))
            day_end = from_zone.localize(datetime.combine(current_date, datetime.strptime('16:00', '%H:%M').time()))

            from_timestamp = int(day_start.timestamp()) * 1000
            to_timestamp = int(day_end.timestamp()) * 1000 - 1

            more_data = True
            next_timestamp = from_timestamp
            day_dataframes = []

            while more_data:
                if requests_this_minute >= 5:
                    elapsed = time.time() - minute_start_time
                    if elapsed < 60:
                        sleep_time = 60 - elapsed + 1  # Add 1 sec buffer
                        print(f"⏳ Rate limit hit. Sleeping for {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    requests_this_minute = 0
                    minute_start_time = time.time()  # reset minute start


                params = {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": MAX_LIMIT,
                    "apiKey": API_KEY
                }

                url = f"{URL_PATTERN}/{next_timestamp}/{to_timestamp}"

                response = requests.get(url, params=params)
                requests_this_minute += 1
                response.raise_for_status()
                data = response.json()

                if "results" in data and data["results"]:
                    df_raw = pd.DataFrame(data["results"])
                    df_formatted = format_df(df_raw)
                    day_dataframes.append(df_formatted)

                    if len(data["results"]) < MAX_LIMIT:
                        more_data = False
                    else:
                        next_timestamp = int(df_raw["t"].iloc[-1]) + 1
                else:
                    more_data = False

            if day_dataframes:
                daily_df = pd.concat(day_dataframes, ignore_index=True)
                rows_fetched = len(daily_df)
                total_rows += rows_fetched
                print(f"{date_str} - Fetched {rows_fetched} rows")
                all_data.append(daily_df)
            else:
                print(f"{date_str} - No data fetched.")

            current_date += timedelta(days=1)

    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Saving fetched data so far...")
        if all_data:
            new_df = pd.concat(all_data, ignore_index=True)
            new_df = new_df[FINAL_COLUMNS]


            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset=["Date"], inplace=True)
            combined_df.sort_values("Date", inplace=True)

            combined_df.to_csv(OUTPUT_FILE, index=False)
            print(f"⚡️ Partial data saved to {OUTPUT_FILE} with {len(combined_df)} rows")
        sys.exit(1)

    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        new_df = new_df[FINAL_COLUMNS]

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["Date"], inplace=True)
        combined_df.sort_values("Date", inplace=True)

        combined_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nAll data saved to {OUTPUT_FILE}")
        print(f"Total rows in file: {len(combined_df)}")
    else:
        print("\nNo new data fetched! Existing data unchanged.")

if __name__ == "__main__":
    main()
