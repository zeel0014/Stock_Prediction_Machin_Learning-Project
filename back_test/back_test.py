import pandas as pd
import joblib

# ================================
# STEP 1: Data Load aur Model Load
# ================================

print("🔹 STEP 1: Data & Model Load...")

df = pd.read_csv(r'C:\Users\SPC11\Desktop\Projects\stock_prediction\feature_engineer\output_features.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

model = joblib.load(r'C:\Users\SPC11\Desktop\Projects\stock_prediction\train_data\xgb_model_final.pkl')
print("✅ Model Loaded")

# ================================
# STEP 2: Features aur Target Setup
# ================================
feature_cols = [
    'Return_1min', 'Return_3min', 'Return_5min',
    'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI_14',
    'Range', 'VWAP_Diff', 'Volume_SMA_10'
]

X = df[feature_cols]
df['Actual_Move'] = df['Close'].shift(-1) - df['Close']

# ================================
# STEP 3: Predict Proba aur Signal Generate
# ================================
print("🔹 STEP 3: Predict Proba & Signals...")

probas = model.predict_proba(X)[:, 1]
df['Proba'] = probas

threshold = 0.4
df['Signal'] = df['Proba'].apply(lambda x: 1 if x > threshold else 0)

# ================================
# STEP 4: P&L Calculate
# ================================
print("🔹 STEP 4: P&L Calculation...")

df['PnL'] = df['Signal'] * df['Actual_Move']
df['Equity_Curve'] = df['PnL'].cumsum()

df.to_csv('backtest_output.csv', index=False)
print("✅ Backtest output saved: backtest_output.csv")

# ================================
# STEP 5: Risk Metrics
# ================================
print("🔹 STEP 5: Risk Metrics Calculation...")

total_trades = (df['Signal'] != 0).sum()
win_trades = ((df['Signal'] != 0) & (df['PnL'] > 0)).sum()
loss_trades = ((df['Signal'] != 0) & (df['PnL'] <= 0)).sum()
win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0

total_pnl = df.loc[df['Signal'] != 0, 'PnL'].sum()
avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

returns = df.loc[df['Signal'] != 0, 'PnL']
mean_return = returns.mean()
std_return = returns.std()
sharpe_ratio = mean_return / std_return if std_return > 0 else 0

equity = df['Equity_Curve']
rolling_max = equity.cummax()
drawdown = (equity - rolling_max)
max_drawdown = drawdown.min()

# ================================
# STEP 6: Print Summary
# ================================
print("\n🔹 Backtest Metrics 🔹")
print(f"Total Trades: {total_trades}")
print(f"Wins: {win_trades}")
print(f"Losses: {loss_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total PnL: {total_pnl:.2f}")
print(f"Avg PnL per Trade: {avg_pnl:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.2f}")

print("\n✅ FULL Backtest & Risk Metrics Complete!")

