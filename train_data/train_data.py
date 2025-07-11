import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ========================================
# CONFIG
# ========================================

INPUT_FILE = r'C:\Users\SPC11\Desktop\Projects\stock_prediction\feature_engineer\output_features.csv'
N_SPLITS = 5

MAX_DEPTH = 4          
LEARNING_RATE = 0.03   
N_ESTIMATORS = 300     

THRESHOLD = 0.4        

# ========================================
# STEP 1: Load Data
# ========================================

print("\nSTEP 1: Data Load kar rahe hain...")
df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"Total rows: {df.shape[0]}")

# ========================================
# STEP 2: X aur y Split
# ========================================

print("\nSTEP 2: X aur y split...")

feature_cols = ['Return_1min', 'Return_3min', 'Return_5min', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI_14', 'Range', 'VWAP_Diff', 'Volume_SMA_10']

X = df[feature_cols]
y = df['Label']

print(f"X shape: {X.shape} | y shape: {y.shape}")

# ========================================
# STEP 3: TimeSeriesSplit Setup
# ========================================

print(f"\nSTEP 3: TimeSeriesSplit setup, Splits: {N_SPLITS} ...")
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

fold = 1
accuracy_list = []

for train_idx, test_idx in tscv.split(X):
    print(f"\n📈 Fold {fold}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ========================================
    # STEP 4: XGBoost Train with Tuning
    # ========================================

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)

    # ========================================
    # STEP 5A: Predict (Default 0.5)
    # ========================================

    y_pred_default = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_default)
    accuracy_list.append(acc)

    print(f"Accuracy: {acc:.4f}")
    print("✅ Confusion Matrix (Default):")
    print(confusion_matrix(y_test, y_pred_default))
    print(classification_report(y_test, y_pred_default))

    # ========================================
    # STEP 5B: Predict (Custom threshold 0.4)
    # ========================================

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_proba > THRESHOLD).astype(int)

    print(f"✅ Confusion Matrix (Custom Threshold={THRESHOLD}):")
    print(confusion_matrix(y_test, y_pred_custom))
    print(classification_report(y_test, y_pred_custom))

    fold += 1

# ========================================
# STEP 5: Average Accuracy
# ========================================

print("\n✅ Walk-forward test complete!")
print(f"✅ Average Accuracy: {sum(accuracy_list)/len(accuracy_list):.4f}")

print(f"\n📌 Tuning used → max_depth: {MAX_DEPTH} | learning_rate: {LEARNING_RATE} | n_estimators: {N_ESTIMATORS} | threshold: {THRESHOLD}")

# Assume 'model' naam ka tera tuned XGBClassifier hai
joblib.dump(model, "xgb_model_final.pkl")
print("✅ Final model saved: xgb_model_final.pkl")

