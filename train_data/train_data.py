import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ========================================
# CONFIG
# ========================================
INPUT_FILE = 'output_features_v2.csv'
N_SPLITS = 5

print("\nSTEP 1: Data Load...")
df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Total rows: {df.shape[0]}")

# ========================================
# STEP 2: Features & Target
# ========================================
print("\nSTEP 2: X aur y split...")

feature_cols = [
    'Return_1min', 'Return_3min', 'Return_5min',
    'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI_14',
    'Range', 'VWAP_Diff', 'Volume_SMA_10'  # ✅ Body aur Vol_Spike hata diya
]

X = df[feature_cols]
y = df['Label']

print(f"X shape: {X.shape} | y shape: {y.shape}")

# ========================================
# STEP 3: TimeSeriesSplit
# ========================================
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

fold = 1
accuracy_list = []

for train_idx, test_idx in tscv.split(X):
    print(f"\n📈 Fold {fold}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ========================================
    # STEP 4: XGBoost with tuned hyperparameters
    # ========================================
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        learning_rate=0.05,
        max_depth=4,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)

    # ========================================
    # STEP 5A: Default prediction
    # ========================================
    y_pred_default = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_default)
    accuracy_list.append(acc)

    print(f"Accuracy: {acc:.4f}")
    print("✅ Confusion Matrix (Default):")
    print(confusion_matrix(y_test, y_pred_default))
    print(classification_report(y_test, y_pred_default))

    # ========================================
    # STEP 5B: Threshold tuning
    # ========================================
    y_proba = model.predict_proba(X_test)[:, 1]
    custom_threshold = 0.4
    y_pred_custom = (y_proba > custom_threshold).astype(int)

    print(f"✅ Confusion Matrix (Custom Threshold={custom_threshold}):")
    print(confusion_matrix(y_test, y_pred_custom))
    print(classification_report(y_test, y_pred_custom))

    # ========================================
    # STEP 5C: Feature Importance
    # ========================================
    importances = model.feature_importances_
    print("📊 Feature Importances:")
    for col, score in zip(feature_cols, importances):
        print(f"{col}: {score:.4f}")

    plt.figure(figsize=(8,6))
    plt.barh(feature_cols, importances)
    plt.title(f"Feature Importance Fold {fold}")
    plt.show()

    fold += 1

# ========================================
# STEP 6: Final Accuracy
# ========================================
print("\n✅ Walk-forward test complete!")
print(f"✅ Average Accuracy: {sum(accuracy_list)/len(accuracy_list):.4f}")