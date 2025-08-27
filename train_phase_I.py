import numpy as np
import pandas as pd
import joblib, os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- Config ----------------
SAVE_DIR = r"C:\Users\Faheem\Desktop\Project\Code\PPOP_Phase_II\models"
os.makedirs(SAVE_DIR, exist_ok=True)

target_col = "factor_level_IU_dL"
cat_features = ["severity"]
num_features = ["age", "weight_kg", "infusion_dose_IU",
                "time_since_last_infusion_hr", "tsi_sq",
                "log_dose", "dose_per_kg"]

seq_features = ["infusion_dose_IU", "time_since_last_infusion_hr", target_col]

# ---------------- Load Data ----------------
df = pd.read_csv("C:\\Users\\Faheem\\Desktop\\Project\\Dataset\\synthetic_hemophilia_data.csv")  
df = df.dropna(subset=[target_col])

# ---------------- Feature Engineering ----------------
df["tsi_sq"] = df["time_since_last_infusion_hr"] ** 2
df["log_dose"] = np.log1p(df["infusion_dose_IU"])
df["dose_per_kg"] = df["infusion_dose_IU"] / df["weight_kg"]

# ---------------- Split ----------------
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df[num_features + cat_features]
y_train = train_df[target_col]
X_val = val_df[num_features + cat_features]
y_val = val_df[target_col]

# ---------------- Preprocessor for XGB ----------------
preproc_xgb = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

X_train_proc = preproc_xgb.fit_transform(X_train)
X_val_proc = preproc_xgb.transform(X_val)

# ---------------- Train XGBoost ----------------
dtrain = xgb.DMatrix(X_train_proc, label=y_train)
dval = xgb.DMatrix(X_val_proc, label=y_val)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=20
)

joblib.dump(preproc_xgb, os.path.join(SAVE_DIR, "preproc_xgb.pkl"))
booster.save_model(os.path.join(SAVE_DIR, "xgb_booster.json"))

# ---------------- Train RF (Bleed Risk) ----------------
train_df["bleed_risk"] = (train_df[target_col] < 50).astype(int)
val_df["bleed_risk"] = (val_df[target_col] < 50).astype(int)

X_train_rf = train_df[num_features + cat_features + [target_col]]
y_train_rf = train_df["bleed_risk"]
X_val_rf = val_df[num_features + cat_features + [target_col]]
y_val_rf = val_df["bleed_risk"]

preproc_rf = ColumnTransformer([
    ("num", StandardScaler(), num_features + [target_col]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

X_train_rf_proc = preproc_rf.fit_transform(X_train_rf)
X_val_rf_proc = preproc_rf.transform(X_val_rf)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train_rf_proc, y_train_rf)

# Evaluate RF
proba_val = rf_model.predict_proba(X_val_rf_proc)[:, 1]
auc = roc_auc_score(y_val_rf, proba_val)
print(f"Validation AUC (RF bleed risk): {auc:.3f}")

# Pick best threshold
best_thr = 0.5
best_score = 0
for thr in np.linspace(0.1, 0.9, 9):
    preds = (proba_val >= thr).astype(int)
    score = (preds == y_val_rf).mean()
    if score > best_score:
        best_score = score
        best_thr = thr
print(f"Best threshold: {best_thr:.2f}, Accuracy: {best_score:.3f}")

joblib.dump(preproc_rf, os.path.join(SAVE_DIR, "preproc_rf.pkl"))
joblib.dump(rf_model, os.path.join(SAVE_DIR, "rf_model.pkl"))
with open(os.path.join(SAVE_DIR, "rf_threshold.txt"), "w") as f:
    f.write(str(best_thr))

# ---------------- Train LSTM (Phase II) ----------------
scaler = MinMaxScaler()
seq_data = df[seq_features].copy()
seq_scaled = scaler.fit_transform(seq_data)

def create_sequences(data, window=5):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, :])
        y.append(data[i+window, -1])  # predict factor level
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(seq_scaled, window=5)
X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

lstm_model = Sequential([
    LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dense(32, activation="relu"),
    Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mse")

es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=30, batch_size=32,
    callbacks=[es],
    verbose=1
)

lstm_model.save(os.path.join(SAVE_DIR, "lstm_model.keras"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "lstm_scaler.pkl"))

print(f"✅ Full training complete. All models saved to {SAVE_DIR}")