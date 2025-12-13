import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

##FUNCIONS D'UTILITAT##
def remove_correlated_features(X, y, threshold=0.95):
    df_tmp = X.copy()
    df_tmp['recidiva'] = y.values

    corr = df_tmp.corr().abs()
    feature_cols = X.columns

    vars_to_drop = set()

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c1, c2 = feature_cols[i], feature_cols[j]
            if corr.loc[c1, c2] > threshold:
                if corr.loc[c1, 'recidiva'] > corr.loc[c2, 'recidiva']:
                    vars_to_drop.add(c1)
                else:
                    vars_to_drop.add(c2)

    return X.drop(columns=vars_to_drop), list(vars_to_drop)

def add_kmeans_features(X_train, X_val, vars_kmeans, n_clusters=2):
    Xtr = X_train[vars_kmeans].copy()
    Xva = X_val[vars_kmeans].copy()

    for col in vars_kmeans:
        mean_val = Xtr[Xtr[col] != -1][col].mean()
        Xtr[col] = Xtr[col].replace(-1, mean_val)
        Xva[col] = Xva[col].replace(-1, mean_val)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(Xtr_s)

    X_train = X_train.copy()
    X_val = X_val.copy()

    X_train['cluster_group'] = kmeans.predict(Xtr_s)
    X_val['cluster_group'] = kmeans.predict(Xva_s)

    # distàncies (molt bona idea la teva)
    dtr = kmeans.transform(Xtr_s)
    dva = kmeans.transform(Xva_s)

    for i in range(n_clusters):
        X_train[f'dist_cluster_{i}'] = dtr[:, i]
        X_val[f'dist_cluster_{i}'] = dva[:, i]

    return X_train, X_val, kmeans, scaler

# --- 1. CARREGAR DADES ---
# Assegura't que el fitxer està a la mateixa carpeta
try:
    df = pd.read_csv("dataset.csv", sep=",")
    print(f"Dimensions originals: {df.shape}")
except FileNotFoundError:
    print("Error: No s'ha trobat el fitxer 'dataset_train.csv'. Revisa el nom o la ruta.")
    # Aturem l'execució si no hi ha fitxer
    raise

# --- 2. NETEJA I PREPARACIÓ (DATA CLEANING) ---

# Filtrar només els casos vàlids per al target 'recidiva'
df = df[df['recidiva'].isin([0, 1])]

# LLISTA DE VARIABLES FUTURES
leakage_keywords = [
    'recid', 'muerte', 'exitus', 'libre', 'fecha', 'dias', 'causa', 'estado', 'f_', 'fn', 'inicio','final', 'ini_', 'visita_'
]

cols_to_drop = []
for col in df.columns:
    # Si conté una paraula clau i NO és el target 'recidiva', l'eliminem
    if any(keyword in col.lower() for keyword in leakage_keywords) and col != 'recidiva':
        cols_to_drop.append(col)

# Afegim identificadors i camps de text lliure
cols_to_drop.extend(['codigo_participante', 'usuario_reg1', 'comentarios', 'ap_comentarios'])

# Eliminem duplicats en la llista i apliquem la neteja
cols_to_drop = list(set(cols_to_drop))
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

print(f"Dimensions després de netejar leakage: {df_clean.shape}")

# --- 3. PREPROCESSAMENT (ENCODING) ---

# 1. Identifiquem les columnes de text
object_cols = df_clean.select_dtypes(include=['object']).columns

# 2. Apliquem One-Hot Encoding (substitueix el bucle 'for')
# Això crea noves columnes binàries i elimina les originals de text
df_clean = pd.get_dummies(df_clean, 
                          columns=object_cols, 
                          drop_first=True, 
                          dtype=int)

# 3. Neteja final de nuls (per a la resta de variables numèriques)
df_clean = df_clean.fillna(-1)
print(f"Dimensions després de l'encoding: {df_clean.shape}")

# --- 4. ENTRENAMENT AMB 5-FOLD CROSS-VALIDATION ---

# y = target (el que vols predir)
y = df_clean['recidiva']

# X = totes les altres variables
X = df_clean.drop(columns=['recidiva'])

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=73
)
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=73
)

auc_scores = []
brier_scores = []
best_iterations = []
   

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    print(f"\n=== Fold {fold+1} ===")

    X_train = X_train_full.iloc[train_idx]
    y_train = y_train_full.iloc[train_idx]

    X_val = X_train_full.iloc[val_idx]
    y_val = y_train_full.iloc[val_idx]

    # 1. correlacions
    X_train, dropped = remove_correlated_features(X_train, y_train)
    X_val = X_val.drop(columns=dropped)

    vars_kmeans = ['edad', 'imc', 'valor_de_ca125','tiempo_qx','perdida_hem_cc','tamano_tumoral','n_total_ganCent']
    # 2. KMeans
    X_train, X_val, kmeans, scaler = add_kmeans_features(
        X_train, X_val, vars_kmeans
    )

    # 3. MODEL (XGBoost)
    #compta quants casos negatius (0) hi ha al train, compta quants casos positius (1) hi ha al train i calcula el ratio negatius / positiu
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=73,
    early_stopping_rounds=100   # 👈 AQUÍ
)

    model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
    )


    val_probs = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, val_probs)
    brier = brier_score_loss(y_val, val_probs)

    auc_scores.append(auc)
    brier_scores.append(brier)
    best_iterations.append(model.best_iteration)

    print(f"AUC: {auc:.3f} | Brier: {brier:.3f} | Trees: {model.best_iteration}")

print("\n=== CROSS-VALIDATION SUMMARY ===")
print(f"AUC mitjana  : {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
print(f"Brier mitjà : {np.mean(brier_scores):.3f}")
print(f"Iteracions mitjanes: {int(np.mean(best_iterations))}")


final_n_estimators = int(np.mean(best_iterations))

final_model = XGBClassifier(
    n_estimators=final_n_estimators,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=(y_train_full == 0).sum() / (y_train_full == 1).sum(),
    random_state=73
)

final_model.fit(X_train_full, y_train_full)

test_probs = final_model.predict_proba(X_test)[:, 1]

auc_test = roc_auc_score(y_test, test_probs)
brier_test = brier_score_loss(y_test, test_probs)

print("\n=== TEST FINAL ===")
print(f"AUC TEST   : {auc_test:.3f}")
print(f"Brier TEST : {brier_test:.3f}")


calibrated_model = CalibratedClassifierCV(
    final_model,
    method="isotonic",
    cv=5
)

calibrated_model.fit(X_train_full, y_train_full)


prob_true, prob_pred = calibration_curve(
    y_test,
    calibrated_model.predict_proba(X_test)[:, 1],
    n_bins=10
)

plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("Probabilitat predita")
plt.ylabel("Freqüència real")
plt.title("Calibration Curve")
plt.show()


probs = calibrated_model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0, 1, 200)

for t in thresholds:
    preds = (probs >= t).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()
    sensitivity = tp / (tp + fn + 1e-9)

    if sensitivity >= 0.9:
        print("Llindar recomanat:", round(t, 3))
        break

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train_full)

shap.summary_plot(shap_values, X_train_full)

joblib.dump({
    "model": calibrated_model,
    "features": X.columns.tolist(),
    "threshold_clinic": t
}, "xgb_clinic_model.joblib")

