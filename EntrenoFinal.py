import numpy as np
import pandas as pd

# Canvi important aquí: afegim RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

## --- FUNCIONS D'UTILITAT --- ##
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
    # Fem còpia per evitar warnings de SettingWithCopy
    Xtr = X_train[vars_kmeans].copy()
    Xva = X_val[vars_kmeans].copy()

    for col in vars_kmeans:
        # Calculem mitjana sobre train per imputar -1
        mean_val = Xtr[Xtr[col] != -1][col].mean()
        Xtr[col] = Xtr[col].replace(-1, mean_val)
        Xva[col] = Xva[col].replace(-1, mean_val)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(Xtr_s)

    # Tornem a copiar els dataframes originals per afegir columnes
    X_train_out = X_train.copy()
    X_val_out = X_val.copy()

    X_train_out['cluster_group'] = kmeans.predict(Xtr_s)
    X_val_out['cluster_group'] = kmeans.predict(Xva_s)

    # Distàncies als centroides
    dtr = kmeans.transform(Xtr_s)
    dva = kmeans.transform(Xva_s)

    for i in range(n_clusters):
        X_train_out[f'dist_cluster_{i}'] = dtr[:, i]
        X_val_out[f'dist_cluster_{i}'] = dva[:, i]

    return X_train_out, X_val_out, kmeans, scaler

# --- 1. CARREGAR DADES ---
try:
    df = pd.read_csv("dataset.csv", sep=",")
    print(f"Dimensions originals: {df.shape}")
except FileNotFoundError:
    print("Error: No s'ha trobat el fitxer 'dataset.csv'.")
    raise

# --- 2. NETEJA I PREPARACIÓ ---
df = df[df['recidiva'].isin([0, 1])]

# Paraules clau de leakage
leakage_keywords = [
    'recid', 'muerte', 'exitus', 'libre', 'fecha', 'dias', 
    'causa', 'estado', 'f_', 'fn', 'inicio','final', 'ini_', 'visita_'
]

cols_to_drop = []
for col in df.columns:
    if any(keyword in col.lower() for keyword in leakage_keywords) and col != 'recidiva':
        cols_to_drop.append(col)

cols_to_drop.extend(['codigo_participante', 'usuario_reg1', 'comentarios', 'ap_comentarios'])
cols_to_drop = list(set(cols_to_drop))
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

print(f"Dimensions després de netejar leakage: {df_clean.shape}")

# --- 3. PREPROCESSAMENT (ENCODING) ---
object_cols = df_clean.select_dtypes(include=['object']).columns
df_clean = pd.get_dummies(df_clean, columns=object_cols, drop_first=True, dtype=int)
df_clean = df_clean.fillna(-1)

print(f"Dimensions després de l'encoding: {df_clean.shape}")

# --- 4. PREPARACIÓ DEL DATASET ---
y = df_clean['recidiva']
X = df_clean.drop(columns=['recidiva'])

# Eliminació manual de variables sospitoses extra
LEAKY_FEATURES = [
    'Tratamiento_sistemico',
    'Tratamiento_RT',
    'Reseccion_macroscopica_complet'
]
X = X.drop(columns=LEAKY_FEATURES, errors='ignore')

# Split inicial (Train / Test Holdout)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=73
)

# --- 5. VALIDACIÓ ROBUSTA: REPEATED STRATIFIED K-FOLD ---

# Configuració: 5 Folds repetits 10 vegades = 50 entrenaments
n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=73)

auc_scores = []
brier_scores = []
best_iterations = []

print(f"\n🚀 Iniciant Repeated Stratified K-Fold ({n_splits*n_repeats} iteracions)...")

for fold, (train_idx, val_idx) in enumerate(rskf.split(X_train_full, y_train_full)):
    
    # Índexs del fold actual
    X_train = X_train_full.iloc[train_idx]
    y_train = y_train_full.iloc[train_idx]
    
    X_val = X_train_full.iloc[val_idx]
    y_val = y_train_full.iloc[val_idx]

    # A. Correlacions (Dins del loop per evitar leakage)
    X_train, dropped_cols = remove_correlated_features(X_train, y_train)
    X_val = X_val.drop(columns=dropped_cols)

    # B. KMeans (Feature Engineering)
    # Assegura't que aquestes variables existeixen al teu dataset
    vars_kmeans = [c for c in ['edad', 'imc', 'valor_de_ca125','tiempo_qx','perdida_hem_cc','tamano_tumoral','n_total_ganCent'] if c in X_train.columns]
    
    if vars_kmeans:
        X_train, X_val, _, _ = add_kmeans_features(X_train, X_val, vars_kmeans)

    # C. Model XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=3,            # Profunditat baixa per estabilitat
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=73 + fold, # Canviem la seed lleugerament a cada fold (opcional però recomanat)
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # D. Mètriques
    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    brier = brier_score_loss(y_val, val_probs)

    auc_scores.append(auc)
    brier_scores.append(brier)
    best_iterations.append(model.best_iteration)

    # Imprimim només cada 5 folds per no saturar la pantalla
    if (fold + 1) % 5 == 0:
        print(f"Iteració {fold+1}/{n_splits*n_repeats} -> AUC: {auc:.3f} | Brier: {brier:.3f}")

print("\n=== RESUM DE VALIDACIÓ ROBUSTA ===")
print(f"✅ AUC mitjana : {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
print(f"✅ Brier mitjà : {np.mean(brier_scores):.3f}")
print(f"🌲 Arbres òptims (mitjana): {int(np.mean(best_iterations))}")

# --- 6. ENTRENAMENT FINAL I TEST ---

# OJO: Hem d'aplicar el preprocessament (KMeans i correlacions) a TOT el conjunt 
# X_train_full i X_test abans d'entrenar el model final.
# Farem servir la lògica de l'última iteració o una nova execució sobre el full train.

print("\n⚙️ Processant el conjunt complet per al model final...")

# 1. Eliminem correlacions sobre tot el train
X_train_final, dropped_final = remove_correlated_features(X_train_full, y_train_full)
X_test_final = X_test.drop(columns=dropped_final)

# 2. Afegim KMeans sobre tot el train i apliquem al test
if vars_kmeans:
    X_train_final, X_test_final, kmeans_final, scaler_final = add_kmeans_features(
        X_train_final, X_test_final, vars_kmeans
    )

# 3. Entrenem el model final
final_n_estimators = int(np.mean(best_iterations))
print(f"Entrenant model final amb {final_n_estimators} arbres...")

final_model = XGBClassifier(
    n_estimators=final_n_estimators,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=(y_train_full == 0).sum() / (y_train_full == 1).sum(),
    random_state=73
)

final_model.fit(X_train_final, y_train_full)

# Predicció al TEST SET (el que mai ha vist)
test_probs = final_model.predict_proba(X_test_final)[:, 1]
auc_test = roc_auc_score(y_test, test_probs)
brier_test = brier_score_loss(y_test, test_probs)

print("\n=== RESULTATS TEST FINAL ===")
print(f"🏆 AUC TEST   : {auc_test:.3f}")
print(f"🎯 Brier TEST : {brier_test:.3f}")

# --- 7. CALIBRATGE I GRÀFIQUES ---

calibrated_model = CalibratedClassifierCV(
    final_model,
    method="isotonic",
    cv=5 
)
calibrated_model.fit(X_train_final, y_train_full)

# Corba de Calibratge
prob_true, prob_pred = calibration_curve(
    y_test,
    calibrated_model.predict_proba(X_test_final)[:, 1],
    n_bins=10
)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="El teu model")
plt.plot([0,1], [0,1], linestyle="--", color="gray", label="Calibratge Perfecte")
plt.xlabel("Probabilitat predita")
plt.ylabel("Freqüència real (Recidiva)")
plt.title("Corba de Calibratge")
plt.legend()
plt.show()

# Cerca del millor llindar
probs_calibrated = calibrated_model.predict_proba(X_test_final)[:, 1]
thresholds = np.linspace(0, 1, 200)

best_threshold = 0.5
found = False

print("\nBuscant llindar per Sensibilitat > 0.90...")
for t in thresholds:
    preds = (probs_calibrated >= t).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()
    sensitivity = tp / (tp + fn + 1e-9)

    if sensitivity < 0.9 and not found:
        # Just quan baixa de 0.9, el t anterior era l'últim bo
        best_threshold = t
        found = True
        print(f"Llindar límit trobat: {t:.3f} (Sensibilitat aprox: {sensitivity:.2f})")
        break
    
    if not found and sensitivity >= 0.9:
         best_threshold = t

print(f"Llindar seleccionat: {best_threshold:.3f}")

# SHAP
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train_final)
shap.summary_plot(shap_values, X_train_final)

# Guardar Model i Objectes necessaris
joblib.dump({
    "model": calibrated_model,
    "features": X_train_final.columns.tolist(),
    "threshold": best_threshold,
    "kmeans": kmeans_final if vars_kmeans else None,
    "scaler": scaler_final if vars_kmeans else None,
    "dropped_cols": dropped_final
}, "xgb_clinic_model_robust.joblib")

print("Model guardat correctament.")