import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from xgboost import XGBClassifier

# --- FUNCIONS D'UTILITAT ---

def remove_correlated_features(X, y, threshold=0.95):
    """
    Elimina característiques amb una correlació superior al llindar especificat.
    Entre dues variables correlacionades, conserva la que té major correlació amb l'objectiu (y).
    """
    df_tmp = X.copy()
    df_tmp['recidiva'] = y.values

    corr = df_tmp.corr().abs()
    feature_cols = X.columns
    vars_to_drop = set()

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c1, c2 = feature_cols[i], feature_cols[j]
            # Si la correlació supera el llindar
            if corr.loc[c1, c2] > threshold:
                # Elimina la variable que tingui menys correlació amb el target
                if corr.loc[c1, 'recidiva'] > corr.loc[c2, 'recidiva']:
                    vars_to_drop.add(c2)
                else:
                    vars_to_drop.add(c1)

    return X.drop(columns=vars_to_drop), list(vars_to_drop)

def add_kmeans_features(X_train, X_val, vars_kmeans, n_clusters=2):
    """
    Genera noves característiques basades en clústers (KMeans) sobre un subconjunt de variables.
    Calcula el clúster assignat i les distàncies als centroides.
    """
    # Còpies per evitar SettingWithCopyWarning
    Xtr = X_train[vars_kmeans].copy()
    Xva = X_val[vars_kmeans].copy()

    # Imputació simple de valors -1 amb la mitjana del train (excloent els -1)
    for col in vars_kmeans:
        mean_val = Xtr[Xtr[col] != -1][col].mean()
        Xtr[col] = Xtr[col].replace(-1, mean_val)
        Xva[col] = Xva[col].replace(-1, mean_val)

    # Escaladejat de les dades per al càlcul de distàncies
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    # Entrenament del KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(Xtr_s)

    X_train_out = X_train.copy()
    X_val_out = X_val.copy()

    # Assignació de clústers
    X_train_out['cluster_group'] = kmeans.predict(Xtr_s)
    X_val_out['cluster_group'] = kmeans.predict(Xva_s)

    # Càlcul de distàncies als centroides
    dtr = kmeans.transform(Xtr_s)
    dva = kmeans.transform(Xva_s)

    for i in range(n_clusters):
        X_train_out[f'dist_cluster_{i}'] = dtr[:, i]
        X_val_out[f'dist_cluster_{i}'] = dva[:, i]

    return X_train_out, X_val_out, kmeans, scaler

# --- 1. CÀRREGA DE DADES ---
try:
    df = pd.read_csv("dataset.csv", sep=",")
    print(f"Dimensions originals: {df.shape}")
except FileNotFoundError:
    print("Error: No s'ha trobat el fitxer 'dataset.csv'.")
    raise

# --- 2. NETEJA I GESTIÓ DE LEAKAGE ---
df = df[df['recidiva'].isin([0, 1])]

# Llista de termes sospitosos de fuita de dades (data leakage)
leakage_keywords = [
    'recid', 'muerte', 'exitus', 'libre', 'fecha', 'dias', 
    'causa', 'estado', 'f_', 'fn', 'inicio','final', 'ini_', 'visita_'
]

cols_to_drop = []
for col in df.columns:
    if any(keyword in col.lower() for keyword in leakage_keywords) and col != 'recidiva':
        cols_to_drop.append(col)

# Eliminació de columnes administratives o de text lliure
cols_to_drop.extend(['codigo_participante', 'usuario_reg1', 'comentarios', 'ap_comentarios'])
cols_to_drop = list(set(cols_to_drop))
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

print(f"Dimensions després de netejar leakage: {df_clean.shape}")

# --- 3. PREPROCESSAMENT (ENCODING) ---
# One-Hot Encoding per a variables categòriques i imputació bàsica de nuls
object_cols = df_clean.select_dtypes(include=['object']).columns
df_clean = pd.get_dummies(df_clean, columns=object_cols, drop_first=True, dtype=int)
df_clean = df_clean.fillna(-1)

print(f"Dimensions després de l'encoding: {df_clean.shape}")

# --- 4. PREPARACIÓ DEL DATASET ---
y = df_clean['recidiva']
X = df_clean.drop(columns=['recidiva'])

# Eliminació manual de variables específiques
features_to_exclude = [
    'Tratamiento_sistemico',
    'Tratamiento_RT',
    'Reseccion_macroscopica_complet'
]
X = X.drop(columns=features_to_exclude, errors='ignore')

# Divisió inicial (Train/Test Holdout) estratificada
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=73
)

# --- 5. VALIDACIÓ CREUADA ROBUSTA (RSKF) ---
n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=73)

auc_scores = []
brier_scores = []
best_iterations = []

print(f"\nIniciant Repeated Stratified K-Fold ({n_splits*n_repeats} iteracions)...")

for fold, (train_idx, val_idx) in enumerate(rskf.split(X_train_full, y_train_full)):
    
    X_train = X_train_full.iloc[train_idx]
    y_train = y_train_full.iloc[train_idx]
    
    X_val = X_train_full.iloc[val_idx]
    y_val = y_train_full.iloc[val_idx]

    # A. Selecció de característiques (dins del loop per evitar leakage)
    X_train, dropped_cols = remove_correlated_features(X_train, y_train)
    X_val = X_val.drop(columns=dropped_cols)

    # B. Feature Engineering (KMeans)
    vars_kmeans = [c for c in ['edad', 'imc', 'valor_de_ca125','tiempo_qx','perdida_hem_cc','tamano_tumoral','n_total_ganCent'] if c in X_train.columns]
    
    if vars_kmeans:
        X_train, X_val, _, _ = add_kmeans_features(X_train, X_val, vars_kmeans)

    # C. Entrenament XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=3,           # Profunditat conservadora per evitar overfitting
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=73 + fold, 
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # D. Avaluació
    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    brier = brier_score_loss(y_val, val_probs)

    auc_scores.append(auc)
    brier_scores.append(brier)
    best_iterations.append(model.best_iteration)

    if (fold + 1) % 5 == 0:
        print(f"Iteració {fold+1}/{n_splits*n_repeats} -> AUC: {auc:.3f} | Brier: {brier:.3f}")

print("\n=== RESULTATS VALIDACIÓ CREUADA ===")
print(f"AUC mitjana : {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
print(f"Brier mitjà : {np.mean(brier_scores):.3f}")
print(f"Arbres òptims (mitjana): {int(np.mean(best_iterations))}")

# --- 6. ENTRENAMENT FINAL I TEST ---
print("\nProcessant el conjunt complet d'entrenament i generant model final...")

# 1. Aplicació del preprocessament a tot el conjunt d'entrenament i test
X_train_final, dropped_final = remove_correlated_features(X_train_full, y_train_full)
X_test_final = X_test.drop(columns=dropped_final)

if vars_kmeans:
    X_train_final, X_test_final, kmeans_final, scaler_final = add_kmeans_features(
        X_train_final, X_test_final, vars_kmeans
    )

# 2. Entrenament del model amb els millors hiperparàmetres trobats
final_n_estimators = int(np.mean(best_iterations))
print(f"Entrenant amb {final_n_estimators} estimadors...")

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

# Predicció al conjunt de test (Holdout)
test_probs = final_model.predict_proba(X_test_final)[:, 1]
auc_test = roc_auc_score(y_test, test_probs)
brier_test = brier_score_loss(y_test, test_probs)

print("\n=== RESULTATS TEST FINAL ===")
print(f"AUC TEST   : {auc_test:.3f}")
print(f"Brier TEST : {brier_test:.3f}")

# --- 7. CALIBRATGE I EXPLICABILITAT ---

# Calibratge isotònic sobre les prediccions del model final
calibrated_model = CalibratedClassifierCV(
    final_model,
    method="isotonic",
    cv=5 
)
calibrated_model.fit(X_train_final, y_train_full)

# Gràfic de la corba de calibratge
prob_true, prob_pred = calibration_curve(
    y_test,
    calibrated_model.predict_proba(X_test_final)[:, 1],
    n_bins=10
)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0,1], [0,1], linestyle="--", color="gray", label="Calibratge Perfecte")
plt.xlabel("Probabilitat predita")
plt.ylabel("Freqüència real (Recidiva)")
plt.title("Corba de Calibratge")
plt.legend()
plt.show()

# Optimització del llindar de decisió (Sensibilitat mínima > 0.90)
probs_calibrated = calibrated_model.predict_proba(X_test_final)[:, 1]
thresholds = np.linspace(0, 1, 200)

best_threshold = 0.5
found = False

print("\nCercant llindar per a Sensibilitat > 0.90...")
for t in thresholds:
    preds = (probs_calibrated >= t).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()
    sensitivity = tp / (tp + fn + 1e-9)

    # Si la sensibilitat cau per sota de 0.9, el llindar anterior era el límit
    if sensitivity < 0.9 and not found:
        best_threshold = t
        found = True
        print(f"Llindar límit trobat: {t:.3f} (Sensibilitat aprox: {sensitivity:.2f})")
        break
    
    if not found and sensitivity >= 0.9:
         best_threshold = t

print(f"Llindar final seleccionat: {best_threshold:.3f}")

# Anàlisi SHAP (Explicabilitat)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train_final)
shap.summary_plot(shap_values, X_train_final)

# Guardat del model i objectes de transformació
vars_kmeans_list = vars_kmeans if vars_kmeans else []

joblib.dump({
    "model": calibrated_model,
    "features": X_train_final.columns.tolist(),
    "threshold": best_threshold,
    "kmeans": kmeans_final if vars_kmeans else None,
    "scaler": scaler_final if vars_kmeans else None,
    "dropped_cols": dropped_final,
    "vars_kmeans": vars_kmeans_list
}, "xgb_clinic_model_robust.joblib")

print("Model i objectes auxiliars guardats correctament.")