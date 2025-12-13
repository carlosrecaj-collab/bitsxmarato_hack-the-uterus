import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold  # <--- NUEVO IMPORT
from xgboost import XGBClassifier

# =============================================================================
# 1. CLASES Y FUNCIONES DE INGENIERÍA DE DATOS (EL "PIPELINE")
# =============================================================================

class KMeansFeaturizer:
    """
    Clase personalizada para manejar la creación de features de Clustering
    de forma segura dentro de un pipeline.
    """
    def __init__(self, vars_kmeans, n_clusters=2):
        self.vars_kmeans = vars_kmeans
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.imputer = SimpleImputer(strategy='mean') 

    def fit(self, X):
        if not self.vars_kmeans:
            return self
            
        # Filtramos solo las columnas necesarias que existan en X
        available_vars = [v for v in self.vars_kmeans if v in X.columns]
        if not available_vars:
            return self
            
        X_sub = X[available_vars].copy()
        
        # 1. Imputar
        X_sub = self.imputer.fit_transform(X_sub)
        # 2. Escalar
        X_scaled = self.scaler.fit_transform(X_sub)
        # 3. Entrenar K-means
        self.kmeans.fit(X_scaled)
        return self

    def transform(self, X):
        X_new = X.copy()
        available_vars = [v for v in self.vars_kmeans if v in X.columns]
        
        if not available_vars:
            # Si no hay variables, devolvemos X sin cambios (evita errores)
            return X_new

        X_sub = X_new[available_vars].copy()
        
        # Aplicar transformaciones guardadas
        X_sub = self.imputer.transform(X_sub)
        X_scaled = self.scaler.transform(X_sub)
        
        # Generar Features
        try:
            cluster_labels = self.kmeans.predict(X_scaled)
            X_new['cluster_group'] = cluster_labels
            
            dists = self.kmeans.transform(X_scaled)
            for i in range(self.n_clusters):
                X_new[f'dist_cluster_{i}'] = dists[:, i]
        except:
            pass # Si falla por dimension mismatch (raro), devolvemos sin clusters
            
        return X_new

def get_correlated_features_to_drop(X, y, threshold=0.95):
    """
    Versión optimizada usando álgebra matricial.
    """
    # Calculamos matriz de correlación absoluta
    corr_matrix = X.corr().abs()
    
    # Seleccionamos triángulo superior
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    # Pre-calculamos correlación con target para desempatar
    target_corr = X.corrwith(y).abs()
    
    for column in upper.columns:
        if any(upper[column] > threshold):
            correlated_feats = upper.index[upper[column] > threshold].tolist()
            for feat in correlated_feats:
                if target_corr.get(column, 0) > target_corr.get(feat, 0):
                    to_drop.add(feat)
                else:
                    to_drop.add(column)
    return list(to_drop)

def _apply_ohe(df, ohe, cat_cols, num_cols):
    """Auxiliar para aplicar OHE y devolver DataFrame con nombres correctos"""
    if not cat_cols:
        return df[num_cols]
        
    ohe_arr = ohe.transform(df[cat_cols])
    feat_names = ohe.get_feature_names_out(cat_cols)
    df_ohe = pd.DataFrame(ohe_arr, columns=feat_names, index=df.index)
    return pd.concat([df[num_cols], df_ohe], axis=1)

def fit_preprocessing(X_tr, y_tr):
    """
    Entrena los transformadores sobre datos de entrenamiento.
    """
    transformers = {}
    
    # 1. Identificar columnas
    cat_cols = X_tr.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_tr.select_dtypes(exclude=['object']).columns.tolist()
    
    # 2. One Hot Encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
    ohe.fit(X_tr[cat_cols])
    
    transformers['ohe'] = ohe
    transformers['cat_cols'] = cat_cols
    transformers['num_cols'] = num_cols
    
    X_tr_proc = _apply_ohe(X_tr, ohe, cat_cols, num_cols)
    
    # 3. K-Means Features
    vars_posibles = ['edad', 'imc', 'valor_de_ca125','tiempo_qx','perdida_hem_cc','tamano_tumoral','n_total_ganCent']
    vars_kmeans = [c for c in vars_posibles if c in X_tr_proc.columns]
    
    kmeans_featurizer = KMeansFeaturizer(vars_kmeans, n_clusters=2)
    kmeans_featurizer.fit(X_tr_proc)
    transformers['kmeans'] = kmeans_featurizer
    
    X_tr_proc = kmeans_featurizer.transform(X_tr_proc)
    
    # 4. Imputación final (-1 para XGBoost)
    X_tr_proc = X_tr_proc.fillna(-1)

    # 5. --- ELIMINAR COLUMNAS CONSTANTES (FIX ERROR DIVISION 0) ---
    var_th = VarianceThreshold(threshold=0)
    var_th.fit(X_tr_proc)
    transformers['var_th'] = var_th
    
    # Nos quedamos solo con las columnas que varían
    cols_kept = X_tr_proc.columns[var_th.get_support()]
    X_tr_proc = X_tr_proc[cols_kept]

    # 6. Eliminación de Correlacionadas (Ahora es seguro sin warnings)
    cols_to_drop = get_correlated_features_to_drop(X_tr_proc, y_tr, threshold=0.95)
    transformers['cols_to_drop'] = cols_to_drop
    
    X_tr_final = X_tr_proc.drop(columns=cols_to_drop)
    
    return X_tr_final, transformers

def transform_data(X_new, transformers):
    """
    Aplica los transformadores YA ENTRENADOS a nuevos datos.
    """
    # 1. OHE
    X_proc = _apply_ohe(X_new, transformers['ohe'], transformers['cat_cols'], transformers['num_cols'])
    
    # 2. KMeans
    X_proc = transformers['kmeans'].transform(X_proc)
    
    # 3. Nulos
    X_proc = X_proc.fillna(-1)
    
    # 4. --- Variance Threshold (Aplicar filtro aprendido) ---
    # Usamos las columnas que sobrevivieron en el fit
    cols_kept_by_var = X_proc.columns[transformers['var_th'].get_support()]
    X_proc = X_proc[cols_kept_by_var]
    
    # 5. Drop Correlated
    # Solo eliminamos las columnas que existen
    cols_to_drop = [c for c in transformers['cols_to_drop'] if c in X_proc.columns]
    X_final = X_proc.drop(columns=cols_to_drop)
    
    return X_final

# =============================================================================
# 2. CARGA Y LIMPIEZA INICIAL
# =============================================================================

try:
    df = pd.read_csv("dataset.csv", sep=",")
    print(f"Dimensions originals: {df.shape}")
except FileNotFoundError:
    # Generamos un dataset dummy si no existe, solo para que el código corra si lo copias
    print("AVISO: No s'ha trobat el fitxer. Generant dades dummy per prova...")
    df = pd.DataFrame(np.random.rand(200, 20), columns=[f'var_{i}' for i in range(20)])
    df['edad'] = np.random.randint(20, 80, 200)
    df['recidiva'] = np.random.randint(0, 2, 200)
    df['genero'] = np.random.choice(['M', 'F'], 200)

# Filtrar target
df = df[df['recidiva'].isin([0, 1])]

# Variables de Leakage
leakage_keywords = [
    'recid', 'muerte', 'exitus', 'libre', 'fecha', 'dias', 'causa', 'estado', 'f_', 'fn', 'inicio','final', 'ini_', 'visita_'
]
cols_to_drop_manual = ['codigo_participante', 'usuario_reg1', 'comentarios', 'ap_comentarios']

for col in df.columns:
    if any(keyword in col.lower() for keyword in leakage_keywords) and col != 'recidiva':
        cols_to_drop_manual.append(col)

# Variables específicas
cols_to_drop_manual.extend(['Tratamiento_sistemico', 'Tratamiento_RT', 'Reseccion_macroscopica_complet'])

# Aseguramos que solo borramos las que existen
cols_to_drop_manual = [c for c in cols_to_drop_manual if c in df.columns]
df_clean = df.drop(columns=cols_to_drop_manual)
print(f"Dimensions després de neteja manual: {df_clean.shape}")

# Separar X e y
y = df_clean['recidiva']
X = df_clean.drop(columns=['recidiva'])

# =============================================================================
# 3. SPLIT TRAIN / TEST
# =============================================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=73
)

# =============================================================================
# 4. CROSS-VALIDATION
# =============================================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=73)
auc_scores = []
brier_scores = []
best_iterations = []

print("\n=== INICIANDO CROSS-VALIDATION ===")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    X_tr_fold = X_train_full.iloc[train_idx]
    y_tr_fold = y_train_full.iloc[train_idx]
    X_val_fold = X_train_full.iloc[val_idx]
    y_val_fold = y_train_full.iloc[val_idx]
    
    # --- PIPELINE ---
    X_tr_ready, transformers_fold = fit_preprocessing(X_tr_fold, y_tr_fold)
    X_val_ready = transform_data(X_val_fold, transformers_fold)
    
    # --- MODELO ---
    # Control de división por cero en scale_pos_weight
    n_pos = (y_tr_fold == 1).sum()
    n_neg = (y_tr_fold == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=73,
        early_stopping_rounds=100
    )

    model.fit(
        X_tr_ready, y_tr_fold,
        eval_set=[(X_val_ready, y_val_fold)],
        verbose=False
    )
    
    val_probs = model.predict_proba(X_val_ready)[:, 1]
    auc = roc_auc_score(y_val_fold, val_probs)
    brier = brier_score_loss(y_val_fold, val_probs)
    
    auc_scores.append(auc)
    brier_scores.append(brier)
    best_iterations.append(model.best_iteration)
    
    print(f"Fold {fold+1}: AUC={auc:.3f} | Brier={brier:.3f} | Cols={X_tr_ready.shape[1]}")

print("\n=== CV RESUMEN ===")
print(f"AUC Promedio:   {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
print(f"Brier Promedio: {np.mean(brier_scores):.3f}")

# =============================================================================
# 5. ENTRENAMIENTO FINAL (TRAIN FULL + TEST)
# =============================================================================

print("\n=== ENTRENANDO MODELO FINAL ===")

X_train_final_ready, final_transformers = fit_preprocessing(X_train_full, y_train_full)
X_test_ready = transform_data(X_test, final_transformers)

final_n_estimators = int(np.mean(best_iterations))

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

final_model.fit(X_train_final_ready, y_train_full)

test_probs = final_model.predict_proba(X_test_ready)[:, 1]
print(f"AUC TEST FINAL   : {roc_auc_score(y_test, test_probs):.3f}")
print(f"Brier TEST FINAL : {brier_score_loss(y_test, test_probs):.3f}")

# =============================================================================
# 6. CALIBRACIÓN Y SHAP
# =============================================================================

calibrated_model = CalibratedClassifierCV(
    final_model,
    method="isotonic",
    cv=5
)
calibrated_model.fit(X_train_final_ready, y_train_full)

probs_calibrated = calibrated_model.predict_proba(X_test_ready)[:, 1]

# Umbral
thresholds = np.linspace(0, 1, 200)
best_threshold = 0.5
for t in thresholds:
    preds = (probs_calibrated >= t).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()
    sensitivity = tp / (tp + fn + 1e-9)
    if sensitivity < 0.9:
        break
    best_threshold = t

print(f"Llindar recomanat (Sens >= 0.9): {best_threshold:.3f}")

# SHAP
try:
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_train_final_ready)
    shap.summary_plot(shap_values, X_train_final_ready, show=False)
    plt.title("SHAP Summary Plot")
    plt.show()
except Exception as e:
    print(f"Error generando SHAP (probablemente dataset muy pequeño o version mismatch): {e}")

# =============================================================================
# 7. GUARDAR
# =============================================================================

artifact = {
    "model_xgb": final_model,
    "model_calibrated": calibrated_model,
    "transformers": final_transformers,
    "features_final": X_train_final_ready.columns.tolist(),
    "threshold_clinic": best_threshold
}

joblib.dump(artifact, "xgb_clinic_production_model.joblib")
print("\nModelo guardado. Listo.")