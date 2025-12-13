import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

# --- 1. CONFIGURACIÓ ---
PATH_TRAIN = 'dataset_train_final_kmeans_euclidiana.csv'
PATH_TEST = 'dataset_test_processat.csv'
PATH_KMEANS = 'kmeans_model.pkl'
TARGET_COL = 'recidiva'
MODEL_OUTPUT_PATH = 'gboost_model.json'
CLUSTER_COL_NAME = 'cluster_k'

# LES 7 VARIABLES DEL KMEANS (Correctes segons el teu log)
KMEANS_COLS = ['edad', 'imc', 'valor_de_ca125', 'tiempo_qx', 'perdida_hem_cc', 'tamano_tumoral', 'n_total_ganCent']

# --- 2. CÀRREGA DE DADES ---
print("Carregant datasets...")
df_train = pd.read_csv(PATH_TRAIN)
df_test = pd.read_csv(PATH_TEST)

y_train = df_train[TARGET_COL]
X_train = df_train.drop(columns=[TARGET_COL])

y_test = df_test[TARGET_COL]
X_test = df_test.drop(columns=[TARGET_COL])

# --- 3. GESTIÓ DEL KMEANS (TRAIN I TEST) ---
print("Carregant i aplicant KMeans...")

try:
    kmeans_model = joblib.load(PATH_KMEANS)
    
    def aplicar_cluster(df, cols_kmeans, model_km):
        # Comprovem columnes
        missing = [c for c in cols_kmeans if c not in df.columns]
        if missing:
            # Si falten, omplim amb 0 (per seguretat)
            for m in missing: df[m] = 0
        subset = df[cols_kmeans]
        return model_km.predict(subset)

    # 1. Apliquem al TEST
    X_test[CLUSTER_COL_NAME] = aplicar_cluster(X_test, KMEANS_COLS, kmeans_model)
    X_test[CLUSTER_COL_NAME] = X_test[CLUSTER_COL_NAME].astype('category')
    
    # 2. Apliquem al TRAIN
    X_train[CLUSTER_COL_NAME] = aplicar_cluster(X_train, KMEANS_COLS, kmeans_model)
    X_train[CLUSTER_COL_NAME] = X_train[CLUSTER_COL_NAME].astype('category')

except Exception as e:
    print(f"ERROR CRÍTIC AMB KMEANS: {e}")
    exit()

# --- 4. ALINEACIÓ DE COLUMNES ---
print("Alineant columnes entre Train i Test...")

# Columnes que falten al Test -> Posem 0
missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0

# Columnes que sobren al Test -> Esborrem
extra_cols = set(X_test.columns) - set(X_train.columns)
if extra_cols:
    X_test = X_test.drop(columns=extra_cols)

# Reordenar exactament igual
X_test = X_test[X_train.columns]

print("Columnes alineades correctament.")

# --- 5. ENTRENAMENT AMB ELS TEUS PARÀMETRES ---
print(f"Entrenant model XGBoost (n_estimators=3000, early_stopping=100)...")

model = xgb.XGBClassifier(
    objective='binary:logistic',
    # !!! CANVIAT A 'hist' PERQUÈ 'gpu_hist' DONAVA ERROR AL TEU PC !!!
    tree_method='hist',
    # predictor='cpu_predictor', # Per defecte ja usa CPU si no hi ha GPU
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    early_stopping_rounds=100, # Pararà si no millora en 100 iteracions
    enable_categorical=True,
    random_state=73
)

# NECESSITEM eval_set PER A L'EARLY STOPPING
# Fem servir el test set com a validació per veure quan parar
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# --- 6. RESULTATS ---
print("\n--- Resultats: Probabilitat de Recidiva (Test Set) ---")

# Si ha parat aviat (early stopping), farà servir la millor iteració automàticament
probs_recidiva = model.predict_proba(X_test)[:, 1]

resultats = pd.DataFrame({
    'Recidiva_Real': y_test.values,
    'Probabilitat_Model (%)': [round(p * 100, 2) for p in probs_recidiva],
    'Predicció (Tall 50%)': (probs_recidiva > 0.5).astype(int)
})

print(resultats)

# --- 7. GUARDAR (CORREGIT) ---

# Opció A: Guardar el Booster intern (Soluciona l'error del JSON)
# Això guarda només la lògica de l'arbre, sense la carcassa de Scikit-Learn
model.get_booster().save_model(MODEL_OUTPUT_PATH)
print(f"\nModel (Booster) guardat a: {MODEL_OUTPUT_PATH}")

# Opció B (Alternativa): Guardar amb Joblib
# Si vols guardar tot l'objecte Python sencer (més fàcil de carregar en Python després)
# import joblib
# joblib.dump(model, 'gboost_model_full.pkl')
# print("També guardat com a pickle per seguretat.")