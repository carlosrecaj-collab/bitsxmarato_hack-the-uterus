import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# --- 1. CARREGAR DADES ---
# Assegura't que el fitxer està a la mateixa carpeta
try:
    df = pd.read_csv("dataset_train.csv", sep=",")
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

# --- 4. MATRIU DE CORRELACIÓ ---

corr_matrix = df_clean.corr().abs() # Utilitzem valor absolut per simplificar comparacions

# --- 5. ELIMINACIÓ DE MULTICOLINEALITAT (> 0.95) ---
# REQUISIT: Eliminar la variable de la parella que tingui MÉS correlació amb 'recidiva'

vars_to_drop = set()
feature_cols = df_clean.columns.drop('recidiva')

# Iterem per totes les parelles de variables
for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
        
        col_1 = feature_cols[i]
        col_2 = feature_cols[j]
        
        # Correlació entre les dues variables predictores
        corr_between = corr_matrix.loc[col_1, col_2]
        
        if corr_between > 0.95:
            # Correlació de cadascuna amb el target (recidiva)
            corr_1_target = corr_matrix.loc[col_1, 'recidiva']
            corr_2_target = corr_matrix.loc[col_2, 'recidiva']
            
            # Lògica: Eliminar la que té correlació més ALTA amb recidiva
            if corr_1_target > corr_2_target:
                vars_to_drop.add(col_1)
            else:
                vars_to_drop.add(col_2)

# Crear el DataFrame final
df_final = df_clean.drop(columns=vars_to_drop)
# Definim el nom del nou fitxer
nom_fitxer_sortida = 'dataset_train_processat.csv'

# Exportem a CSV
# index=False: Evita que es guardi la columna d'índexs numèrics
# sep=',': Assegura que el separador sigui la coma (estàndard)
df_final.to_csv(nom_fitxer_sortida, index=False, sep=',')

print(f"Fitxer exportat correctament: {nom_fitxer_sortida}")

# --- 6. RESULTATS I VISUALITZACIÓ ---
print("\n--- RESULTAT DE LA NETEJA DE CORRELACIONS ---")
print(f"Variables eliminades per alta colinealitat ({len(vars_to_drop)}):")
print(list(vars_to_drop))
print(f"Dimensions finals del dataset: {df_final.shape}")

# Veure el nou mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(df_final.corr(), cmap='coolwarm', center=0)
plt.title('Matriu de Correlació Final (Sense variables > 0.95)')
plt.show()