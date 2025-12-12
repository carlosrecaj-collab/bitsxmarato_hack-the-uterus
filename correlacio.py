import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# --- 1. CARREGAR DADES ---
# Assegura't que el nom del fitxer coincideix amb el que tens a la carpeta
file_path = 'IQ_Cancer_Endometrio_merged_NMSP.xlsx - IQ_Cancer_Endometrio_merged_NMS.csv'
df = pd.read_csv("dataset.csv", sep=",")


print(f"Dimensions originals: {df.shape}")

# --- 2. NETEJA I PREPARACIÓ (DATA CLEANING) ---

# Filtrar només els casos vàlids per al target 'recidiva' (0 = No, 1 = Sí)
# Ignorem valors com 'Desconocido' o buits si n'hi ha
df = df[df['recidiva'].isin([0, 1])]

# LLISTA DE FUITES D'INFORMACIÓ (DATA LEAKAGE)
# Aquestes variables contenen informació del futur (post-diagnòstic inicial) que el model no hauria de saber.
# Ex: Data de la mort, tractament de la recidiva, estat actual, etc.
leakage_keywords = [
    'recid', 'muerte', 'exitus', 'libre', 'fecha', 'dias', 'causa', 'estado', 'f_'
]

# Identifiquem columnes sospitoses automàticament
cols_to_drop = []
for col in df.columns:
    # Si conté una paraula clau i NO és el target 'recidiva', l'eliminem
    if any(keyword in col.lower() for keyword in leakage_keywords) and col != 'recidiva':
        cols_to_drop.append(col)

# Afegim identificadors i camps de text lliure que no són útils per al model numèric
cols_to_drop.extend(['codigo_participante', 'usuario_reg1', 'comentarios', 'ap_comentarios'])

# Eliminem duplicats en la llista i apliquem la neteja
cols_to_drop = list(set(cols_to_drop))
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

print(f"Dimensions després de netejar leakage: {df_clean.shape}")

# --- 3. PREPROCESSAMENT (ENCODING) ---

# Els models necessiten números. Convertim les variables categòriques (text) a números.
# Utilitzem LabelEncoder per simplicitat.
object_cols = df_clean.select_dtypes(include=['object']).columns

for col in object_cols:
    le = LabelEncoder()
    # Convertim a string per gestionar possibles valors nuls (NaN) de forma uniforme
    df_clean[col] = df_clean[col].astype(str)
    df_clean[col] = le.fit_transform(df_clean[col])

# Omplim valors nuls numèrics restants amb un valor neutre (ex: -1 o 0) o deixem que el model ho gestioni (els models d'arbres ho suporten)
df_clean = df_clean.fillna(-1)

# --- 4. MATRIU DE CORRELACIÓ ---

# Calculem la correlació de totes les variables amb 'recidiva'
corr_matrix = df_clean.corr()
target_corr = corr_matrix['recidiva'].abs().sort_values(ascending=False)

print("\n--- Top 10 Variables més correlacionades amb Recidiva ---")
print(target_corr.head(10))

# Generar el gràfic (Heatmap)
top_features = target_corr.head(15).index.tolist()
plt.figure(figsize=(12, 10))
sns.heatmap(df_clean[top_features].corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Matriu de Correlació (Top Variables)')
plt.show()

