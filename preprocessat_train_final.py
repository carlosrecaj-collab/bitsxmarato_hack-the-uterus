import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


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
#########################################################################
#                       MAPA DE CALOR (CORRELACIONS)                    #
#########################################################################
print("Generant mapa de calor...")
plt.figure(figsize=(12, 10))
# Calculem la correlació només de les columnes numèriques
corr_matrix = df_final.select_dtypes(include=['number']).corr()
# Dibuixem el mapa
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Mapa de Calor: Correlacions del Dataset")
plt.show()

#######################################################################################################
#AQUÍ VEUREM QUINES VARIABLES APLICA KMENA, RCIDIVA NO ESTARA I ALTRES VAIRABLES COM LA CIUTAT TAMPOC #
#######################################################################################################

# --- ESCÀNER ---

print(f"{'COLUMNA':<30} | {'Nº VALORS ÚNICS':<15} | {'EXEMPLE (MIN-MAX)':<20}")
print("-" * 75)

candidates = []

# Excloem el target 'recidiva'
for col in df_final.columns:
    if col == 'recidiva':
        continue
    
    n_unique = df_final[col].nunique()
    min_val = df_final[col].min()
    max_val = df_final[col].max()
    
    print(f"{col:<30} | {n_unique:<15} | {min_val} - {max_val}")
 

 ### SELECIÓ DEL KMEANS
 
vars_kmeans = ['edad', 'imc', 'valor_de_ca125','tiempo_qx','perdida_hem_cc','tamano_tumoral','n_total_ganCent']

#EL -1 son poc importants el treiem  posme la mitjana

X_cluster = df_final[vars_kmeans].copy()

for col in vars_kmeans:
    # Calculem la mitjana ignorant els -1
    mean_val = X_cluster[X_cluster[col] != -1][col].mean()
    # Substituïm els -1 per la mitjana
    X_cluster[col] = X_cluster[col].replace(-1, mean_val)

##AQUÍ JA ES POT FER EL KMEANS serà un cluster amb vars_kemenas variables
##tindrem una column final feta pel kmmena que ens donara info extra

#INICI eina
scaler = StandardScaler()
#calcula mitasja i deviaci´ñoe standard
X_scaled = scaler.fit_transform(X_cluster)

# Provem amb 4 grups diferents(solen ser perfils típics: lleu, moderat, greu, molt greu)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_final['cluster_group'] = kmeans.fit_predict(X_scaled)

print("Variables usades:", vars_kmeans)
print(df_final['cluster_group'].value_counts())


################################################################################
#      OBSERVACIÓ: HI HA 1 PARCIENT QUE E SUN CAS EXTREM , L'ELIMINEM          #
################################################################################


# --- INVESTIGACIÓ DEL CAS EXTREM ---

# 1. Aïllem el pacient del grup 2 (l'outlier)
outlier = df_final[df_final['cluster_group'] == 2]

print("\n" + "="*50)
print("       ANAILISI DEL PACIENT OUTLIER")
print("="*50)

# 2. Mostrem els seus valors
print("Aquestes són les dades del pacient:")
print(outlier[vars_kmeans].T) # Posem .T per veure-ho vertical (més fàcil de llegir)

print("\n" + "-"*50)
print("COMPARATIVA AMB LA MITJANA DELS ALTRES PACIENTS")
print("-"*50)

# 3. Calculem la mitjana de TOTS els altres per comparar
resta_pacients = df_final[df_final['cluster_group'] != 2]
print(resta_pacients[vars_kmeans].mean())



################################################################################
#      OBSERVACIÓ:  L'ELIMINEM                                                 #
################################################################################

print("Eliminant pacient  ...")

# 1. ELIMINEM EL PACIENT 
# Nota: Fem servir .copy() per evitar avisos de Python
df_final = df_final[df_final['cluster_group'] != 2].copy()

# 2. RE-IMPUTACIÓ DE MITJANES
# Molt important: Ara que hem tret el valor extrem (1486), la mitjana del CA125 baixarà.
# Hem de tornar a calcular la mitjana 'real' pels altres pacients que tinguin -1.

X_cluster = df_final[vars_kmeans].copy()

for col in vars_kmeans:
    # Calculem la mitjana (ignorant els -1) de les dades netes
    mean_val = X_cluster[X_cluster[col] != -1][col].mean()
    # Omplim els buits
    X_cluster[col] = X_cluster[col].replace(-1, mean_val)

# 3. ESCALAT
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 4. K-MEANS DEFINITIU (2 GRUPS)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df_final['cluster_group'] = kmeans.fit_predict(X_scaled)

print("\n" + "="*40)
print(" RESULTAT FINAL (GRUPS EQUILIBRATS)")
print("="*40)
print(df_final['cluster_group'].value_counts())
print("\nAra tenim 2 grups un lleu/moderat i un cas greu.")

########################################
## ANALISI MULTIGRUPS 
########################################

# --- ANÀLISI DELS PERFILS ---
print("\n" + "="*50)
print(" COMPARATIVA DELS DOS PERFILS (MITJANES)")
print("="*50)

# Agrupem per cluster i calculem la mitjana de cada variable
perfils = df_final.groupby('cluster_group')[vars_kmeans].mean()

# Mostrem la taula transposada (més fàcil de llegir)
print(perfils.T)

print("\n" + "-"*50)
print("INTERPRETACIÓ RÀPIDA:")
print("Mira la variable 'valor_de_ca125' i 'tamano_tumoral'.")
print("El grup que tingui aquests valors MÉS ALTS és el grup de Risc/Greu.")
print("El grup de més risc sera el grup 0 ")


# --- VERIFICACIÓ DE LES NOVES DADES ---
#----AQUÍ S'HA AFEGIT EL CLUSTER GROUP AL DATA SET"

##################################################
#       DISTANCIA EUCLIDIANA (SOFT CLUSTERING)   #
##################################################

print("\n" + "="*60)
print(" CALCULANT DISTÀNCIES ALS CENTRES (SOFT CLUSTERING)")
print("="*60)

# 1. Calculem la matriu de distàncies
# Això crea una matriu on cada fila té la distància al Grup 0 i al Grup 1
distancies = kmeans.transform(X_scaled)

# 2. Afegim les columnes al DataFrame
for i in range(kmeans.n_clusters):
    col_name = f'dist_cluster_{i}'
    df_final[col_name] = distancies[:, i]
    print(f" -> Variable creada: {col_name}")

# ==========================================
# GUARDAR EL MODEL (MOLT IMPORTANT!)
# ==========================================

# Guardem el model per poder calcular aquestes mateixes distàncies al test
joblib.dump(kmeans, 'kmeans_model.pkl')
print("Model KMeans guardat com 'kmeans_model.pkl'")

##################################################
#           VERIFICACIÓ    i        GUARDAR      #
##################################################

print("\n" + "="*60)
print(" ESTAT FINAL DEL DATASET (PRIMERES 5 FILES)")
print("="*60)

# Mostrem només les columnes noves per comprovar que s'han creat bé
cols_noves = ['cluster_group', 'dist_cluster_0', 'dist_cluster_1']
print(df_final[cols_noves].head())

print("\n" + "="*60)
print(" GUARDANT FITXER DEFINITIU...")
print("="*60)

# Guardem el CSV amb el nom nou
df_final.to_csv("dataset_train_final_kmeans_euclidiana.csv", index=False)

print(" Creat correctament.")
print(" llest per al model.")

