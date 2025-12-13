import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



### NOU DATA SET (CARLOS)
df_final = pd.read_csv("dataset_train_processat.csv")

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

