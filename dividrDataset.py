import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar el teu dataset
# Assegura't de canviar 'les_teves_dades.csv' pel nom real del fitxer
df = pd.read_csv('dataset.csv')

# 2. Realitzar la divisió
# test_size=0.2  -> Significa que el 10% de les dades van a test, el 90% a train.
# random_state=73 -> Assegura que la divisió sigui sempre la mateixa (reproduïble).
train_df, test_df = train_test_split(df, test_size=0.10, random_state=73)

# 3. Guardar els nous fitxers CSV
# index=False evita que es guardi una columna extra amb l'índex numèric
train_df.to_csv('dataset_train.csv', index=False)
test_df.to_csv('dataset_test.csv', index=False)

print(f"Fet! Train: {len(train_df)} files, Test: {len(test_df)} files.")