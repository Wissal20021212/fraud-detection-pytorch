import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
df = pd.read_csv("data/creditcard.csv")

# Afficher les 5 premières lignes
print(df.head())

# Afficher les dimensions
print("Shape:", df.shape)

# Afficher les informations générales
print(df.info())

# Compter les transactions normales et frauduleuses
print(df["Class"].value_counts())

# Calculer le pourcentage de chaque classe
print("\nClass distribution (%):")
print(df["Class"].value_counts(normalize=True) * 100)

# Vérifier les valeurs manquantes
print("\nMissing values:")
print(df.isnull().sum().sum())

# Statistiques descriptives
print("\nStatistical summary:")
print(df[["Time", "Amount"]].describe())

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Class")

plt.title("Distribution of Normal and Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.show()

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=50,
    log_scale=(False, True)
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency (log scale)")

plt.show()