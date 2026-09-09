# ============================================================
# PROGETTO DATA ANALYSIS - CALIFORNIA HOUSING
# ============================================================

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cartella in cui si trova questo script: così il CSV si trova sempre,
# indipendentemente da dove viene lanciato lo script (terminale, VS Code, ecc.)
CARTELLA_SCRIPT = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. CARICAMENTO DEL DATASET
# ============================================================

df = pd.read_csv(os.path.join(CARTELLA_SCRIPT, "housing.csv"))


# ============================================================
# 2. ESPLORAZIONE DEI DATI
# ============================================================

print("\n--- PRIME 5 RIGHE ---")
print(df.head())

print("\n--- DIMENSIONI DEL DATASET ---")
print(df.shape)

print("\n--- INFORMAZIONI SUL DATASET ---")
df.info()

print("\n--- STATISTICHE DESCRITTIVE ---")
print(df.describe())


# ============================================================
# 3. DISTRIBUZIONE DEL VALORE DELLE ABITAZIONI
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(df["MedHouseVal"], bins=30)

plt.xlabel("Valore mediano delle case")
plt.ylabel("Numero di osservazioni")
plt.title("Distribuzione del valore delle abitazioni")

plt.show()
plt.close()


# ============================================================
# 4. RELAZIONE TRA REDDITO E VALORE DELLE ABITAZIONI
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["MedInc"],
    df["MedHouseVal"],
    alpha=0.3
)

plt.xlabel("Reddito mediano")
plt.ylabel("Valore mediano delle case")
plt.title("Reddito e valore delle abitazioni")

plt.show()
plt.close()


# ============================================================
# 5. TUTTE LE CORRELAZIONI
# ============================================================
# (include già reddito, età e stanze: non serve calcolarle a parte)

correlazioni = df.corr(numeric_only=True)

print("\n--- CORRELAZIONI CON IL VALORE DELLE CASE ---")

print(
    correlazioni["MedHouseVal"]
    .sort_values(ascending=False)
)


# ============================================================
# 6. MATRICE DI CORRELAZIONE
# ============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlazioni,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matrice di correlazione")

plt.show()
plt.close()


# ============================================================
# 7. ANALISI GEOGRAFICA
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df["Longitude"],
    df["Latitude"],
    c=df["MedHouseVal"],
    alpha=0.5
)

plt.xlabel("Longitudine")
plt.ylabel("Latitudine")
plt.title("Distribuzione geografica del valore delle abitazioni")

plt.colorbar(
    label="Valore mediano delle abitazioni"
)

plt.show()
plt.close()


# ============================================================
# 8. CONTROLLO VALORI MANCANTI
# ============================================================

print("\n--- VALORI MANCANTI ---")

print(df.isnull().sum())


# ============================================================
# 9. CONTROLLO DUPLICATI
# ============================================================

print("\n--- RIGHE DUPLICATE ---")

print(
    "Righe duplicate:",
    df.duplicated().sum()
)


# ============================================================
# 10. CONTROLLO DEGLI OUTLIER
# ============================================================

print("\n--- STATISTICHE PER INDIVIDUARE OUTLIER ---")

print(df.describe().T)


# ============================================================
# 11. VALORI PIÙ ALTI DI AVEOCCUP
# ============================================================

print("\n--- 10 VALORI PIÙ ALTI DI AVEOCCUP ---")

print(
    df["AveOccup"]
    .sort_values(ascending=False)
    .head(10)
)


# ============================================================
# 12. QUANTE OSSERVAZIONI HANNO AVEOCCUP > 10?
# ============================================================

print("\n--- OUTLIER AVEOCCUP ---")

print(
    "Osservazioni con AveOccup > 10:",
    (df["AveOccup"] > 10).sum()
)


# ============================================================
# 13. ANALISI DELLE OSSERVAZIONI ANOMALE
# ============================================================

print("\n--- OSSERVAZIONI CON AVEOCCUP > 10 ---")

print(
    df[df["AveOccup"] > 10][
        ["AveOccup", "MedInc", "MedHouseVal"]
    ]
    .sort_values("AveOccup", ascending=False)
)


# ============================================================
# 14. RIMOZIONE DEGLI OUTLIER ESTREMI
# ============================================================

righe_prima = len(df)

df = df[df["AveOccup"] <= 50]


# ============================================================
# 15. CONTROLLO FINALE DEL DATASET
# ============================================================

print("\n--- DATASET DOPO LA RIMOZIONE DEGLI OUTLIER ---")

print("Nuove dimensioni:", df.shape)

print(
    "Osservazioni rimosse:",
    righe_prima - len(df)
)
# ============================================================
# 16. CONTROLLO FINALE DEL DATASET PULITO
# ============================================================
df.to_csv(os.path.join(CARTELLA_SCRIPT, "housing_clean.csv"), index=False)
print("\n--- CONTROLLO FINALE DATASET PULITO ---")

print("Dimensioni:", df.shape)

print("\nValori mancanti:")
print(df.isnull().sum())

print("\nStatistiche descrittive:")
print(df.describe())
# ============================================================
# DAY 3 - PREPARAZIONE DEI DATI PER IL MACHINE LEARNING
# ============================================================

# Variabili utilizzate dal modello
X = df.drop("MedHouseVal", axis=1)

# Variabile che vogliamo prevedere
y = df["MedHouseVal"]

print("\n--- FEATURES (X) ---")
print(X.head())

print("\n--- TARGET (y) ---")
print(y.head())

print("\nDimensioni X:", X.shape)
print("Dimensioni y:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n--- TRAIN / TEST SPLIT ---")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# ============================================================
# STEP 3 - PRIMO MODELLO DI MACHINE LEARNING
# ============================================================

from sklearn.linear_model import LinearRegression

# Creazione del modello
model = LinearRegression()

# Addestramento del modello
model.fit(X_train, y_train)

print("\n--- MODELLO ADDESTRATO ---")
print("Linear Regression completata")

# ============================================================
# STEP 4 - PRIME PREVISIONI
# ============================================================

y_pred = model.predict(X_test)

print("\n--- PRIME PREVISIONI ---")
print(y_pred[:10])

# ============================================================
# STEP 5 - VALUTAZIONE DEL MODELLO: MAE
# ============================================================

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("\n--- MAE ---")
print("Errore medio assoluto:", mae)

# ============================================================
# STEP 6 - VALUTAZIONE DEL MODELLO: RMSE
# ============================================================

from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("\n--- RMSE ---")
print("Root Mean Squared Error:", rmse)

# ============================================================
# STEP 7 - VALUTAZIONE DEL MODELLO: R²
# ============================================================

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("\n--- R² ---")
print("R²:", r2)

# ============================================================
# STEP 8 - RIEPILOGO DELLA BASELINE
# ============================================================

print("\n--- RISULTATI LINEAR REGRESSION ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# ============================================================
# GIORNO 4 - RANDOM FOREST
# ============================================================

from sklearn.ensemble import RandomForestRegressor
# ============================================================
# STEP 2 - CREAZIONE RANDOM FOREST
# ============================================================

model_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("\n--- RANDOM FOREST CREATA ---")
print("Numero di alberi:", model_rf.n_estimators)

# ============================================================
# STEP 3 - ADDESTRAMENTO RANDOM FOREST
# ============================================================

model_rf.fit(X_train, y_train)

print("\n--- RANDOM FOREST ADDESTRATA ---")
print("Addestramento completato")

# ============================================================
# STEP 4 - PREVISIONI RANDOM FOREST
# ============================================================

y_pred_rf = model_rf.predict(X_test)

print("\n--- PRIME PREVISIONI RANDOM FOREST ---")
print(y_pred_rf[:10])

# ============================================================
# STEP 5 - VALUTAZIONE RANDOM FOREST: MAE
# ============================================================

mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("\n--- MAE RANDOM FOREST ---")
print("Errore medio assoluto:", mae_rf)

# ============================================================
# STEP 6 - VALUTAZIONE RANDOM FOREST
# ============================================================

# RMSE
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5

# R²
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- RISULTATI RANDOM FOREST ---")
print(f"MAE:  {mae_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²:   {r2_rf:.4f}")

# ============================================================
# STEP 7 - IMPORTANZA DELLE FEATURE
# ============================================================

importances = model_rf.feature_importances_

feature_importance = pd.Series(
    importances,
    index=X.columns
).sort_values(ascending=False)

print("\n--- IMPORTANZA DELLE FEATURE ---")
print(feature_importance)

# ============================================================
# DAY 5 - GRADIENT BOOSTING
# ============================================================

from sklearn.ensemble import GradientBoostingRegressor

# Creazione del modello
model_gb = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)

print("\n--- GRADIENT BOOSTING CREATO ---")
print("Numero di alberi:", model_gb.n_estimators)

# ============================================================
# STEP 2 - ADDESTRAMENTO GRADIENT BOOSTING
# ============================================================

model_gb.fit(X_train, y_train)

print("\n--- GRADIENT BOOSTING ADDESTRATO ---")
print("Addestramento completato")

# ============================================================
# STEP 3 - PREVISIONI GRADIENT BOOSTING
# ============================================================

y_pred_gb = model_gb.predict(X_test)

print("\n--- PRIME PREVISIONI GRADIENT BOOSTING ---")
print(y_pred_gb[:10])

# ============================================================
# STEP 4 - VALUTAZIONE GRADIENT BOOSTING
# ============================================================

mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = mean_squared_error(y_test, y_pred_gb) ** 0.5
r2_gb = r2_score(y_test, y_pred_gb)

print("\n--- RISULTATI GRADIENT BOOSTING ---")
print(f"MAE:  {mae_gb:.4f}")
print(f"RMSE: {rmse_gb:.4f}")
print(f"R²:   {r2_gb:.4f}")

# ============================================================
# STEP 5 - CONFRONTO DEI MODELLI
# ============================================================

risultati = pd.DataFrame({
    "Modello": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        mae,
        mae_rf,
        mae_gb
    ],
    "RMSE": [
        rmse,
        rmse_rf,
        rmse_gb
    ],
    "R2": [
        r2,
        r2_rf,
        r2_gb
    ]
})

print("\n--- CONFRONTO DEI MODELLI ---")
print(risultati)

# ============================================================
# STEP 6 - GRAFICO CONFRONTO MODELLI
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    risultati["Modello"],
    risultati["R2"]
)

plt.ylabel("R²")
plt.xlabel("Modello")
plt.title("Confronto delle performance dei modelli")

plt.ylim(0, 1)
plt.xticks(rotation=15)

plt.show()
plt.close()

# ============================================================
# STEP 7 - VALORI REALI VS PREVISIONI
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred_rf,
    alpha=0.3
)

plt.xlabel("Valore reale")
plt.ylabel("Valore previsto")
plt.title("Random Forest - Valori reali vs valori previsti")

plt.show()
plt.close()

# ============================================================
# STEP 8 - CONCLUSIONI
# ============================================================

print("\n--- CONCLUSIONI ---")

print("Il modello migliore è la Random Forest.")
print(f"MAE: {mae_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²: {r2_rf:.4f}")

print("\nLa feature più importante è:")
print(feature_importance.index[0])

print(f"Importanza: {feature_importance.iloc[0]:.4f}")