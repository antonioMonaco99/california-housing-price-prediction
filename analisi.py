# ============================================================
# PROGETTO DATA ANALYSIS & MACHINE LEARNING - CALIFORNIA HOUSING
# ============================================================

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.3)
plt.xlabel("Reddito mediano")
plt.ylabel("Valore mediano delle case")
plt.title("Reddito e valore delle abitazioni")
plt.show()
plt.close()


# ============================================================
# 5. CORRELAZIONI
# ============================================================

correlazioni = df.corr(numeric_only=True)

print("\n--- CORRELAZIONI CON IL VALORE DELLE CASE ---")
print(correlazioni["MedHouseVal"].sort_values(ascending=False))


# ============================================================
# 6. MATRICE DI CORRELAZIONE
# ============================================================

plt.figure(figsize=(10, 8))
sns.heatmap(correlazioni, annot=True, cmap="coolwarm", fmt=".2f")
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
plt.colorbar(label="Valore mediano delle abitazioni")
plt.show()
plt.close()


# ============================================================
# 8. VALORI MANCANTI E DUPLICATI
# ============================================================

print("\n--- VALORI MANCANTI ---")
print(df.isnull().sum())

print("\n--- RIGHE DUPLICATE ---")
print("Righe duplicate:", df.duplicated().sum())


# ============================================================
# 9. CONTROLLO DEGLI OUTLIER (AveOccup)
# ============================================================

print("\n--- 10 VALORI PIÙ ALTI DI AVEOCCUP ---")
print(df["AveOccup"].sort_values(ascending=False).head(10))

print("\n--- OSSERVAZIONI CON AVEOCCUP > 10 ---")
print(
    df[df["AveOccup"] > 10][["AveOccup", "MedInc", "MedHouseVal"]]
    .sort_values("AveOccup", ascending=False)
)


# ============================================================
# 10. RIMOZIONE DEGLI OUTLIER ESTREMI
# ============================================================

righe_prima = len(df)
df = df[df["AveOccup"] <= 50]

print("\n--- DATASET DOPO LA RIMOZIONE DEGLI OUTLIER ---")
print("Nuove dimensioni:", df.shape)
print("Osservazioni rimosse:", righe_prima - len(df))


# ============================================================
# 11. SALVATAGGIO DATASET PULITO
# ============================================================

df.to_csv(os.path.join(CARTELLA_SCRIPT, "housing_clean.csv"), index=False)

print("\n--- DATASET PULITO SALVATO ---")
print("Dimensioni:", df.shape)
print("Valori mancanti:\n", df.isnull().sum())


# ============================================================
# 12. PREPARAZIONE DEI DATI PER IL MACHINE LEARNING
# ============================================================

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\n--- TRAIN / TEST SPLIT ---")
print("X_train:", X_train.shape, "| X_test:", X_test.shape)


# ============================================================
# 13. LINEAR REGRESSION
# ============================================================

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n--- RISULTATI LINEAR REGRESSION ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# ============================================================
# 14. RANDOM FOREST
# ============================================================

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- RISULTATI RANDOM FOREST ---")
print(f"MAE:  {mae_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²:   {r2_rf:.4f}")

feature_importance = pd.Series(
    model_rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\n--- IMPORTANZA DELLE FEATURE (RANDOM FOREST) ---")
print(feature_importance)


# ============================================================
# 15. GRADIENT BOOSTING
# ============================================================

model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = mean_squared_error(y_test, y_pred_gb) ** 0.5
r2_gb = r2_score(y_test, y_pred_gb)

print("\n--- RISULTATI GRADIENT BOOSTING ---")
print(f"MAE:  {mae_gb:.4f}")
print(f"RMSE: {rmse_gb:.4f}")
print(f"R²:   {r2_gb:.4f}")


# ============================================================
# 16. CONFRONTO DEI MODELLI
# ============================================================

risultati = pd.DataFrame({
    "Modello": ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "MAE": [mae, mae_rf, mae_gb],
    "RMSE": [rmse, rmse_rf, rmse_gb],
    "R2": [r2, r2_rf, r2_gb]
})

print("\n--- CONFRONTO DEI MODELLI ---")
print(risultati)

plt.figure(figsize=(8, 5))
plt.bar(risultati["Modello"], risultati["R2"])
plt.ylabel("R²")
plt.xlabel("Modello")
plt.title("Confronto delle performance dei modelli")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.xlabel("Valore reale")
plt.ylabel("Valore previsto")
plt.title("Random Forest - Valori reali vs valori previsti")
plt.show()
plt.close()


# ============================================================
# 17. CONCLUSIONI
# ============================================================

print("\n--- CONCLUSIONI ---")
print("Il modello migliore è la Random Forest.")
print(f"MAE: {mae_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²: {r2_rf:.4f}")

print("\nLa feature più importante è:", feature_importance.index[0])
print(f"Importanza: {feature_importance.iloc[0]:.4f}")
