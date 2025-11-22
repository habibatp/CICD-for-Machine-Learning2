import os
import pandas as pd
import matplotlib.pyplot as plt
import skops.io as sio

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ==========================
# 0. Chemins / Préparation
# ==========================

# ⚠️ Ton fichier s'appelle drug200.csv
DATA_PATH = os.path.join("Data", "drug200.csv")
MODEL_DIR = "Model"
RESULTS_DIR = "Results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("📂 Dossier courant :", os.getcwd())
print("📂 Chemin attendu pour le CSV :", DATA_PATH)
print("📂 Contenu du dossier Data :", os.listdir("Data"))
print(f"📂 Chargement du dataset depuis : {DATA_PATH}")

# ==========================
# 1. Chargement des données
# ==========================

drug_df = pd.read_csv(DATA_PATH)

# Mélange pour éviter un ordre biaisé (avec random_state pour la reproductibilité)
drug_df = drug_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("🔍 Aperçu des 5 premières lignes :")
print(drug_df.head())
print("\nColonnes :", drug_df.columns.tolist())

# ==========================
# 2. Train / Test Split
# ==========================

X = drug_df.drop("Drug", axis=1).values  # variables explicatives
y = drug_df["Drug"].values               # variable cible

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=125,
    stratify=y,   # pour garder la même répartition des classes
)

print("\n📊 Dimensions :")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

# ==========================
# 3. Construction du pipeline
# ==========================

# Indices des colonnes (dans l'ordre du CSV original)
# 0 : Age (numérique)
# 1 : Sex (catégorielle)
# 2 : BP (catégorielle)
# 3 : Cholesterol (catégorielle)
# 4 : Na_to_K (numérique)

cat_col = [1, 2, 3]
num_col = [0, 4]

transform = ColumnTransformer(
    transformers=[
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=10, random_state=125)),
    ]
)

print("\n=== Pipeline entraîné (structure) ===")
print(pipe)

# ==========================
# 4. Entraînement du modèle
# ==========================

pipe.fit(X_train, y_train)
print("\n✅ Entraînement terminé.")

# ==========================
# 5. Évaluation du modèle
# ==========================

predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(
    f"\n📈 Résultats sur le test : "
    f"Accuracy = {accuracy:.2f}  |  F1 macro = {f1:.2f}"
)

# ==========================
# 6. Matrice de confusion
# ==========================

cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)

disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Drug Classification")
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "model_results.png")
plt.savefig(cm_path, dpi=120)
plt.close()

print(f"🖼 Matrice de confusion sauvegardée dans : {cm_path}")

# ==========================
# 7. Sauvegarde des métriques
# ==========================

metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as outfile:
    outfile.write(f"Accuracy = {accuracy:.4f}\n")
    outfile.write(f"F1_macro = {f1:.4f}\n")

print(f"📝 Métriques sauvegardées dans : {metrics_path}")

# ==========================
# 8. Sauvegarde du pipeline (skops)
# ==========================

model_path = os.path.join(MODEL_DIR, "drug_pipeline.skops")
sio.dump(pipe, model_path)

print(f"💾 Modèle sauvegardé dans : {model_path}")
