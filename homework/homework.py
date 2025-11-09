# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}


# flake8: noqa: E501


import os
import json
import gzip
import pickle
from typing import List, Dict, Union

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import warnings
from sklearn.exceptions import ConvergenceWarning

# Silenciar warnings de convergencia del MLP para una salida más limpia
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------- Rutas ---------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(ROOT, os.pardir))

TRAIN_ZIP = os.path.join(BASE, "files/input/train_data.csv.zip")
TEST_ZIP = os.path.join(BASE, "files/input/test_data.csv.zip")
MODEL_OUT = os.path.join(BASE, "files/models/model.pkl.gz")
METRICS_OUT = os.path.join(BASE, "files/output/metrics.json")

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)

# ---------------------- Carga y limpieza ------------------------------

def load_csv_zip(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="zip")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Renombrar la variable objetivo
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})

    # Eliminar ID si existe
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # EDUCATION: >4 y 0 -> others (4)
    if "EDUCATION" in df.columns:
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
        df.loc[df["EDUCATION"] == 0, "EDUCATION"] = 4

    # Eliminar registros con NA
    df = df.dropna(axis=0).reset_index(drop=True)

    # Downcast de tipos para eficiencia
    for c in df.select_dtypes(include=["int64", "int32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")

    return df


train_df = clean_df(load_csv_zip(TRAIN_ZIP))
test_df = clean_df(load_csv_zip(TEST_ZIP))

# ----------------------------- Split ---------------------------------
assert "default" in train_df.columns and "default" in test_df.columns
X_train = train_df.drop(columns=["default"])
y_train = train_df["default"].astype(int)
X_test = test_df.drop(columns=["default"])
y_test = test_df["default"].astype(int)

# ---------- Columnas categóricas vs numéricas para OHE ----------------
# OHE solo a SEX, EDUCATION, MARRIAGE (las PAY_* se tratan como ordinales/numéricas)
categorical_cols: List[str] = [
    c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in X_train.columns
]
numeric_cols: List[str] = [c for c in X_train.columns if c not in categorical_cols]

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ),
        ("num", "passthrough", numeric_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,  # salida densa necesaria para PCA
)

# Estimar número máximo de características post-preprocess (y por ende de PCA)
# para acotar k en SelectKBest de forma segura.
_pre_fit = preprocess.fit_transform(X_train)
max_features_after_prep = _pre_fit.shape[1]

# PCA con todas las componentes (equivalente a n_components_ = min(n_samples, n_features))
pca = PCA(n_components=None, svd_solver="full", random_state=42)

# Construcción del pipeline final
pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("pca", pca),
        ("scaler", StandardScaler()),  # escala a [0, 1]
        ("select", SelectKBest(f_classif, k=20)),  # valor base; se afina en grid
        (
            "clf",
            MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=42,
            ),
        ),
    ]
)

# ---------------------- Búsqueda de hiperparámetros -------------------
# k de SelectKBest no debe exceder las características después de PCA.
k_candidates: List[Union[int, str]] = [10, 20, 30, 50, "all"]
k_grid = [k for k in k_candidates if k == "all" or (isinstance(k, int) and k <= max_features_after_prep)]
if not k_grid:
    k_grid = ["all"]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
param_grid = {
    "select__k": k_grid,
    "clf__hidden_layer_sizes": [(64,), (128,)],
    "clf__alpha": [1e-4, 1e-3],
    # Mantener activación y tasa de aprendizaje fijas para no disparar el costo de CV
    # "clf__activation": ["relu"],
    # "clf__learning_rate_init": [1e-3],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=1,           # conservador y estable
    pre_dispatch=1,
    refit=True,
    verbose=0,
)

grid.fit(X_train, y_train)

# ------------------------- Guardar modelo -----------------------------
# IMPORTANTE: guardamos el objeto GridSearchCV completo
with gzip.open(MODEL_OUT, "wb") as f:
    pickle.dump(grid, f)

# --------------------------- Métricas ---------------------------------

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

# Predicciones con el mejor modelo
ytr_pred = grid.predict(X_train)
yte_pred = grid.predict(X_test)

train_metrics = {"type": "metrics", "dataset": "train", **compute_metrics(y_train, ytr_pred)}
test_metrics = {"type": "metrics", "dataset": "test", **compute_metrics(y_test, yte_pred)}


def cm_dict(y_true, y_pred) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "true_0": {"predicted_0": tn, "predicted_1": fp},
        "true_1": {"predicted_0": fn, "predicted_1": tp},
    }

cm_train = {"type": "cm_matrix", "dataset": "train", **cm_dict(y_train, ytr_pred)}
cm_test = {"type": "cm_matrix", "dataset": "test", **cm_dict(y_test, yte_pred)}

with open(METRICS_OUT, "w", encoding="utf-8") as f:
    f.write(json.dumps(train_metrics, ensure_ascii=False) + "\n")
    f.write(json.dumps(test_metrics, ensure_ascii=False) + "\n")
    f.write(json.dumps(cm_train, ensure_ascii=False) + "\n")
    f.write(json.dumps(cm_test, ensure_ascii=False) + "\n")

# --------------------------- Info consola -----------------------------
print("Mejores hiperparámetros:", grid.best_params_)
print("Mejor balanced_accuracy (CV):", grid.best_score_)
# Nota: score() del estimador devuelve accuracy por defecto
print("Accuracy train:", grid.score(X_train, y_train))
print("Accuracy test :", grid.score(X_test, y_test))
print(f"Modelo guardado en: {MODEL_OUT}")
print(f"Métricas guardadas en: {METRICS_OUT}")
