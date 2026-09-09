"""
Modelo XGBoost: 56 clasificadores binarios (¿sale el número X?).
Features: temporales, secuenciales, históricas, lags.
"""
import argparse
import joblib
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from config import (
    CSV_HISTORICO, MODELO_XGB, RANGO_NUMEROS, NUM_BOLAS_POR_SORTEO,
    TRAIN_TEST_SPLIT, MIN_SORTEOS_PARA_MODELO
)


def cargar_datos() -> pd.DataFrame:
    if not CSV_HISTORICO.exists():
        raise FileNotFoundError(f"Falta: {CSV_HISTORICO}")
    
    df = pd.read_csv(CSV_HISTORICO)
    
    # Crear fecha_dt si no existe (el scraper no la guarda)
    if "fecha_dt" not in df.columns:
        if "fecha" in df.columns:
            df["fecha_dt"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
        else:
            # Intentar detectar la columna de fecha automáticamente
            for col in df.columns:
                if "fecha" in col.lower() or "date" in col.lower():
                    df["fecha_dt"] = pd.to_datetime(df[col], format="%d/%m/%Y", errors="coerce")
                    break
    
    df = df.dropna(subset=["fecha_dt"])
    
    if len(df) < MIN_SORTEOS_PARA_MODELO:
        print(f"⚠️  Solo {len(df)} sorteos. Recomendados: {MIN_SORTEOS_PARA_MODELO}")
    return df


def crear_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Crea 56 columnas binarias: y_1, y_2, ..., y_56."""
    cols_num = [f"n{i}" for i in range(1, NUM_BOLAS_POR_SORTEO + 1)]
    for n in range(1, RANGO_NUMEROS + 1):
        df[f"y_{n}"] = df[cols_num].eq(n).any(axis=1).astype(int)
    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de features temporales e históricas."""
    df = df.copy()
    cols_num = [f"n{i}" for i in range(1, NUM_BOLAS_POR_SORTEO + 1)]

    # Temporales
    df["dia_semana"] = df["fecha_dt"].dt.dayofweek
    df["mes"] = df["fecha_dt"].dt.month
    df["semana_anio"] = df["fecha_dt"].dt.isocalendar().week.astype(int)
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)

    # Secuenciales
    df["sorteo_idx"] = np.arange(len(df))
    df["dias_desde_anterior"] = df["fecha_dt"].diff().dt.days.fillna(7)

    # Históricas: frecuencia en ventanas móviles
    for ventana in [30, 90, 180]:
        for n in range(1, RANGO_NUMEROS + 1):
            aparecio = df[cols_num].eq(n).any(axis=1).astype(int)
            df[f"freq_{n}_v{ventana}"] = aparecio.rolling(ventana, min_periods=1).mean().shift(1)

    # Lags: ¿salió en los últimos K sorteos?
    for lag in [1, 2, 3]:
        for n in range(1, RANGO_NUMEROS + 1):
            aparecio = df[cols_num].eq(n).any(axis=1).astype(int)
            df[f"lag_{n}_{lag}"] = aparecio.shift(lag).fillna(0).astype(int)

    # Contexto del sorteo anterior
    df["suma_anterior"] = df[cols_num].sum(axis=1).shift(1)
    df["pares_anterior"] = (df[cols_num] % 2 == 0).sum(axis=1).shift(1)

    return df


def preparar_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Separa features, targets y hace split temporal."""
    df = crear_targets(df)
    df = crear_features(df)

    feature_cols = [c for c in df.columns if c.startswith(("dia_", "mes", "semana", "es_",
                                                            "sorteo_", "dias_", "freq_",
                                                            "lag_", "suma_", "pares_"))]
    target_cols = [f"y_{n}" for n in range(1, RANGO_NUMEROS + 1)]

    df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"[modelo] Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"[modelo] Features: {len(feature_cols)}")
    return df_train, df_test, feature_cols, target_cols


def entrenar_modelos(df_train: pd.DataFrame, feature_cols: List[str],
                     target_cols: List[str]) -> dict:
    """Entrena un XGBoost por cada número."""
    modelos = {}
    X_train = df_train[feature_cols].values

    print(f"[modelo] Entrenando {len(target_cols)} clasificadores...")
    for i, target in enumerate(target_cols):
        y_train = df_train[target].values
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale = n_neg / max(n_pos, 1)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        modelos[target] = model

        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(target_cols)}")

    return modelos


def evaluar_modelos(modelos: dict, df_test: pd.DataFrame,
                    feature_cols: List[str], target_cols: List[str]) -> dict:
    """Evalúa cada clasificador en test."""
    X_test = df_test[feature_cols].values
    aucs = []

    print(f"\n[modelo] Evaluando en test ({len(df_test)} sorteos)...")
    for target in target_cols:
        y_test = df_test[target].values
        y_prob = modelos[target].predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5
        aucs.append(auc)

    print(f"AUC promedio: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Predicción agregada: top-6
    probs = np.zeros((len(df_test), RANGO_NUMEROS))
    for i, target in enumerate(target_cols):
        probs[:, i] = modelos[target].predict_proba(X_test)[:, 1]

    top6_pred = np.argsort(probs, axis=1)[:, -6:] + 1
    hits = []
    for j in range(len(df_test)):
        reales_set = set()
        for n in range(1, RANGO_NUMEROS + 1):
            if df_test.iloc[j][f"y_{n}"] == 1:
                reales_set.add(n)
        pred_set = set(top6_pred[j])
        hits.append(len(reales_set & pred_set))

    hits = np.array(hits)
    print(f"\n=== Hits del modelo ===")
    for k in range(7):
        pct = (hits >= k).mean() * 100
        print(f"  ≥{k} aciertos: {pct:.1f}%")

    return {}


def guardar_modelos(modelos: dict, feature_cols: List[str]) -> None:
    """Guarda modelos y metadata."""
    joblib.dump({"modelos": modelos, "feature_cols": feature_cols}, MODELO_XGB)
    print(f"[modelo] Guardado en: {MODELO_XGB}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    df = cargar_datos()
    df_train, df_test, feature_cols, target_cols = preparar_dataset(df)
    modelos = entrenar_modelos(df_train, feature_cols, target_cols)
    guardar_modelos(modelos, feature_cols)

    if not args.no_eval:
        evaluar_modelos(modelos, df_test, feature_cols, target_cols)


if __name__ == "__main__":
    main()