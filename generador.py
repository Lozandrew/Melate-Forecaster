"""
Generador de combinaciones optimizadas.
Combina: probabilidades XGBoost + filtros estadísticos + penalización populares.
"""
import argparse
import joblib
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import (
    CSV_HISTORICO, MODELO_XGB, RESULTADO_PREDICCION,
    RANGO_NUMEROS, NUM_BOLAS_POR_SORTEO,
    N_COMBINACIONES_RECOMENDADAS, RANGO_SUMA_MIN, RANGO_SUMA_MAX,
    PARES_MIN, PARES_MAX
)


def cargar_modelo():
    """Carga el modelo entrenado. Si no existe, usa fallback uniforme."""
    if not MODELO_XGB.exists():
        print("[generador] No hay modelo, usando frecuencia histórica")
        return None, None
    data = joblib.load(MODELO_XGB)
    return data["modelos"], data["feature_cols"]


def preparar_features_ultimo_sorteo(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Prepara las features del último sorteo conocido."""
    from modelo import crear_features, crear_targets
    df = crear_targets(df)
    df = crear_features(df)
    df = df.dropna(subset=feature_cols)
    return df[feature_cols].iloc[-1:].values


def predecir_probabilidades(modelos: dict, X: np.ndarray) -> np.ndarray:
    """Obtiene probabilidad de cada número (1-56)."""
    probs = np.zeros(RANGO_NUMEROS)
    for i in range(RANGO_NUMEROS):
        target = f"y_{i+1}"
        probs[i] = modelos[target].predict_proba(X)[0, 1]
    return probs


def penalizar_populares(combinacion: np.ndarray) -> float:
    """Penaliza combinaciones que mucha gente juega."""
    score = 0.0

    # Cumpleaños (1-31) sobre-representados
    n_cumple = sum(1 for x in combinacion if x <= 31)
    if n_cumple >= 5:
        score -= 3
    elif n_cumple >= 4:
        score -= 1

    # Secuencias obvias
    sorted_c = sorted(combinacion)
    diffs = [sorted_c[i+1] - sorted_c[i] for i in range(len(sorted_c)-1)]
    if all(d == 1 for d in diffs):
        score -= 10
    if all(d == diffs[0] for d in diffs):
        score -= 5

    # Múltiplos
    if all(x % 5 == 0 for x in combinacion):
        score -= 5
    if all(x % 7 == 0 for x in combinacion):
        score -= 5

    # Todos del mismo rango
    decenas = [x // 10 for x in combinacion]
    if len(set(decenas)) == 1:
        score -= 4

    # Muchos consecutivos
    if sum(1 for i in range(len(sorted_c)-1) if sorted_c[i+1] - sorted_c[i] == 1) >= 3:
        score -= 2

    return score


def validar_combinacion(combinacion: np.ndarray) -> bool:
    """Verifica filtros estadísticos típicos."""
    suma = combinacion.sum()
    pares = sum(1 for x in combinacion if x % 2 == 0)

    if not (RANGO_SUMA_MIN <= suma <= RANGO_SUMA_MAX):
        return False
    if not (PARES_MIN <= pares <= PARES_MAX):
        return False
    if not (any(x < 10 for x in combinacion) and any(x > 40 for x in combinacion)):
        return False
    return True


def generar_combinacion_muestreo(probs: np.ndarray, n_intentos: int = 10000) -> np.ndarray:
    """Muestrea combinación ponderada por probabilidades."""
    probs_norm = probs / probs.sum()
    numeros = np.arange(1, RANGO_NUMEROS + 1)

    for _ in range(n_intentos):
        cand = np.sort(np.random.choice(numeros, size=NUM_BOLAS_POR_SORTEO,
                                        replace=False, p=probs_norm))
        if validar_combinacion(cand):
            return cand

    return np.sort(np.random.choice(numeros, size=NUM_BOLAS_POR_SORTEO,
                                    replace=False, p=probs_norm))


def generar_recomendaciones(n: int = N_COMBINACIONES_RECOMENDADAS) -> List[Tuple[np.ndarray, float]]:
    """Genera N combinaciones recomendadas."""
    df = pd.read_csv(CSV_HISTORICO)
    
    # === FIX: Forzar conversión a datetime ===
    if "fecha_dt" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
    elif "fecha" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
    else:
        for col in df.columns:
            if "fecha" in col.lower() or "date" in col.lower():
                df["fecha_dt"] = pd.to_datetime(df[col], errors="coerce")
                break
    
    if not pd.api.types.is_datetime64_any_dtype(df["fecha_dt"]):
        df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
    
    df = df.dropna(subset=["fecha_dt"])

    modelos, feature_cols = cargar_modelo()

    if modelos is not None:
        X = preparar_features_ultimo_sorteo(df, feature_cols)
        probs = predecir_probabilidades(modelos, X)
        modo = "XGBoost"
    else:
        cols = [f"n{i}" for i in range(1, NUM_BOLAS_POR_SORTEO + 1)]
        nums = df[cols].values.flatten()
        freq, _ = np.histogram(nums, bins=range(1, RANGO_NUMEROS + 2))
        probs = freq.astype(float)
        probs = probs / probs.sum()
        modo = "Frecuencia histórica"

    print(f"[generador] Modo: {modo}")
    print(f"[generador] Top 10 números por probabilidad:")
    top_idx = np.argsort(probs)[::-1][:10]
    for i in top_idx:
        print(f"  Número {i+1:2d}: {probs[i]:.4f}")

    combinaciones = []
    vistos = set()
    intentos = 0
    max_intentos = n * 50

    while len(combinaciones) < n and intentos < max_intentos:
        cand = generar_combinacion_muestreo(probs)
        key = tuple(cand)
        if key not in vistos:
            score = probs[cand - 1].sum() + penalizar_populares(cand)
            combinaciones.append((cand, score))
            vistos.add(key)
        intentos += 1

    combinaciones.sort(key=lambda x: x[1], reverse=True)
    return combinaciones


def guardar_resultado(combinaciones: List[Tuple[np.ndarray, float]]) -> None:
    """Guarda recomendaciones en CSV."""
    data = []
    for i, (comb, score) in enumerate(combinaciones, 1):
        row = {"rank": i, "score": f"{score:.4f}"}
        for j, n in enumerate(comb, 1):
            row[f"n{j}"] = n
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(RESULTADO_PREDICCION, index=False)
    print(f"[generador] Guardado en: {RESULTADO_PREDICCION}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=N_COMBINACIONES_RECOMENDADAS)
    args = parser.parse_args()

    if not CSV_HISTORICO.exists():
        print(f"[generador] Error: falta {CSV_HISTORICO}")
        return

    combinaciones = generar_recomendaciones(args.n)

    print(f"\n{'='*60}")
    print(f"  RECOMENDACIONES PARA PRÓXIMO SORTEO")
    print(f"  Generadas: {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*60}")
    for i, (comb, score) in enumerate(combinaciones, 1):
        nums_str = " ".join(f"{n:2d}" for n in comb)
        print(f"  #{i}  [{nums_str}]  score={score:.4f}")
    print(f"{'='*60}")

    guardar_resultado(combinaciones)


if __name__ == "__main__":
    main()