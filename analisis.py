"""
Análisis estadístico del histórico del Melate.
Detecta sesgos físicos, patrones temporales y valida aleatoriedad.
"""
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import CSV_HISTORICO, OUTPUT_DIR, RANGO_NUMEROS, NUM_BOLAS_POR_SORTEO


def cargar_datos() -> pd.DataFrame:
    if not CSV_HISTORICO.exists():
        raise FileNotFoundError(f"Ejecuta scraper.py primero. Falta: {CSV_HISTORICO}")
    
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
    
    # Verificación final
    if not pd.api.types.is_datetime64_any_dtype(df["fecha_dt"]):
        df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
    
    df = df.dropna(subset=["fecha_dt"])
    print(f"[analisis] Cargados {len(df)} sorteos")
    return df


def aplanar_numeros(df: pd.DataFrame) -> np.ndarray:
    """Convierte las 6 columnas n1-n6 en un array 1D."""
    cols = [f"n{i}" for i in range(1, NUM_BOLAS_POR_SORTEO + 1)]
    return df[cols].values.flatten()


def test_uniformidad_chi2(df: pd.DataFrame) -> Tuple[float, float]:
    """Prueba χ² de uniformidad: ¿cada número sale con la misma frecuencia?"""
    nums = aplanar_numeros(df)
    freq, _ = np.histogram(nums, bins=range(1, RANGO_NUMEROS + 2))
    esperado = np.full(RANGO_NUMEROS, len(nums) / RANGO_NUMEROS)
    chi2, p = stats.chisquare(freq, f_exp=esperado)

    print(f"\n=== Test χ² de Uniformidad ===")
    print(f"Chi² = {chi2:.2f}, p-valor = {p:.4f}")
    if p < 0.05:
        print("⚠️  RECHAZA H0: hay sesgo significativo")
    else:
        print("✓ No se rechaza H0: máquina parece justa")
    return chi2, p


def test_runs(df: pd.DataFrame) -> Tuple[float, float]:
    """Test de rachas: ¿los números altos/bajos aparecen aleatoriamente?"""
    nums = aplanar_numeros(df)
    mediana = np.median(nums)
    binario = (nums > mediana).astype(int)

    n1 = binario.sum()
    n0 = len(binario) - n1
    rachas = 1 + sum(binario[i] != binario[i-1] for i in range(1, len(binario)))

    mu = (2 * n0 * n1) / (n0 + n1) + 1
    sigma = np.sqrt((2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) /
                    ((n0 + n1)**2 * (n0 + n1 - 1)))
    z = (rachas - mu) / sigma if sigma > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"\n=== Test de Rachas ===")
    print(f"Rachas observadas: {rachas}, esperadas: {mu:.1f}")
    print(f"Z = {z:.2f}, p-valor = {p:.4f}")
    return z, p


def test_paridad(df: pd.DataFrame) -> Tuple[float, float]:
    """¿La proporción par/impar es 50/50?"""
    nums = aplanar_numeros(df)
    pares = (nums % 2 == 0).sum()
    impares = len(nums) - pares
    chi2, p = stats.chisquare([pares, impares], f_exp=[len(nums)/2, len(nums)/2])

    print(f"\n=== Test de Paridad ===")
    print(f"Pares: {pares}, Impares: {impares}")
    print(f"Chi² = {chi2:.2f}, p-valor = {p:.4f}")
    return chi2, p


def analisis_frecuencia(df: pd.DataFrame) -> pd.DataFrame:
    """Frecuencia de cada número, ordenada."""
    nums = aplanar_numeros(df)
    freq, _ = np.histogram(nums, bins=range(1, RANGO_NUMEROS + 2))
    df_freq = pd.DataFrame({
        "numero": range(1, RANGO_NUMEROS + 1),
        "frecuencia": freq,
        "porcentaje": freq / len(nums) * 100,
    }).sort_values("frecuencia", ascending=False)

    print(f"\n=== Top 10 más frecuentes ===")
    print(df_freq.head(10).to_string(index=False))
    return df_freq


def analisis_temporal(df: pd.DataFrame) -> None:
    """Distribución de la suma por día de la semana."""
    cols = [f"n{i}" for i in range(1, 7)]
    df["suma"] = df[cols].sum(axis=1)
    df["dia_semana"] = df["fecha_dt"].dt.day_name()

    print(f"\n=== Suma promedio por día ===")
    print(df.groupby("dia_semana")["suma"].agg(["mean", "std", "count"]))


def analisis_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis de gaps: ¿cuántos sorteos pasa un número sin salir?"""
    cols = [f"n{i}" for i in range(1, 7)]
    gaps = {n: [] for n in range(1, RANGO_NUMEROS + 1)}

    for n in range(1, RANGO_NUMEROS + 1):
        aparece = df[cols].eq(n).any(axis=1).values
        ultimo = -1
        for t, ok in enumerate(aparece):
            if ok:
                if ultimo >= 0:
                    gaps[n].append(t - ultimo - 1)
                ultimo = t

    todos_gaps = []
    for n, g in gaps.items():
        todos_gaps.extend(g)

    todos_gaps = np.array(todos_gaps)
    if len(todos_gaps) == 0:
        return pd.DataFrame()

    print(f"\n=== Análisis de Gaps ===")
    print(f"Gap promedio: {todos_gaps.mean():.2f} sorteos")
    print(f"Gap máximo: {todos_gaps.max()}")
    return pd.DataFrame({"gap": todos_gaps})


def generar_reporte_grafico(df: pd.DataFrame) -> None:
    """Genera reporte visual completo."""
    nums = aplanar_numeros(df)
    cols = [f"n{i}" for i in range(1, 7)]
    df["suma"] = df[cols].sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Frecuencia por número
    freq, _ = np.histogram(nums, bins=range(1, RANGO_NUMEROS + 2))
    axes[0, 0].bar(range(1, RANGO_NUMEROS + 1), freq, color="steelblue")
    axes[0, 0].axhline(freq.mean(), color="red", linestyle="--", label="Esperado")
    axes[0, 0].set_title("Frecuencia por número")
    axes[0, 0].legend()

    # 2. Distribución de la suma
    axes[0, 1].hist(df["suma"], bins=30, color="coral", edgecolor="black")
    axes[0, 1].set_title("Distribución de la suma")

    # 3. Heatmap de co-ocurrencias
    pares = np.zeros((RANGO_NUMEROS, RANGO_NUMEROS))
    for _, row in df.iterrows():
        nums_sorteo = sorted([row[f"n{i}"] for i in range(1, 7)])
        for i in range(len(nums_sorteo)):
            for j in range(i + 1, len(nums_sorteo)):
                a, b = nums_sorteo[i] - 1, nums_sorteo[j] - 1
                pares[a, b] += 1
                pares[b, a] += 1
    sns.heatmap(pares, ax=axes[1, 0], cmap="YlOrRd")
    axes[1, 0].set_title("Co-ocurrencia entre números")

    # 4. Serie temporal de sumas
    axes[1, 1].plot(df["fecha_dt"], df["suma"], marker=".", alpha=0.6)
    axes[1, 1].axhline(df["suma"].mean(), color="red", linestyle="--")
    axes[1, 1].set_title("Suma por sorteo")

    plt.tight_layout()
    ruta = OUTPUT_DIR / "reporte_analisis.png"
    plt.savefig(ruta, dpi=120)
    print(f"\n[analisis] Reporte guardado en: {ruta}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    df = cargar_datos()
    test_uniformidad_chi2(df)
    test_runs(df)
    test_paridad(df)
    analisis_frecuencia(df)
    analisis_temporal(df)
    analisis_gaps(df)

    if args.plot:
        generar_reporte_grafico(df)


if __name__ == "__main__":
    main()