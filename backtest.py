"""
Backtest: simula ganancias históricas con diferentes estrategias.
"""
import argparse
import joblib
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    CSV_HISTORICO, MODELO_XGB, OUTPUT_DIR, COSTO_BOLETO,
    RANGO_NUMEROS, NUM_BOLAS_POR_SORTEO, SORTEOS_BACKTEST, MONTO_INICIAL
)

PREMIOS = {
    6: 50_000_000,
    5: 50_000,
    4: 600,
    3: 30,
}


def cargar_datos() -> pd.DataFrame:
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
    return df


def cargar_modelo():
    if not MODELO_XGB.exists():
        return None, None
    data = joblib.load(MODELO_XGB)
    return data["modelos"], data["feature_cols"]


def calcular_premio(aciertos: int) -> float:
    if aciertos >= 6:
        return PREMIOS[6]
    elif aciertos == 5:
        return PREMIOS[5]
    elif aciertos == 4:
        return PREMIOS[4]
    elif aciertos == 3:
        return PREMIOS[3]
    return 0.0


def simular_estrategia(df: pd.DataFrame, estrategia: str,
                       n_sorteos: int, modelos=None, feature_cols=None,
                       semilla: int = 42) -> Dict:
    """Simula una estrategia durante n_sorteos."""
    rng = np.random.default_rng(semilla)
    nums = np.arange(1, RANGO_NUMEROS + 1)
    cols_num = [f"n{i}" for i in range(1, NUM_BOLAS_POR_SORTEO + 1)]

    df_test = df.tail(n_sorteos).reset_index(drop=True)

    probs_serie = None
    if estrategia == "modelo" and modelos is not None:
        from modelo import crear_features, crear_targets
        df_full = crear_targets(df.copy())
        df_full = crear_features(df_full)
        df_full = df_full.dropna(subset=feature_cols)
        probs_serie = []
        X = df_full[feature_cols].values
        for i in range(RANGO_NUMEROS):
            target = f"y_{i+1}"
            probs_serie.append(modelos[target].predict_proba(X)[:, 1])
        probs_serie = np.array(probs_serie).T

    if estrategia == "frecuencia":
        nums_all = df[cols_num].values.flatten()
        freq, _ = np.histogram(nums_all, bins=range(1, RANGO_NUMEROS + 2))
        probs_freq = freq / freq.sum()

    dinero = MONTO_INICIAL
    historial = []
    aciertos_total = {k: 0 for k in [3, 4, 5, 6]}

    for t in range(len(df_test)):
        row = df_test.iloc[t]
        reales = set(int(row[f"n{i}"]) for i in range(1, 7))

        if estrategia == "aleatorio":
            pred = set(rng.choice(nums, size=NUM_BOLAS_POR_SORTEO, replace=False))
        elif estrategia == "frecuencia":
            pred = set(rng.choice(nums, size=NUM_BOLAS_POR_SORTEO,
                                  replace=False, p=probs_freq))
        elif estrategia == "modelo" and probs_serie is not None:
            p = probs_serie[t]
            p = p / p.sum()
            pred = set(rng.choice(nums, size=NUM_BOLAS_POR_SORTEO, replace=False, p=p))
        else:
            pred = set(rng.choice(nums, size=NUM_BOLAS_POR_SORTEO, replace=False))

        aciertos = len(reales & pred)
        premio = calcular_premio(aciertos)
        dinero += premio - COSTO_BOLETO

        if aciertos >= 3:
            aciertos_total[aciertos] += 1

        historial.append({
            "sorteo": t + 1,
            "fecha": row["fecha_dt"],
            "aciertos": aciertos,
            "premio": premio,
            "costo": COSTO_BOLETO,
            "dinero": dinero,
        })

    df_hist = pd.DataFrame(historial)
    return {
        "historial": df_hist,
        "aciertos": aciertos_total,
        "dinero_final": dinero,
        "roi": (dinero - MONTO_INICIAL) / MONTO_INICIAL * 100,
        "estrategia": estrategia,
    }


def comparar_estrategias(df: pd.DataFrame, n_sorteos: int,
                         n_simulaciones: int = 5) -> None:
    """Compara las 3 estrategias."""
    modelos, feature_cols = cargar_modelo()

    estrategias = ["aleatorio", "frecuencia"]
    if modelos is not None:
        estrategias.insert(0, "modelo")

    resultados = {}
    for estr in estrategias:
        rois = []
        aciertos_acum = {k: 0 for k in [3, 4, 5, 6]}
        for sim in range(n_simulaciones):
            r = simular_estrategia(df, estr, n_sorteos, modelos, feature_cols,
                                   semilla=sim)
            rois.append(r["roi"])
            for k in [3, 4, 5, 6]:
                aciertos_acum[k] += r["aciertos"][k]

        resultados[estr] = {
            "roi_promedio": np.mean(rois),
            "roi_std": np.std(rois),
            "aciertos": {k: v // n_simulaciones for k, v in aciertos_acum.items()},
        }

    print(f"\n{'='*70}")
    print(f"  BACKTEST: {n_sorteos} sorteos × {n_simulaciones} simulaciones")
    print(f"{'='*70}")
    print(f"{'Estrategia':<15} {'ROI promedio':<15} {'ROI std':<10} "
          f"{'3 aciertos':<12} {'4 aciertos':<12} {'5 aciertos':<12}")
    print(f"{'-'*70}")
    for estr, r in resultados.items():
        print(f"{estr:<15} {r['roi_promedio']:>+13.2f}% {r['roi_std']:>8.2f}% "
              f"{r['aciertos'][3]:>10} {r['aciertos'][4]:>10} {r['aciertos'][5]:>10}")
    print(f"{'='*70}")

    plt.figure(figsize=(12, 6))
    for estr in estrategias:
        r = simular_estrategia(df, estr, n_sorteos, modelos, feature_cols)
        plt.plot(r["historial"]["sorteo"], r["historial"]["dinero"],
                 label=estr, linewidth=2)

    plt.axhline(MONTO_INICIAL, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Sorteo")
    plt.ylabel("Dinero (MXN)")
    plt.title(f"Backtest: evolución del capital ({n_sorteos} sorteos)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "backtest_comparacion.png"
    plt.savefig(ruta, dpi=120)
    print(f"\n[backtest] Gráfico guardado en: {ruta}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=SORTEOS_BACKTEST)
    parser.add_argument("--sims", type=int, default=5)
    args = parser.parse_args()

    if not CSV_HISTORICO.exists():
        print(f"[backtest] Error: falta {CSV_HISTORICO}")
        return

    df = cargar_datos()
    if len(df) < args.n:
        args.n = len(df)

    comparar_estrategias(df, args.n, args.sims)


if __name__ == "__main__":
    main()