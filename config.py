"""
Configuración central del proyecto Melate Forecaster.
"""
from pathlib import Path
from datetime import datetime

# === Rutas ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Crear carpetas automáticamente si no existen
for d in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Archivos ===
CSV_HISTORICO = DATA_DIR / "melate_historico.csv"
MODELO_XGB = MODELS_DIR / "modelo_xgb.joblib"
RESULTADO_PREDICCION = OUTPUT_DIR / f"prediccion_{datetime.now():%Y%m%d_%H%M}.csv"

# === Parámetros del Melate ===
NUM_BOLAS_POR_SORTEO = 6
RANGO_NUMEROS = 56  # 1 a 56
COSTO_BOLETO = 30   # MXN

# === Parámetros del modelo ===
TRAIN_TEST_SPLIT = 0.8
MIN_SORTEOS_PARA_MODELO = 500

# === Scraping ===
URL_PRONOSTICOS = "https://pronosticos.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUA"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# === Generador ===
N_COMBINACIONES_RECOMENDADAS = 5
RANGO_SUMA_MIN = 140
RANGO_SUMA_MAX = 220
PARES_MIN = 2
PARES_MAX = 4

# === Backtest ===
SORTEOS_BACKTEST = 200
MONTO_INICIAL = 10000