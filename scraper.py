"""
Scraper de resultados oficiales del Melate.
Versión optimizada: Descarga directa del CSV histórico oficial.
"""
import argparse
import io
from typing import List, Dict
import warnings

import requests
import pandas as pd

# Suprimir advertencias de SSL (necesario por certificados de sitios .gob.mx)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import CSV_HISTORICO, NUM_BOLAS_POR_SORTEO


# URL directa de descarga del histórico (encontrada vía Inspect)
URL_DESCARGA_CSV = "https://pronosticos.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUA"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def descargar_csv_oficial() -> pd.DataFrame:
    """Descarga el CSV histórico directamente del endpoint oficial."""
    print(f"[scraper] Descargando histórico desde endpoint oficial...")
    
    headers = {"User-Agent": USER_AGENT}
    try:
        # verify=False para evitar errores de certificado del servidor gubernamental
        response = requests.get(URL_DESCARGA_CSV, headers=headers, timeout=30, verify=False)
        response.raise_for_status()
        
        # Intentar leer el CSV. Los sitios .gob.mx a veces usan encoding 'latin-1' o 'utf-8-sig'
        try:
            df = pd.read_csv(io.BytesIO(response.content), encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(response.content), encoding='latin-1')
            
        print(f"[scraper] ¡Descarga exitosa! Filas crudas: {len(df)}")
        print(f"[scraper] Columnas encontradas: {list(df.columns)}")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"[scraper] Error al descargar: {e}")
        return pd.DataFrame()


def estandarizar_dataframe(df_crudo: pd.DataFrame) -> pd.DataFrame:
    """Limpia, renombra y normaliza el DataFrame al formato del proyecto."""
    if df_crudo.empty:
        return df_crudo

    # Mapeo exacto de las columnas del CSV de Pronósticos
    mapeo_columnas = {
        'FECHA': 'fecha',
        'R1': 'n1',
        'R2': 'n2',
        'R3': 'n3',
        'R4': 'n4',
        'R5': 'n5',
        'R6': 'n6',
        'R7': 'adicional',
        'BOLSA': 'bolsa',
        'CONCURSO': 'concurso',
        'NPRODUCTO': 'producto'
    }

    # Renombrar solo las columnas que existan
    columnas_existentes = {k: v for k, v in mapeo_columnas.items() if k in df_crudo.columns}
    df = df_crudo.rename(columns=columnas_existentes)

    # Verificar columnas requeridas
    cols_requeridas = ['fecha', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    for col in cols_requeridas:
        if col not in df.columns:
            print(f"[scraper] ⚠️ Error: No se encontró la columna '{col}'")
            print(f"[scraper] Columnas disponibles: {list(df.columns)}")
            return pd.DataFrame()

    # === DIAGNÓSTICO DE LA COLUMNA PRODUCTO ===
    if 'producto' in df.columns:
        print(f"\n[scraper] 🔍 Análisis de columna 'producto':")
        print(f"[scraper] Tipo de dato: {df['producto'].dtype}")
        print(f"[scraper] Primeros 10 valores únicos:")
        print(df['producto'].unique()[:10])
        
        # Intentar filtrar de forma más inteligente
        df['producto_str'] = df['producto'].astype(str).str.upper().str.strip()
        
        # Buscar cualquier valor que contenga "MELATE"
        melate_mask = df['producto_str'].str.contains('MELATE', na=False)
        if melate_mask.sum() > 0:
            print(f"[scraper] ✓ Encontrados {melate_mask.sum()} registros que contienen 'MELATE'")
            df = df[melate_mask].copy()
        else:
            print(f"[scraper] ⚠️ No se encontró 'MELATE' en los productos.")
            print(f"[scraper] 💡 Usando TODOS los registros (asumiendo que ya son solo de Melate)")
            # No filtramos, usamos todos los datos
    else:
        print(f"[scraper] ℹ️ No existe columna 'producto', usando todos los registros")

    # Limpiar columna temporal si existe
    if 'producto_str' in df.columns:
        df = df.drop(columns=['producto_str'])

    # Limpiar y convertir tipos de datos
    df['fecha'] = df['fecha'].astype(str).str.strip()
    df["fecha_dt"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
    
    # Eliminar filas sin fecha válida
    df = df.dropna(subset=["fecha_dt"])
    
    # Convertir columnas numéricas a enteros
    for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['n1', 'n2', 'n3', 'n4', 'n5', 'n6'])
    
    # Asegurar que los números estén ordenados de menor a mayor
    cols_num = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    df[cols_num] = df[cols_num].apply(lambda row: sorted(row), axis=1, result_type="broadcast")

    # Eliminar duplicados y ordenar por fecha
    df = df.drop_duplicates(subset=["fecha"]).sort_values("fecha_dt").reset_index(drop=True)
    
    # Seleccionar solo las columnas que necesitamos
    cols_finales = ['fecha', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    if 'adicional' in df.columns:
        cols_finales.append('adicional')
        
    return df[cols_finales]

def cargar_csv_existente() -> pd.DataFrame:
    """Carga el CSV histórico local si ya existe."""
    if CSV_HISTORICO.exists():
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
        print(f"[scraper] CSV local existente cargado: {len(df)} sorteos")
        return df
    return pd.DataFrame()


def guardar_csv(df: pd.DataFrame) -> None:
    """Guarda el DataFrame, asegurando que fecha_dt sea datetime."""
    # 1. Crear fecha_dt si no existe
    if "fecha_dt" not in df.columns and "fecha" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
    
    # 2. Asegurar que sea datetime (sin usar .dt.strftime para evitar errores)
    if "fecha_dt" in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df["fecha_dt"]):
                df["fecha_dt"] = pd.to_datetime(df["fecha_dt"], errors="coerce")
        except Exception:
            pass  # Si falla, que lo guarde como está
    
    # 3. Guardar (pandas guardará el datetime como string ISO automáticamente)
    df.to_csv(CSV_HISTORICO, index=False)
    print(f"[scraper] ✅ Guardado exitosamente: {len(df)} sorteos en {CSV_HISTORICO}")

def main():
    parser = argparse.ArgumentParser(description="Scraper de resultados del Melate")
    parser.add_argument("--force", action="store_true", help="Forzar descarga sin fusionar con local")
    args = parser.parse_args()

    print("="*50)
    print(" INICIANDO DESCARGA DE HISTÓRICO MELATE")
    print("="*50)
    
    df_crudo = descargar_csv_oficial()
    
    if df_crudo.empty:
        print("\n[scraper] ❌ No se pudieron descargar los datos.")
        print("[scraper] 💡 Alternativa: Descarga el CSV manualmente desde el navegador")
        print("[scraper] y guárdalo como 'melate_historico.csv' en la carpeta 'data/'.")
        return

    df_nuevo = estandarizar_dataframe(df_crudo)
    
    if df_nuevo.empty:
        print("\n[scraper] ❌ Error al procesar el CSV descargado.")
        print("[scraper] Revisa que el formato coincida con lo esperado.")
        return

    if args.force:
        df_final = df_nuevo
    else:
        # Fusionar con datos locales previos (por si tienes datos extra o corregidos)
        df_existente = cargar_csv_existente()
        if not df_existente.empty:
            df_combined = pd.concat([df_existente, df_nuevo], ignore_index=True)
            df_final = df_combined.drop_duplicates(subset=["fecha"]).sort_values("fecha_dt").reset_index(drop=True)
        else:
            df_final = df_nuevo

    guardar_csv(df_final)
    print(f"\n[scraper] 🎉 ¡Proceso completado! Total de sorteos en base de datos: {len(df_final)}")
    print("\nÚltimos 5 sorteos:")
    print(df_final[['fecha', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()