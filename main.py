"""
MELATE FORECASTER - Panel de Control
"""
import sys
import os
import time
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich import box

# FIX WINDOWS: Forzar UTF-8 en todos los subprocess
os.environ["PYTHONIOENCODING"] = "utf-8"

console = Console()
error_console = Console(stderr=True, style="bold red")

SCRIPTS = {
    "datos":    {"archivo": "scraper.py",  "args": "",          "icono": "📥", "nombre": "Descargar datos"},
    "analizar": {"archivo": "analisis.py", "args": "--plot",    "icono": "🔍", "nombre": "Análisis estadístico"},
    "entrenar": {"archivo": "modelo.py",   "args": "",          "icono": "🧠", "nombre": "Entrenar modelo"},
    "predecir": {"archivo": "generador.py","args": "-n 5",      "icono": "🎯", "nombre": "Generar recomendaciones"},
    "simular":  {"archivo": "backtest.py", "args": "-n 200",    "icono": "📈", "nombre": "Backtest"},
    "app":      {"archivo": "app.py",      "args": "streamlit", "icono": "", "nombre": "App web"},
}

FLUJO_COMPLETO = ["datos", "analizar", "entrenar", "predecir", "simular"]

def banner():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]  MELATE FORECASTER  🎰[/bold cyan]\n"
        "[dim]Sistema de Análisis y Predicción[/dim]",
        border_style="cyan",
        padding=(1, 3),
    ))
    console.print()

def mostrar_menu():
    tabla = Table(title="Selecciona una opción", box=box.ROUNDED, border_style="blue", title_style="bold white", show_lines=True)
    tabla.add_column("Opción", style="bold cyan", justify="center")
    tabla.add_column("Acción", style="white")
    tabla.add_column("Descripción", style="dim")
    tabla.add_row("1", "  Descargar datos", "Baja histórico oficial")
    tabla.add_row("2", "🔍  Analizar",         "Tests estadísticos")
    tabla.add_row("3", "🧠  Entrenar modelo",  "XGBoost con +4,000 sorteos")
    tabla.add_row("4", "🎯  Predecir",         "Genera 5 combinaciones")
    tabla.add_row("5", "📈  Simular (backtest)","Compara estrategias")
    tabla.add_row("6", "🚀  Ejecutar TODO",    "Flujo completo en orden")
    tabla.add_row("7", "📱  Abrir App Web",    "Dashboard Streamlit")
    tabla.add_row("0", "❌  Salir",            "Cerrar programa")
    console.print(tabla)
    console.print()

def ejecutar_script(clave: str) -> bool:
    info = SCRIPTS[clave]
    archivo = info["archivo"]
    args = info["args"]
    icono = info["icono"]
    nombre = info["nombre"]

    console.print(Panel(f"[bold]{icono}  {nombre}[/bold]\n[dim]Ejecutando: {archivo} {args}[/dim]", border_style="cyan"))

    if clave == "app":
        console.print("[yellow] Abriendo app en http://localhost:8501 ...[/yellow]")
        console.print("[dim]Presiona Ctrl+C para cerrar[/dim]\n")
        subprocess.run([sys.executable, "-m", "streamlit", "run", archivo])
        return True

    if not Path(archivo).exists():
        error_console.print(f"  ❌ No se encontró: {archivo}")
        return False

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task(f"[cyan]{nombre}...", total=None)
        try:
            # FIX: Forzar UTF-8 y reemplazar caracteres problemáticos
            resultado = subprocess.run(
                [sys.executable, archivo] + args.split(),
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=600,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}
            )
            progress.update(task, completed=100)
            if resultado.returncode == 0:
                console.print(f"[green]  ✅ {nombre} completado[/green]")
                if resultado.stdout:
                    for linea in resultado.stdout.strip().split("\n")[-5:]:
                        console.print(f"     [dim]{linea}[/dim]")
                return True
            else:
                console.print(f"[red]  ❌ {nombre} falló[/red]")
                if resultado.stderr:
                    for linea in resultado.stderr.strip().split("\n")[-5:]:
                        console.print(f"     [red]{linea}[/red]")
                return False
        except subprocess.TimeoutExpired:
            console.print(f"[red]  ⏰ {nombre} tardó demasiado (>10 min)[/red]")
            return False
        except Exception as e:
            console.print(f"[red]  💥 Error: {e}[/red]")
            return False

def ejecutar_flujo_completo():
    console.print(Panel("[bold magenta]🚀  FLUJO COMPLETO  🚀[/bold magenta]\nSe ejecutarán todos los pasos en orden.", border_style="magenta"))
    resultados = {}
    for i, clave in enumerate(FLUJO_COMPLETO, 1):
        info = SCRIPTS[clave]
        console.print(f"\n[bold]Paso {i}/{len(FLUJO_COMPLETO)}: {info['icono']} {info['nombre']}[/bold]")
        resultados[clave] = ejecutar_script(clave)
        if i < len(FLUJO_COMPLETO):
            console.print("[dim]   Esperando 2 segundos...[/dim]")
            time.sleep(2)

    console.print("\n" + "="*60)
    tabla_resumen = Table(title="Resumen del flujo", box=box.ROUNDED)
    tabla_resumen.add_column("Paso", style="bold")
    tabla_resumen.add_column("Resultado", justify="center")
    for clave in FLUJO_COMPLETO:
        info = SCRIPTS[clave]
        estado = "[green]✅ OK[/green]" if resultados[clave] else "[red]❌ Falló[/red]"
        tabla_resumen.add_row(f"{info['icono']} {info['nombre']}", estado)
    console.print(tabla_resumen)
    exitosos = sum(1 for v in resultados.values() if v)
    console.print(f"\n[bold]Resultado: {exitosos}/{len(resultados)} pasos exitosos[/bold]")
    if exitosos == len(resultados):
        console.print("[bold green] ¡Todo completado! Abre la app con: python main.py --app[/bold green]")

def menu_interactivo():
    while True:
        banner()
        mostrar_menu()
        opcion = Prompt.ask("[bold cyan]Tu opción[/bold cyan]", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="0")
        console.print()
        if opcion == "0":
            console.print("[bold] ¡Hasta luego![/bold]")
            break
        elif opcion == "1": ejecutar_script("datos")
        elif opcion == "2": ejecutar_script("analizar")
        elif opcion == "3": ejecutar_script("entrenar")
        elif opcion == "4": ejecutar_script("predecir")
        elif opcion == "5": ejecutar_script("simular")
        elif opcion == "6":
            if Confirm.ask("¿Ejecutar TODO el flujo?"): ejecutar_flujo_completo()
        elif opcion == "7": ejecutar_script("app")
        console.print()
        Prompt.ask("[dim]Presiona Enter para continuar[/dim]")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Melate Forecaster - Panel de Control")
    parser.add_argument("--todo", action="store_true")
    parser.add_argument("--datos", action="store_true")
    parser.add_argument("--analizar", action="store_true")
    parser.add_argument("--entrenar", action="store_true")
    parser.add_argument("--predecir", action="store_true")
    parser.add_argument("--simular", action="store_true")
    parser.add_argument("--app", action="store_true")
    args = parser.parse_args()
    banner()
    if not any(vars(args).values()):
        menu_interactivo()
        return
    if args.todo: ejecutar_flujo_completo()
    elif args.datos: ejecutar_script("datos")
    elif args.analizar: ejecutar_script("analizar")
    elif args.entrenar: ejecutar_script("entrenar")
    elif args.predecir: ejecutar_script("predecir")
    elif args.simular: ejecutar_script("simular")
    elif args.app: ejecutar_script("app")

if __name__ == "__main__":
    main()