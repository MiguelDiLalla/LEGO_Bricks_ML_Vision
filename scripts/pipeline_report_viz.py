import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === Módulo 1: Extracción de Métricas ===
def load_metrics_from_yolo(results_path):
    """
    Carga métricas generadas por YOLO desde un archivo JSON.

    Parameters:
    - results_path (str): Ruta al archivo JSON con las métricas.

    Returns:
    - dict: Diccionario con las métricas cargadas.
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"[ERROR] No se encontró el archivo de métricas: {results_path}")

    with open(results_path, "r") as f:
        metrics = json.load(f)
    
    print(f"[INFO] Métricas cargadas desde {results_path}")
    return metrics

# === Módulo 2: Visualización de Métricas ===
def plot_training_metrics(metrics, output_dir):
    """
    Genera gráficos para métricas de entrenamiento como precisión, recall y pérdida.

    Parameters:
    - metrics (dict): Diccionario con métricas de entrenamiento.
    - output_dir (str): Carpeta donde se guardarán los gráficos.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gráfico de Precisión y Recall
    epochs = range(1, len(metrics["precision"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["precision"], label="Precision", marker="o")
    plt.plot(epochs, metrics["recall"], label="Recall", marker="o")
    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.title("Precisión y Recall por Época")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()

    # Gráfico de Pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["loss"], label="Loss", marker="o", color="red")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Curva de Pérdida por Época")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    print(f"[INFO] Gráficos generados en {output_dir}")

# === Módulo 3: Generación de Reporte ===
def generate_summary_report(metrics, output_dir):
    """
    Genera un resumen en CSV con las métricas principales por época.

    Parameters:
    - metrics (dict): Diccionario con métricas de entrenamiento.
    - output_dir (str): Carpeta donde se guardará el CSV resumen.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Crear un DataFrame para el resumen
    df = pd.DataFrame({
        "Epoch": range(1, len(metrics["precision"]) + 1),
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Loss": metrics["loss"]
    })

    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Resumen de métricas guardado en {csv_path}")

# === Ejecución Principal ===
def main():
    """
    Pipeline principal para cargar métricas, generar gráficos y guardar resúmenes.
    """
    # Configuración de rutas
    results_path = "results/metrics.json"  # Ruta al archivo de métricas
    output_dir = "results/visualizations"  # Carpeta para los gráficos y resúmenes

    try:
        # Cargar métricas
        metrics = load_metrics_from_yolo(results_path)

        # Generar visualizaciones
        plot_training_metrics(metrics, output_dir)

        # Generar resumen
        generate_summary_report(metrics, output_dir)

        print("[INFO] Pipeline de visualización completado exitosamente.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un problema: {e}")

if __name__ == "__main__":
    main()
