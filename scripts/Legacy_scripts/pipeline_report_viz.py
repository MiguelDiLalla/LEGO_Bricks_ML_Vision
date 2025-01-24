import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fpdf import FPDF

# === Módulo 1: Carga de Parámetros y Métricas ===
def load_args_and_metrics(args_path, metrics_path):
    """
    Carga los argumentos del entrenamiento y las métricas generadas.

    Parameters:
    - args_path (str): Ruta al archivo args.yaml.
    - metrics_path (str): Ruta al archivo JSON con las métricas.

    Returns:
    - dict: Diccionario con los argumentos del entrenamiento.
    - dict: Diccionario con las métricas cargadas.
    """
    import yaml

    if not os.path.exists(args_path):
        raise FileNotFoundError(f"[ERROR] No se encontró el archivo de parámetros: {args_path}")
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"[ERROR] No se encontró el archivo de métricas: {metrics_path}")

    with open(args_path, "r") as f:
        args = yaml.safe_load(f)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    print("[INFO] Parámetros y métricas cargados correctamente.")
    return args, metrics

# === Módulo 2: Visualización de Gráficos ===
def plot_precision_recall_curve(metrics, output_dir):
    """
    Genera la curva de Precision-Recall.

    Parameters:
    - metrics (dict): Diccionario con métricas de entrenamiento.
    - output_dir (str): Carpeta donde se guardarán los gráficos.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(metrics['recall'], metrics['precision'], label=f"mAP@0.5: {metrics['map']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    output_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Curva Precision-Recall guardada en {output_path}")

# === Módulo 3: Generación de Reporte PDF ===
def generate_pdf_report(args, metrics, output_dir, images_dir):
    """
    Genera un reporte en formato PDF.

    Parameters:
    - args (dict): Parámetros del entrenamiento.
    - metrics (dict): Diccionario con métricas de entrenamiento.
    - output_dir (str): Carpeta donde se guardará el PDF.
    - images_dir (str): Carpeta donde se encuentran los gráficos generados.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Portada
    pdf.add_page()
    pdf.set_font("Arial", size=20)
    pdf.cell(200, 10, txt="Reporte de Entrenamiento", ln=True, align='C')

    # Sección de Parámetros
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Parámetros de Entrenamiento:", ln=True)
    for key, value in args.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)

    # Sección de Métricas
    pdf.add_page()
    pdf.cell(0, 10, "Métricas de Entrenamiento:", ln=True)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            pdf.cell(0, 10, f"{key}: {value:.3f}", ln=True)

    # Gráficos
    pdf.add_page()
    pdf.cell(0, 10, "Gráficos Generados:", ln=True)
    for image in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image)
        if os.path.isfile(image_path) and image.endswith(".png"):
            pdf.image(image_path, x=10, w=190)

    # Guardar PDF
    pdf_path = os.path.join(output_dir, "training_report.pdf")
    pdf.output(pdf_path)
    print(f"[INFO] Reporte PDF generado en {pdf_path}")

# === Pipeline Principal ===
def main():
    """
    Pipeline principal para cargar métricas, generar gráficos y guardar reportes.
    """
    # Configuración de rutas
    args_path = "args.yaml"
    metrics_path = "results/metrics.json"
    output_dir = "results/report"
    images_dir = "results/visualizations"

    try:
        # Cargar parámetros y métricas
        args, metrics = load_args_and_metrics(args_path, metrics_path)

        # Generar gráficos
        plot_precision_recall_curve(metrics, images_dir)

        # Generar reporte PDF
        generate_pdf_report(args, metrics, output_dir, images_dir)

        print("[INFO] Pipeline completado exitosamente.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un problema: {e}")

if __name__ == "__main__":
    main()
