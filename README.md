# LEGO Bricks ML Vision

**LEGO Bricks ML Vision** es un paquete de Python diseñado para detectar y visualizar piezas de LEGO en imágenes usando YOLOv8 y pipelines personalizados. Este proyecto aborda el problema de identificar piezas individuales en entornos desordenados y proporciona herramientas de visualización para análisis y presentaciones profesionales.

---

## **Características Principales**

- **Detección de Objetos**: Identificación de piezas de LEGO utilizando YOLOv8.
- **Preprocesamiento de Datos**: Herramientas para redimensionar imágenes y asegurar consistencia en los nombres de archivos.
- **Conversión de Anotaciones**: Transformación de formatos de LabelMe a YOLO.
- **Visualización**: Generación de grids anotados, comparaciones y organización de carpetas para presentaciones.
- **Pipeline Modular**: Diseño adaptable para escalabilidad y personalización.

---

## **Requisitos del Sistema**

- **Versión de Python**: >= 3.8
- **Sistemas Operativos Compatibles**: Windows, macOS, Linux
- **Hardware Recomendado**:
  - CPU (mínimo)
  - GPU (opcional, recomendado para entrenamiento)

---

## **Instalación**

1. Instala el paquete directamente desde PyPI:

   ```bash
   pip install lego-bricks-ml-vision
   ```

2. Alternativamente, clona el repositorio:

   ```bash
   git clone https://github.com/MiguelDiLalla/LEGO_Bricks_ML_Vision.git
   cd LEGO_Bricks_ML_Vision
   pip install -e .
   ```

---

## **Uso**

### Configuración del Entorno

Ejecuta el siguiente script para preparar el entorno:

```python
from scripts.pipeline import setup_environment
setup_environment()
```

### Ejecución del Pipeline Principal

El pipeline principal realiza detección, preprocesamiento, entrenamiento y pruebas:

```bash
run-pipeline
```

### Generación de Visualizaciones

Crea visualizaciones profesionales para análisis:

```bash
run-visualize
```

---

## **Detalles de Scripts y Funciones**

### **Pipeline Principal (`pipeline.py`)**

- `setup_environment()`: Configura el entorno necesario.
- `download_dataset_from_kaggle()`: Descarga y extrae datasets desde Kaggle.
- `preprocess_images()`: Redimensiona imágenes.
- `labelme_to_yolo()`: Convierte anotaciones LabelMe a formato YOLO.
- `train_yolo_pipeline()`: Entrena un modelo YOLO con datos procesados.
- `test_model_on_real_images()`: Evalúa el modelo entrenado en imágenes reales.
- `visualize_results()`: Genera un grid de imágenes anotadas.

### **Visualización y Presentaciones (`visualize_presentation.py`)**

- `create_dataset_grid()`: Genera grids de imágenes del dataset.
- `annotate_model_results()`: Anota imágenes con resultados del modelo.
- `generate_comparison_grid()`: Crea comparaciones antes/después de la detección.
- `organize_presentation_folders()`: Organiza subcarpetas para presentaciones.

---

## **Ejemplos Prácticos**

### Preprocesamiento de Imágenes

```python
from scripts.pipeline import preprocess_images
preprocess_images("datasets/raw", "datasets/processed")
```

### Entrenamiento de Modelo YOLO

```python
from scripts.pipeline import train_yolo_pipeline
train_yolo_pipeline("datasets", annotations_format="YOLO", epochs=50, img_size=256)
```

### Generación de Grids

```python
from scripts.visualize_presentation import create_dataset_grid
create_dataset_grid("datasets/processed_images", "presentation/dataset_samples")
```

---

## **Estructura del Proyecto**

```
LEGO_Bricks_ML_Vision/
├── data/              # Datos crudos y procesados
├── scripts/           # Scripts principales
│   ├── pipeline.py    # Pipeline de entrenamiento
│   ├── visualize_presentation.py  # Visualización y presentación
├── presentation/      # Visualizaciones generadas
├── requirements.txt   # Dependencias del proyecto
├── setup.py           # Configuración del paquete
├── README.md          # Documentación del proyecto
```

---

## **Contribuciones**

Las contribuciones son bienvenidas. Sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una rama para tu contribución:

   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```

3. Envía un pull request explicando tus cambios.

---

## **Licencia**

Este proyecto está bajo la Licencia Apache 2.0.
