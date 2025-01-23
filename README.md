# LEGO Bricks ML Vision: Documentación

Este proyecto se centra en la detección y clasificación de piezas de LEGO mediante pipelines modulares optimizados para entrenar modelos YOLO. La versión actual soporta los scripts principales `pipeline_setup.py` y `pipeline_train.py`.

## **Características Principales**

- **Pipeline de Preprocesamiento (`pipeline_setup.py`)**:
  - Configuración del entorno para ejecución en Kaggle, Colab o local.
  - Creación de estructuras de carpetas para preparar datasets de entrenamiento.
  - Aumentación de datos usando Albumentations.
  - Generación de archivos `dataset.yaml` compatibles con YOLO.

- **Pipeline de Entrenamiento (`pipeline_train.py`)**:
  - Entrenamiento de modelos YOLO con hiperparámetros predefinidos.
  - Optimización automática de hiperparámetros usando Optuna.
  - Integración de visualización de progreso durante el entrenamiento.

## **Requisitos del Sistema**

- **Python**: >= 3.8
- **Dependencias Clave**:
  ```
  torch==2.4.1+cpu
  ultralytics==8.2.99
  pillow==11.1.0
  matplotlib==3.8.4
  kaggle==1.6.17
  albumentations==1.3.0
  ```

Instala las dependencias desde `requirements.txt`:
```bash
pip install -r requirements.txt
```

## **Uso del Proyecto**

### 1. Configuración del Entorno
Ejecuta `pipeline_setup.py` para preparar tu dataset:
```bash
python pipeline_setup.py
```
- Esto descargará y estructurará el dataset.
- Generará aumentaciones y un archivo `dataset.yaml` compatible con YOLO.

### 2. Entrenamiento del Modelo
Ejecuta `pipeline_train.py` para entrenar el modelo YOLO:
```bash
python pipeline_train.py
```
Por defecto, entrena con hiperparámetros definidos. Para optimización automática con Optuna:
```bash
python pipeline_train.py --optuna-mode
```

## **Ejemplo de Docker**
El pipeline puede ejecutarse en un contenedor Docker. Usa la siguiente plantilla para crear tu imagen Docker:

**Dockerfile**:
```Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "pipeline_train.py"]
```

Construcción y ejecución:
```bash
docker build -t lego-ml .
docker run --rm -v $(pwd):/app lego-ml
```

## **Estructura del Proyecto**
```plaintext
LEGO_Bricks_ML_Vision/
├── scripts/
│   ├── pipeline_setup.py    # Preprocesamiento del dataset
│   ├── pipeline_train.py    # Entrenamiento del modelo YOLO
├── requirements.txt         # Dependencias del proyecto
├── Dockerfile               # Imagen Docker opcional
├── README.md                # Documentación del proyecto
└── results/                 # Carpeta para resultados y visualizaciones
```

## **Contribuciones**
¡Las contribuciones son bienvenidas! Sigue estos pasos:
1. Realiza un fork del repositorio.
2. Crea una rama para tu funcionalidad:
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. Envía un pull request con tus cambios.

## **Próximos Pasos**
- Implementar un pipeline de evaluación y visualización avanzado.
- Optimizar compatibilidad y eficiencia para Docker.
- Explorar estrategias de fine-tuning con más clases de piezas LEGO.

## **Licencia**
Este proyecto está licenciado bajo Apache License 2.0.
