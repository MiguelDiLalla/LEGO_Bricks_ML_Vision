![Hero Banner](https://migueldilalla.github.io/assets/branding-elements/brickssifier-herobanner.jpg)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org) [![YOLOv8](https://img.shields.io/badge/YOLOv8-8.1+-00FFFF.svg)](https://github.com/ultralytics/ultralytics) [![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-red.svg)](https://opencv.org) [![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io) [![Albumentations](https://img.shields.io/badge/Albumentations-2.0+-brightgreen.svg)](https://albumentations.ai) [![Rich](https://img.shields.io/badge/Rich-13.0+-9933CC.svg)](https://github.com/Textualize/rich) [![Click](https://img.shields.io/badge/Click-8.1+-yellow.svg)](https://click.palletsprojects.com)

</div>

# 🧱 Proyecto: Brickssifier_Studwise


> *La IA está aprendiendo de nosotros. ¿Pero qué le estamos enseñando?*

## 📌 La Pregunta:

> *"Si puedo reconocer la mayoría de las piezas LEGO de un vistazo, ¿puedo enseñar a un algoritmo de visión por computadora a hacer lo mismo?"*

Este proyecto es un intento compacto pero ambicioso de responder a esa pregunta — construyendo un pipeline real de aprendizaje automático que clasifica piezas LEGO basándose en imágenes tomadas desde arriba. Integra dos modelos YOLOv8 ajustados y un algoritmo de post-procesamiento geométrico para la inferencia de dimensiones.

Mientras que sistemas comerciales potentes como Brickognize o BrickIt logran resultados impresionantes con miles de piezas, también operan a una escala respaldada por equipos, servidores y conjuntos de datos de órdenes de magnitud mayores que los míos:

### 🪄 Sistemas Comerciales vs. Proyectos Personales

* **Brickognize** (Tramacsoft) utiliza Mask R-CNN con generación sintética de imágenes para alimentar una herramienta comercial que identifica más de 85,000 piezas y sets de LEGO.

* **BrickIt** (equipo startup) se basa en una CNN optimizada para móviles para la detección en tiempo real de piezas en montones, proporcionando retroalimentación visual inmediata en una pulida aplicación móvil.

En contraste, Brickssifier es un ejercicio personal de ingeniería: construido desde cero, entrenado con ~2000 imágenes y desplegado con curiosidad y cuidado.

---

## 🧪 Mi Enfoque: Cuando el ML se Encuentra con la Geometría

Después de fallar en entrenar un clasificador multiclase confiable, reduje mi objetivo a un **conjunto reducido de 14 clases básicas de piezas**, y reestructuré la tarea en un pipeline de pasos modulares.

### 📊 Clases de Piezas Soportadas

El sistema actualmente clasifica las siguientes dimensiones de piezas LEGO:

```python
STUDS_TO_DIMENSIONS_MAP = {
        1: "1x1",
        2: "2x1",
        3: "3x1",
        4: ["2x2", "4x1"],
        6: ["3x2", "6x1"],
        8: ["4x2", "8x1"],
        10: "10x1",
        12: ["6x2", "12x1"],
        16: ["4x4", "8x2"]
    }
```

Esta asignación muestra cómo el conteo de studs por sí solo no es suficiente para la clasificación (por ejemplo, 4 studs podrían ser 1x4, 2x2, o 4x1), lo que hace necesario el enfoque geométrico.

### 🚀 Pipeline de Clasificación de Dimensiones de Piezas
```python
1. Detectar piezas LEGO en una imagen con YOLOv8 (modelo 1)
2. Para cada caja detectada:
   a. Recortar la región de la imagen original (array NumPy)
   b. Detectar studs dentro del recorte usando YOLOv8 (modelo 2)
   c. Extraer coordenadas centrales (x, y) de todos los studs
   d. Pasar los puntos a un algoritmo geométrico basado en regresión
   e. Predecir la dimensión vista desde arriba (ej., 2x4 o 1x8)
```
Esta estrategia híbrida — combinando modelos de detección y geometría clásica — permite desambiguar tipos de piezas que comparten el mismo conteo de studs.

---

## 🧱 Dataset 1: Detección de Piezas (Crudo y Etiquetado)

Este dataset contiene más de 2000 imágenes de piezas LEGO individuales capturadas bajo iluminación natural con fondos variados.

- 📸 Anotado usando **LabelMe**
- 🔁 Convertido al **formato YOLO** utilizando scripts personalizados
- 🧩 Contiene 14 clases base (ej., 2x2, 1x4, 2x4...)

### 📦 Disponibilidad del Dataset

El dataset completo anotado está disponible para descarga y uso en tus propios proyectos.

> 🧷 **[Descargar Dataset desde Kaggle](https://www.kaggle.com/datasets/migueldilalla/spiled-lego-bricks)**

### 📷 Ejemplo en Cuadrícula (Imágenes Anotadas)
![Ejemplos de Piezas Etiquetadas](https://migueldilalla.github.io/assets/own-projects-resources/readme_dataset_samples/bricks/output_grid_7.webp)

---

## 🔍 Dataset 2: Detección de Studs (Entradas Recortadas)

Cada imagen de pieza recortada es reetiquetada con las **posiciones visibles de los studs**:
- Puntos clave marcados manualmente
- Transformados en **cajas delimitadoras**
- Convertidos al **formato de puntos clave YOLO** usando scripts auxiliares

> 🔧 Herramientas de conversión disponibles pronto.

> 📦 **[Descargar Dataset de Studs desde Kaggle](https://www.kaggle.com/datasets/migueldilalla/labeledstuds-lego-bricks)**

### 📷 Ejemplo en Cuadrícula (Pieza Recortada + Anotaciones de Studs)
> *(Las imágenes se cargarán desde la carpeta `assets/dataset_studs/`)*


![Ejemplos de Studs Etiquetados](https://migueldilalla.github.io//assets/own-projects-resources/readme_dataset_samples/studs/output_grid_10.webp)


---
## 🔍 Recortador de Piezas de Clase Única

Este modelo fue diseñado para manejar imágenes con piezas apiladas. Esto significa que el modelo debería poder detectar piezas en un montón. El modelo fue entrenado con un dataset de más de 2000 imágenes de piezas LEGO individuales capturadas bajo iluminación natural con fondos variados.

![Ejemplo de inferencia del modelo de piezas](https://migueldilalla.github.io/assets/own-projects-resources/readme_model_samples/readme-model1-display.webp)

## 📐 Clasificador de Geometría de Studs: Resumen del Algoritmo

Cuando un conteo de studs corresponde a múltiples dimensiones posibles, se utiliza lógica espacial para desambiguar el tipo de pieza.

### 🎓 Análisis de Patrones Basado en Regresión

- Extraer puntos centrales de los studs: `(x1, y1), (x2, y2), ...`
- Ajustar una línea de regresión lineal a los puntos
- Medir la desviación de cada punto respecto a la línea
- Si la desviación < umbral → studs alineados → forma lineal (ej., 1x8)
- Si no → patrón de cuadrícula → forma 2D (ej., 2x4)

> ![Ejemplo de inferencia del modelo de studs y Clasificación de pieza 2x2](https://migueldilalla.github.io/assets/own-projects-resources/readme_model_samples/readme-model2_2x2-display.webp)

> ![Ejemplo de inferencia del modelo de studs y Clasificación de pieza 4x1](https://migueldilalla.github.io/assets/own-projects-resources/readme_model_samples/readme-model2_4X1-display.webp)

---

## 🧪 Notebooks de Entrenamiento de Modelos

Todos los modelos fueron entrenados en **Notebooks de Kaggle**, utilizando Ultralytics YOLOv8 (`yolov8n.pt`) con amplia aumentación de datos.

- 📄 Modelo de detección de piezas de clase única
- 📄 Modelo de detección de studs

> 🔗 **[Notebook de Kaggle: Ajuste del Detector de Piezas](https://www.kaggle.com/code/migueldilalla/brickssifier-studwise-project-models-trainer/edit)**

> 🔗 **[Notebook de Kaggle: Ajuste del Detector de Studs](https://www.kaggle.com/code/migueldilalla/brickssifier-studwise-project-models-trainer/edit)**

---

## 🖼️ Demo de la Aplicación Streamlit

Puedes probar el pipeline completo de forma interactiva en la web. Sube una imagen, obtén el resultado con anotaciones, metadatos y predicción de la pieza.

- 🔧 Construida con Streamlit + OpenCV + EXIF
- ⚙️ Incluye una huella digital de metadatos por inferencia

> 🌐 **[Prueba la Aplicación en Streamlit](https://migueldilalla-lego-classify.streamlit.app/)**

---

## 🖥️ Interfaz de Línea de Comandos (CLI)

Próximamente pondré a disposición instrucciones detalladas para utilizar la interfaz CLI incluida en este repositorio. Con esta herramienta podrás:

- 🔄 Realizar inferencias localmente en tus propias imágenes de piezas LEGO
- 🧩 Reproducir el procesamiento completo de los datasets utilizados
- 📊 Generar visualizaciones de resultados con metadatos incorporados
- 🔍 Experimentar con diferentes parámetros de detección y clasificación

Esta interfaz CLI ofrece todas las funcionalidades principales del proyecto en un entorno local, permitiendo mayor control y personalización que la versión web.

> ⏳ **Disponible próximamente**: Documentación completa con ejemplos de uso.

---

## 💬 Notas, Reflexiones y Agradecimientos

- Este proyecto fue construido como un **hito de aprendizaje** — no un producto comercial.
- Trabajé solo con recursos mínimos, haciendo de cada paso un ejercicio de creatividad, depuración y pensamiento claro.
- Aprendí sobre: etiquetado de imágenes, ajuste de modelos ML, detección de puntos clave, diseño CLI/UX e ingeniería de metadatos.

> 🙏 Gracias a la comunidad de código abierto y a las herramientas que hicieron esto posible: YOLOv8, LabelMe, Albumentations, Rich, Streamlit.

📬 ¡Siéntete libre de explorar el repositorio, hacerle fork o contactarme!

🔗 [Mi Portafolio](https://migueldilalla.github.io/)  
💼 [Mi LinkedIn](https://www.linkedin.com/in/MiguelDiLalla/)  
📦 [Repositorio del Proyecto](https://github.com/MiguelDiLalla/Brickssifier_Studwise)

---

© Miguel Di Lalla — LEGO® es una marca registrada del Grupo LEGO, que no patrocina ni respalda este proyecto.