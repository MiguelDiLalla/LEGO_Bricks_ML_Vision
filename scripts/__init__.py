from .pipeline import (
    setup_environment,
    get_kaggle_credentials,
    download_dataset_from_kaggle,
    validate_directories,
    preprocess_images,
    labelme_to_yolo,
    train_yolo_pipeline,
    test_model_on_real_images,
    visualize_results,
)

from .visualize_presentation import (
    create_dataset_grid,
    annotate_model_results,
    generate_comparison_grid,
    organize_presentation_folders,
)

__all__ = [
    "setup_environment",
    "get_kaggle_credentials",
    "download_dataset_from_kaggle",
    "validate_directories",
    "preprocess_images",
    "labelme_to_yolo",
    "train_yolo_pipeline",
    "test_model_on_real_images",
    "visualize_results",
    "create_dataset_grid",
    "annotate_model_results",
    "generate_comparison_grid",
    "organize_presentation_folders",
]

"""
scripts: Módulos para el preprocesamiento, entrenamiento y visualización en el proyecto de detección de LEGO.
Incluye herramientas para la generación de presentaciones visuales y la organización automática de carpetas.
"""
