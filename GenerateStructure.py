# -*- coding: utf-8 -*-

import os

def generate_structure_file(output_file="estructura.txt"):
    """
    Genera un archivo con la estructura de directorios solo si el directorio actual
    contiene una carpeta 'src' o está relacionado con ella.

    Args:
        output_file (str): Nombre del archivo para guardar la estructura.
    """
    # Directorio actual
    current_dir = os.getcwd()

    # Verificar si existe la carpeta 'src' en el directorio actual
    if "src" in os.listdir(current_dir):
        # Crear la estructura en el directorio actual
        target_dir = current_dir
        print(f"Directorio 'src' encontrado en el directorio actual: {target_dir}")
    else:
        # Subir al directorio padre y verificar si 'src' es una carpeta hermana
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if "src" in os.listdir(parent_dir):
            # Crear la estructura en el directorio padre
            target_dir = parent_dir
            print(f"Directorio 'src' encontrado como carpeta hermana. Generando estructura en: {target_dir}")
        else:
            print("No se encontró la carpeta 'src'. No se generará el archivo de estructura.")
            return

    # Generar la estructura
    with open(output_file, "w") as f:
        for root, dirs, files in os.walk(target_dir):
            level = root.replace(target_dir, "").count(os.sep)
            indent = " " * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = " " * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")

    print(f"Estructura generada exitosamente en: {output_file}")

# Ejecutar la función
if __name__ == "__main__":
    generate_structure_file()
