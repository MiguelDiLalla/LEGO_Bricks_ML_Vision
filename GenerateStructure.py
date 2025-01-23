# -*- coding: utf-8 -*-

import os

def generate_structure_file(output_file="estructura.txt"):
    """
    Genera un archivo con la estructura de directorios del directorio actual.

    Args:
        output_file (str): Nombre del archivo para guardar la estructura.
    """
    # Directorio actual
    current_dir = os.getcwd()

    # Generar la estructura
    with open(output_file, "w") as f:
        for root, dirs, files in os.walk(current_dir):
            level = root.replace(current_dir, "").count(os.sep)
            indent = " " * 4 * level
            f.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = " " * 4 * (level + 1)
            for file in files:
                f.write(f"{subindent}{file}\n")

    print(f"Estructura generada exitosamente en: {output_file}")

# Ejecutar la función
if __name__ == "__main__":
    generate_structure_file()
