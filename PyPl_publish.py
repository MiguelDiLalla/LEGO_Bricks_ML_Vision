import os
import subprocess

def main():
    print("Bienvenido al asistente de publicación de PyPI.")
    version = input("Por favor, ingrese el nuevo número de versión (por ejemplo, 0.1.2): ")

    # Actualizar el archivo setup.py con la nueva versión
    setup_file = "setup.py"
    with open(setup_file, "r") as file:
        lines = file.readlines()

    with open(setup_file, "w") as file:
        for line in lines:
            if line.strip().startswith("version="):
                file.write(f"    version=\"{version}\",\n")
            else:
                file.write(line)

    print(f"Versión actualizada a {version} en setup.py")

    # Reconstruir el paquete
    print("Construyendo el paquete...")
    subprocess.run(["rm", "-r", "dist/"])
    subprocess.run(["python", "-m", "build"])

    # Subir el paquete a PyPI
    print("Subiendo el paquete a PyPI...")
    subprocess.run(["python", "-m", "twine", "upload", "dist/*"])

    print("Publicación completada. ¡Gracias por usar el asistente!")

if __name__ == "__main__":
    main()
