import os

def create_project_structure():
    """
    Create a folder structure for an image data project.

    The base path is automatically set to the current working directory.
    """
    # Set the base directory to the current working directory
    base_path = os.getcwd()

    # Define the folder structure
    structure = {
        "data": [
            "raw",
            "processed",
            "annotations"
        ],
        "src": [
            "data_preprocessing",
            "models",
            "visualization"
        ],
        "models": [
            "checkpoints",
            "final"
        ],
        "results": [
            "predictions",
            "logs"
        ],
        "docs": [],
        "notebooks": [],
        "tests": [],
        "presentation": []
    }

    # Function to create folders recursively
    def create_folders(base, folders):
        for folder, subfolders in folders.items():
            folder_path = os.path.join(base, folder)
            os.makedirs(folder_path, exist_ok=True)
            for subfolder in subfolders:
                os.makedirs(os.path.join(folder_path, subfolder), exist_ok=True)

    # Create the structure
    create_folders(base_path, structure)

    print(f"Project structure created successfully at: {base_path}")

# Example usage
if __name__ == "__main__":
    create_project_structure()
