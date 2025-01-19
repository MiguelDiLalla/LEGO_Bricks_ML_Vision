import os
import shutil
import json

# Define source and base destination paths
source_folder = "UNORGANIZED_folder"
base_destination = os.getcwd()

# Define mapping of file types to target folders
folder_mapping = {
    "models/checkpoints": [".pt"],
    "data/raw": [".jpg", ".png"],
    "data/annotations": [".json", ".txt"],
    "results/cropped_bricks": ["cropped_bricks"],
    "notebooks": [".ipynb"],
    "docs": [".pdf", ".md"]
}

# Define specific context-based mapping for key project elements
context_mapping = {
    "stud_detection_model": "models/checkpoints",
    "raw_images": "data/raw",
    "annotations": "data/annotations",
    "cropped_bricks": "results/cropped_bricks"
}

# Save original and new paths for reference
reference_file = os.path.join(base_destination, "path_references.json")
path_references = {}

def organize_files(source, destination, mapping, context):
    """
    Organize files from the source folder into the destination folder
    according to the provided folder mapping and context mapping.

    Parameters:
        source (str): Path to the source folder.
        destination (str): Path to the base destination folder.
        mapping (dict): A dictionary mapping target folders to file extensions.
        context (dict): A dictionary mapping logical contexts to folders.

    Returns:
        None
    """
    for root, _, files in os.walk(source):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()

            # Check for context mapping
            for context_key, context_folder in context.items():
                if context_key in file_path.lower():
                    target_folder = os.path.join(destination, context_folder)
                    os.makedirs(target_folder, exist_ok=True)
                    shutil.copy(file_path, target_folder)
                    path_references[file_path] = os.path.join(target_folder, file)
                    print(f"Copied {file} to {target_folder} based on context")
                    break
            else:
                # Check for folder mapping based on extensions
                for folder, extensions in mapping.items():
                    if file_extension in extensions:
                        target_folder = os.path.join(destination, folder)
                        os.makedirs(target_folder, exist_ok=True)
                        shutil.copy(file_path, target_folder)
                        path_references[file_path] = os.path.join(target_folder, file)
                        print(f"Copied {file} to {target_folder}")
                        break

    # Save the path references
    with open(reference_file, "w") as ref_file:
        json.dump(path_references, ref_file, indent=4)

if __name__ == "__main__":
    organize_files(source_folder, base_destination, folder_mapping, context_mapping)
    print("\nFiles organized successfully and path references saved!")
