import os, sys, math, random
from PIL import Image, ImageDraw
from rich.console import Console
from rich.prompt import Prompt
from collections import Counter

# Initialize rich console
console = Console()

# --- Helper Functions ---

def parse_grid_input(grid_str):
    """Parse grid input like '3x2' or a single integer '4' (interpreted as 4x4) and return (rows, cols)."""
    grid_str = grid_str.strip().lower()
    if 'x' in grid_str:
        try:
            parts = grid_str.split('x')
            rows = int(parts[0])
            cols = int(parts[1])
        except Exception:
            raise ValueError("Grid must be in the format 'NxM', e.g. '3x2'.")
    else:
        try:
            rows = cols = int(grid_str)
        except Exception:
            raise ValueError("Please enter a valid integer or grid in 'NxM' format.")
    if rows < 1 or cols < 1:
        raise ValueError("Grid dimensions must be positive integers.")
    return rows, cols

def fallback_grid(n):
    """Generate a fallback grid based on available pairs count n.
       Returns (rows, cols) such that rows*cols is as close as possible to n (and not exceeding n if possible)."""
    rows = int(math.sqrt(n))
    if rows == 0: 
        rows = 1
    cols = math.ceil(n / rows)
    return rows, cols

def calculate_line_thickness(image_size):
    """Calculate appropriate line thickness based on image dimensions.
    
    Args:
        image_size: Width or height of the square image (they are the same)
        
    Returns:
        Line thickness in pixels
    """
    # Base the line thickness on image size - larger images get thicker lines
    # These values were tuned for typical viewing conditions
    if image_size < 200:
        return 2
    elif image_size < 400:
        return 3
    elif image_size < 600:
        return 5
    elif image_size < 800:
        return 7
    else:
        return 9  # For very large images

def annotate_image(image_path, label_path):
    """Open the image, resize to square, and draw YOLO bounding boxes for all classes."""
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        console.log(f"[red]Error opening image: {image_path} - {str(e)}[/red]")
        return None

    orig_w, orig_h = im.size
    
    # Instead of padding, directly resize the image to a square (allows deformation)
    square_size = max(orig_w, orig_h)
    new_im = im.resize((square_size, square_size))
    
    # Calculate appropriate line thickness for this image size
    line_thickness = calculate_line_thickness(square_size)
    
    draw = ImageDraw.Draw(new_im)
    try:
        with open(label_path, "r") as f:
            lines = f.read().strip().splitlines()
    except Exception as e:
        console.log(f"[red]Error reading label file: {label_path} - {str(e)}[/red]")
        return new_im

    class_counts = Counter()
    boxes_drawn = 0
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            class_counts[cls] += 1
        except:
            continue
            
        try:
            x_c = float(parts[1])
            y_c = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except:
            continue
            
        # Apply scaling to the coordinates
        x_center = x_c * square_size
        y_center = y_c * square_size
        box_width = bw * square_size
        box_height = bh * square_size
        
        left = x_center - box_width / 2
        top = y_center - box_height / 2
        right = x_center + box_width / 2
        bottom = y_center + box_height / 2
        
        # Draw bounding box with dynamically calculated line thickness
        draw.rectangle([left, top, right, bottom], outline="#facc15", width=line_thickness)
        boxes_drawn += 1
    
    # Report on boxes by class
    if boxes_drawn == 0:
        console.log(f"[yellow]Warning: No bounding boxes were drawn for {os.path.basename(image_path)}[/yellow]")
    else:
        console.log(f"[green]Drew {boxes_drawn} boxes for {os.path.basename(image_path)} (line thickness: {line_thickness}px)[/green]")
        
    return new_im

def create_grid_image(selected_keys, valid_images, valid_labels, rows, cols, cell_size, grid_id=1):
    """Creates a single grid image from the selected keys."""
    grid_width = cell_size * cols
    grid_height = cell_size * rows
    
    # Annotate images for the selected pairs
    annotated_images = []
    console.print(f"\n[bold]Annotating images for Grid {grid_id}...[/bold]")
    for key in selected_keys:
        img_path = valid_images[key]
        label_path = valid_labels[key]
        console.print(f"[bold cyan]Processing: {os.path.basename(img_path)}[/bold cyan]")
        annotated_im = annotate_image(img_path, label_path)
        if annotated_im is not None:
            # Resize the annotated square image to cell size
            annotated_im = annotated_im.resize((cell_size, cell_size))
            annotated_images.append(annotated_im)
            console.print(f"[green]Successfully annotated: {os.path.basename(img_path)}[/green]")
        else:
            console.print(f"[red]Failed to annotate: {os.path.basename(img_path)}[/red]")

    # Create the grid (collage)
    grid_image = Image.new("RGB", (grid_width, grid_height), (0, 0, 0))
    index = 0
    console.print(f"\n[bold]Creating grid image {grid_id}...[/bold]")
    for r in range(rows):
        for c in range(cols):
            if index < len(annotated_images):
                grid_image.paste(annotated_images[index], (c * cell_size, r * cell_size))
                index += 1
    
    return grid_image

# --- Main Script ---

def main():
    console.rule("[bold green]Annotation & Grid Creation Script[/bold green]")
    
    # Ask for main folder path
    folder = Prompt.ask("Enter the path to the main folder")
    if not os.path.isdir(folder):
        console.print(f"[red]The folder '{folder}' does not exist.[/red]")
        sys.exit(1)
    
    # List subfolders in the main folder (only directories)
    subfolders = [os.path.join(folder, d) for d in os.listdir(folder)
                  if os.path.isdir(os.path.join(folder, d))]
    if len(subfolders) != 2:
        console.print("[red]The main folder must contain exactly two subfolders.[/red]")
        sys.exit(1)
    
    # Determine which subfolder is for images and which for labels based on file extensions.
    images_folder = None
    labels_folder = None
    for sub in subfolders:
        files = os.listdir(sub)
        jpg_count = sum(1 for f in files if f.lower().endswith('.jpg'))
        txt_count = sum(1 for f in files if f.lower().endswith('.txt'))
        if jpg_count >= txt_count and jpg_count > 0:
            images_folder = sub
        elif txt_count > jpg_count:
            labels_folder = sub

    if images_folder is None or labels_folder is None:
        console.print("[red]Could not determine image and label subfolders (ensure one folder mainly contains .jpg and the other .txt files).[/red]")
        sys.exit(1)
    
    console.print(f"[bold]Images folder:[/bold] {images_folder}")
    console.print(f"[bold]Labels folder:[/bold] {labels_folder}")

    # Scan image folder and validate images
    total_image_files = 0
    valid_images = dict()   # key: base filename, value: full image file path
    invalid_image_count = 0
    for filename in os.listdir(images_folder):
        if not filename.lower().endswith('.jpg'):
            continue  # ignore unsupported file types
        total_image_files += 1
        img_path = os.path.join(images_folder, filename)
        try:
            with Image.open(img_path) as im:
                im.verify()
            # On success, store by base filename (without extension)
            base = os.path.splitext(filename)[0]
            valid_images[base] = img_path
        except Exception:
            invalid_image_count += 1

    # Scan labels folder and validate label files
    total_label_files = 0
    valid_labels = dict()   # key: base filename, value: full label file path
    invalid_label_count = 0
    for filename in os.listdir(labels_folder):
        if not filename.lower().endswith('.txt'):
            continue  # ignore unsupported file types
        total_label_files += 1
        label_path = os.path.join(labels_folder, filename)
        valid = True
        try:
            with open(label_path, "r") as f:
                lines = f.read().strip().splitlines()
            for line in lines:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    valid = False
                    break
                # Try to convert to float
                try:
                    [float(p) for p in parts]
                except:
                    valid = False
                    break
        except Exception:
            valid = False

        if valid:
            base = os.path.splitext(filename)[0]
            valid_labels[base] = label_path
        else:
            invalid_label_count += 1

    # Check matching between images and labels (strict base filename matching)
    image_keys = set(valid_images.keys())
    label_keys = set(valid_labels.keys())
    matching_keys = image_keys.intersection(label_keys)
    discarded_image_matches = len(image_keys - matching_keys)
    discarded_label_matches = len(label_keys - matching_keys)

    total_valid_pairs = len(matching_keys)

    # Report summary of comprobations
    console.print("\n[bold underline]Validation Report:[/bold underline]")
    console.print(f"Total image files scanned: {total_image_files}")
    console.print(f"Valid images: {len(valid_images)}   |   Invalid images discarded: {invalid_image_count}")
    console.print(f"Total label files scanned: {total_label_files}")
    console.print(f"Valid label files: {len(valid_labels)}   |   Invalid labels discarded: {invalid_label_count}")
    console.print(f"Matching valid pairs: {total_valid_pairs}")
    console.print(f"Mismatched (unpaired) images discarded: {discarded_image_matches}")
    console.print(f"Mismatched (unpaired) labels discarded: {discarded_label_matches}\n")
    
    if total_valid_pairs == 0:
        console.print("[red]No valid matching pairs found. Exiting.[/red]")
        sys.exit(1)
    
    # Ask for grid size input and validate
    grid_input = Prompt.ask("Enter the desired grid size (e.g. '3x2' or a single integer like '4' for 4x4)")
    try:
        rows, cols = parse_grid_input(grid_input)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    requested_cells = rows * cols
    
    if total_valid_pairs < requested_cells:
        # Clean fallback: choose a nearly square grid based on available pairs.
        fallback_rows, fallback_cols = fallback_grid(total_valid_pairs)
        console.print(f"[yellow]Not enough pairs for the requested grid of {rows}x{cols} ({requested_cells} cells).")
        console.print(f"Falling back to a grid of {fallback_rows}x{fallback_cols} ({fallback_rows * fallback_cols} cells).[/yellow]")
        rows, cols = fallback_rows, fallback_cols
        requested_cells = rows * cols

    # Decide cell size so that maximum grid dimension does not exceed 1200px
    cell_size = math.floor(1200 / max(rows, cols))
    grid_width = cell_size * cols
    grid_height = cell_size * rows

    # Ask for the number of grid images to generate
    num_grids = Prompt.ask(
        "How many different grid images would you like to generate?", 
        default="3"
    )
    try:
        num_grids = int(num_grids)
        if num_grids < 1:
            num_grids = 1
        if num_grids > 10:
            console.print("[yellow]Warning: Limiting to 10 grids maximum.[/yellow]")
            num_grids = 10
    except ValueError:
        console.print("[yellow]Invalid number, defaulting to 3 grids.[/yellow]")
        num_grids = 3
    
    console.print(f"\n[bold]Grid Settings:[/bold] {rows} rows x {cols} columns, each cell {cell_size}px")
    console.print(f"Generating {num_grids} different grid images of size {grid_width}px x {grid_height}px")
    console.print(f"Bounding box line thickness: dynamic based on image size, color: #facc15 (yellow)\n")
    
    # Generate multiple grid images with different random selections
    for grid_index in range(num_grids):
        # Randomly select keys for this grid
        randomized_keys = list(matching_keys)
        random.shuffle(randomized_keys)
        selected_keys = randomized_keys[:requested_cells]
        
        console.print(f"\n[bold underline]Grid {grid_index+1} of {num_grids}[/bold underline]")
        console.print("\n[bold]Selected Images (Randomly Chosen):[/bold]")
        for i, key in enumerate(selected_keys):
            console.print(f"[cyan]{i+1}. {os.path.basename(valid_images[key])}[/cyan]")
        
        # Analyze class distribution in this set of selected images
        all_classes = Counter()
        for key in selected_keys:
            label_path = valid_labels[key]
            try:
                with open(label_path, "r") as f:
                    for line in f.read().strip().splitlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                cls = int(float(parts[0]))
                                all_classes[cls] += 1
                            except:
                                pass
            except:
                pass
        
        console.print("\n[bold]Class distribution in selected images:[/bold]")
        if not all_classes:
            console.print("[yellow]No valid class data found in labels[/yellow]")
        else:
            for cls, count in sorted(all_classes.items()):
                console.print(f"Class {cls}: {count} bounding boxes")
        
        # Create the grid image
        grid_image = create_grid_image(
            selected_keys, valid_images, valid_labels, 
            rows, cols, cell_size, grid_id=grid_index+1
        )
                
        # Save the grid to the main input folder with a unique name
        output_filename = f"output_grid_{grid_index+1}.png"
        output_path = os.path.join(folder, output_filename)
        grid_image.save(output_path)
        console.print(f"[green]Grid image {grid_index+1} successfully saved as:[/green] {output_path}")
    
    # Final summary report of actions taken.
    console.print("\n[bold underline]Final Summary:[/bold underline]")
    console.print(f"Total valid matching pairs: {total_valid_pairs}")
    console.print(f"Grid dimensions: {rows} x {cols} (each cell: {cell_size}px)")
    console.print(f"Generated {num_grids} different grid images")
    console.print(f"Bounding box color: #facc15 (yellow) with dynamic line thickness")
    console.print(f"Images resized to square format (1:1 ratio with deformation)")
    console.print("Invalid files and mismatches have been silently discarded.")

if __name__ == "__main__":
    main()
