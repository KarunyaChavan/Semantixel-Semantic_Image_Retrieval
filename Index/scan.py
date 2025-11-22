from Index.scan_default import fast_scan_for_images
from concurrent.futures import ThreadPoolExecutor
import yaml, csv
from tqdm import tqdm
from PIL import Image

# Try to import Rust-accelerated functions
try:
    from rust_integration import RustScanner, RustCSVHandler, RustImageProcessor
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Note: Rust module not available, using Python implementations")


def process_image(path):
    try:
        with Image.open(path) as img:
            try:
                first_channel = img.split()[0]
                pixel_data = list(first_channel.getdata())
            except AttributeError:
                pixel_data = list(img.getdata())
            total = sum(pixel_data)
            num_pixels = len(pixel_data)
            average = int(total / num_pixels)
        return (path, average)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return (path, 0)


def save_to_csv(image_paths, filename="paths.csv", save_average=False):
    """
    Saves the paths and, optionally, average pixel values of images to a CSV file.

    Args:
        image_paths (list): A list of image file paths to process.
        filename (str, optional): The name of the CSV file to save. Defaults to "image_paths.csv".
        save_average (bool, optional): If True, saves the average pixel value of images. If False, does not save the average. Defaults to False.
    """
    # Handle backslashes in Windows paths
    image_paths = [path.replace("\\", "/") for path in image_paths]
    image_paths = list(set(image_paths))
    
    if save_average:
        if RUST_AVAILABLE:
            # Use Rust for parallel average calculation
            averages = RustImageProcessor.calculate_averages(image_paths)
            paths_and_averages = list(zip(image_paths, averages))
        else:
            # Fall back to Python with ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                paths_and_averages = list(
                    tqdm(
                        executor.map(process_image, image_paths), total=len(image_paths)
                    )
                )
        
        if RUST_AVAILABLE:
            # Use Rust CSV writer
            RustCSVHandler.write_csv(filename, image_paths, [p[1] for p in paths_and_averages])
        else:
            # Use Python CSV writer
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["path", "average"])
                for path, average in paths_and_averages:
                    writer.writerow([path, average])
    else:
        # Write without averages
        if RUST_AVAILABLE:
            RustCSVHandler.write_csv(filename, image_paths, [0] * len(image_paths))
        else:
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["path", "average"])
                for path in tqdm(image_paths):
                    writer.writerow([path, 0])

    print(f"Image paths{' and averages' if save_average else ''} saved to {filename}")


def read_from_csv(filename="paths.csv"):
    """
    Reads image paths and, optionally, their average pixel values from a CSV file.

    Args:
        filename (str, optional): The name of the CSV file to read from. Defaults to "image_paths.csv".

    Returns:
        tuple: Depending on read_average, returns a tuple containing one or two lists: one of the image paths and, if read_average is True, one of their average pixel values.
    """
    if RUST_AVAILABLE:
        # Use Rust for fast CSV reading
        try:
            paths, averages = RustCSVHandler.read_csv(filename)
            return (paths, averages)
        except Exception as e:
            print(f"Rust CSV read failed: {e}, falling back to Python")
    
    # Fall back to Python
    paths = []
    averages = []
    with open("Index/paths.csv", "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader):
            paths.append(row["path"])
            averages.append(int(row["average"]))
    return (paths, averages)


def scan_and_save():
    """
    Scans for images based on the configuration specified in 'config.yaml' and saves the paths to a CSV file.

    The function supports two scanning methods: 'default' and 'Everything'. The 'default' method uses specified
    include and exclude directories to find images, while the 'Everything' method utilizes the Everything SDK for scanning.

    Returns:
        bool: True if the scan and save operation was successful, False otherwise.
    """
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        if config["scan_method"] == "default":
            include_dirs = config["include_directories"]
            exclude_dirs = config["exclude_directories"]
            if len(include_dirs) == 0:
                include_dirs = None
                print("No directories to include")
                return False
            
            # Use Rust scanner for faster performance
            if RUST_AVAILABLE:
                print("Using Rust backend for directory scanning...")
                all_paths = []
                for directory in include_dirs:
                    paths, elapsed_ms = RustScanner.scan_directory(directory, exclude_dirs)
                    all_paths.extend(paths)
                print(f"Scanned {len(all_paths)} images using Rust ({elapsed_ms}ms)")
                paths = all_paths
            else:
                print("Using Python backend for directory scanning...")
                paths, _ = fast_scan_for_images(include_dirs, exclude_dirs)
        elif config["scan_method"] == "Everything":
            from Index.scan_EverythingSDK import search_EverythingSDK

            paths = search_EverythingSDK()
        else:
            print("Error in config.yaml: scan_method must be 'default' or 'Everything'")
            return False

        save_to_csv(paths, "Index/paths.csv", config["deep_scan"])
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
