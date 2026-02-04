import shutil
from pathlib import Path
from collections import defaultdict

# Mapping from diverse folder names to standard class names
CLASS_MAPPING = {
    # Apple variants
    "apple": ["apple 10", "apple 11", "apple 12", "apple 13", "apple 14", "apple 17", "apple 18", 
              "apple 19", "apple 5", "apple 7", "apple 8", "apple 9", "apple core 1", 
              "apple red yellow 2", "apple worm 1", "apple_6", "apple_braeburn_1", 
              "apple_crimson_snow_1", "apple_golden_1", "apple_golden_2", "apple_golden_3",
              "apple_granny_smith_1", "apple_hit_1", "apple_pink_lady_1", "apple_red_1",
              "apple_red_2", "apple_red_3", "apple_red_delicios_1", "apple_red_yellow_1", "apple_rotten_1"],
    
    # Banana variants
    "banana": ["banana 3", "banana 4"],
    
    # Cucumber variants
    "cucumber": ["cucumber 1", "cucumber 10", "cucumber 11", "cucumber 3", "cucumber 4",
                 "cucumber 5", "cucumber 6", "cucumber 7", "cucumber 8", "cucumber 9"],
    
    # Tomato variants
    "tomato": ["tomato 1", "tomato 10", "tomato 5", "tomato 7", "tomato 8", "tomato 9",
               "tomato cherry maroon 1", "tomato cherry orange 1", "tomato cherry red 2",
               "tomato cherry yellow 1", "tomato maroon 2"],
    
    # Pear variants
    "pear": ["pear 1", "pear 10", "pear 11", "pear 12", "pear 13", "pear 3", "pear 5",
             "pear 6", "pear 7", "pear 8", "pear 9"],
    
    # Cherry variants
    "cherry": ["cherry 3", "cherry 4", "cherry 5", "cherry rainier 2", "cherry rainier 3",
               "cherry sour 1", "cherry wax not ripen 1", "cherry wax not ripen 2",
               "cherry wax red 2", "cherry wax red 3"],
    
    # Peach variants
    "peach": ["peach 3", "peach 4", "peach 5", "peach 6"],
    
    # Onion variants
    "onion": ["onion 2", "onion red 2", "onion white peeled 1"],
    
    # Cabbage variants
    "cabbage": ["cabbage red 1", "cabbage_white_1"],
    
    # Carrot
    "carrot": ["carrot_1"],
    
    # Eggplant
    "eggplant": ["eggplant_long_1"],
    
    # Ginger
    "ginger": ["ginger 2"],
    
    # Zucchini variants
    "zucchini": ["zucchini dark 1", "zucchini green 1", "zucchini_1"],
    
    # Other fruits
    "avocado": ["avocado black 1", "avocado black 2", "avocado green 1"],
    "beans": ["beans 1"],
    "blackberry": ["blackberrie 1", "blackberrie 2", "blackberrie half rippen 1", "blackberrie not rippen 1"],
    "cactus_fruit": ["cactus fruit green 1", "cactus fruit red 1"],
    "cashew": ["caju seed 1"],
    "cherimoya": ["cherimoya 1"],
    "gooseberry": ["gooseberry 1"],
    "grape": ["grape not ripen 1"],
    "nectarine": ["nectarine flat 2"],
    "nut": ["nut 1", "nut 2", "nut 3", "nut 4", "nut 5"],
    "papaya": ["papaya 2"],
    "pistachio": ["pistachio 1"],
    "plum": ["plum 4"],
    "quince": ["quince 2", "quince 3", "quince 4"],
}


def normalize_name(name: str) -> str:
    """Normalize folder name to lowercase for matching."""
    return name.lower().strip()


def consolidate_directory(src_dir: Path, dest_dir: Path) -> None:
    """Consolidate images from various class folders into standard class folders."""
    if not src_dir.exists():
        print(f"Skipping {src_dir} (does not exist)")
        return
    
    # Build reverse mapping: original_folder -> standard_class
    folder_to_class = {}
    for standard_class, variants in CLASS_MAPPING.items():
        for variant in variants:
            folder_to_class[normalize_name(variant)] = standard_class
    
    # Collect all source folders
    src_folders = [f for f in src_dir.iterdir() if f.is_dir()]
    
    if not src_folders:
        print(f"No folders found in {src_dir}")
        return
    
    # Group images by target class
    class_images = defaultdict(list)
    unmatched_folders = []
    
    for folder in src_folders:
        normalized = normalize_name(folder.name)
        target_class = folder_to_class.get(normalized)
        
        if target_class:
            # Collect all images from this folder
            images = [f for f in folder.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            class_images[target_class].extend(images)
        else:
            unmatched_folders.append(folder.name)
    
    if unmatched_folders:
        print(f"\nWarning: {len(unmatched_folders)} folders not mapped to standard classes:")
        for name in sorted(unmatched_folders)[:10]:
            print(f"  - {name}")
        if len(unmatched_folders) > 10:
            print(f"  ... and {len(unmatched_folders) - 10} more")
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images to consolidated class folders
    for class_name, images in class_images.items():
        class_dir = dest_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for img in images:
            dest_path = class_dir / img.name
            # Handle duplicate names by adding counter
            counter = 1
            while dest_path.exists():
                stem = img.stem
                dest_path = class_dir / f"{stem}_{counter}{img.suffix}"
                counter += 1
            shutil.copy2(img, dest_path)
        
        print(f"  {class_name}: {len(images)} images")
    
    print(f"\nConsolidated {sum(len(imgs) for imgs in class_images.values())} images into {len(class_images)} classes")


def main():
    base_dir = Path("data")
    
    # Process train, validation, and test sets
    for split in ["train", "validation", "test"]:
        src_dir = base_dir / split
        dest_dir = base_dir / f"{split}_consolidated"
        
        print(f"\n{'='*60}")
        print(f"Processing {split} set:")
        print(f"{'='*60}")
        
        consolidate_directory(src_dir, dest_dir)
    
    print("\n" + "="*60)
    print("Consolidation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify the consolidated folders in data/train_consolidated/, data/validation_consolidated/, data/test_consolidated/")
    print("2. If satisfied, backup original folders and rename consolidated ones:")
    print("   - Rename data/train to data/train_old")
    print("   - Rename data/train_consolidated to data/train")
    print("   - (Same for validation and test)")
    print("3. Re-run training with the consolidated dataset")


if __name__ == "__main__":
    main()
