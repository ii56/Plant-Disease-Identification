import argparse
import os
import shutil
import random
from pathlib import Path
from math import floor
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}


def is_image_file(p: Path):
    return p.suffix.lower() in IMG_EXTS


def list_class_folders(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def safe_makedirs(p: Path, exist_ok=True):
    p.mkdir(parents=True, exist_ok=exist_ok)


def split_for_class(files, train_ratio, val_ratio, test_ratio, rng):
    """
    Given list of files (paths), return three lists (train, val, test).
    Deterministic with rng (random.Random or numpy RandomState-like .shuffle()).
    """
    n = len(files)
    if n == 0:
        return [], [], []
    files_shuffled = list(files)
    rng.shuffle(files_shuffled)

    # compute counts (use floor for train+val then rest to test to avoid rounding issues)
    n_train = int(floor(n * train_ratio))
    n_val = int(floor(n * val_ratio))
    # ensure at least one in train if possible
    if n_train == 0 and n >= 1:
        n_train = 1
        # reduce val if possible
        if n_val > 0:
            n_val -= 1
    n_test = n - n_train - n_val
    if n_test < 0:
        # fallback fix (shouldn't happen if ratios sum <=1)
        n_test = 0
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

    train_files = files_shuffled[:n_train]
    val_files = files_shuffled[n_train:n_train + n_val]
    test_files = files_shuffled[n_train + n_val:]
    return train_files, val_files, test_files


def prepare_split(dataset_root: Path, output_dir: Path,
                  train_ratio: float, val_ratio: float, test_ratio: float,
                  move_files: bool, overwrite: bool, seed: int):
    # sanity
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if output_dir.exists():
        if overwrite:
            print(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {output_dir} (use --overwrite to remove)")

    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    safe_makedirs(train_dir)
    safe_makedirs(val_dir)
    safe_makedirs(test_dir)

    classes = list_class_folders(dataset_root)
    if not classes:
        raise RuntimeError(f"No class subfolders found inside dataset root: {dataset_root}")

    rng = random.Random(seed)

    summary = {}
    for cls in classes:
        cls_name = cls.name
        # gather image files
        files = [p for p in cls.iterdir() if p.is_file() and is_image_file(p)]
        files = sorted(files)  # deterministic order before shuffling
        if len(files) == 0:
            print(f"Warning: class '{cls_name}' contains no image files - skipping.")
            continue

        train_files, val_files, test_files = split_for_class(files, train_ratio, val_ratio, test_ratio, rng)

        # create class subfolders
        target_train = train_dir / cls_name
        target_val = val_dir / cls_name
        target_test = test_dir / cls_name
        safe_makedirs(target_train)
        safe_makedirs(target_val)
        safe_makedirs(target_test)

        # copy / move files
        for p in tqdm(train_files, desc=f"Copying train/{cls_name}", leave=False):
            dest = target_train / p.name
            if move_files:
                shutil.move(str(p), str(dest))
            else:
                shutil.copy2(str(p), str(dest))
        for p in tqdm(val_files, desc=f"Copying val/{cls_name}", leave=False):
            dest = target_val / p.name
            if move_files:
                shutil.move(str(p), str(dest))
            else:
                shutil.copy2(str(p), str(dest))
        for p in tqdm(test_files, desc=f"Copying test/{cls_name}", leave=False):
            dest = target_test / p.name
            if move_files:
                shutil.move(str(p), str(dest))
            else:
                shutil.copy2(str(p), str(dest))

        summary[cls_name] = {
            'total': len(files),
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }

    # print summary
    print("\nDataset split summary:")
    total_images = 0
    total_train = total_val = total_test = 0
    for cls, stats in sorted(summary.items()):
        print(f"  {cls:40s} total={stats['total']:4d}   train={stats['train']:4d}   val={stats['val']:4d}   test={stats['test']:4d}")
        total_images += stats['total']
        total_train += stats['train']
        total_val += stats['val']
        total_test += stats['test']

    print(f"\nTotals: images={total_images}   train={total_train}   val={total_val}   test={total_test}")
    print(f"Output written to: {output_dir}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Prepare train/val/test splits for a dataset with class subfolders.")
    p.add_argument("--dataset_root", required=True, type=Path,
                   help="Path to your original dataset root that contains class subfolders (e.g. PlantVillage/)")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Directory to create with train/val/test subfolders.")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Proportion to use for training (default 0.8)")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Proportion to use for validation (default 0.1)")
    p.add_argument("--test_ratio", type=float, default=0.1, help="Proportion to use for test (default 0.1)")
    p.add_argument("--move", action="store_true", help="Move files instead of copying (destructive).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it exists.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling (default 42).")
    return p.parse_args()


def main():
    args = parse_args()
    prepare_split(args.dataset_root, args.output_dir,
                  args.train_ratio, args.val_ratio, args.test_ratio,
                  move_files=args.move, overwrite=args.overwrite, seed=args.seed)


if __name__ == "__main__":
    main()
