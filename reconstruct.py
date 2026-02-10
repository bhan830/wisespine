# reconstruct.py
import nibabel as nib
import numpy as np
from pathlib import Path

# ---------------------------
# Adjust these paths
# ---------------------------
INPUT_DIR = "./baseline_outputs/sub-gl017"  # Folder containing individual vertebra masks
OUTPUT_FILE = "./baseline_outputs/sub-gl017/sub-gl017_reconstructed.nii.gz"
# ---------------------------

# Define vertebra labels (same as Verse20 original)
VERTS = {
    "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7,
    "T1": 8, "T2": 9, "T3": 10, "T4": 11, "T5": 12, "T6": 13, "T7": 14,
    "T8": 15, "T9": 16, "T10": 17, "T11": 18, "T12": 19,
    "L1": 20, "L2": 21, "L3": 22, "L4": 23, "L5": 24, "L6": 25
}

def main():
    input_dir = Path(INPUT_DIR)
    output_file = Path(OUTPUT_FILE)
    
    # Find at least one mask to get the reference shape and affine
    ref_mask_path = next(input_dir.glob("*.nii.gz"), None)
    if ref_mask_path is None:
        raise FileNotFoundError(f"No .nii.gz masks found in {input_dir}")

    ref_img = nib.load(str(ref_mask_path))
    shape = ref_img.shape
    affine = ref_img.affine
    header = ref_img.header

    # Initialize empty volume
    reconstructed = np.zeros(shape, dtype=np.uint8)

    # Load each vertebra mask and add to reconstructed volume
    for vert_name, label in VERTS.items():
        mask_file = input_dir / f"{ref_mask_path.stem.rsplit('_',1)[0]}_{vert_name}.nii.gz"
        if not mask_file.exists():
            print(f"⚠️ {vert_name} mask not found, skipping")
            continue

        mask_img = nib.load(str(mask_file))
        mask_data = mask_img.get_fdata().astype(bool)
        reconstructed[mask_data] = label
        print(f"✅ Added {vert_name} to reconstructed volume")

    # Save reconstructed multi-label image
    nib.save(nib.Nifti1Image(reconstructed, affine, header), str(output_file))
    print(f"\n🎉 Reconstructed spine saved as {output_file}")

if __name__ == "__main__":
    main()
