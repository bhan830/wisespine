# split_masks.py
import nibabel as nib
import numpy as np
from pathlib import Path

# ---------------------------
# Adjust these paths
# ---------------------------
INPUT_MASK = "/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-02validation/derivatives/sub-gl017/sub-gl017_seg-vert_msk.nii.gz"
OUTPUT_DIR = "./baseline_outputs/sub-gl017"
# ---------------------------

# Define vertebra labels
VERTS = {
    "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7,
    "T1": 8, "T2": 9, "T3": 10, "T4": 11, "T5": 12, "T6": 13, "T7": 14, "T8": 15, "T9": 16, "T10": 17, "T11": 18, "T12": 19,
    "L1": 20, "L2": 21, "L3": 22, "L4": 23, "L5": 24, "L6": 25
}

def main():
    input_path = Path(INPUT_MASK)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load multi-label mask
    img = nib.load(str(input_path))
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    # Split into individual vertebrae
    for name, label in VERTS.items():
        mask = (data == label).astype(np.uint8)
        if mask.sum() == 0:
            print(f"⚠️ No voxels found for {name}, skipping")
            continue

        out_file = output_dir / f"{input_path.stem}_{name}.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine, header), str(out_file))
        print(f"✅ Saved {out_file}")

    print(f"✅ All vertebrae masks saved to {output_dir}")

if __name__ == "__main__":
    main()
