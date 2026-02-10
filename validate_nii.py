#!/usr/bin/env python3
"""
validate_nii.py

Check NIfTI image properties for a single file to ensure:
- Voxel dimensions and resolution
- Affine and orientation
- Data type integrity
- Non-zero voxels in segmentation masks
- NaN or Inf values

Usage:
    python validate_nii.py /path/to/file.nii.gz
"""

import nibabel as nib
import numpy as np
import sys
import os

def validate_nii(nii_path):
    if not os.path.exists(nii_path):
        print(f"❌ File does not exist: {nii_path}")
        return

    print(f"🧪 Validating NIfTI file: {nii_path}")
    
    # Load NIfTI file
    img = nib.load(nii_path)
    data = img.get_fdata()

    # Basic info
    print("\n--- Basic Info ---")
    print("Shape (voxels):", data.shape)
    print("Voxel size (mm):", img.header.get_zooms())
    print("Data type:", data.dtype)

    # Affine & orientation
    print("\n--- Affine & Orientation ---")
    print("Affine matrix:\n", img.affine)
    try:
        orientation = nib.orientations.aff2axcodes(img.affine)
        print("Orientation code:", orientation)
    except Exception as e:
        print("Error computing orientation:", e)

    # Check integrity
    print("\n--- Data Integrity ---")
    print("Min voxel value:", np.min(data))
    print("Max voxel value:", np.max(data))
    print("Number of NaNs:", np.isnan(data).sum())
    print("Number of infinite values:", np.isinf(data).sum())

    # Non-zero voxels (for segmentation masks)
    nonzero_voxels = np.count_nonzero(data)
    print("Number of non-zero voxels:", nonzero_voxels)
    if nonzero_voxels == 0:
        print("⚠️ Warning: segmentation mask may be empty!")

    print("\n✅ Validation complete!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_nii.py /path/to/file.nii.gz")
        sys.exit(1)
    
    nii_file = sys.argv[1]
    validate_nii(nii_file)
