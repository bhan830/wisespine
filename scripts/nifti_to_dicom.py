#!/usr/bin/env python3
"""
Robust NIfTI → DICOM conversion
- Correct geometry (no rotation / distortion)
- Fully valid DICOM (no ITK-SNAP crash)
- Slice-based (clinical standard)
"""


import numpy as np
from pathlib import Path
import nibabel as nib
import pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage
import shutil


# ==============================
# Configuration
# ==============================
RUN_ALL_TRAINING_CASES = False
CASES = ["sub-gl003"]


RAWDATA_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/data/Verse20/dataset-01training/rawdata")
DICOM_BASE_DIR = Path("/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/dicom_outputs")


CLEAR_EXISTING = True




# ==============================
# Helpers
# ==============================


def get_case_list():
    if RUN_ALL_TRAINING_CASES:
        return sorted([p.name for p in RAWDATA_DIR.iterdir() if p.is_dir()])
    return CASES




def compute_geometry(affine):
    """
    Extract spacing + direction cosines from affine
    """
    R = affine[:3, :3]


    spacing = np.linalg.norm(R, axis=0)


    row = R[:, 0] / spacing[0]
    col = R[:, 1] / spacing[1]


    # Convert RAS → LPS (DICOM standard)
    ras_to_lps = np.diag([-1, -1, 1])
    row = ras_to_lps @ row
    col = ras_to_lps @ col


    return spacing, row, col




def validate(data, spacing, nii_spacing):
    print("\n[VALIDATION]")
    print(f"Shape: {data.shape}")
    print(f"Affine spacing: {spacing}")
    print(f"Header spacing: {nii_spacing}")


    diff = np.abs(np.array(spacing) - np.array(nii_spacing))
    if np.any(diff > 1e-3):
        print(f"[WARNING] Spacing mismatch: {diff}")
    else:
        print("[OK] Spacing matches")




# ==============================
# Conversion
# ==============================


def convert_case(input_ct, output_dir, case):


    if output_dir.exists():
        if CLEAR_EXISTING:
            shutil.rmtree(output_dir)
        else:
            print(f"[INFO] Skipping {case}")
            return


    output_dir.mkdir(parents=True, exist_ok=True)


    print(f"\n[RUN] {case}")


    nii = nib.load(str(input_ct))
    data = nii.get_fdata().astype(np.int16)
    affine = nii.affine
    nii_spacing = nii.header.get_zooms()


    spacing, row, col = compute_geometry(affine)
    validate(data, spacing, nii_spacing)


    origin = affine[:3, 3]
    slice_vec = affine[:3, 2]


    ras_to_lps = np.diag([-1, -1, 1])


    study_uid = generate_uid()
    series_uid = generate_uid()


    Z = data.shape[2]


    for i in range(Z):


        if i % 20 == 0:
            print(f"[INFO] Writing slice {i+1}/{Z}")


        slice_data = data[:, :, i]


        # Slice position
        position = origin + i * slice_vec
        position = ras_to_lps @ position


        # ==========================
        # File Meta (REQUIRED FIX)
        # ==========================
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()


        ds = pydicom.Dataset()
        ds.file_meta = file_meta


        # ==========================
        # Required DICOM UIDs
        # ==========================
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID


        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid


        # ==========================
        # Metadata
        # ==========================
        ds.PatientName = "Verse20^Subject"
        ds.PatientID = case
        ds.Modality = "CT"


        ds.InstanceNumber = i + 1


        # ==========================
        # Geometry (CRITICAL)
        # ==========================
        ds.ImagePositionPatient = list(map(float, position))
        ds.ImageOrientationPatient = list(map(float, row)) + list(map(float, col))


        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.SliceThickness = float(spacing[2])
        ds.SpacingBetweenSlices = float(spacing[2])


        # ==========================
        # Image Data
        # ==========================
        ds.Rows, ds.Columns = slice_data.shape


        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"


        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1


        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1


        ds.PixelData = slice_data.tobytes()


        # ==========================
        # Save
        # ==========================
        filename = output_dir / f"{case}_slice_{i:04d}.dcm"


        pydicom.dcmwrite(
            str(filename),
            ds,
            write_like_original=False,
            enforce_file_format=True
        )


    print(f"[SUCCESS] {case} converted")




# ==============================
# Main
# ==============================


def main():
    for case in get_case_list():


        input_ct = RAWDATA_DIR / case / f"{case}_dir-ax_ct.nii.gz"
        output_dir = DICOM_BASE_DIR / case


        if not input_ct.exists():
            print(f"[WARNING] Missing {case}")
            continue


        convert_case(input_ct, output_dir, case)


    print("\n✅ All cases processed.")

if __name__ == "__main__":
    main()
