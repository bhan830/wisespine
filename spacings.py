#!/usr/bin/env python3

import os
import json
import numpy as np
import nibabel as nib

MODEL_NAME = "TotalSegmentator"

PRED_BASE = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/models"
OUTPUT_BASE = "/gscratch/scrubbed/bhan830/wisespine/wisespine_new/baseline_outputs/centroids"

# Proper vertebra order
VERTEBRA_ORDER = [
    "C1","C2","C3","C4","C5","C6","C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5"
]

def compute_centroid(mask_path):
    img = nib.load(mask_path)
    data = img.get_fdata()
    affine = img.affine

    coords = np.argwhere(data > 0)
    if coords.size == 0:
        return None

    voxel_centroid = coords.mean(axis=0)
    world_centroid = nib.affines.apply_affine(affine, voxel_centroid)

    return world_centroid


def main():

    model_dir = os.path.join(PRED_BASE, MODEL_NAME)

    for case in sorted(os.listdir(model_dir)):

        case_dir = os.path.join(model_dir, case)
        if not os.path.isdir(case_dir):
            continue

        print(f"\nProcessing {case}")

        centroids = []
        label_counter = 1

        for vert in VERTEBRA_ORDER:

            filename = f"{case}_{MODEL_NAME}_vertebrae_{vert}.nii.gz"
            mask_path = os.path.join(case_dir, filename)

            if not os.path.exists(mask_path):
                continue

            centroid = compute_centroid(mask_path)

            if centroid is None:
                continue

            centroids.append({
                "label": label_counter,
                "X": float(centroid[0]),
                "Y": float(centroid[1]),
                "Z": float(centroid[2])
            })

            label_counter += 1

        if len(centroids) == 0:
            print(f"[INFO] No centroids found for {case}")
            continue

        output_data = [{"direction": ["L", "P", "S"]}]
        output_data.extend(centroids)

        output_dir = os.path.join(OUTPUT_BASE, MODEL_NAME, case)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir,
            f"{case}_{MODEL_NAME}_centroids.json"
        )

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"✅ Saved: {output_file}")

    print("\n🎉 Centroid extraction complete.")


if __name__ == "__main__":
    main()