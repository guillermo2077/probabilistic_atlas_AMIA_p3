import itk
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
from preprocessing import rescale_intensity


def getProbabilities(tissueModel, intensity):
    assert (
        tissueModel.shape[1] > intensity
        and intensity >= 0
    ), f"Invalid intensity value {intensity}"

    # print(intensity)
    # print(type(intensity))
    # print(tissueModel.shape)
    # print(tissueModel[...][intensity])

    return tissueModel[ : , intensity]


def buildTissueModel(image_dir, mask_dir, save = True):
    # Get sorted filenames
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")])

    assert len(image_files) == len(
        mask_files
    ), "Mismatch between number of images and masks"

    # Initialize histogram counters
    intensity_range = 256  # Assuming 8-bit images
    num_labels = 4
    intensity_label_counts = np.zeros((num_labels, intensity_range))

    # Process all images
    for img_file, mask_file in tqdm(
        zip(image_files, mask_files),
        total=len(image_files),
        desc="Building tissue model",
    ):
        # Load image and mask
        image = itk.array_from_image(rescale_intensity(
            itk.imread(os.path.join(image_dir, img_file), itk.F)
        ))
        mask = itk.array_from_image(
            itk.imread(os.path.join(mask_dir, mask_file), itk.UC)
        )
        
        # Count occurrences of each intensity-label pair
        for label in range(num_labels):
            mask_binary = mask == label
            for intensity in range(intensity_range):
                intensity_binary = image == intensity
                intensity_label_counts[label, intensity] += np.sum(
                    mask_binary & intensity_binary
                )

    # Convert counts to probabilities
    tissue_model = np.zeros_like(intensity_label_counts, dtype=np.float32)
    # Counts for each intensity value
    total_counts = intensity_label_counts.sum(axis=0, keepdims=True)
    tissue_model = intensity_label_counts / total_counts

    # Save the tissue model
    if save:
        np.save(os.path.join("results", "tissue_model.npy"), tissue_model)

    return tissue_model


# Example usage:
if __name__ == "__main__":
    model = buildTissueModel(
        "training-set/training-images", "training-set/training-labels"
    )

    # Visualize the tissue model
    plt.figure(figsize=(12, 8))
    for label in range(model.shape[0]):
        plt.plot(model[label, :], label=f"Label {label}")
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.title("Tissue Model Probability Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig("tissue_model_visualization.png")
