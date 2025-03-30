import itk
import os
import numpy as np
from tqdm import tqdm

def one_hot_encode_labels(label_array, num_classes=3):
    shape = label_array.shape
    one_hot = np.zeros(shape + (num_classes,), dtype=np.float32)
    for i in range(1, num_classes + 1):
        one_hot[..., i - 1] = (label_array == i).astype(np.float32)
    return one_hot

def buildProbabilisticAtlas(image_dir, mask_dir, parameter_file_path):
    # Get sorted filenames
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")])

    assert len(image_files) == len(mask_files), "Mismatch between number of images and masks"

    # Load reference image and mask (first one)
    fixed_image = itk.imread(os.path.join(image_dir, image_files[0]), itk.F)
    result = one_hot_encode_labels(itk.array_from_image(itk.imread(os.path.join(mask_dir, mask_files[0]), itk.UC)))

    # Load parameter file
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(parameter_file_path)
    parameter_object.SetParameter(0, "WriteResultImage", "true")

    for img_file, mask_file in tqdm(zip(image_files[1:], mask_files[1:]), total=len(image_files)-1, desc="Registering"):
        moving_image = itk.imread(os.path.join(image_dir, img_file), itk.F)
        moving_mask = itk.imread(os.path.join(mask_dir, mask_file), itk.UC)

        _ , result_transform = itk.elastix_registration_method(
            fixed_image, moving_image,
            parameter_object=parameter_object,
            log_to_console=False
        )

        result_mask = itk.transformix_filter(
            moving_mask,
            transform_parameter_object=result_transform,
            log_to_console=False
        )

        result = result + one_hot_encode_labels(itk.array_from_image(result_mask))

    result = result / len(image_files)

    # result is shape (Z, Y, X, C) — split and save each channel
    for i in range(result.shape[-1]):
        channel_img = itk.image_from_array(result[..., i])
        itk.imwrite(channel_img, f"atlas_label_{i+1}.nii.gz")

    return result

# Example usage:
# buildProbabilisticAtlas("training-set/training-images", "training-set/training-labels", "Par0010/Par0010affine.txt")