import os
import itk
import numpy as np
from tqdm import tqdm
from buildAtlas import buildProbabilisticAtlas, loadModel
from buildTissueModel import buildTissueModel, getProbabilities
from preprocessing import rescale_intensity


class BayesianPredictor:
    def __init__(self, refereceImage, parameterPath, training_dir=None, model_dir=None):
        self.refereceImage = rescale_intensity(itk.imread(refereceImage, itk.F))
        self.training_dir = training_dir
        self.model_dir = model_dir
        self.parameter_object = itk.ParameterObject.New()
        self.parameter_object.AddParameterFile(parameterPath)
        self.parameter_object.SetParameter(0, "WriteResultImage", "true")

        assert (
            self.training_dir is not None or self.model_dir is not None
        ), "Either training_dir or model_dir must be provided"

        self.train()

    def train(self):
        if self.training_dir is not None:
            self.atlasModel = buildProbabilisticAtlas(
                os.path.join(self.training_dir, "training-images"),
                os.path.join(self.training_dir, "training-labels"),
                "Par0010/Par0010affine.txt",
            )
            self.tissueModel = buildTissueModel(
                os.path.join(self.training_dir, "training-images"),
                os.path.join(self.training_dir, "training-labels"),
            )
        elif self.model_dir is not None:
            self.atlasModel = loadModel(
                [
                    os.path.join(self.model_dir, f"atlas_label_{i+1}.nii.gz")
                    for i in range(4)
                ]
            )
            self.tissueModel = np.load(os.path.join(self.model_dir, "tissue_model.npy"))

    def inference(self, image_path, save=True):
        image = rescale_intensity(itk.imread(image_path, itk.F))
        result = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

        image, _ = itk.elastix_registration_method(
            self.refereceImage,
            image,
            parameter_object=self.parameter_object,
            log_to_console=False,
        )

        # print("Image debug information after registration:")
        # print(f"Shape: {image.shape}")
        # print(f"Data type: {image.dtype}")
        # print(f"Min value: {np.min(image)}")
        # print(f"Max value: {np.max(image)}")
        # print(f"Mean value: {np.mean(image)}")
        # print(f"Standard deviation: {np.std(image)}")

        total_voxels = image.shape[0] * image.shape[1] * image.shape[2]
        with tqdm(total=total_voxels, desc="Processing voxels") as pbar:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        intensity = image[i, j, k]
                        result[i, j, k] = np.argmax(
                            self.atlasModel[i, j, k]
                            * getProbabilities(self.tissueModel, intensity)
                        )
                        pbar.update(1)
        if save:
            itk.imwrite(
                itk.image_from_array(result),
                os.path.join(
                    "results",
                    "inference",
                    f"inferenceResult_{os.path.basename(image_path)}.nii.gz",
                ),
            )

        return result


if __name__ == "__main__":
    predictor = BayesianPredictor(
        "training-set/training-images/1000.nii.gz",
        "Par0010/Par0010affine.txt",
        training_dir="training-set",
    )
    result = predictor.inference("test images/testing-images/1003.nii.gz")
