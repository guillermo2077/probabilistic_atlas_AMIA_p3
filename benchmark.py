import os
import itk
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from predictor import BayesianPredictor

def compute_metrics(prediction, ground_truth, background_class=0):
    # Ensure both prediction and ground_truth are numpy arrays and have the same shape
    assert prediction.shape == ground_truth.shape, "Prediction and ground truth must have the same shape"
    
    # Flatten the volumes to 1D arrays
    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()

    # Exclude background class from metrics
    valid_mask = ground_truth != background_class
    prediction_filtered = prediction[valid_mask]
    ground_truth_filtered = ground_truth[valid_mask]
    
    # Calculate global accuracy
    global_accuracy = accuracy_score(ground_truth_filtered, prediction_filtered)
    
    # Calculate per-class recall and F1 score
    num_classes = np.unique(ground_truth)
    recalls = {}
    f1_scores = {}
    
    for class_id in num_classes:
        # Exclude the background class when computing metrics for other classes
        if class_id == background_class:
            continue
        true_class_mask = ground_truth == class_id
        prediction_class_mask = prediction == class_id
        recall = recall_score(true_class_mask, prediction_class_mask, average='binary')
        f1 = f1_score(true_class_mask, prediction_class_mask, average='binary')
        
        recalls[class_id] = recall
        f1_scores[class_id] = f1
    
    # Confusion matrix for detailed analysis
    cm = confusion_matrix(ground_truth_filtered, prediction_filtered, labels=num_classes)
    
    # Calculate global recall and F1 score
    global_recall = recall_score(ground_truth_filtered, prediction_filtered, average='macro')
    global_f1 = f1_score(ground_truth_filtered, prediction_filtered, average='macro')
    
    return {
        'global_accuracy': global_accuracy,
        'global_recall': global_recall,
        'global_f1_score': global_f1,
        'per_class_recall': recalls,
        'per_class_f1_score': f1_scores,
        'confusion_matrix': cm
    }

def benchmark_prediction(prediction_path, ground_truth_path, transform, ignore_labels=None):
    """
    Evaluate the accuracy of predicted labels against ground truth
    
    Args:
        prediction_path: Path to the predicted label volume (.nii.gz)
        ground_truth_path: Path to the ground truth label volume (.nii.gz)
        transform: ITK transform object to apply to the ground truth labels to match the prediction 
        ignore_labels: List of label values to ignore in the evaluation (e.g. background)
    
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    # Load the volumes
    prediction = itk.array_from_image(itk.imread(prediction_path, itk.UC))
    ground_truth = itk.imread(ground_truth_path, itk.UC)

    ground_truth = itk.array_from_image(itk.transformix_filter(
        ground_truth,
        transform_parameter_object=transform,
        log_to_console=False
    ))

    print(np.unique(ground_truth))

    # TODO Some have 4 and 255? what does this mean?
    ground_truth[ground_truth>3] = 0

    print(np.unique(ground_truth))

    # Create a new array of zeros with the target size
    padded_gt = np.zeros(prediction.shape, dtype=ground_truth.dtype)
    
    # Determine the slices for placing the original array into the padded array
    slices = tuple(slice(0, min(d, prediction.shape[i])) for i, d in enumerate(ground_truth.shape))
    
    # Paste the original array over the zeros in the padded array
    padded_gt [slices] = ground_truth


    # Save middle slice of prediction and ground truth
    middle_slice = prediction.shape[0] // 2
    plt.imsave('prediction_slice.png', prediction[middle_slice])
    plt.imsave('ground_truth_slice.png', padded_gt[middle_slice])    
    
    return compute_metrics(prediction, padded_gt)

if __name__ == "__main__":
    image_files = sorted([os.path.join("test-images", "testing-images", f) for f in os.listdir(os.path.join("test-images", "testing-images")) if f.endswith(".nii.gz")])
    label_files = sorted([os.path.join("test-images", "testing-labels", f) for f in os.listdir(os.path.join("test-images", "testing-labels")) if f.endswith(".nii.gz")])

    predictor = BayesianPredictor(
        "training-set/training-images/1000.nii.gz",
        "Par0010/Par0010affine.txt",
        model_dir="results",
    )

    out_path = os.path.join("results", "inference", f"inference_{os.path.basename(image_files[0])}")

    _, transform = predictor.inference(image_files[0], save_path=out_path, skip_prediction=True)

    metrics = benchmark_prediction(out_path, label_files[0], transform)
    print(metrics)




