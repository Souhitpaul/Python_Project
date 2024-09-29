import os
import glob
import cv2
import numpy as np
import nibabel as nib
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import kurtosis, skew
from skimage.measure import shannon_entropy
import imgaug.augmenters as iaa

# Data Augmentation
def augment_image(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),  # vertical flips
        iaa.Affine(rotate=(-20, 20)),  # random rotations
        iaa.Multiply((0.8, 1.2)),  # change brightness
        iaa.GaussianBlur(sigma=(0, 1.0))  # Gaussian blur
    ])
    return seq.augment_image(image)

# Feature Extraction Functions
def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], symmetric=True, normed=True)
    return [graycoprops(glcm, prop)[0, 0] for prop in ['contrast', 'correlation', 'energy', 'homogeneity']]

def extract_intensity_features(image):
    return [np.mean(image), np.std(image), shannon_entropy(image), kurtosis(image, axis=None), skew(image, axis=None)]

def extract_lbp_features(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    return hist.astype("float") / (hist.sum() + 1e-6)

def extract_features(image):
    return np.hstack([extract_glcm_features(image), extract_intensity_features(image), extract_lbp_features(image)])

# Image Processing Functions
def process_normal_image(image_path):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = augment_image(original_image)  # Apply augmentation

    # Apply darkish filter and Gaussian Blur
    dark_filter_factor = 0.5
    dark_image = np.uint8(original_image * dark_filter_factor)
    blurred_image = cv2.GaussianBlur(dark_image, (15, 11), 20)

    # Apply K-Means Segmentation
    pixel_values = blurred_image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 6
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(blurred_image.shape)

    # Histogram Equalization
    equalized_image = cv2.equalizeHist(segmented_image)

    return extract_features(equalized_image)

def process_nii_image(image_path):
    nii_image = nib.load(image_path)
    middle_slice = nii_image.get_fdata()[:, :, nii_image.shape[2] // 2]
    original_image = np.uint8(cv2.normalize(middle_slice, None, 0, 255, cv2.NORM_MINMAX))
    original_image = augment_image(original_image)  # Apply augmentation

    # Simulate T2-weighted and apply darkish filter + Gaussian blur
    t2_weighted_image = cv2.pow(original_image / 255.0, 10)
    t2_weighted_image = np.uint8(t2_weighted_image * 255)
    dark_filter_factor = 0.5
    dark_t2_weighted_image = np.uint8(t2_weighted_image * dark_filter_factor)
    blurred_image = cv2.GaussianBlur(dark_t2_weighted_image, (5, 15), 10)

    # Apply K-Means Segmentation
    pixel_values = blurred_image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 11
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(blurred_image.shape)

    # Histogram Equalization
    equalized_image = cv2.equalizeHist(segmented_image)

    return extract_features(equalized_image)

# Folder Processing
def process_folder(folder_yes, folder_no):
    X, y = [], []
    for folder, label in [(folder_yes, 1), (folder_no, 0)]:
        image_paths = glob.glob(os.path.join(folder, '*.nii')) + glob.glob(os.path.join(folder, '*.jpg')) + glob.glob(os.path.join(folder, '*.jpeg'))
        for image_path in image_paths:
            features = process_nii_image(image_path) if image_path.endswith('.nii') else process_normal_image(image_path)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Feature Reduction
def reduce_features(X_train, X_test, n_components=10):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_train), pca.transform(X_test)

# SVM Classifier with Hyperparameter Tuning
def apply_svm_with_tuning(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_

# Main Function
def main():
    folder_yes = 'dataset/yes'
    folder_no = 'dataset/no'
    
    X, y = process_folder(folder_yes, folder_no)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature reduction
    X_train_reduced, X_test_reduced = reduce_features(X_train, X_test)

    # Apply SVM with hyperparameter tuning
    best_params, best_model = apply_svm_with_tuning(X_train_reduced, y_train)
    print(f"Best Parameters: {best_params}")

    # Evaluate the model
    y_pred = best_model.predict(X_test_reduced)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Use GPU for TensorFlow if available
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":
    main()









