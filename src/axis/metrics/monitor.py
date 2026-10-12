"""Quick registration-quality summaries used to monitor training.

Image similarity (NCC, mutual information, SSIM) and label-based spot metrics
(spatial cross-correlation, centroid shift, alpha-shape Dice) between a fixed
sample and a warped moving sample. The metrics reported in the manuscript are
in [`axis.metrics.alignment`][axis.metrics.alignment]. Requires the ``metrics`` extra
(``alphashape``, ``shapely``, ``scikit-image``).
"""

import math
from collections import defaultdict

import alphashape
import numpy as np
import scipy.ndimage
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from skimage.draw import polygon
from skimage.metrics import structural_similarity as ssim


### ----------------------- IMAGE EVALUATION FUNCTIONS -----------------------
def normalized_cross_correlation(img_fixed, img_moving):
    img_fixed = img_fixed - np.mean(img_fixed)
    img_moving = img_moving - np.mean(img_moving)
    numerator = np.sum(img_fixed * img_moving)
    denominator = np.sqrt(np.sum(img_fixed**2) * np.sum(img_moving**2))
    return numerator / denominator


def mutual_information(img_fixed, img_moving, bins=256):
    """
    Computes the Mutual Information (MI) between two images.

    Args:
        img_fixed: Fixed/reference image.
        img_moving: Moving/transformed image.
        bins: Number of histogram bins (default: 256).

    Returns:
        Mutual Information (MI) score.
    """

    # Step 1: Compute Joint Histogram
    joint_hist, x_edges, y_edges = np.histogram2d(img_fixed.ravel(), img_moving.ravel(), bins=bins)

    # Step 2: Normalize to Get Joint Probability Distribution
    joint_prob = joint_hist / joint_hist.sum()

    # Step 3: Compute Marginal Distributions
    prob_fixed = joint_prob.sum(axis=1)  # Summing along y-axis gives P(fixed)
    prob_moving = joint_prob.sum(axis=0)  # Summing along x-axis gives P(moving)

    # Step 4: Compute Entropies from Probability Distributions
    h_fixed = -np.sum(prob_fixed * np.log2(prob_fixed + 1e-9))  # Entropy of fixed image
    h_moving = -np.sum(prob_moving * np.log2(prob_moving + 1e-9))  # Entropy of moving image
    h_joint = -np.sum(joint_prob * np.log2(joint_prob + 1e-9))  # Joint entropy

    # Step 5: Compute Mutual Information
    mi = h_fixed + h_moving - h_joint

    return mi


def compute_ssim(img_fixed, img_moving):
    return ssim(img_fixed, img_moving, data_range=img_moving.max() - img_moving.min(), channel_axis=-1)


### ----------------------- SPOT EVALUATION FUNCTIONS -----------------------
def spatial_cross_correlation(coords_A, labels_A, coords_B, labels_B, grid_size=200, sigma=2):
    """
    Computes and visualizes spatial cross-correlation between two sets of labeled points.

    Args:
        coords_A, labels_A: Coordinates and labels for dataset A.
        coords_B, labels_B: Coordinates and labels for dataset B.
        grid_size: Size of the grid for density estimation.
        sigma: Standard deviation for Gaussian smoothing.
        visualize: Whether to generate visualization plots.

    Returns:
        Mean normalized spatial cross-correlation score.
    """
    # Step 1: Normalize coordinates to fit within [0, grid_size-1]
    all_coords = np.vstack((coords_A, coords_B))
    x_min, y_min = np.min(all_coords, axis=0)
    x_max, y_max = np.max(all_coords, axis=0)

    def normalize(coords):
        return ((coords - [x_min, y_min]) / ([x_max - x_min, y_max - y_min]) * (grid_size - 1)).astype(int)

    norm_coords_A = normalize(coords_A)
    norm_coords_B = normalize(coords_B)

    # Step 2: Compute density maps and correlation scores
    unique_classes = np.unique(labels_A)
    correlation_scores = {}
    density_maps_A = {}
    density_maps_B = {}

    for c in unique_classes:
        density_A = np.zeros((grid_size, grid_size))
        density_B = np.zeros((grid_size, grid_size))

        for x, y in norm_coords_A[labels_A == c]:
            density_A[y, x] += 1  # Now safely within [0, grid_size-1]
        for x, y in norm_coords_B[labels_B == c]:
            density_B[y, x] += 1  # Now safely within [0, grid_size-1]

        # Apply Gaussian smoothing
        density_A = scipy.ndimage.gaussian_filter(density_A, sigma=sigma)
        density_B = scipy.ndimage.gaussian_filter(density_B, sigma=sigma)

        # Normalize correlation using energy normalization

        numerator = np.sum(density_A * density_B)
        denominator = np.sqrt(np.sum(density_A**2) * np.sum(density_B**2))
        normalized_correlation = numerator / (denominator + 1e-9)  # Avoid division by zero

        # Store results
        correlation_scores[c] = normalized_correlation
        density_maps_A[c] = density_A
        density_maps_B[c] = density_B

    return np.mean(list(correlation_scores.values()))


def compute_region_based_centroid_shift_balanced(region_masks_A, region_masks_B):

    shifts = []

    # Iterate over class labels in region_masks_A
    for class_label in region_masks_A.keys():
        coords_A = region_masks_A[class_label]
        coords_B = region_masks_B[class_label]

        centroids_A = np.mean(coords_A, axis=0)
        centroids_B = np.mean(coords_B, axis=0)

        dx = centroids_A[0] - centroids_B[0]  # 1488.26794226 - 1544.3325
        dy = centroids_A[1] - centroids_B[1]  # 418.61993267 - 432.871365
        dist = math.sqrt(dx**2 + dy**2)

        # Compute the centroid shifts for matched pairs
        shifts.append(dist)

    return np.mean(np.asarray(shifts)) if shifts else 0.0


def mean_centroid_shift(coords_A, labels_A, coords_B, labels_B):

    regions_A = {}
    regions_B = {}
    labels = np.unique(labels_A)
    for class_label in labels:
        region_coords_A = coords_A[(labels_A == class_label)]
        region_coords_B = coords_B[(labels_B == class_label)]

        regions_A[class_label] = region_coords_A
        regions_B[class_label] = region_coords_B

    region_mean_shift = compute_region_based_centroid_shift_balanced(regions_A, regions_B)

    return region_mean_shift


def alpha_shape(points, alpha=0.5):
    """
    Computes the alpha shape (concave hull) of a set of points.

    Args:
        points (np.ndarray): Array of (x, y) coordinates.
        alpha (float): Alpha parameter controlling shape tightness.

    Returns:
        shapely.geometry.Polygon or MultiPolygon: The outer boundary polygon.
    """
    if len(points) < 3:
        return None  # A polygon cannot be formed

    return alphashape.alphashape(points, alpha)


def create_boundary_mask(coords, shape, alpha=0.5):
    mask = np.zeros(shape, dtype=np.uint8)
    boundary = alpha_shape(coords, alpha)
    if boundary is None:
        return mask  # No valid boundary found

    # Handle different geometry types
    if isinstance(boundary, Polygon):
        boundary = [np.array(boundary.exterior.coords)]
    elif isinstance(boundary, MultiPolygon):
        boundary = [np.array(poly.exterior.coords) for poly in boundary.geoms]
    elif isinstance(boundary, GeometryCollection):
        boundary = [np.array(geom.exterior.coords) for geom in boundary.geoms if isinstance(geom, Polygon)]

    if not boundary:  # If no valid polygons were extracted
        return mask

    for poly in boundary:
        rr, cc = polygon(poly[:, 1], poly[:, 0], shape)
        mask[rr, cc] = 1  # Fill the region

    return mask


def dice_coefficient(mask_A, mask_B):
    """
    Computes the Dice coefficient between two binary masks.

    Args:
        mask_A (np.ndarray): Binary mask for dataset A.
        mask_B (np.ndarray): Binary mask for dataset B.

    Returns:
        float: Dice similarity coefficient.
    """
    intersection = np.sum(mask_A & mask_B)
    sum_masks = np.sum(mask_A) + np.sum(mask_B)
    return (2.0 * intersection) / sum_masks if sum_masks > 0 else 0.0


def compute_region_dice_score(coords_A, labels_A, coords_B, labels_B, mask_shape=(512, 512), alpha=0.01):
    """
    Computes the Dice score for each class label using the exact outer boundaries.

    Args:
        region_masks_A: Dictionary mapping class labels to (coordinates, region labels) for dataset A.
        region_masks_B: Dictionary mapping class labels to (coordinates, region labels) for dataset B.
        mask_shape (tuple): Fixed shape for binary masks (height, width).
        alpha (float): Alpha parameter for boundary tightness.

    Returns:
        dict: Class-wise Dice scores.
        float: Average Dice score across all classes.
    """

    region_dice_scores = {}
    labels = np.unique(labels_B)

    regions_A = {}
    regions_B = {}
    for class_label in labels:
        # class_label=6
        classid_A = labels_A == class_label
        region_coords_A = coords_A[classid_A]

        classid_B = labels_B == class_label
        region_coords_B = coords_B[classid_B]

        mask_A = create_boundary_mask(region_coords_A, mask_shape, alpha)
        mask_B = create_boundary_mask(region_coords_B, mask_shape, alpha)

        dice_score = dice_coefficient(mask_A, mask_B)
        region_dice_scores[class_label] = dice_score

        regions_A[class_label] = (region_coords_A, class_label)
        regions_B[class_label] = (region_coords_B, class_label)
    average_dice_score = np.mean(list(region_dice_scores.values())) if region_dice_scores else 0.0

    return average_dice_score


def evaluate_single_pair(img_fixed, img_moving, coords_A, coords_B, labels_A, labels_B):

    results = {
        "Class-wise Dice Coefficient": compute_region_dice_score(coords_A, labels_A, coords_B, labels_B),
        "Spatial Cross-Correlation": spatial_cross_correlation(coords_A, labels_A, coords_B, labels_B),
        "Mean Centroid Shift": mean_centroid_shift(
            coords_A,
            labels_A,
            coords_B,
            labels_B,
        ),
        "Mutual Information": mutual_information(img_fixed, img_moving),
        "SSIM": compute_ssim(img_fixed, img_moving),
        "NCC": normalized_cross_correlation(img_fixed, img_moving),
    }
    print(results)
    return results


def evaluate_multiple_pairs(pairs):
    """
    Evaluate and average metrics across multiple image/coordinate/label pairs.

    Args:
        pairs: list of tuples, each containing
               (img_fixed, img_moving, coords_A, coords_B, labels_A, labels_B)

    Returns:
        A dictionary of average evaluation metrics.
    """
    metrics_accumulator = defaultdict(list)

    for i, (img_fixed, img_moving, coords_A, coords_B, labels_A, labels_B) in enumerate(pairs):
        results = evaluate_single_pair(img_fixed, img_moving, coords_A, coords_B, labels_A, labels_B)
        for k, v in results.items():
            metrics_accumulator[k].append(v)

    averaged_metrics = {k: np.mean(v) for k, v in metrics_accumulator.items()}
    return averaged_metrics
