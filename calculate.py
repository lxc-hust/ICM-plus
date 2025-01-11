import numpy as np
import os
import cv2
import argparse
from scipy.ndimage import filters
from scipy.ndimage import distance_transform_edt


def matte_mse(pred_matte, gt_matte):
    ''' Mean Squared Error '''
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_mse = np.mean(np.power(pred_matte - gt_matte, 2))
    return error_mse * 0.001


def matte_sad(pred_matte, gt_matte):
    '''
    Sum of Absolute Differences

    pred_matte : np.array, shape : [h,w]
    gt_matte   : np.array, shape : [h,w]
    '''
    assert (len(pred_matte.shape) == len(gt_matte.shape))

    if pred_matte.dtype == np.uint8:
        pred_matte = pred_matte.astype(np.float32) / 255.0
    if gt_matte.dtype == np.uint8:
        gt_matte = gt_matte.astype(np.float32) / 255.0

    height, width = pred_matte.shape
    total_pixels = height * width
    error_sad = np.sum(np.abs(pred_matte - gt_matte)) / total_pixels

    return error_sad


def matte_grad(pred_matte, gt_matte):
    ''' Error measure with Gradient '''
    assert (len(pred_matte.shape) == len(gt_matte.shape))

    if pred_matte.dtype == np.uint8:
        pred_matte = pred_matte / 255.0
    if gt_matte.dtype == np.uint8:
        gt_matte = gt_matte / 255.0

    predict_grad = filters.gaussian_filter(pred_matte, 1.4, order=1)
    gt_grad = filters.gaussian_filter(gt_matte, 1.4, order=1)
    error_grad = np.sum(np.power(predict_grad - gt_grad, 2))
    return error_grad


def matte_conn(pred_matte, gt_matte, theta=0.15):
    """
    Calculate the connectivity error between the predicted and ground truth alpha mattes.

    Parameters:
        pred_matte (np.ndarray): Predicted alpha matte, values in range [0, 1].
        gt_matte (np.ndarray): Ground truth alpha matte, values in range [0, 1].
        theta (float): Threshold for connectivity error computation.

    Returns:
        float: Connectivity error.
    """
    # Ensure input mattes are numpy arrays
    pred_matte = np.clip(np.asarray(pred_matte), 0, 1)
    gt_matte = np.clip(np.asarray(gt_matte), 0, 1)

    # Define fully opaque regions (alpha = 1)
    source_region = gt_matte == 1

    # Compute the connectivity for the ground truth
    li_gt = distance_transform_edt(~source_region)
    li_gt[gt_matte < 1] = 0

    # Compute the connectivity for the predicted matte
    li_pred = distance_transform_edt(~source_region)
    li_pred[pred_matte < 1] = 0

    # Calculate connectivity differences
    di = pred_matte - li_pred
    gt_di = gt_matte - li_gt

    # Degree of connectivity for predicted and ground truth mattes
    conn_pred = 1 - (np.clip(di, 0, theta) / theta)
    conn_gt = 1 - (np.clip(gt_di, 0, theta) / theta)

    # Compute pixel-wise connectivity differences
    connectivity_error = np.sum(np.abs(conn_pred - conn_gt)) / np.prod(gt_matte.shape)

    return connectivity_error * 1000


def main(pred_folder, gt_folder):
    sad_errors = []
    mse_errors = []
    grad_errors = []
    conn_errors = []

    for filename in os.listdir(pred_folder):
        pred_path = os.path.join(pred_folder, filename)
        gt_path = os.path.join(gt_folder, filename)

        pred_matte = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_matte = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        sad_error = matte_sad(pred_matte, gt_matte)
        mse_error = matte_mse(pred_matte, gt_matte)
        grad_error = matte_grad(pred_matte, gt_matte)
        conn_error = matte_conn(pred_matte, gt_matte)

        sad_errors.append(sad_error)
        mse_errors.append(mse_error)
        grad_errors.append(grad_error)
        conn_errors.append(conn_error)

    print("SAD Errors Mean:", np.mean(sad_errors))
    print("MSE Errors Mean:", np.mean(mse_errors))
    print("Gradient Errors Mean:", np.mean(grad_errors))
    print("Conn Errors Mean:", np.mean(conn_errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Matte Metrics")
    parser.add_argument('--pred_folder', type=str, required=True, help="Path to predicted matte folder")
    parser.add_argument('--gt_folder', type=str, required=True, help="Path to ground truth matte folder")

    args = parser.parse_args()
    main(args.pred_folder, args.gt_folder)



