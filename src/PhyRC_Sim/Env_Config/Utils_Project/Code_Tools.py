import os
import numpy as np
import math
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Slerp, Rotation
import torch

def get_unique_filename(base_filename, extension=".png"):
    counter = 0
    filename = f"{base_filename}_{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_filename}_{counter}{extension}"
        
    return filename

def float_truncate(num):
    '''
    Keep four decimal places.
    '''
    result = math.trunc(num * 1e4) / 1e4
    
    return result


def dense_trajectory_points_generation(start_pos:np.ndarray, end_pos:np.ndarray, start_quat:np.ndarray=None, end_quat:np.ndarray=None, num_points:int=50):
    '''
    generate dense trajectory points for inverse kinematics control.
    '''
    # ---- 1. Generate five sample points (including start and end) ----
    distance = np.linalg.norm(end_pos - start_pos)
    # distance = torch.norm(torch.tensor(end_pos).to("cuda:0") - torch.tensor(start_pos)).item()
    # print(distance)
    initial_sample_points_num = 5
    initial_sample_points = np.linspace(start_pos, end_pos, initial_sample_points_num)

    # ---- 2. Fit B-spline to sample points to generate smooth trajectory ----
    tck, u = splprep(initial_sample_points.T, s=0)  # B-spline fitting
    u_new = np.linspace(0, 1, num_points)  # Finer sampling
    interp_pos = np.array(splev(u_new, tck)).T  # Interpolated smooth trajectory
    # print(interp_pos.shape)
    
    # ---- 3. Perform spherical linear interpolation (Slerp) for rotation quaternions ----
    if start_quat is not None and end_quat is not None:
        rotations = Rotation.from_quat([start_quat, end_quat])  # Convert quaternions to rotation objects
        slerp = Slerp([0, 1], rotations) # Slerp interpolator
        interp_times = np.linspace(0, 1, num_points)  # Interpolation time points
        interp_rotations = slerp(interp_times).as_quat() # Interpolation result converted to quaternions
        print(interp_rotations.shape)
    
        return interp_pos, interp_rotations
    
    return interp_pos

def normalize_columns(data):
    """
    Normalize each column of a (N, M) array to the range [0, 1].

    Args:
        data: np.ndarray, shape (N, M)

    Returns:
        norm_data: np.ndarray, shape (N, M) - Column-wise normalized array
    """
    min_vals = np.min(data, axis=0)     # Minimum value of each column → shape (M,)
    max_vals = np.max(data, axis=0)     # Maximum value of each column → shape (M,)
    norm_data = (data - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero
    return norm_data

import matplotlib.pyplot as plt

def plot_column_distributions(data, save_path="column_distributions.png"):
    """
    Plot histograms of each column in a (N, 2) normalized array and save the figure to file.

    Args:
        data: np.ndarray of shape (N, 2)
        save_path: str, path to save the image
    """
    plt.figure(figsize=(12, 4))

    # Column 0
    plt.subplot(1, 2, 1)
    plt.hist(data[:, 0], bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Column 0")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Column 1
    plt.subplot(1, 2, 2)
    plt.hist(data[:, 1], bins=50, color='salmon', edgecolor='black')
    plt.title("Distribution of Column 1")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(save_path)  # Save figure
    plt.close()  # Close the figure to free memory