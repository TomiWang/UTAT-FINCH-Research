# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:58:56 2025

The purpose of this code is to solve the Linear Mixing Model (LMM),
a mathematical model used in hyperspectral imaging. Hyperspectral sensors collect
light reflected from Earth's surface with many spectral bands within the visible
spectrum. These spectal bands are collected as FINCH observes the area with mixed
materials (e.g. soil, crop residue, vegetation), called unique spectral signatures.
The goal of LMM is to analyze and determine how much of each material is present in
the observed spectrum. We are unmixing hyperspectral data!

The core equation looks like this:
                                x = S * a
Where:
    - x (Abundance vector): The observed spectrum (1D array), what the spectrum observes. Observes each
                            from one pixel, this is the unique "colour fingerprint". Dimensions [1 * Bands]
    - S (Endmember Matrix): The endmember matrix (2D array), each column represents the unique spectal
                            ignature of material (endmember). Dimensions: [Bands * Endmembers]
    - a (Measured Reflectance): The abundance vector (1D array), shows how much of each material is present
                                in the mixture per pixel. Dimensions [Endmembers * 1]

@author: Tomi Wang :)
"""

import sys
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize
from scipy.spatial import ConvexHull
from numpy.linalg import lstsq, cholesky, qr, svd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------------
# Function: load_and_preprocess_data
# Topics Covered:
#    - Data Loading and Preprocessing for LMM.
#    - Normalization of spectral data.
#    - Handling missing data.
# ---------------------------------------------------------------------------------
def load_and_preprocess_data(file_path: str,
                             normalize: bool = True,
                             fill_missing: bool = True) -> np.ndarray:
    """
    Loads and preprocesses hyperspectral data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV data file.
        normalize (bool): If True, each band is normalized to [0, 1].
        fill_missing (bool): If True, fill missing values with column means.

    Returns:
        np.ndarray: Preprocessed data array.
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

    # Select numeric columns only
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        raise ValueError("The provided data file does not contain numeric columns.")

    # Fill missing values with the column mean if necessary
    if fill_missing:
        numeric_data = numeric_data.fillna(numeric_data.mean())

    data_array = numeric_data.to_numpy()
    if data_array.shape[0] < 1 or data_array.shape[1] < 1:
        raise ValueError("Data must have at least one sample and one band.")

    # Normalize each column to [0, 1] (avoid division by zero)
    if normalize:
        min_vals = np.min(data_array, axis=0)
        max_vals = np.max(data_array, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        data_array = (data_array - min_vals) / range_vals

    return data_array


# ---------------------------------------------------------------------------------
# Function: prepare_matrices
# Topics Covered:
#    - LMM formulation: setting up the mixing equation x = S * a.
#    - Endmember Determination: Extracting endmembers using the N-FINDR algorithm
#      (a minimum volume simplex approach).
# ---------------------------------------------------------------------------------
def prepare_matrices(spectral_data: np.ndarray,
                     endmember_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares the endmember matrix (S) and the observed mixed spectrum (x)
    for the linear mixing model. Assumes the input data has samples as rows
    and spectral bands as columns. The mixed spectrum is computed as the mean
    spectrum over all samples.

    Parameters:
        spectral_data (np.ndarray): Hyperspectral data (samples x bands).
        endmember_count (int): Number of endmembers to extract.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - S: Endmember matrix with shape (bands, endmembers).
            - x: Mixed spectrum (1D array of length bands).
    """
    num_samples, num_bands = spectral_data.shape
    if num_samples < endmember_count:
        raise ValueError(f"Not enough samples to select {endmember_count} endmembers from {num_samples} samples.")

    # Use N-FINDR to extract endmembers.
    # n_findr returns an array of selected endmember spectra with shape (endmember_count, bands)
    endmember_candidates = n_findr(spectral_data, endmember_count)
    S = endmember_candidates.T  # Transpose to get shape (bands, endmembers)

    # Define the observed spectrum as the mean over all samples.
    x = np.mean(spectral_data, axis=0)

    return S, x


# ---------------------------------------------------------------------------------
# Function: linear_mixing_model
# Topics Covered:
#    - LMM formulation and solving for abundance vector (a).
#    - Unconstrained Least Squares Estimation (LSE):
#         * Basic LSE using lstsq, Cholesky, QR, or SVD.
#    - Constrained LSE:
#         * Enforcing sum-to-one and nonnegativity constraints.
#    - Regularized LSE:
#         * Adding a penalty term (lambda_reg) to stabilize the solution.
#    - Optional PCA for dimensionality reduction (geometrical interpretation).
# ---------------------------------------------------------------------------------
def linear_mixing_model(S: np.ndarray,
                        x: np.ndarray,
                        method: str = "unconstrained",
                        constraints: Optional[Dict[str, bool]] = None,
                        lambda_reg: float = 0.1,
                        optimization: Optional[str] = None) -> np.ndarray:
    """
    This function solves the LMM model by estimating the abundance vector a, based on the
    chosen method (LSE). This is the core function!! Which includes:
        1. Unconstrained:
            - Uses standard/basic Least Squares Estimation (LSE) to find a.
            - It does not impose any restrictions, so the values of a can be negative or
              add up to more than 1 (which is not realistic in real-world scenerios)
            - Equation: a = (S^T * S)^(-1) * S^T * x
        2. Constrained:
            - Does impose constraints on a:
                - Sum-to-one: The abundances must sum exactly to 1,
                  (e.g., 50% soil, 30% crop residue, 20% vegetation -> total = 1).
                - Nonnegativity: Abundances should be non-negative
                  (no negative materials in a mixture!)
            - Uses optimization methods to solve for a under these constraints.
            - Equation: minimize ||x - S * a||^2 subject to a_i >= 0 and Σa_i = 1
        3. Regularized:
            - Adds a penalty (regularization) to the solution to enforce stability/
              smoothness.
            - Very useful when data is noisy.
            - However, it does not enforce sum-to-one or nonnegativity
            - Equation: a = (S^T * S + λ_reg * I)^(-1) * S^T * x
            
    How the function works:
        - Input parameters:
            - S: Endmember matrix (columns = materials' spectral signatures).
            - x: Observed mixed spectrum (1D array).
            - Methods: The solving methods ("unconstrained", "constrained", "regularized").
            - Constraints: For constrained methods, specify sum_to_one or nonnegativity.
            - lambda_reg: Regularization weight (only for "regularized").
        - Output parameters:
            - a: The estimated abundance vector.
    """
    # Optional PCA: For dimensionality reduction (Geometrical Interpretation).
    if constraints and constraints.get("use_pca", False):
        if S.shape[0] != x.shape[0]:
            raise ValueError("The number of bands (rows) in S must match length of x for PCA.")
        # Stack S and x so that each column represents a spectral signature.
        combined = np.column_stack((S, x))
        n_components = constraints.get("pca_components", min(S.shape[1], S.shape[0]))
        pca = PCA(n_components=n_components)
        # Transpose so that PCA operates on features; then transpose back.
        combined_pca = pca.fit_transform(combined.T).T
        S = combined_pca[:, :-1]
        x = combined_pca[:, -1]

    a = None
    if method == "unconstrained":
        # --------------------------------------------------
        # Unconstrained LSE: Solve x = S * a without constraints.
        # Optimization Techniques Covered:
        #    - Cholesky decomposition.
        #    - QR decomposition.
        #    - Singular Value Decomposition (SVD).
        # --------------------------------------------------
        if optimization == "cholesky":
            G = S.T @ S
            y = S.T @ x
            L = cholesky(G)
            a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        elif optimization == "qr":
            Q, R = qr(S, mode='reduced')
            a = np.linalg.solve(R, Q.T @ x)
        elif optimization == "svd":
            U, Sigma, VT = svd(S, full_matrices=False)
            a = VT.T @ np.linalg.solve(np.diag(Sigma), U.T @ x)
        else:
            a, _, _, _ = lstsq(S, x, rcond=None)
    elif method == "constrained":
        # --------------------------------------------------
        # Constrained LSE: Enforce:
        #   - Sum-to-one constraint.
        #   - Nonnegativity constraint.
        # (A deterministic approach for fill-fraction estimation.)
        # --------------------------------------------------
        constraints_list = []
        bounds = None
        if constraints:
            if constraints.get("sum_to_one", False):
                constraints_list.append({"type": "eq", "fun": lambda a: np.sum(a) - 1})
            if constraints.get("nonnegativity", False):
                bounds = [(0, None)] * S.shape[1]
            else:
                bounds = None
        # Special case: if only nonnegativity is imposed, use NNLS
        if constraints and constraints.get("nonnegativity", False) and not constraints.get("sum_to_one", False):
            a, _ = nnls(S, x)
            return a

        def objective(a):
            return np.linalg.norm(x - S @ a) ** 2

        initial_guess = np.full(S.shape[1], 1.0 / S.shape[1])
        result = minimize(objective, initial_guess, constraints=constraints_list, bounds=bounds)
        if result.success:
            a = result.x
        else:
            raise ValueError("Optimization failed for constrained linear mixing model.")
    elif method == "regularized":
        # --------------------------------------------------
        # Regularized LSE: Incorporate a penalty term to enhance stability.
        # (Useful when data is noisy.)
        # --------------------------------------------------
        def objective(a):
            return np.linalg.norm(x - S @ a) ** 2 + lambda_reg * np.linalg.norm(a) ** 2

        initial_guess = np.full(S.shape[1], 1.0 / S.shape[1])
        result = minimize(objective, initial_guess, bounds=[(0, None)] * S.shape[1])
        if result.success:
            a = result.x
        else:
            raise ValueError("Optimization failed for regularized linear mixing model.")
    else:
        raise ValueError("Invalid method specified. Choose 'unconstrained', 'constrained', or 'regularized'.")

    return a


# ---------------------------------------------------------------------------------
# Function: plot_spectral_curves
# Topics Covered:
#    - Visualization of spectral curves.
#    - Useful for inspecting the unique spectral signatures of samples.
# ---------------------------------------------------------------------------------
def plot_spectral_curves(spectral_data: np.ndarray,
                         sample_indices: Optional[List[int]] = None,
                         title: str = "Spectral Curves") -> None:
    """
    Plots spectral curves from the hyperspectral data.

    Parameters:
        spectral_data (np.ndarray): 2D array (samples x bands).
        sample_indices (List[int], optional): Indices of samples to plot. Defaults to all samples.
        title (str): Title of the plot.
    """
    if sample_indices is None:
        sample_indices = list(range(spectral_data.shape[0]))

    plt.figure(figsize=(10, 6))
    for idx in sample_indices:
        plt.plot(spectral_data[idx], label=f"Sample {idx}")
    plt.title(title)
    plt.xlabel("Bands")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    

# ---------------------------------------------------------------------------------
# Function: plot_convex_hull
# Topics Covered:
#    - Geometrical Interpretation: Visualizing the convex hull (simplex geometry)
#      of endmember signatures.
# ---------------------------------------------------------------------------------
def plot_convex_hull(endmembers: np.ndarray,
                     labels: List[str],
                     title: str = "Convex Hull of Endmembers") -> None:
    """
    Plots the convex hull of the endmembers. Expects the endmember data in 2D or 3D.

    Parameters:
        endmembers (np.ndarray): Array of shape (num_endmembers, dimensions) where dimensions is 2 or 3.
        labels (List[str]): List of labels for each endmember.
        title (str): Title of the plot.
    """
    if endmembers.shape[1] > 3:
        print("Warning: Convex hull visualization is limited to 2D/3D.")
        return
    try:
        hull = ConvexHull(endmembers)
    except Exception as e:
        print(f"Error computing convex hull: {e}")
        return

    plt.figure()
    if endmembers.shape[1] == 2:
        for simplex in hull.simplices:
            plt.plot(endmembers[simplex, 0], endmembers[simplex, 1], 'k-')
        plt.scatter(endmembers[:, 0], endmembers[:, 1], c='red', label="Endmembers")
    elif endmembers.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for simplex in hull.simplices:
            ax.plot(endmembers[simplex, 0],
                    endmembers[simplex, 1],
                    endmembers[simplex, 2], 'k-')
        ax.scatter(endmembers[:, 0], endmembers[:, 1], endmembers[:, 2],
                   c='red', label="Endmembers")

    plt.title(title)
    plt.legend()
    plt.show()
    

# ---------------------------------------------------------------------------------
# Function: ppi
# Topics Covered:
#    - Endmember Determination:
#         * Pixel Purity Index (PPI) algorithm for extracting extreme points.
# ---------------------------------------------------------------------------------
def ppi(S: np.ndarray,
        num_iterations: int = 1000,
        threshold: float = 0.99) -> np.ndarray:
    """
    Pixel Purity Index (PPI) algorithm for endmember extraction.

    Parameters:
        S (np.ndarray): Spectral data with shape (bands x pixels).
        num_iterations (int): Number of random projections.
        threshold (float): Purity threshold (unused in this implementation).

    Returns:
        np.ndarray: Array of indices corresponding to extreme points.
    """
    num_bands, num_pixels = S.shape
    extreme_points = set()

    for _ in range(num_iterations):
        random_vector = np.random.rand(num_bands)
        projections = S.T @ random_vector
        max_idx = np.argmax(projections)
        min_idx = np.argmin(projections)
        extreme_points.add(max_idx)
        extreme_points.add(min_idx)

    return np.array(list(extreme_points))


# ---------------------------------------------------------------------------------
# Function: n_findr
# Topics Covered:
#    - Endmember Determination:
#         * N-FINDR algorithm, a minimum volume simplex method.
# ---------------------------------------------------------------------------------
def n_findr(spectral_data: np.ndarray,
            num_endmembers: int,
            max_iter: int = 100) -> np.ndarray:
    """
    N-FINDR algorithm for endmember extraction.

    Parameters:
        spectral_data (np.ndarray): Input data (samples x bands).
        num_endmembers (int): Number of endmembers to extract.
        max_iter (int): Maximum number of iterations for improvement.

    Returns:
        np.ndarray: Selected endmembers (num_endmembers x bands).
    """
    num_samples, _ = spectral_data.shape
    selected_indices = np.random.choice(num_samples, num_endmembers, replace=False)
    # Perturb the data slightly to avoid degenerate cases.
    epsilon = 1e-6
    spectral_data_perturbed = spectral_data + np.random.uniform(-epsilon, epsilon, spectral_data.shape)

    current_volume = volume(spectral_data_perturbed[selected_indices, :])
    for _ in range(max_iter):
        improved = False
        for i in range(num_samples):
            for j in range(num_endmembers):
                temp_indices = selected_indices.copy()
                temp_indices[j] = i
                temp_volume = volume(spectral_data_perturbed[temp_indices, :])
                if temp_volume > current_volume:
                    selected_indices = temp_indices
                    current_volume = temp_volume
                    improved = True
        if not improved:
            break

    return spectral_data[selected_indices, :]


# ---------------------------------------------------------------------------------
# Function: volume
# Topics Covered:
#    - Geometrical Interpretation:
#         * Computes convex hull volume used in the N-FINDR algorithm.
# ---------------------------------------------------------------------------------
def volume(points: np.ndarray) -> float:
    """
    Computes the volume of the convex hull of the given points.

    Parameters:
        points (np.ndarray): Array of shape (num_points x dimensions).

    Returns:
        float: The volume of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception as e:
        print(f"Warning: Convex hull computation failed due to {e}. Setting volume to 0.")
        return 0.0


# ---------------------------------------------------------------------------------
# Function: smacc
# Topics Covered:
#    - Endmember Determination:
#         * Sequential Maximum Angle Convex Cone (SMACC) algorithm.
# ---------------------------------------------------------------------------------
def smacc(spectral_data: np.ndarray,
          num_endmembers: int) -> np.ndarray:
    """
    Sequential Maximum Angle Convex Cone (SMACC) algorithm for endmember extraction.

    Parameters:
        spectral_data (np.ndarray): Input data (samples x bands).
        num_endmembers (int): Number of endmembers to extract.

    Returns:
        np.ndarray: Extracted endmembers (num_endmembers x bands).
    """
    num_samples, num_bands = spectral_data.shape
    if num_endmembers > num_bands:
        raise ValueError("Number of endmembers must not exceed the number of spectral bands.")

    endmembers = []
    residual = spectral_data.copy()

    for _ in range(num_endmembers):
        norms = np.linalg.norm(residual, axis=1)
        max_idx = np.argmax(norms)
        endmember = spectral_data[max_idx, :]
        endmembers.append(endmember)
        # Remove the projection of the selected endmember
        projection = (residual @ endmember) / (endmember @ endmember)
        residual = residual - np.outer(projection, endmember)

    return np.array(endmembers)


# ---------------------------------------------------------------------------------
# Function: plot_abundances
# Topics Covered:
#    - Visualization of Abundance Estimations:
#         * Comparing unconstrained, constrained, and regularized LMM solutions.
# ---------------------------------------------------------------------------------
def plot_abundances(a_unconstrained: np.ndarray,
                    a_constrained: np.ndarray,
                    a_regularized: np.ndarray,
                    labels: List[str]) -> None:
    """
    Plots the abundance estimations from different methods.

    Parameters:
        a_unconstrained (np.ndarray): Abundances from the unconstrained method.
        a_constrained (np.ndarray): Abundances from the constrained method.
        a_regularized (np.ndarray): Abundances from the regularized method.
        labels (List[str]): Labels for each endmember.
    """
    x_pos = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, a_unconstrained, marker='o', label="Unconstrained", linestyle='-', linewidth=2)
    plt.plot(x_pos, a_constrained, marker='s', label="Constrained", linestyle='-', linewidth=2)
    plt.plot(x_pos, a_regularized, marker='^', label="Regularized", linestyle='-', linewidth=2)
    plt.xticks(x_pos, labels)
    plt.xlabel("Endmembers")
    plt.ylabel("Abundance")
    plt.title("Abundance Estimations for Different Methods")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    

# ---------------------------------------------------------------------------------
# Function: main
# Topics Covered:
#    - Integration of all components:
#         * Data preprocessing.
#         * Endmember extraction.
#         * Solving LMM using different LSE approaches.
#         * Visualization of spectral curves, abundance estimates, and convex hull.
# ---------------------------------------------------------------------------------
def main() -> None:
    # Load and preprocess the hyperspectral data.
    file_path = input("Enter the path to your data file: ").strip()
    try:
        spectral_data = load_and_preprocess_data(file_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    print("Shape of preprocessed spectral data:", spectral_data.shape)

    # Ask for the number of endmembers.
    try:
        endmember_count = int(input("Enter the number of endmembers to select: ").strip())
    except ValueError:
        print("Invalid input for the number of endmembers.")
        sys.exit(1)

    # Prepare the matrices S and x.
    try:
        S, x = prepare_matrices(spectral_data, endmember_count)
    except Exception as e:
        print(f"Error in preparing matrices: {e}")
        sys.exit(1)

    # Perform Linear Spectral Unmixing using different methods.
    try:
        a_unconstrained_cholesky = linear_mixing_model(S, x,
                                                       method="unconstrained",
                                                       optimization="cholesky")
    except Exception as e:
        print(f"Error in unconstrained method: {e}")
        sys.exit(1)

    # Constrained LSE with sum-to-one and nonnegativity (deterministic fill-fraction estimation)
    constraints = {"sum_to_one": True, "nonnegativity": True, "use_pca": False}
    try:
        a_constrained = linear_mixing_model(S, x,
                                            method="constrained",
                                            constraints=constraints)
    except Exception as e:
    # Regularized LSE: Adds a penalty term to handle noise.
        print(f"Error in constrained method: {e}")
        sys.exit(1)

    try:
        a_regularized = linear_mixing_model(S, x,
                                            method="regularized",
                                            lambda_reg=0.1)
    except Exception as e:
        print(f"Error in regularized method: {e}")
        sys.exit(1)

    # Generate labels for visualization.
    labels = [f"Endmember {i+1}" for i in range(S.shape[1])]

    # Visualize spectral curves for a subset of samples.
    plot_spectral_curves(spectral_data, sample_indices=[0, 1, 2],
                         title="Spectral Curves (Samples 0, 1, 2)")

    # Plot abundance estimations.
    plot_abundances(a_unconstrained_cholesky, a_constrained, a_regularized, labels)

    # Visualize convex hull of endmembers.
    # Since S has shape (bands, endmembers), we transpose it so that each row is an endmember.
    endmembers = S.T
    if endmembers.shape[1] > 3:
        # If the endmember signatures have more than 3 dimensions, reduce to 3 via PCA for plotting.
        pca = PCA(n_components=3)
        endmembers_reduced = pca.fit_transform(endmembers)
    else:
        endmembers_reduced = endmembers

    if endmembers_reduced.shape[1] >= 2:
        plot_convex_hull(endmembers_reduced, labels, title="Convex Hull of Endmembers")
    else:
        print(f"Convex hull plotting requires at least 2 dimensions. Current dimensions: {endmembers_reduced.shape[1]}")
        
        
if __name__ == "__main__":
    main()
    

"""
Conclusion:
Why is this significant for FINCH?

- Endmembers: Materials like soil, crop residue, and vegetation have unique spectral
              signatures (columns of S).
- Mixed Spectrum (x): The sensor observes the combined reflection from multiple materials
              in a pixel.
- Goal & Applications:
    1. Crop Residue Mapping: Identify areas with crop residue left on the ground after harvest.
        - How: Estimate the abundances of mixed materials (e.g. soil, crop residue, and vegetation) present in every pixel,
               ensuring they sum to 1.
    2. Soil Quality Analysis: Monitor soil health and its composition.
        - How: Determine the fraction of soil in each pixel, which can be used to study erosion or fertility.
    3. Vegetation Monitoring: Measure vegetation cover or health (e.g. crop yield estimation).
        - How: Extract vegetation proportions from the pixel abundances.
    4. Environmental Monitoring: Detect changes in land cover or classify land use (e.g. agricultural land vs barren land).
        - How: Use endmember abundances to classify land cover types.
        
This code helps analyze hyperspectral data by estimating material compositions.
The constrained and regularized methods are especially useful for real-world
data as they enforce physical realism.

Helps farmers! :)
"""