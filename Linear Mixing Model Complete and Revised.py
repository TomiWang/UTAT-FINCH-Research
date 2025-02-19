# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:58:56 2025

Project: Field Imaging Nanosatellite for Crop residue Hyperspectral mapping (FINCH)

The purpose of this code is to solve the Linear Mixing Model (LMM),
a mathematical model used in hyperspectral imaging. Hyperspectral sensors collect
light reflected from Earth's surface with many spectral bands within the visible
spectrum. These spectral bands are collected as FINCH observes the area with mixed
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
# Function: parse_columns_input
# Topics Covered:
#    - My helper function lol.
#    - This is important. Come here if data columns are changed.
#    - Column headers are 400, 410, 420 etc. They serve as labels
#      not counted as part of the data :thumbs_up:. They are not
#      counted as part of any rows.
# ---------------------------------------------------------------------------------
def parse_columns_input(columns_str: str) -> List[str]:
    """
    Parses a strings for column names. The strings can be a comma-separated list of column names
    or include a range specification using a hyphen. For example, "400-440" will be expanded
    to ["400", "410", "420", "430", "440"] assuming a step of 10.
    """
    columns = []
    # Split the input by commas to allow both individual entries and ranges
    for token in columns_str.split(","):
        token = token.strip()
        if not token:
            continue  # skip empty tokens
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range specification: {token}")
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError:
                raise ValueError(f"Invalid numbers in range: {token}")
            if start > end:
                raise ValueError("Start of range must be less than or equal to end.")
            # Assume a step of 10 (adjust if needed)
            for x in range(start, end + 1, 10):
                columns.append(str(x))
        else:
            columns.append(token)
    return columns


# ---------------------------------------------------------------------------------
# Function: parse_rows_input
# Topics Covered:
#    - My second helper function, lol.
#    - This is important. Come here if data rows are changed.
#    - The column headers are not part of the rows.
#      Row 0 is the very first data row after the headers.
# ---------------------------------------------------------------------------------
def parse_rows_input(rows_str: str) -> List[int]:
    """
    Parses a string for row indices. The string can be a comma-separated
    list of indices or include a range specification using a hyphen.
    For example:
       "0-5"  -> [0, 1, 2, 3, 4, 5]
       "0-5, 10, 12-15" -> [0, 1, 2, 3, 4, 5, 10, 12, 13, 14, 15]
    """
    rows = []
    # Split the input by commas to allow both individual entries and ranges
    for token in rows_str.split(","):
        token = token.strip()
        if not token:
            continue  # skip empty tokens

        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range specification: {token}")
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError:
                raise ValueError(f"Invalid numbers in range: {token}")
            if start > end:
                raise ValueError("Start of range must be less than or equal to end.")

            # Default step of 1
            for x in range(start, end + 1):
                rows.append(x)
        else:
            # Single index
            try:
                row_index = int(token)
                rows.append(row_index)
            except ValueError:
                raise ValueError(f"Invalid row index: {token}")

    return rows


# ---------------------------------------------------------------------------------
# Function: load_and_preprocess_data
# Topics Covered:
#    - Data Loading and Preprocessing for LMM.
#    - Normalization of spectral data.
#    - Handling missing data.
# ---------------------------------------------------------------------------------
def load_and_preprocess_data(file_path: str,
                             columns: List[str],
                             normalize: bool = False, # These parts just set their defaults lol
                             fill_missing: bool = True) -> np.ndarray:
    """
    This function loads data from a CSV file by selecting specified columns and handles missing values.
    It also has an optional normalization of the data!!
    
    After loading, the function reads the CSV file using pandas and interprets the data by selecting the
    columns (which are the spectral bands). Selected columns are entered by the user. If there are missing
    values the function will fill the missing values by taking the mean of each column, provided the parameter
    is set to be true. Data is normalized if the parameter is set to be true and then take each column's values
    to fall between 0 and 1. This is done by finding the max and min values for each columns, subtract the min
    value from each element in that column, and then devide the result of each element by the range given as
    max-min so that values are scaled between 0 and 1. If the range falls to be 0, the code will set the range
    to 1.
    
    e.g Min 5, Max 15, Range 15 - 5 = 10 :grin_emoji:
    
    Parameters:
        file_path (str): The path to the CSV file containing the data.
        columns (List[str]): The list of columns (spectral bands) selected from the CSV.
        normalize (bool, optional): Optionally to normalize the data. Defaults to False.
        fill_missing (bool, optional): Optionally to fill missing data with the column mean. Defaults to True.
    
    Returns:
        np.ndarray: A 2D numpy array with rows as samples and columns as spectral bands!
    
    Raises:
        ValueError: Prompted if the file can't be loaded or no columns are specified or if the resulting data
                    is empty or has invalid dimensions.
    """
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

    # Use only the specified columns. So select your own :thumbs_up:
    if not columns:
        raise ValueError("No columns were specified. yo don't forget to provide a list of columns to load.")
    data = data[columns]
    if data.empty:
        raise ValueError("No data found in the specified columns.")

    # Fills in missing values with the column mean if needed.
    if fill_missing:
        data = data.fillna(data.mean())
    data_array = data.to_numpy()

    if data_array.shape[0] < 1 or data_array.shape[1] < 1:
        raise ValueError("Data must have at least one sample and one band.")

    # Normalize each column to [0, 1] (avoids division by zero).
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
    This function prepares the matrices required for solving the Linear Mixing Model.
    
    The function extracts a set of pure spectral signatures (endmembers) from the data using
    the N-FINDR algorithm. It then constructs the endmember matrix S and defines the observed spectrum x as
    the mean of the spectral data. This is the core equation.
    
    Parameters:
        spectral_data (np.ndarray): The preprocessed spectral data (samples x bands).
        endmember_count (int): The number of endmembers (distinct materials) to extract.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - S: Endmember matrix with dimensions (bands x endmembers).
            - x: Observed spectrum as a 1D array of length equal to the number of bands.
    
    Raises:
        ValueError: Prompted if there are not enough samples to extract the specified number of endmembers.
    """
    num_samples, num_bands = spectral_data.shape
    if num_samples < endmember_count:
        raise ValueError(f"Not enough samples to select {endmember_count} endmembers from {num_samples} samples.")

    # Uses N-FINDR to extract endmembers.
    # n_findr returns an array of selected endmember spectra with shape (endmember_count, bands)
    endmember_candidates = n_findr(spectral_data, endmember_count)
    S = endmember_candidates.T  # Transpose to get shape (bands, endmembers)

    # Defines the observed spectrum as the mean over all samples.
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
        raise ValueError("Invalid method specified. Choose 'unconstrained', 'constrained', or 'regularized'. Gotta pick.")

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
    This function plots the spectral curves for a set of samples from the data.
    
    Each spectral curve represents the reflectance values across different bands for a single sample,
    providing the visual fingerprint of that sample. This visualization helps me- you probably too to understanding the
    unique spectral signatures and differences between samples.
    
    Parameters:
        spectral_data (np.ndarray): 2D array of spectral data (samples x bands).
        sample_indices (Optional[List[int]], optional): List of sample indices to plot. If None, all samples are plotted. :thumbs_up:
        title (str, optional): Title of the plot. :)
    
    Returns:
        None
    
    Notes:
        - The x-axis represents the spectral bands.
        - The y-axis represents the reflectance values.
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
#    - Geometrical Interpretation: Visualizing the convex hull (simplex geometry).
#      of endmember signatures.
# ---------------------------------------------------------------------------------
def plot_convex_hull(endmembers: np.ndarray,
                     labels: List[str],
                     title: str = "Convex Hull of Endmembers") -> None:
    """
    This functions visualizes the convex hull formed by the endmember signatures in either 2D or 3D space. Very cool.
    
    A convex hull is the smallest convex shape that encloses all the endmembers, which are depicted as points. This
    visualization is useful for understanding the geometrical relationships between the endmember signatures. It shows
    how the pure material signatures (endmembers) span the spectral space and forms a simplex in the case
    of three materials.
    
    Parameters:
        endmembers (np.ndarray): A 2D array of endmember points. Each row is an endmember and each column is a dimension.
        labels (List[str]): Labels for each endmember to be used in the plot legend.
        title (str, optional): Title of the plot. :)
    
    Returns:
        None
    
    Notes:
        - The convex hull computation is supported only for 2D and 3D data. If the endmembers have more than 3 dimensions,
          there is gonna be a problem lol. Dw, I made sure to put a warning display.
    """
    if endmembers.shape[1] > 3:
        print("Warning!! Convex hull visualization is limited to 2D/3D. :pensive_emoji:")
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
    This function implements the Pixel Purity Index (PPI) algorithm to identify extreme points in the data.
    
    The PPI algorithm projects the data onto many randomly generated directions (random vectors) and records
    the indices of the pixels that achieve the maximum and minimum projections. The pixels that frequently
    appear as extreme are considered pure or "endmember" candidates.
    
    Parameters:
        S (np.ndarray): Endmember matrix with dimensions (bands x pixels).
        num_iterations (int, optional): Number of random projections to perform. Defaults to 1000.
        threshold (float, optional): A threshold value (not used explicitly here) that could be used to filter extreme points. Defaults to 0.99.
    
    Returns:
        np.ndarray: Array of unique indices corresponding to the identified extreme (pure) pixels.
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
    This function implements the N-FINDR algorithm to extract endmembers from the data.
    
    N-FINDR works by searching for set of pixels that define the simplex with the maximum volume in the spectral space.
    A simplex is the simplest possible shape (a triangle in 2D, tetrahedron in 3D, etc.) that can enclose the data points.
    The algorithm starts with a random selection of endmembers and iteratively replaces them with other pixels if a specific
    swap increases the volume of the simplex, which is a measure of the diversity of the endmembers.
    
    Parameters:
        spectral_data (np.ndarray): 2D array of spectral data (samples x bands).
        num_endmembers (int): The number of endmembers to extract.
        max_iter (int, optional): Maximum number of iterations for the iterative search. Defaults to 100.
    
    Returns:
        np.ndarray: A 2D array containing the selected endmember signatures with dimensions (num_endmembers x bands).
    
    Notes:
        - A small random perturbation (epsilon) is added to the spectral data to avoid degenerate cases
         (i.e., cases where points lie on a lower-dimensional space).
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
    This function calculates the volume of the convex hull defined by a set of points.
    
    The volume is used to measure the simplex size in the N-FINDR algorithm. A larger volume would suggest a better
    spread of the endmember signatures, meaning the pure materials are more distinct from each other. In case the convex hull
    cannot be computed (e.g., due to degenerate or collinear points), the function returns 0.
    
    Parameters:
        points (np.ndarray): A 2D array where each row represents a point in spectral space.
    
    Returns:
        float: The volume of the convex hull. Returns 0 if computation fails.
    
    Notes:
        - The ConvexHull function from scipy.spatial is used for the calculation.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception as e:
        print(f"Warning!! Convex hull computation failed due to {e}. Setting volume to 0.")
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
    This function implements the Sequential Maximum Angle Convex Cone (SMACC) algorithm to extract endmembers from the data.
    
    The SMACC algorithm selects endmembers by iteratively choosing the pixel with the maximum norm (i.e., the most "extreme"
    in terms of magnitude) from the residual data. After selecting an endmember, its projection is removed from the data,
    which helps in finding subsequent endmembers that are distinct. The idea is to maximize the angular separation between
    the selected endmembers, ensuring that each one represents a unique spectral signature. Super cool.
    
    Parameters:
        spectral_data (np.ndarray): 2D array of spectral data (samples x bands).
        num_endmembers (int): The number of endmembers to extract.
    
    Returns:
        np.ndarray: A 2D array containing the selected endmember signatures with dimensions (num_endmembers x bands).
    
    Raises:
        ValueError: If the number of requested endmembers exceeds the number of spectral bands.
    """
    num_samples, num_bands = spectral_data.shape
    if num_endmembers > num_bands:
        raise ValueError("Number of endmembers must not exceed the number of spectral bands. :raised_eyebrow:")

    endmembers = []
    residual = spectral_data.copy()

    for _ in range(num_endmembers):
        norms = np.linalg.norm(residual, axis=1)
        max_idx = np.argmax(norms)
        endmember = spectral_data[max_idx, :]
        endmembers.append(endmember)
        # Removes the projection of the selected endmember
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
    This function plots and compares the abundance vectors obtained from different LMM solving methods
    (remember, unconstrained, constrained, and regularized).
    
    The abundance vector 'a' represents the proportion of each material (endmember) present in the observed spectrum.
    This function creates a line plot where each method's estimated abundances are plotted against the endmember labels,
    allowing for a visual comparison of the three methods.
    
    Parameters:
        a_unconstrained (np.ndarray): Abundance vector obtained from the unconstrained method.
        a_constrained (np.ndarray): Abundance vector obtained from the constrained method.
        a_regularized (np.ndarray): Abundance vector obtained from the regularized method.
        labels (List[str]): Labels for the endmembers (e.g., ["Endmember 1", "Endmember 2", ...]).
    
    Returns:
        None
    
    Notes:
        - The x-axis represents the endmembers.
        - The y-axis represents the abundance (fraction) values.
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
    """
    This is the main function that integrates all the steps for data analysis using the Linear Mixing Model!!
    
    Workflow:
    1. Data Loading & Preprocessing:
       - Prompts the user for the CSV file path, columns to use, and optional row selection. Very convenient.
       - Loads the hyperspectral data, fills missing values, and optionally normalizes the data.
    2. Endmember Extraction:
       - Asks for the number of endmembers and uses the N-FINDR algorithm to extract pure spectral signatures.
       - Prepares the endmember matrix S and computes the observed spectrum x (as the mean of the data).
    3. Solving the Mixing Model:
       - Estimates the abundance vector using three approaches:
           a) Unconstrained Least Squares (with many numerical optimization techniques).
           b) Constrained Least Squares (enforces sum-to-one and nonnegativity constraints).
           c) Regularized Least Squares (adds a penalty to handle noise).
    4. Visualization:
       - Plots spectral curves for a subset of samples.
       - Compares the abundance estimations from the three methods.
       - Visualizes the convex hull of the endmember signatures (reduces dimensions via PCA if needed).
    
    This function orchestrates the entire process of unmixing hyperspectral data, which is very very importnat
    for applications such as our mission in crop residue mapping, soil quality analysis, and vegetation monitoring.
    
    Returns:
        None
    
    Notes:
        - yo mkae sure to input file paths, column selections, row selections, and the number of endmembers.
        - But dw, I placed error handling to make sure that the process stops with a message if any step fails. :thumbs_up:
    """
    # Load and preprocess the hyperspectral data. Gimme your csv file.
    file_path = input("Enter the path to your data file: ").strip()
    
    # Prompt for columns (comma-separated or range). tell this section what column you want.
    columns_str = input("Enter the columns to select (comma-separated or range): ").strip()
    if not columns_str:
        raise ValueError("You must specify columns.")
    columns = parse_columns_input(columns_str)

    try:
        # This has normalization. Put False to disable.
        spectral_data = load_and_preprocess_data(file_path, columns=columns, normalize=False, fill_missing=True)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    print("Shape of preprocessed spectral data:", spectral_data.shape)
    
    # Prompt for rows (comma-seperated or range or just enter nothing if all). tell this section what rows you want.
    row_str = input("Enter the rows to select: ").strip()
    if row_str:
        rows = parse_rows_input(row_str)
        spectral_data = spectral_data[rows, :]
        print("Shape after row selection:", spectral_data.shape)
    else:
        print("If no row selection is provided; use all rows.")

    # Ask for the number of endmembers too :)
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
Summary:
1. Data Loading & Preprocessing: The code reads hyperspectral data from a CSV file. It then selects numeric columns, fills
in any missing values using the column means, and normalizes the data so that each spectral band ranges between 0 and 1.

2. Endmember Extraction: This step identifies endmembers which are pure spectral signatures representing distinct materials.
The N-FINDR algorithm (among PPI and SMACC) is used to extract these endmembers from the datatset.

3. Setting up the Linear Mixing Model: The observed mixed spectrum (x) is computed as the mean of the spectra. The endmember
Matrix (S) is formed by stacking the pure spectral signatures (each column corresponds to one material).

4. Solving The Mixing Model: The code estimates the abundance vector (a), which tells you the proportion of each material
present. The three approaches are unconstrained least squares, contstrained least squares, regularized least squares.

5. Visualization: The code plots the spectral curves for selected samples to show unique signatures. Abundance estimates
are displayed from the different methods. Geomatric relationships (via convex hull) of the endmember signatures are visualized,
and PCA is optionally used for dimensionality reduction if needed.


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