# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:02:20 2025

The purpose of this code is to perform Occam's Inversion, a regularized inversion technique used to
interpret hypersspectral imaging data collected by FINCH. This application derives a smoothed or constrained model that fits observed data

@author: Tomi Wang :)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from scipy.sparse import diags
from scipy.optimize import minimize
from typing import List, Tuple, Optional

# Configure logging: change level to DEBUG for more detailed output.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class OccamsInversion:
    # ---------------------------------------------------------------------------------
    # __init__
    # Topics Covered:
    #   - Parameterization with Derivatives: Initializes the model,
    #     sets up the derivative matrix.
    #   - Regularization Techniques: Sets the regularization parameter (alpha).
    # ---------------------------------------------------------------------------------
    def __init__(self, 
                 data: np.ndarray, 
                 uncertainties: np.ndarray, 
                 depth_intervals: np.ndarray, 
                 order: int = 2, 
                 alpha: float = 1.0) -> None:
        """
        Initialize the Occam's inversion with data, uncertainties, depth intervals, 
        derivative order, and regularization parameter.
        
        Parameters:
            data (np.ndarray): Observed data values.
            uncertainties (np.ndarray): Uncertainties associated with the data.
            depth_intervals (np.ndarray): Depth or layer intervals.
            order (int): Order of the derivative used for roughness (1 or 2). Default is 2.
            alpha (float): Regularization parameter. Default is 1.0.
        """
        self.depth_intervals = depth_intervals
        self.num_layers = len(depth_intervals)
        self.order = order
        self.alpha = alpha

        # Ensure that data and uncertainties have the same number of entries as depth_intervals.
        if len(data) != self.num_layers or len(uncertainties) != self.num_layers:
            logging.warning("Mismatched data/uncertainties and depth intervals detected. "
                            "Interpolating to match the number of layers.")
            self.data = np.interp(
                depth_intervals, 
                np.linspace(depth_intervals[0], depth_intervals[-1], len(data)), 
                data
            )
            self.uncertainties = np.interp(
                depth_intervals, 
                np.linspace(depth_intervals[0], depth_intervals[-1], len(uncertainties)), 
                uncertainties
            )
        else:
            self.data = data
            self.uncertainties = uncertainties

        # Initialize the model to the average data value.
        self.model = np.full(self.num_layers, np.mean(self.data))
        # Build the derivative matrix for the regularization term.
        self.derivative_matrix = self._construct_derivative_matrix()

        # Save the original forward model (in case of temporary overrides).
        self.original_forward_model = self.forward_model


    # ---------------------------------------------------------------------------------
    # _construct_derivative_matrix
    # Topics Covered:
    #   - Parameterization with Derivatives: Constructs the derivative operator
    #     (first or second order) to measure model smoothness.
    # ---------------------------------------------------------------------------------
    def _construct_derivative_matrix(self) -> "diags":
        """
        Construct the derivative matrix based on the specified order.
        
        Returns:
            scipy.sparse.diags: Sparse derivative operator matrix.
        """
        if self.order == 1:
            diagonals = [-np.ones(self.num_layers - 1), np.ones(self.num_layers - 1)]
            return diags(diagonals, [0, 1], shape=(self.num_layers - 1, self.num_layers))
        elif self.order == 2:
            diagonals = [
                np.ones(self.num_layers - 2),
                -2 * np.ones(self.num_layers - 2),
                np.ones(self.num_layers - 2),
            ]
            return diags(diagonals, [-1, 0, 1], shape=(self.num_layers - 2, self.num_layers))
        else:
            raise ValueError("Only first or second derivatives are supported.")


    # ---------------------------------------------------------------------------------
    # _roughness
    # Topics Covered:
    #   - Regularization Techniques: Computes the roughness (smoothness penalty)
    #     by applying the derivative operator.
    # ---------------------------------------------------------------------------------
    def _roughness(self, model: np.ndarray) -> float:
        """
        Compute the roughness (regularization) term of the model.
        
        Parameters:
            model (np.ndarray): Current model parameters.
        
        Returns:
            float: Sum of squared derivatives (roughness).
        """
        diff = self.derivative_matrix @ model
        return np.linalg.norm(diff) ** 2


    # ---------------------------------------------------------------------------------
    # _misfit
    # Topics Covered:
    #   - Data Misfit: Measures how well the model predictions match the observed data.
    # ---------------------------------------------------------------------------------
    def _misfit(self, model: np.ndarray) -> float:
        """
        Compute the misfit between observed data and model predictions.
        
        Parameters:
            model (np.ndarray): Current model parameters.
        
        Returns:
            float: Sum of squared normalized residuals.
        """
        predicted = self.forward_model(model)
        residual = (self.data - predicted) / self.uncertainties
        return np.sum(residual ** 2)


    # ---------------------------------------------------------------------------------
    # forward_model
    # Topics Covered:
    #   - Forward Modeling: Represents the mapping from model parameters to predicted data.
    #     (Currently implemented as an identity function; can be replaced.)
    # ---------------------------------------------------------------------------------
    def forward_model(self, model: np.ndarray) -> np.ndarray:
        """
        The forward model mapping model parameters to predicted data.
        This placeholder function is currently the identity function and can
        be replaced with a more complex model.
        
        Parameters:
            model (np.ndarray): Current model parameters.
        
        Returns:
            np.ndarray: Predicted data.
        """
        return model


    # ---------------------------------------------------------------------------------
    # _objective_function
    # Topics Covered:
    #   - Set up the Objective Function: Combines the data misfit and the roughness penalty
    #     to balance data fitting and model smoothness.
    # ---------------------------------------------------------------------------------
    def _objective_function(self, model: np.ndarray) -> float:
        """
        Combined objective function: misfit plus a roughness (regularization) penalty.
        
        Parameters:
            model (np.ndarray): Current model parameters.
        
        Returns:
            float: Value of the objective function.
        """
        misfit = self._misfit(model)
        roughness = self._roughness(model)
        return misfit + self.alpha * roughness


    # ---------------------------------------------------------------------------------
    # _compute_gradient
    # Topics Covered:
    #   - Step-Size Optimization: Computes the gradient needed for gradient descent.
    #   - Regularization Techniques: Differentiates the roughness term.
    # ---------------------------------------------------------------------------------
    def _compute_gradient(self, model: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function with respect to the model.
        Note: For a nonlinear forward model, the chain rule (via the Jacobian) should be applied.
        
        Parameters:
            model (np.ndarray): Current model parameters.
        
        Returns:
            np.ndarray: Gradient of the objective function.
        """
        # Assuming forward_model is linear (or identity). Adjust if using a nonlinear forward model.
        gradient_misfit = -2 * (self.data - self.forward_model(model)) / (self.uncertainties ** 2)
        gradient_roughness = 2 * self.derivative_matrix.T @ (self.derivative_matrix @ model)
        return gradient_misfit + self.alpha * gradient_roughness


    # ---------------------------------------------------------------------------------
    # _optimize_step_size
    # Topics Covered:
    #   - Step-Size Optimization: Dynamically adjusts the step size in gradient descent
    #     to ensure a reduction in the objective function.
    # ---------------------------------------------------------------------------------
    def _optimize_step_size(self, 
                            gradient: np.ndarray, 
                            current_model: np.ndarray,
                            bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
                            max_step: float = 1.0, 
                            min_step: float = 1e-4) -> float:
        """
        Optimize the step size for a gradient descent update.
        
        Parameters:
            gradient (np.ndarray): Gradient of the objective function.
            current_model (np.ndarray): Current model parameters.
            bounds (List[Tuple[Optional[float], Optional[float]]], optional): 
                Bounds for each model parameter.
            max_step (float): Maximum allowable step size.
            min_step (float): Minimum allowable step size.
        
        Returns:
            float: The step size that reduces the objective function.
        """
        step_size = max_step
        for _ in range(10):  # Limit iterations for efficiency.
            proposed_model = current_model - step_size * gradient
            if bounds is not None:
                lower_bounds = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
                upper_bounds = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
                proposed_model = np.clip(proposed_model, lower_bounds, upper_bounds)

            if self._objective_function(proposed_model) < self._objective_function(current_model):
                return step_size

            step_size *= 0.5  # Reduce step size if the objective doesn't improve.
            if step_size < min_step:
                break

        return step_size


    # ---------------------------------------------------------------------------------
    # inversion
    # Topics Covered:
    #   - Constrained Optimization: Uses bounds and an equality constraint on misfit
    #     to ensure physical meaningfulness of the model.
    #   - Set up the Objective Function: Minimizes a combination of misfit and roughness.
    # ---------------------------------------------------------------------------------
    def inversion(self, 
                  target_misfit: float, 
                  bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None) -> np.ndarray:
        """
        Perform inversion by minimizing the objective function with a constraint on the misfit.
        
        Parameters:
            target_misfit (float): Desired misfit value.
            bounds (List[Tuple[Optional[float], Optional[float]]], optional): 
                Bounds for the model parameters.
        
        Returns:
            np.ndarray: Inverted model parameters.
        """
        def constraint_function(model: np.ndarray) -> float:
            return self._misfit(model) - target_misfit

        constraints = {"type": "eq", "fun": constraint_function}

        # Default to non-negative parameters if bounds are not provided.
        if bounds is None:
            bounds = [(0, None) for _ in range(self.num_layers)]

        result = minimize(
            self._objective_function,
            self.model,
            constraints=constraints,
            bounds=bounds,
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if not result.success:
            raise RuntimeError("Inversion did not converge: " + result.message)

        self.model = result.x
        return self.model


    # ---------------------------------------------------------------------------------
    # iterative_refinement
    # Topics Covered:
    #   - Iterative Refinement: Uses gradient descent to update the model step-by-step.
    #   - Step-Size Optimization: Determines an optimal step size during each iteration.
    # ---------------------------------------------------------------------------------
    def iterative_refinement(self, 
                               max_iterations: int = 10, 
                               tolerance: float = 1e-4,
                               bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None) -> np.ndarray:
        """
        Refine the model iteratively using gradient descent with a dynamically optimized step size.
        
        Parameters:
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence tolerance for model updates.
            bounds (List[Tuple[Optional[float], Optional[float]]], optional): 
                Bounds for the model parameters.
        
        Returns:
            np.ndarray: Refined model parameters.
        """
        current_model = self.model.copy()

        for iteration in range(max_iterations):
            logging.info(f"Refinement Iteration {iteration + 1}")

            gradient = self._compute_gradient(current_model)
            step_size = self._optimize_step_size(gradient, current_model, bounds)
            logging.info(f"Optimized Step Size: {step_size}")

            updated_model = current_model - step_size * gradient

            if bounds is not None:
                lower_bounds = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
                upper_bounds = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
                updated_model = np.clip(updated_model, lower_bounds, upper_bounds)

            model_difference = np.linalg.norm(updated_model - current_model)
            logging.info(f"Model difference: {model_difference}")

            if model_difference < tolerance:
                logging.info("Convergence reached during refinement.")
                current_model = updated_model
                break

            current_model = updated_model

        self.model = current_model
        return self.model
    
    
    # ---------------------------------------------------------------------------------
    # lagrange_inversion
    # Topics Covered:
    #   - Regularization Techniques: Uses a Lagrange multiplier approach to balance the misfit and roughness.
    # ---------------------------------------------------------------------------------
    def lagrange_inversion(self, tolerance: float) -> np.ndarray:
        """
        Perform inversion using a Lagrange multiplier approach to balance misfit and roughness.
        
        Parameters:
            tolerance (float): Target misfit level.
        
        Returns:
            np.ndarray: Inverted model parameters.
        """
        lambda_value = 0.0
        step_size = 1e-2

        while True:
            def lagrange_objective(model: np.ndarray) -> float:
                return self._roughness(model) + lambda_value * (self._misfit(model) - tolerance) ** 2

            result = minimize(
                lagrange_objective,
                self.model,
                bounds=[(0, None) for _ in range(self.num_layers)],
                method="SLSQP",
                options={"maxiter": 500, "ftol": 1e-8},
            )

            if not result.success:
                raise RuntimeError("Lagrange inversion did not converge: " + result.message)

            misfit = self._misfit(result.x)
            if abs(misfit - tolerance) < 1e-4:
                self.model = result.x
                return self.model

            lambda_value += step_size * (misfit - tolerance)


    # ---------------------------------------------------------------------------------
    # linearize_and_invert
    # Topics Covered:
    #   - Linearization: Approximates the nonlinear forward model using its Jacobian.
    #   - Constrained Optimization: Uses the inversion method on the linearized model.
    # ---------------------------------------------------------------------------------
    def linearize_and_invert(self, 
                             target_misfit: float, 
                             max_iterations: int = 10, 
                             bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None) -> np.ndarray:
        """
        For nonlinear forward models, iteratively linearize and invert until convergence.
        
        Parameters:
            target_misfit (float): Desired misfit level.
            max_iterations (int): Maximum number of linearization iterations.
            bounds (List[Tuple[Optional[float], Optional[float]]], optional): 
                Bounds for the model parameters.
        
        Returns:
            np.ndarray: Final inverted model.
        """
        current_model = self.model.copy()

        for iteration in range(max_iterations):
            logging.info(f"Linearization Iteration {iteration + 1}")

            # Compute the Jacobian of the forward model at the current model.
            jacobian = self._compute_jacobian(current_model)

            # Define a linearized forward model.
            def linear_forward_model(model: np.ndarray) -> np.ndarray:
                return jacobian @ model

            # Temporarily override the forward model.
            original_forward_model = self.forward_model
            self.forward_model = linear_forward_model

            try:
                updated_model = self.inversion(target_misfit, bounds=bounds)
            except RuntimeError as e:
                logging.error("Linearized inversion failed: " + str(e))
                self.forward_model = original_forward_model  # Restore original forward model.
                break

            # Restore the original forward model.
            self.forward_model = original_forward_model

            if np.linalg.norm(updated_model - current_model) < 1e-4:
                logging.info("Convergence reached after linearization.")
                current_model = updated_model
                break

            current_model = updated_model

        self.model = current_model
        return self.model


    # ---------------------------------------------------------------------------------
    # _compute_jacobian
    # Topics Covered:
    #   - Linearization: Provides a (placeholder) Jacobian for the forward model.
    # ---------------------------------------------------------------------------------
    def _compute_jacobian(self, model: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix for the forward model.
        This placeholder returns an identity matrix and should be updated if a more complex 
        forward model is used.
        
        Parameters:
            model (np.ndarray): Current model parameters.
        
        Returns:
            np.ndarray: Approximated Jacobian matrix.
        """
        # Example placeholder: identity matrix.
        return np.eye(self.num_layers)
    
 
# ---------------------------------------------------------------------------------
# load_and_preprocess_data
# Topics Covered:
#   - Data Preparation: Loads CSV data, fills missing values, and normalizes the data.
# ---------------------------------------------------------------------------------
def load_and_preprocess_data(file_path: str,
                             normalize: bool = True,
                             fill_missing: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and preprocesses Occam's inversion data from a CSV file.

    The CSV file is expected to contain at least the columns:
        - "depth": The depth or layer intervals.
        - "data": The observed data.
    Optionally, an "uncertainty" column can be provided. If not, uncertainties are set to 0.1.

    Parameters:
        file_path (str): Path to the CSV file.
        normalize (bool): If True, normalize the 'data' column to [0, 1].
        fill_missing (bool): If True, fill missing values with the column mean.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - depth_intervals: 1D array of depths.
            - data: 1D array of observed data.
            - uncertainties: 1D array of uncertainties.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")

    required_columns = ['depth', 'data']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file must contain a '{col}' column.")

    if fill_missing:
        df = df.fillna(df.mean())

    depth_intervals = df['depth'].to_numpy()
    data = df['data'].to_numpy()

    if 'uncertainty' in df.columns:
        uncertainties = df['uncertainty'].to_numpy()
    else:
        uncertainties = np.full_like(data, 0.1, dtype=float)

    if normalize:
        # Normalize the data values to the [0, 1] range.
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val != 0:
            data = (data - min_val) / (max_val - min_val)

    return depth_intervals, data, uncertainties


# ---------------------------------------------------------------------------------
# plot_data_vs_depth
# Topics Covered:
#   - Visualization: Plots observed data and, optionally, the inverted model vs. depth.
# ---------------------------------------------------------------------------------
def plot_data_vs_depth(depth: np.ndarray,
                       observed: np.ndarray,
                       model: Optional[np.ndarray] = None,
                       title: str = "Observed Data and Inverted Model") -> None:
    """
    Plots the observed data and, optionally, the inverted model versus depth.

    Parameters:
        depth (np.ndarray): 1D array of depth intervals.
        observed (np.ndarray): 1D array of observed data.
        model (Optional[np.ndarray]): 1D array of inverted model values. Defaults to None.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(depth, observed, 'bo-', label='Observed Data')
    if model is not None:
        plt.plot(depth, model, 'r--', label='Inverted Model')
    plt.xlabel("Depth")
    plt.ylabel("Data Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------
# main
# Topics Covered:
#   - Data Preparation: Loads and preprocesses CSV data.
#   - Constrained Optimization & Iterative Refinement: Performs inversion.
#   - Visualization: Plots the results.
# ---------------------------------------------------------------------------------
# Example Usage
def main() -> None:
    """
    Main routine for Occam's inversion.

    This function:
      - Prompts for a CSV file path containing 'depth', 'data', and optionally 'uncertainty' columns.
      - Loads and preprocesses the data.
      - Initializes the Occam's inversion instance.
      - Performs iterative refinement inversion.
      - Plots the observed data and the inverted model versus depth.
    """
    file_path = input("Enter the path to your data CSV file: ").strip()
    try:
        depth_intervals, data, uncertainties = load_and_preprocess_data(file_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    logging.info(f"Data loaded successfully. Number of layers: {len(depth_intervals)}")

    # Initialize Occam's inversion instance (tweak order and alpha as needed).
    inversion = OccamsInversion(data, uncertainties, depth_intervals, order=2, alpha=0.5)

    # Define bounds (for example, non-negative model values).
    bounds = [(0, None) for _ in range(len(depth_intervals))]

    # Perform iterative refinement.
    try:
        refined_model = inversion.iterative_refinement(max_iterations=10, tolerance=1e-4, bounds=bounds)
        logging.info("Inversion completed successfully.")
    except RuntimeError as e:
        print(f"Inversion failed: {e}")
        return

    # Plot observed data and the inverted model.
    plot_data_vs_depth(depth_intervals, data, refined_model, title="Observed Data and Inverted Model")


if __name__ == "__main__":
    main()


