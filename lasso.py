# Import necessary libraries
import numpy as np  # For numerical computations
import pandas as pd  # For handling tabular data
import matplotlib.pyplot as plt  # For data visualization

# Set random seed for reproducibility
np.random.seed(42)  # Ensures results are consistent across runs

def generate_synthetic_returns(n_assets=5, n_days=500):
    """
    Generate synthetic financial returns data.
    
    Args:
    - n_assets: Number of assets (stocks).
    - n_days: Number of days of returns data.
    
    Returns:
    - df: DataFrame of synthetic returns.
    - true_cov: True covariance matrix.
    - true_prec: True precision (inverse covariance) matrix.
    """
    # Create names for the assets
    asset_names = [f'Stock_{i+1}' for i in range(n_assets)]
    
    # Generate a random covariance matrix
    A = np.random.randn(n_assets, n_assets)  # Random matrix
    true_cov = A.T @ A  # Ensure positive-definite covariance matrix
    true_prec = np.linalg.inv(true_cov)  # Compute the true precision matrix
    
    # Generate synthetic returns data
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),  # Mean returns are zero
        cov=true_cov,  # Covariance structure
        size=n_days  # Number of days of data
    )
    
    # Create a DataFrame with dates as the index
    dates = pd.date_range(start='2023-01-01', periods=n_days)  # Generate a date range
    df = pd.DataFrame(returns, columns=asset_names, index=dates)  # Combine into DataFrame
    
    return df, true_cov, true_prec

# Define a class for Graphical Lasso implementation
class GraphicalLasso:
    def __init__(self, S, alpha):
        """
        Initialize the Graphical Lasso solver.
        
        Args:
        - S: Sample covariance matrix.
        - alpha: Regularization parameter.
        """
        self.S = S  # Store the sample covariance matrix
        self.alpha = alpha  # Store the regularization parameter
        self.p = S.shape[0]  # Number of variables (dimensionality)
        
    def direct_solve(self, max_iter=100, tol=1e-4):
        """
        Solve using the Direct Coordinate Descent method.
        
        Args:
        - max_iter: Maximum number of iterations.
        - tol: Tolerance for convergence.
        
        Returns:
        - Precision matrix (inverse covariance).
        - Number of iterations taken to converge.
        """
        W = self.S.copy()  # Initialize W with the sample covariance matrix
        print("Solving with Direct Method...\n")
        
        for it in range(max_iter):  # Iterate up to the maximum number of iterations
            W_old = W.copy()  # Keep a copy of the previous matrix for convergence check
            
            for i in range(self.p):  # Loop over each variable
                indices = list(range(i)) + list(range(i+1, self.p))  # Indices excluding i
                W11 = W[np.ix_(indices, indices)]  # Submatrix excluding row/column i
                s12 = self.S[indices, i]  # Covariance vector excluding i
                
                # Solve for beta using the subproblem
                beta = np.linalg.solve(W11, s12)  # Solve linear system
                beta_new = np.sign(beta) * np.maximum(np.abs(beta) - self.alpha, 0)  # Apply soft-thresholding
                
                # Update W with the new solution
                W[i, indices] = W11 @ beta_new
                W[indices, i] = W[i, indices]  # Symmetric update
                W[i, i] = self.S[i, i] + self.alpha  # Diagonal update
            
            # Check for convergence
            diff = np.max(np.abs(W - W_old))  # Maximum change in W
            if diff < tol:  # Convergence condition
                break
        
        return np.linalg.inv(W), it + 1  # Return the precision matrix and iterations
    
    def admm_solve(self, max_iter=100, tol=1e-4, rho=1.0):
        """
        Solve using the Alternating Direction Method of Multipliers (ADMM).
        
        Args:
        - max_iter: Maximum number of iterations.
        - tol: Tolerance for convergence.
        - rho: Augmented Lagrangian parameter.
        
        Returns:
        - Precision matrix (inverse covariance).
        - Number of iterations taken to converge.
        """
        X = np.eye(self.p)  # Initialize primal variable X as an identity matrix
        Z = np.eye(self.p)  # Initialize auxiliary variable Z
        U = np.zeros((self.p, self.p))  # Initialize dual variable U
        print("Solving with ADMM Method...\n")
        
        for it in range(max_iter):  # Iterate up to the maximum number of iterations
            # Update X by solving the eigenvalue decomposition subproblem
            W = Z - U  # Adjusted variable
            eigvals, eigvecs = np.linalg.eigh(rho * (W + W.T) / 2 - self.S)  # Symmetric eigen-decomposition
            eigvals = (eigvals + np.sqrt(eigvals**2 + 4 * rho)) / (2 * rho)  # Update eigenvalues
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T  # Recompose X
            
            # Update Z with soft-thresholding
            A = X + U  # Adjusted variable
            Z_old = Z.copy()  # Store previous Z for convergence check
            Z = np.sign(A) * np.maximum(np.abs(A) - self.alpha / rho, 0)  # Soft-thresholding
            
            # Update U (dual variable)
            U = U + X - Z  # Update dual variable
            
            # Check for convergence
            diff = np.max(np.abs(Z - Z_old))  # Maximum change in Z
            if diff < tol:  # Convergence condition
                break
        
        return np.linalg.inv(X), it + 1  # Return the precision matrix and iterations

# Generate synthetic returns data
n_assets = 5  # Number of assets
returns_df, true_cov, true_prec = generate_synthetic_returns(n_assets=n_assets)  # Synthetic data

# Print first few rows of the synthetic returns dataset
print("\nSynthetic Returns Dataset (First 5 Rows):")
print(returns_df.head())

# Compute sample covariance matrix from returns data
sample_cov = returns_df.cov().values  # Covariance of the returns
print("\nSample Covariance Matrix:")
print(pd.DataFrame(sample_cov).round(3))

# Print the true precision matrix
print("\nTrue Precision Matrix:")
print(pd.DataFrame(true_prec).round(3))

# Initialize the Graphical Lasso solver
alpha = 0.1  # Regularization parameter
solver = GraphicalLasso(sample_cov, alpha)  # Instantiate solver

# Solve using Direct Method
direct_prec, direct_iters = solver.direct_solve()  # Compute precision using Direct Method

# Solve using ADMM Method
admm_prec, admm_iters = solver.admm_solve()  # Compute precision using ADMM

# Print results for both methods
print("\nDirect Method Precision Matrix:")
print(pd.DataFrame(direct_prec).round(3))  # Display Direct Method precision matrix
print(f"Direct Method Converged in {direct_iters} iterations.")

print("\nADMM Method Precision Matrix:")
print(pd.DataFrame(admm_prec).round(3))  # Display ADMM precision matrix
print(f"ADMM Method Converged in {admm_iters} iterations.")

# Visualize precision matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create subplots

# Plot matrices
for matrix, title, ax in [
    (true_prec, "True Precision", axes[0]),
    (direct_prec, "Direct Method Precision", axes[1]),
    (admm_prec, "ADMM Method Precision", axes[2]),
]:
    im = ax.imshow(matrix, cmap='coolwarm', aspect='auto')  # Display matrix
    ax.set_title(title)  # Title
    plt.colorbar(im, ax=ax)  # Add colorbar

plt.tight_layout()  # Adjust layout
plt.show()  # Display plots
