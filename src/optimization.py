"""
Portfolio Optimization with Multiple Constraints.

This module implements a portfolio optimization strategy considering multiple constraints
including NSFR, LCR, and TLAC ratios.
"""

from typing import Tuple, List, Dict
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass, field

# Constants
DATA_PATH = os.path.join(os.path.dirname(__file__), '../input/Cov_matrix.xlsx')
SYNTHETIC_DATA_PATH = os.path.join(os.path.dirname(__file__), '../input/synthetic_data.xlsx')

@dataclass
class OptimizationParameters:
    """Parameters for the optimization problem."""
    RSF: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.65, 0, 0.15]))
    HQLA: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.25, 1, 0.15]))
    RWA: np.ndarray = field(default_factory=lambda: np.array([1, 0.35, 0, 0.20]))
    ASF: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.9, 0.95, 1, 1]))
    T: int = 360
    DRE: float = 0.18
    Eta: float = 0.63
    Total_Bal: float = 1500000
    NSFR_ratio: float = 1.1
    LCR_ratio: float = 1.2
    TLAC_ratio: float = 0.18
    Haircut: float = 0.02
    borrow_rate: float = 0.0017

class DataGenerator:
    """Handles data generation and loading for the optimization problem."""
    
    @staticmethod
    def generate_synthetic_data(filepath: str, n: int = 100) -> pd.DataFrame:
        """Generate synthetic rate data."""
        np.random.seed(42)
        data = pd.DataFrame({
            'a1 rate': np.random.uniform(0.01, 0.03, n),
            'a2 rate': np.random.uniform(0.01, 0.03, n),
            'a3 rate': np.random.uniform(0.01, 0.03, n),
            'a4 rate': np.random.uniform(0.01, 0.03, n),
            'l1 rate': np.random.uniform(0.01, 0.03, n),
            'l2 rate': np.random.uniform(0.01, 0.03, n),
            'l3 rate': np.random.uniform(0.01, 0.03, n),
            'l4 rate': np.random.uniform(0.01, 0.03, n),
            'l5 rate': np.random.uniform(0.01, 0.03, n)
        })
        data.index.name = 'Index'
        return data

    @staticmethod
    def generate_synthetic_cov(filepath: str, n_assets: int = 9) -> pd.DataFrame:
        """Generate synthetic covariance matrix."""
        np.random.seed(42)
        A = np.random.rand(n_assets, n_assets)
        cov = np.dot(A, A.transpose())
        cov_df = pd.DataFrame(
            cov,
            columns=[f'Asset{i+1}' for i in range(n_assets)],
            index=[f'Asset{i+1}' for i in range(n_assets)]
        )
    # Do not write to disk; return covariance DataFrame in-memory
        return cov_df

    @staticmethod
    def load_data(filepath: str = SYNTHETIC_DATA_PATH) -> pd.DataFrame:
        """Load rate data from file or generate if not exists."""
        # Always generate synthetic data in-memory to avoid file I/O.
        data = DataGenerator.generate_synthetic_data(filepath)
        return data.fillna(0)

    @staticmethod
    def load_cov(filepath: str = DATA_PATH) -> pd.DataFrame:
        """Load covariance matrix from file or generate if not exists."""
        # Always generate synthetic covariance in-memory to avoid file I/O.
        cov = DataGenerator.generate_synthetic_cov(filepath)
        # Ensure symmetric covariance
        cov = cov.copy()
        for i in range(cov.shape[0]):
            cov.iloc[i, :] = cov.iloc[:, i]
        return cov

class PortfolioOptimizer:
    """Handles the portfolio optimization process."""

    def __init__(self):
        self.params = OptimizationParameters()
        self.data = DataGenerator.load_data()
        self.cov = DataGenerator.load_cov()
        self._setup_rates()

    def _setup_rates(self):
        """Extract rates from data."""
        self.a1_rate = self.data['a1 rate'].values
        self.a2_rate = self.data['a2 rate'].values
        self.a3_rate = self.data['a3 rate'].values
        self.a4_rate = self.data['a4 rate'].values
        self.l1_rate = self.data['l1 rate'].values
        self.l2_rate = self.data['l2 rate'].values
        self.l3_rate = self.data['l3 rate'].values
        self.l4_rate = self.data['l4 rate'].values
        self.l5_rate = self.data['l5 rate'].values

    def get_returns_df(self) -> pd.DataFrame:
        """Create returns DataFrame."""
        return pd.DataFrame({
            'A1': self.a1_rate,
            'A2': self.a2_rate,
            'A3': self.a3_rate/360,
            'A4': self.a4_rate,
            'L1': -self.l1_rate/360,
            'L2': -self.l2_rate/360,
            'L3': -self.l3_rate/360,
            'L4': -self.l4_rate,
            'L5': -self.l5_rate
        })

    def obj_mean_var(self, x: np.ndarray) -> float:
        """Objective function for mean-variance optimization."""
        returns = self.get_returns_df()
        return np.sqrt(x.T@(self.cov/np.sqrt(360))@x) - np.sum(returns.sum()*x)

    def calculate_CFs(self, x: np.ndarray) -> float:
        """Calculate cash flows."""
        returns = self.get_returns_df()
        return -np.sum(returns.sum()*x)

    def calculate_net_flows(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate net cash flows."""
        xa_cf = (
            self.a1_rate*x[0] +
            self.a2_rate*x[1] +
            self.a3_rate*x[2]/360 +
            self.a4_rate*x[3]
        )
        xl_cf = (
            self.l1_rate*x[4]/360 +
            self.l2_rate*x[5]/360 +
            self.l3_rate*x[6]/360 +
            self.l4_rate*x[7] +
            self.l5_rate*x[8]
        )
        xnet_cf = xa_cf - xl_cf
        return xnet_cf, xa_cf - xl_cf

    def get_constraints(self) -> List[Dict]:
        """Generate all optimization constraints."""
        p = self.params
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]+x[3]-p.Total_Bal},  # cons1
            {'type': 'eq', 'fun': lambda x: x[4]+x[5]+x[6]+x[7]+x[8]-p.Total_Bal},  # cons2
            {'type': 'ineq', 'fun': lambda x: sum(p.ASF*x[4:])/sum(p.RSF*x[:4]) - p.NSFR_ratio},  # cons3
            {'type': 'ineq', 'fun': lambda x: x[7]/sum(p.RWA*x[:4]) - p.TLAC_ratio},  # cons4
            {'type': 'ineq', 'fun': self._lcr_constraint},  # cons5
            {'type': 'ineq', 'fun': self._coverage_constraint},  # cons6
            {'type': 'ineq', 'fun': lambda x: x[2]-0.1*sum(x[:4])},  # cons7
            {'type': 'ineq', 'fun': lambda x: 0.6*p.Total_Bal - (x[4]+x[5]+x[6])},  # cons8
            {'type': 'eq', 'fun': lambda x: x[5]+x[6] - p.Eta*(x[4]+x[5]+x[6])},  # cons9
            {'type': 'eq', 'fun': lambda x: x[5] - p.DRE*(x[5]+x[6])}  # cons10
        ]
        return constraints

    def _lcr_constraint(self, x: np.ndarray) -> float:
        """LCR ratio constraint calculation."""
        p = self.params
        xnet_cf, _ = self.calculate_net_flows(x)
        HQLA_v = sum(p.HQLA * x[:4])
        TNCO = np.array([])
        
        for t in range(p.T):
            if t <= p.T-30:
                value = x[4] + x[5]*1/12 + x[6]*1/30 + sum(xnet_cf[t:t+30])
            else:
                value = x[4] + x[5]*1/12 + x[6]*1/30 + sum(xnet_cf[t:])
            TNCO = np.append(TNCO, value)
            
        return HQLA_v/TNCO - p.LCR_ratio

    def _coverage_constraint(self, x: np.ndarray) -> float:
        """Coverage constraint calculation."""
        p = self.params
        xnet_cf, _ = self.calculate_net_flows(x)
        xcum_cf = xnet_cf.cumsum()
        return x[3]*(1-p.Haircut)*(1-p.borrow_rate) + xcum_cf



    def solve_optimization(self) -> Dict:
        """
        Solve the portfolio optimization problem.
        Returns a dictionary containing the solution and analysis.
        """
        p = self.params

        # Setting up bounds for each variable
        bnds = tuple([(0, p.Total_Bal)]*9)
        
        # Starting point for minimization 
        x0 = np.array([p.Total_Bal/9]*9)
        
        # Get constraints
        constraints = self.get_constraints()
        
        # Solving using SLSQP algorithm
        solution = minimize(
            self.obj_mean_var,
            x0,
            bounds=bnds,
            constraints=constraints,
            method='SLSQP'
        )
        
        if not solution.success:
            raise ValueError(f"Optimization failed: {solution.message}")
            
        x = solution.x
        
        # Calculate repo amounts and interest
        repo_analysis = self._calculate_repo_analysis(x)
        
        # Prepare results
        results = {
            'solution': x,
            'long_asset_weights': x[:4]/sum(x[:4]),
            'long_liability_weights': x[4:]/sum(x[4:]),
            'pnl_without_repo': -self.calculate_CFs(x),
            'pnl_with_repo': -self.calculate_CFs(x) + repo_analysis['total_interest'],
            'repo_analysis': repo_analysis,
            'optimization_success': solution.success,
            'optimization_message': solution.message,
            'objective_value': self.obj_mean_var(x)
        }
        
        return results
        
    def _calculate_repo_analysis(self, x: np.ndarray) -> Dict:
        """Calculate repo amounts and interest."""
        p = self.params
        
        # Calculate cumulative cash flows
        xnet_cf, _ = self.calculate_net_flows(x)
        xcum_cf = xnet_cf.cumsum()
        
        # Find negative cash flows
        neg_cfs = xcum_cf[xcum_cf < 0]
        
        # Create repo analysis DataFrame
        repo_df = pd.DataFrame(index=np.array(['t'+'{}'.format(i) for i in range(len(neg_cfs)+1)]))
        repo_df['NCNC'] = [0] + neg_cfs.tolist()
        repo_df['Repo Amt'] = (repo_df['NCNC']/(1-p.Haircut)).shift(-1)
        repo_df['Interest'] = repo_df['Repo Amt'] * p.borrow_rate
        
        return {
            'dataframe': repo_df,
            'negative_cashflows': neg_cfs,
            'repo_amounts': repo_df['Repo Amt'].dropna(),
            'interest_charges': repo_df['Interest'].dropna(),
            'total_interest': repo_df['Interest'].sum()
        }

# Standalone script to create the dataset and run optimization
if __name__ == "__main__":
    try:
        # Create and run optimizer
        optimizer = PortfolioOptimizer()
        results = optimizer.solve_optimization()

        # Print results
        print("\nOptimization Results:")
        print("-"*50)
        print(f"Optimization Status: {results['optimization_message']}")
        print(f"\nLong Asset Portfolio Weights:")
        for i, w in enumerate(results['long_asset_weights']):
            print(f"Asset {i+1}: {w:.4f}")
            
        print(f"\nLong Liability Portfolio Weights:")
        for i, w in enumerate(results['long_liability_weights']):
            print(f"Liability {i+1}: {w:.4f}")
            
        print(f"\nPerformance Metrics:")
        print(f"PnL without repo: {results['pnl_without_repo']:.2f}")
        print(f"Final PnL: {results['pnl_with_repo']:.2f}")
        print(f"Objective value: {results['objective_value']:.6f}")
        
        print("\nRepo Analysis:")
        print(results['repo_analysis']['dataframe'])
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        raise
