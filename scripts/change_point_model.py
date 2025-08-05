import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt

def detect_change_points(df, n_changepoints=5):
    prices = df['Price'].values
    n_obs = len(prices)
    
    with pm.Model() as model:
        # Priors for change points (uniform distribution)
        changepoints = pm.Uniform(
            "changepoints",
            lower=0,
            upper=n_obs,
            shape=n_changepoints
        )
        
        # Correct way to sort in PyMC v5+
        sorted_cp = pm.Deterministic("sorted_cp", pt.sort(changepoints))
        
        # Segment means and standard deviation
        segment_means = pm.Normal("segment_means", mu=0, sigma=10, shape=n_changepoints+1)
        segment_sd = pm.HalfNormal("segment_sd", sigma=1)
        
        # Determine regime for each observation
        regime = np.zeros(n_obs, dtype=int)
        for i in range(n_changepoints):
            regime = regime + (np.arange(n_obs) >= sorted_cp[i])
        
        # Likelihood
        pm.Normal(
            "likelihood",
            mu=segment_means[regime],
            sigma=segment_sd,
            observed=prices
        )
        
        # Sample from posterior (no C++ compiler needed)
        trace = pm.sample(
            draws=1000,  # Reduced for faster testing
            tune=1000,
            chains=2,
            target_accept=0.9,
            return_inferencedata=True,
            cores=1  # Avoid multiprocessing issues
        )
    
    return trace

def analyze_results(trace, df):
    # Extract change points
    cp_samples = trace.posterior["sorted_cp"].values
    cp_mean = np.mean(cp_samples, axis=(0, 1)).astype(int)
    detected_dates = df.iloc[cp_mean]["Date"]
    
    # Plot results
    az.plot_trace(trace, var_names=["segment_means", "segment_sd"])
    plt.savefig('../outputs/figures/trace_plot.png')
    plt.close()
    
    return detected_dates

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('./data/processed/cleaned_oil_prices.csv', parse_dates=['Date'])
    
    print("Running change point detection (pure Python mode)...")
    try:
        trace = detect_change_points(df)
        change_points = analyze_results(trace, df)
        print("✅ Detected change points at:", change_points.values)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Try reducing n_changepoints or increasing tune/draws")