# Model comparison after calibration

import numpy as np
import pandas as pd
import xarray as xr
import os 
from datetime import datetime
import matplotlib.pyplot as plt

from DENV_models_pySODM import JumpProcess_SEIR2_beta_by_Temp_sf_BirthDeath_reported_v2 as SEIR2_v2
from Scaling_functions_beta_t import generate_scaling_factors
from time_dep_parameters import time_dependent_beta
from Utilities import load_epi_and_scaling_factors, convert_params_to_weeks

from Model_vs_data_post_calibration import create_draw_fnc, sim_with_draws

# pysodm dependencies
from pySODM.optimization.utils import variance_analysis
from pySODM.optimization.objective_functions import  ll_negative_binomial

base_dir = "/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/SSH" # for results from the ssh, for local "/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model"
scaling_factors = ["seasonal_forcing", "mordecai_aeg", "lambrechts", "perkins"]
dir_per_sf = ["optimization_SF", "optimization_Mord", "optimization_Lamb", "optimization_Perk"]
mcmc_files = {"seasonal_forcing": "/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/SSH/sampler_output_SF/SeasonalForcing_rverstra_SAMPLES_2025-02-28.nc", 
              "mordecai_aeg": "/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/SSH/sampler_output_Mord/Mordecai_rverstra_SAMPLES_2025-02-28.nc", 
              "lambrechts": "/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/SSH/sampler_output_Lamb/Lambrechts_rverstra_SAMPLES_2025-02-28.nc", 
              "perkins": "/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/SSH/sampler_output_Perk/Perkins_rverstra_SAMPLES_2025-02-28.nc", }


time_unit = "W"         # unit of time for simulation (must be the same as for calibration)
tau = 1                 # steps in each unit of time (1 week now)
n = 50                  # repeat simulations with new draw of posterior distr parameters

fig_path = "/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/"

##################
# load the mcmc data & check that they all have the same calibration start- and end_date
##################
mcmc_data, start_dates, end_dates = {}, [], []

for key, filepath in mcmc_files.items():
    try:
        ds = xr.open_dataset(filepath)
        mcmc_data[key] = ds
        start_date, end_date = ds.attrs.get("start_calibration"), ds.attrs.get("end_calibration")

        if not (start_date and end_date):
            print(f"Warning: {key} is missing start_calibration or end_calibration.")
        
        start_dates.append(start_date)
        end_dates.append(end_date)
        print(f"Loaded {key}: start_date={start_date}, end_date={end_date}")

    except Exception as e:
        print(f"Error loading {key}: {e}")

# Get unique start and end dates as either str or datetime
unique_start_dates = list(set(start_dates))  # Convert to set to get unique values, then back to list
unique_end_dates = list(set(end_dates))

# Check if there is only one unique date
if len(unique_start_dates) == 1 and len(unique_end_dates) == 1:
    start_date = unique_start_dates[0]
    end_date = unique_end_dates[0]
else:
    print(f"⚠️ Mismatch detected: start_dates={unique_start_dates}, end_dates={unique_end_dates}")

################### 
# load the epi data 
###################
counts_filtered, scaling_factors_filtered = load_epi_and_scaling_factors(start_date=start_date, end_date=end_date, time_unit= "W")

####################################
# Simulate the model per scaling factor
####################################

model_outputs = {}
params={'alpha':182.5, 'b':2.77e-05, 'd':2.45e-05, 'sigma':6, 'gamma':7, 'psi': 1.5, 'beta_0' : 0.3, 'sf' : scaling_factors_filtered['seasonal_forcing'], 'rho' : 0.10} 

# convert daily rates into weekly rates
if time_unit == "W": 
    params_weekly = convert_params_to_weeks(params)
    params = params_weekly

for scaling_factor, mcmc_file in mcmc_files.items(): 
    out = sim_with_draws(scaling_factor_name= scaling_factor, scaling_factors_filtered = scaling_factors_filtered, counts_filtered = counts_filtered,
                        params_to_draw= ["alpha", "beta_0", "sigma", "gamma", "rho"], 
                        mcmc_file = mcmc_file,
                        params = params, tau = tau, time_unit=time_unit, n=n)
    
    model_outputs[scaling_factor] = out


#############################################################
# check mean-variance relation to choose likelihood function
#############################################################

results, ax = variance_analysis(counts_filtered, resample_frequency='3W')
alpha = results.loc['negative binomial', 'theta']
print(results)
plt.show()
plt.close()

############################################
# Calculate model selection criteria
############################################

# # filter the ymodel to only include dates for which we have data in counts --> MOET DIT?
# simulation_results_filtered = {
#     model_name: results.sel(date=common_dates)
#     for model_name, results in model_outputs.items()
# }

log_likelihoods = {}
RMSEs = {}

for model_name, results in model_outputs.items():
    y_model = results["I_rep"]  # Model output (shape: draws x timepoints)
    
    # Compute log-likelihood for each draw and take the mean log-likelihood
    log_lik_values = [ll_negative_binomial(y_model[i, :], counts_filtered, alpha) for i in range(y_model.shape[0])]
    
    # Compute the RME for each draw
    RMSE_values = [np.sqrt(np.mean((y_model[i, :] - counts_filtered) ** 2)) for i in range(y_model.shape[0])]
    
    # Store mean log-likelihood across draws
    log_likelihoods[model_name] = np.mean(log_lik_values)
    RMSEs[model_name] = np.mean(RMSE_values)

# Print results
for model, log_lik in log_likelihoods.items():
    print(f"{model}: Log-Likelihood = {log_lik:.2f}")

for model, RMSE in RMSEs.items():
    print(f"{model}: RMSE = {RMSE:.2f}")


######################################
# PLOT THE RESULTS 
######################################

# Extract the date from the first filename in mcmc_files
first_file = next(iter(mcmc_files.values()))  
extracted_date = first_file.split("SAMPLES_")[-1].split(".nc")[0]

# Construct the identifier
identifier = f"rverstra_{extracted_date}"

def plot_simulation_results(simulation_results, counts_filtered, log_likelihoods, RMSEs):
    """
    Plots I_rep over time for each model in a 2x2 grid, showing all draws, real data, 
    and log-likelihood values.

    Parameters:
    - simulation_results: dict, where keys are model names and values contain simulation outputs 
      with dimensions (date, draws).
    - counts_filtered: pd.Series, real case data indexed by date.
    - log_likelihoods: dict, log-likelihood values for each model.
    - RMSEs: dict, RMSE values for each model. 

    Returns:
    - fig: matplotlib figure object.
    """
    num_models = len(simulation_results)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))  # 2x2 grid
    axes = axes.flatten()  # Flatten to iterate easily

    for idx, (model_name, results) in enumerate(simulation_results.items()):
        ax = axes[idx]
        dates = results["date"].values  # Convert xarray DataArray to NumPy array
        num_draws = results["I_rep"].shape[0]  # Number of draws

        # Plot individual trajectories for I_rep
        for i in range(num_draws):
            ax.plot(dates, results["I_rep"][i, :], color="red", alpha=0.1)

        # Overlay mean trend for I_rep
        ax.plot(dates, np.mean(results["I_rep"], axis=0), color="red", linewidth=2, label="Mean I_rep")

        # Overlay real data (counts_filtered)
        common_dates = counts_filtered.index.intersection(dates)  # Ensure matching dates
        ax.scatter(common_dates, counts_filtered.loc[common_dates], 
                   color="black", label="Real Data", zorder=3, marker="o", s=20)

        # Add log-likelihood to the plot as an extra legend entry
        log_likelihood = log_likelihoods.get(model_name, "N/A")
        rmse = RMSEs.get(model_name, "N/A")

        extra_legend = (
            f"Log-Likelihood: {log_likelihood:.2f}" if isinstance(log_likelihood, (int, float)) else "N/A"
        )
        extra_legend += (
            f"\nRMSE: {rmse:.2f}" if isinstance(rmse, (int, float)) else "\nRMSE: N/A"
        )

        # Formatting
        ax.set_title(f"Simulation Results for {model_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Reported Infections (I_rep)")
        ax.legend(title=extra_legend, loc="upper right")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    return fig  # Return figure object
    
# Call function
fig = plot_simulation_results(model_outputs, counts_filtered, log_likelihoods, RMSEs)
plotname = f"{identifier}_logLikelihoodPlot.pdf"
plotpath = os.path.join(fig_path, plotname)
fig.savefig(plotpath, dpi=600, bbox_inches="tight")

