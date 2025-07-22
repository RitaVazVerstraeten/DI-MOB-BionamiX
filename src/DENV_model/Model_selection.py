# Model selection
import numpy as np
import pandas as pd
import json
import os 

# import datetime

import matplotlib.pyplot as plt

# pysodm dependencies
from pySODM.optimization.utils import variance_analysis
from pySODM.optimization.objective_functions import  ll_negative_binomial

################
## Load model ##
################

from DENV_models_pySODM import JumpProcess_SEIR2_beta_by_Temp_sf_BirthDeath_reported_v2 as SEIR2_v2

fig_path = "/home/rita/PyProjects/DI-MOB-BionamiX/results/DENV_model/optimization/"

################
## Load data ## -> ydata
################

## Reading the dataframe
epi_data_original = pd.read_excel("/home/rita/PyProjects/DI-MOB-BionamiX/data/WP1_Epidemiological_Data/DENV/DENV_12-23_cleaned_Area_CP_Mz_IDCODE_v18112024_forsharing.xlsx")

epi_data = epi_data_original 
# turn object columns into categorical variables
epi_data[epi_data.select_dtypes(['object']).columns] = epi_data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
# Create a counts dataframe per date
counts = epi_data['XDate_Positivity'].value_counts()
# rename XDate_Positivity to date
counts.rename_axis('date', inplace=True)
# sort by date: 
counts = counts.sort_index()
# Explicitly converting it to a pd.Series
counts = pd.Series(counts)

##########################################
# Model parameters and initial conditions
##########################################

# Core demographics for start_date = 2012-01-01 in Cienfuegos
####################################################################
demo = pd.read_excel("/home/rita/PyProjects/DI-MOB-BionamiX/data/WP1_ Demographic_Data/Population counts and proportions Municipio Cfgos_v17092024.xlsx")

# Generate scaling factors from meteo data
####################################################################
from Scaling_functions_beta_t import generate_scaling_factors

# LOAD THE METEO DATA FOR SCALING FACTORS
# Paths and file
file_path = "/media/rita/New Volume/Documenten/DI-MOB/Data Sharing/WP2_Meteorological_Data"
file = "Meteorologicas_2012_2022_withSE_v04102024_v1.xlsx"

# Load and preprocess the meteorological data
meteo = pd.read_excel(os.path.join(file_path, file))
meteo = meteo.drop(columns=["Canta_Rana", "Los_Jimenez", "La_Ceiba"], errors='ignore')
meteo.columns = meteo.columns.str.strip().str.replace(' ', '_').str.lower()

if 'date' not in meteo.columns:
    raise ValueError("Column 'date' not found in the dataset.")
meteo['date'] = pd.to_datetime(meteo['date'], errors='coerce')
meteo.set_index('date', inplace=True)

# Extract temperature and rainfall series
temperature_series = meteo["temp_med"]
rainfall_series = meteo["precip_ponderada"]

# Handle missing values
if temperature_series.isna().any() or rainfall_series.isna().any():
    print("Warning: Missing values detected. Filling with interpolation.")
    temperature_series = temperature_series.interpolate()
    rainfall_series = rainfall_series.interpolate()

# GENERATE THE SCALING FACTORS
scaling_factors, time = generate_scaling_factors(
    temperature_series=temperature_series,
    rainfall_series=rainfall_series,
    dates=temperature_series.index,
    scaling_methods= ["seasonal_forcing", "mordecai_aeg", "lambrechts", "perkins"]
)

for key, df in scaling_factors.items():
    df.index = meteo.index

# Find the intersection of indices
###################################
common_dates = counts.index.intersection(meteo.index)
# Filter counts to only common dates
common_dates = common_dates[common_dates >= '2012-06-01']
counts_filtered = counts.loc[common_dates].sort_index()
# # Filter each DataFrame in scaling_factors to only common dates
# scaling_factors_filtered = {key: df.loc[common_dates] for key, df in scaling_factors.items()}

start_date = pd.to_datetime('2012-06-01')
end_date = counts_filtered.index[-1]
scaling_factors_filtered = {
    key: df.sort_index()[df.index >= start_date]
    for key, df in scaling_factors.items()
}


# # remove date as index 
# scaling_factors = {k: v.reset_index() for k, v in scaling_factors.items()}

################################################
# GET POSTERIOR DISTRIBUTIONS FROM MCMC RESULTS
################################################

mcmc_files = {
    'model_SF': '/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/optimization_SF/sampler_output_SF/SeasonalForcing_rverstra__SAMPLES_2025-02-11.json',

    'model_Lamb': '/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/optimization_Lamb/sampler_output_Lamb/Lambrechts_rverstra__SAMPLES_2025-02-11.json',

    'model_Perk': '/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/optimization_Perk/sampler_output_Perk/Perkins_rverstra__SAMPLES_2025-02-11.json',

    'model_Mord': '/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/optimization_Mord/sampler_output_Mord/Mordecai_rverstra__SAMPLES_2025-02-11.json'
}

def load_mcmc_posteriors(mcmc_files, parameter_keys):
    """
    Load the posterior distributions of specified parameters from multiple MCMC JSON files.
    
    Parameters:
        mcmc_files (dict): Dictionary with model names as keys and file paths as values.
        parameter_keys (list): List of parameter names to extract.
    
    Returns:
        dict: A dictionary containing parameter samples for each model.
    """
    posterior_samples = {}
    
    for model, file_path in mcmc_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                mcmc_data = json.load(f)
            
            # Extract only the specified parameter keys
            param_samples = {k: mcmc_data[k] for k in parameter_keys if k in mcmc_data}
            
            posterior_samples[model] = param_samples
            print(f"Loaded {len(param_samples)} parameters from {model}.")
        else:
            print(f"Warning: File not found for {model}: {file_path}")
    
    return posterior_samples

# Define the keys that correspond to parameters (update this list based on your JSON structure)
parameter_keys = ['alpha', 'beta_0', 'sigma', 'rho']  

# Load the posterior distributions
posterior_samples = load_mcmc_posteriors(mcmc_files, parameter_keys)

# Print summary of loaded parameters
for model, params in posterior_samples.items():
    print(f"\n{model}: Loaded {len(params)} parameters")
    for param, values in params.items():
        print(f"  {param}: {len(values)} samples")

#############################################################
# SIMULATE EACH MODEL WITH PARAMETERS SAMPLED FROM POSTERIOR -> ymodel
#############################################################

# DEFINE A DRAW_FNC TO DRAW FROM POSTERIORS
def create_draw_fnc(posterior_samples):
    """
    Factory function to create a draw_fnc that samples from a given model's posterior distribution.

    Parameters:
        posterior_samples (dict): Dictionary containing posterior samples for each model.

    Returns:
        function: A draw function that can be passed to `model_temp.sim()`
    """
    def draw_fnc(parameters):
        """Draws parameters from the posterior distributions for use in model simulation."""
        for param, values in posterior_samples.items():
            parameters[param] = np.random.choice(values)  # Sample from posterior
        
        # Print for verification
        # print(f"Drawn parameters: {parameters}")

        return parameters

    return draw_fnc


# # Example: Creating a draw function for a specific model (e.g., 'model_SF')
# draw_fnc_SF = create_draw_fnc(posterior_samples['model_SF'])

# rename the scaling factors in scaling_factors to match the model names in the mcmcm output
scaling_factors_filtered = {
    "model_SF": scaling_factors_filtered["seasonal_forcing"],
    "model_Mord": scaling_factors_filtered["mordecai_aeg"],
    "model_Perk": scaling_factors_filtered["perkins"],
    "model_Lamb": scaling_factors_filtered["lambrechts"],
}


##################################################
# Function to initialize - simulate each model 
##################################################

######## I STILL NEED TO FIGURE OUT WHETHER TIME_DEP_BETA SHOULD BE IMPLEMENTED HERE, BECAUSE I BELIEVE THAT BETA_T IS NOW CONSTANT AT BETA_0 ...
def init_sim_model(scaling_factor_name):
    '''
    Function where the model get's initialized and simulated using the correct scaling factor values + parameter posterior distributions from that scaling_factor's mcmc 

    Input 
    -----
    scaling_factor_model: string containing the name of the calibrated model. Can be model_SF, model_Lamb, model_Perk, model_Mord

    scaling_factor_SF: string containing the name under which the scaling factors are stored. Can be "seasonal_forcing", "mordecai_aeg", "lambrechts", or "perkins" .

    Output
    ------
    Model simulation in xarray.Dataset
    '''
    # select a random rho from the posterior
    rho = np.mean(posterior_samples[scaling_factor_name]["rho"])
    print(f"\nFrom {scaling_factor_name} rho = ", round(rho, 2))
    # set the parameters
    params={'alpha':182.5, 'b':2.77e-05, 'd':2.45e-05, 'sigma':6, 'gamma':7, 'psi': 1.5, 'beta_0' : 0.3, 'sf' : scaling_factors_filtered[scaling_factor_name], 'rho' : rho} 

    # set the initial conditions
    # zet initiele conditie gelijk aan 1/rho * I_reported en verdeel deze mensen over I1, I2, I12, en I21: 
    initial_infected = round((1/params["rho"])*counts[start_date], 0)

    # Equal probabilities for 4 infectious compartments
    probabilities = [0.25, 0.25, 0.25, 0.25]

    # Stochastic division using multinomial
    [init_I1, init_I2, init_I12, init_I21] = np.random.multinomial(initial_infected, probabilities)
    print("init_I1", init_I1, "\ninit_I2", init_I2, "\ninit_I12", init_I12, "\ninit_I21", init_I21)

    # Define initial condition
    initN = demo["Population_Muni"][0]
    initS = initN - (init_I1 + init_I2 + init_I12 + init_I21)

    initial_states = {
        'S': np.array([initS]),
        'I1': np.array([init_I1]),
        'I2': np.array([init_I2]),
        'I12': np.array([init_I12]),
        'I21': np.array([init_I21])
    }
    # initialize the model 
    model = SEIR2_v2(initial_states=initial_states, parameters=params)

    # simulate the model + draw_fnc
    draw_fnc = create_draw_fnc(posterior_samples= posterior_samples[scaling_factor_name])    
    out = model.sim(time=[time[0], time[-1]], N =n , draw_function=draw_fnc, tau=tau, output_timestep=1)
    # store the results

    return out

tau = 1.0
n = 100
# # test run on SF model 
# out = init_sim_model(scaling_factor_name= "model_SF")

# Loop over all the scaling factors and store the model simulation results
simulation_results = {}

for model_name, posterior in posterior_samples.items():
    sim_result = init_sim_model(scaling_factor_name = model_name)
    # Store results
    simulation_results[model_name] = sim_result
    print(f"\nSimulation completed for {model_name} with {n} iterations")


########################
# AIC and BIC variables
########################

k = len(SEIR2_v2.parameters)  # Number of parameters
n = len(counts)  # Sample size

######################
# log-likelihood function selection
######################

if __name__ == '__main__':    
    #############################################################
    # check mean-variance relation to choose likelihood function
    #############################################################

    results, ax = variance_analysis(counts, resample_frequency='3W')
    alpha = results.loc['negative binomial', 'theta']
    #print(results)
    plt.show()
    plt.close()

# Negative binomial is most appropriate -> lowest AIC with argument = 0.885382
alpha = results["theta"]["negative binomial"]

################################################
# Calculate log-likelihood for negative binomial
################################################

# filter the ymodel to only include dates for which we have data in counts
simulation_results_filtered = {
    model_name: results.sel(date=common_dates)
    for model_name, results in simulation_results.items()
}

# Example: Compute log-likelihood for all models
log_likelihoods = {}
RMSEs = {}

for model_name, results in simulation_results_filtered.items():
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

#############################################
# Plot the results
#############################################

# Extract the date from the first filename in mcmc_files
first_file = next(iter(mcmc_files.values()))  
extracted_date = first_file.split("SAMPLES_")[-1].split(".json")[0]

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
fig = plot_simulation_results(simulation_results, counts_filtered, log_likelihoods, RMSEs)
plotname = f"{identifier}_logLikelihoodPlot.pdf"
plotpath = os.path.join(fig_path, plotname)
fig.savefig(plotpath, dpi=300, bbox_inches="tight")