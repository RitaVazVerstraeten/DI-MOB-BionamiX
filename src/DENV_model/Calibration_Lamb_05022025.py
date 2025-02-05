# Calibration of JumpProcess_SEIR2_beta_by_Temp_sf_BirthDeath_reported_v2 model parameters with pySODM
# Using scaling factor = Lambrechts 

# General purpose packages
import corner
import pandas as pd
import numpy as np
import os
import json

import datetime

import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_negative_binomial_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial, ll_poisson

# # OPTIONAL: Load the "autoreload" extension so that package code can change
# %load_ext autoreload
# # OPTIONAL: always reload modules so that as you change code in src, it gets loaded
# %autoreload 2

os.chdir('/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model')

##############
## Settings ##
##############
import multiprocessing as mp

tau = 1.0                                        # Timestep of Tau-Leaping algorithm
# alpha = 0.03                                    # Overdispersion factor (based on COVID-19)
# end_calibration = '2018-03-01'                  # Enddate of calibration
n_pso = 30                                      # Number of PSO iterations
multiplier_pso = 10                             # PSO swarm size
n_mcmc = 500                                    # Number of MCMC iterations
multiplier_mcmc = 5                            # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 100                                   # Print diagnostics every print_n iterations
discard = 50                                    # Discard first `discard` iterations as burn-in
thin = 10                                       # Thinning factor emcee chains
n = 100                                         # Repeated simulations used in visualisations
# processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())) # Retrieve CPU count  
processes = 3                                   # 3 so that if I run all 4 scaling factor scripts simultaneously, I can use my 12 cores. 

# Variables
samples_path='optimization/sampler_output_Lamb/'
fig_path='optimization/sampler_output_Lamb/'
identifier = 'Lambrechts_rverstra_2025-02-04' # Give any output of this script an ID
run_date = str(datetime.date.today())

####################################
# Load epi dataset for calibration
####################################

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

if __name__ == '__main__':    
    #############################################################
    # check mean-variance relation to choose likelihood function
    #############################################################

    results, ax = variance_analysis(counts, resample_frequency='W')
    alpha = results.loc['negative binomial', 'theta']
    #print(results)
    plt.show()
    plt.close()

# Negative binomial is most appropriate -> lowest AIC with argument = 0.885382

################
## Load model ##
################

from DENV_models_pySODM import JumpProcess_SEIR2_beta_by_Temp_sf_BirthDeath_reported_v2 as SEIR2_v2


##########################
## Define scaling factors
##########################

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
    #print("Warning: Missing values detected. Filling with interpolation.")
    temperature_series = temperature_series.interpolate()
    rainfall_series = rainfall_series.interpolate()

# GENERATE THE SCALING FACTORS
scaling_factors, time = generate_scaling_factors(
    temperature_series=temperature_series,
    rainfall_series=rainfall_series,
    dates=temperature_series.index,
    scaling_methods= ['lambrechts']
)

for key, df in scaling_factors.items():
    df.index = meteo.index

###############################################
# Get overlap between dates sf and epidemiology
###############################################

# Find the intersection of indices
common_dates = counts.index.intersection(meteo.index)
# Filter counts to only common dates
counts_filtered = counts.loc[common_dates]
# Filter each DataFrame in scaling_factors to only common dates
scaling_factors_filtered = {key: df.loc[common_dates] for key, df in scaling_factors.items()}

# Sort counts_filtered by date and reset index
counts_filtered = counts_filtered.sort_index()
start_date = counts_filtered.index[0]
end_date = counts_filtered.index[-1]

###############################################
# Set-up model parameters & initial conditions
###############################################

demo = pd.read_excel("/home/rita/PyProjects/DI-MOB-BionamiX/data/WP1_ Demographic_Data/Population counts and proportions Municipio Cfgos_v17092024.xlsx")

initN = np.rint(demo["Population_Muni"][0])

# Sort each DataFrame in scaling_factors_filtered by date and reset index
scaling_factors_filtered = {
    key: df.sort_index()
    for key, df in scaling_factors_filtered.items()
}

params={'alpha':182.5, 'b':2.77e-05, 'd':2.45e-05, 'sigma':6, 'gamma':7, 'psi': 1.5, 'beta_0' : 0.3, 'sf' : scaling_factors_filtered['lambrechts'], 'rho' : 0.10} 

# zet initiele conditie gelijk aan 1/rho * I_reported en verdeel deze mensen over I1, I2, I12, en I21: 
initial_infected = (1/params["rho"])*counts[start_date]

# Equal probabilities for 4 infectious compartments
probabilities = [0.25, 0.25, 0.25, 0.25]

# Stochastic division using multinomial
[init_I1, init_I2, init_I12, init_I21] = np.random.multinomial(initial_infected, probabilities)


# Define initial condition
initN = initN
initS = initN - (init_I1 + init_I2 + init_I12 + init_I21)

initial_states = {
    'S': np.array([initS]),
    'I1': np.array([init_I1]),
    'I2': np.array([init_I2]),
    'I12': np.array([init_I12]),
    'I21': np.array([init_I21])
}

####################################################################
# for scaling_factor = seasonal forcing 
####################################################################
from time_dep_parameters import time_dependent_beta

# set time-dependent parameters
time_dep_beta = time_dependent_beta(sf=params["sf"])

# initialize model
model = SEIR2_v2(initial_states=initial_states, 
        parameters=params, 
        time_dependent_parameters={"beta_0": time_dep_beta})

if __name__ == '__main__':

    a = 0.885382
    # Define dataset
    counts_filtered.index.name = 'date'
    d = pd.Series(counts_filtered)
    data=[d, ]

    states = ["I_rep",]
    # select likelihood function and arguments based on results from variance_analysis function
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [a,]
    # Calibated parameters and bounds
    pars = ['alpha', 'beta_0', 'sigma','rho'] #TCI, baseline infectioun rate, IIP, and fraction of reported cases 
    labels = ['$alpha$' ,'$beta_0$', '$sigma$', '$rho$']
    bounds = [(0.01, 720), (0.1, 0.4), (4.00, 7.00), (0, 1)]

    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,labels=labels)

    ####################
    # PSO / Nelder-Mead
    ####################

    # Initial guess --> pso
    theta = pso.optimize(objective_function, swarmsize=3*18, max_iter=5, processes=10, debug=True)[0]

    theta = nelder_mead.optimize(objective_function, theta, 0.10*np.ones(len(theta)), processes=10, max_iter=5)[0]
    
    ##########
    ## MCMC ##
    ##########

    # Extract expanded bounds and labels
    expanded_labels = objective_function.expanded_labels 
    expanded_bounds = objective_function.expanded_bounds  

    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.30*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=expanded_bounds)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)

    # change format of start and end_date
    start_calibration = start_date.strftime("%Y-%m-%d")
    end_calibration = end_date.strftime("%Y-%m-%d")

    settings={'start_calibration': start_calibration, 'end_calibration': end_calibration,'n_chains': nwalkers, 'starting_estimate': list(theta), 'labels': expanded_labels, 'tau': tau}

    # Sample n_mcmc iterations
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier,
                                objective_function,
                                objective_function_kwargs={'simulation_kwargs': {'tau':tau}},
                                fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                settings_dict=settings)       


# Generate a sample dictionary and save it as .json for long-term storage
# Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard, thin=thin)
# Look at the resulting distributions in a cornerplot
CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
fig = corner.corner(sampler.get_chain(discard=discard, thin=2, flat=True), labels=expanded_labels, **CORNER_KWARGS)
for idx,ax in enumerate(fig.get_axes()):
    ax.grid(False)
plt.show()
plt.close()
