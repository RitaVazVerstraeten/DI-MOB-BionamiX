# Calibration of JumpProcess_SEIR2_beta_by_Temp_sf_BirthDeath_reported_v2 model parameter with pySODM

# General purpose packages
import corner
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_negative_binomial_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial, ll_poisson

# OPTIONAL: Load the "autoreload" extension so that package code can change
%load_ext autoreload
# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
%autoreload 2

os.chdir('/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model')

####################################
# Load epi dataset for calibration
####################################

## Reading the dataframe
epi_data_original = pd.read_excel("/home/rita/PyProjects/DI-MOB-BionamiX/data/WP1_Epidemiological_Data/DENV/DENV_12-23_cleaned_Area_CP_Mz_IDCODE_v18112024_forsharing.xlsx")

print(epi_data_original.head())

# get the datatypes of all variables
epi_data_original.info()

epi_data = epi_data_original 
# turn object columns into categorical variables
epi_data[epi_data.select_dtypes(['object']).columns] = epi_data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
epi_data.info()

# Create a counts dataframe per date
counts = epi_data['XDate_Positivity'].value_counts()
# rename XDate_Positivity to date
counts.rename_axis('date', inplace=True)
# sort by date: 
counts = counts.sort_index()
# Explicitly converting it to a pd.Series
counts = pd.Series(counts)

#############################################################
# check mean-variance relation to choose likelihood function
#############################################################

results, ax = variance_analysis(counts, resample_frequency='W')
alpha = results.loc['negative binomial', 'theta']
print(results)
plt.show()
plt.close()

# Negative binomial is most appropriate -> lowest AIC with argument = 0.885382

################
## Load model ##
################

# from DENV_models_pySODM import JumpProcess_SEIR2_beta_by_Temp_sf_BirthDeath_reported_v2 as SEIR2

from DENV_models_pySODM import JumpProcess_SEIR2_beta_by_Temp_sf_BirthDeath_reported_v3 as SEIR2
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

###############################################
# Get overlap between dates sf and epidemiology
###############################################
# ISSUE: the indices of the params cannot be TimeDates because the t that get's created in the model is in integers. So, I need to reset the index of sf to integers temporariliy: 


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

# counts_filtered = counts_filtered.reset_index(drop=True)


# Sort each DataFrame in scaling_factors_filtered by date and reset index
scaling_factors_filtered = {
    key: df.sort_index().reset_index(drop=True)
    for key, df in scaling_factors_filtered.items()
}


####################################################################
# Core demographics for start_date = 2012-01-01 in Cienfuegos
####################################################################
demo = pd.read_excel("/home/rita/PyProjects/DI-MOB-BionamiX/data/WP1_ Demographic_Data/Population counts and proportions Municipio Cfgos_v17092024.xlsx")

initN = np.rint(demo["Population_Muni"][0])

#################
## Setup model ##
#################

scaling_factors_filtered = scaling_factors_filtered['seasonal_forcing'].values.flatten()
print(type(scaling_factors_filtered)) # numpy.ndarray -> should be OK

params={'alpha':182.5, 'b':2.77e-05, 'd':2.45e-05, 'sigma':6, 'gamma':7, 'psi': 1.5, 'beta_0' : 0.3, 'sf' : scaling_factors_filtered, 'rho' : 0.10} 

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

# initialize the model 
# Initialize model
model = SEIR2(initial_states=initial_states, parameters=params)



# I still need to check whether this works with the date-type indexes!! 

#############################################################
# Setting up the posterior probability
#############################################################


    #####################
    ## PSO/Nelder-Mead ##
    #####################

if __name__ == '__main__': # makes sure this is only run when the script is run directly (not when a function is imported into another file)

    alpha = 0.885382
    # Define dataset
    counts_filtered.index.name = 'date'
    d = pd.Series(counts_filtered)
    data=[d, ]

    # Print the data type of the index
    # print(f"Index data type: {counts[start_date:end_date].index.dtype}")
    
    # The issues was with the sf in the params variable
    # params['sf'] = params['sf'].values.flatten()
    states = ["I_rep",]
    # select likelihood function and arguments based on results from variance_analysis function
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha,]
    # Calibated parameters and bounds
    pars = ['alpha', 'beta_0', 'sigma','rho'] #TCI, baseline infectioun rate, IIP, and fraction of reported cases 
    labels = ['$alpha$' ,'$beta_0$', '$sigma$', '$rho$']
    bounds = [(0, 720), (0.1, 0.4), (4.00, 7.00), (0, 1)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    model = SEIR2
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,labels=labels)


# I STOPPED HERE # with the error "TypeError: list indices must be integers or slices, not str" -> doesn't matter whether the sf indices are integers or DateTime or whether I flatten it, the error remains!

print(model.parameters)

parameters_model = model.parameters
for param_name in parameters_model:
    if isinstance(parameters_model[param_name], bool):
        print(f"Problem with parameter: {param_name}")