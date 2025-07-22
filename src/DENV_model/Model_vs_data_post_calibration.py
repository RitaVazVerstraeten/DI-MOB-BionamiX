# Script to visualize calibration results

# import dependencies
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import os 
import re
import matplotlib.pyplot as plt

# import model and dependencies
from DENV_models_pySODM import JumpProcess_SEIR2_beta_by_Temp_sf_BirthDeath_reported_v2 as SEIR2_v2
from Scaling_functions_beta_t import generate_scaling_factors
from time_dep_parameters import time_dependent_beta
from Utilities import convert_params_to_weeks

# simulation settings
tau = 1
time_unit = "W"
n = 50
scaling_factor_name = "seasonal_forcing" # options are ["seasonal_forcing", "mordecai_aeg", "lambrechts", "perkins"]

mcmc_file = "/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/SSH/sampler_output_SF/SeasonalForcing_rverstra_SAMPLES_2025-02-28.nc"
path_figure = "/home/rita/PyProjects/DI-MOB-BionamiX/src/DENV_model/optimization_SF"

# Extract filename from path
filename = os.path.basename(mcmc_file)

# Regular expression pattern to match YYYY-MM-DD
date_pattern = r"\d{4}-\d{2}-\d{2}"

# Search for the date in the filename
match = re.search(date_pattern, filename)

if match:
    run_date = match.group(0)  # Extracted date as a string
    print(f"Extracted run date: {run_date}")
else:
    print("No date found in filename.")

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

def sim_with_draws(scaling_factor_name, scaling_factors_filtered, counts_filtered, params_to_draw, mcmc_file, params, tau, time_unit, n = 1): 
    """
    This function will simulate the SEIR2_v2 model with parameters drawn from the MCMC results. 

    Input
    -----
    Scaling_factor_name: a string containing the scaling factor you which to use for the model simulation. The options are "seasonal_forcing", "lambrechts", "mordecai", and "perkins".

    params_to_draw: a list of strings containing the parameters you wish to draw from the mcmc results. Example: ["alpha", "beta_0", "sigma", "gamma", "rho"] 

    mcmc_folder: the path to the file with mcmc results. Make sure this folder refers to the same model as the scaling_factor_name does!

    Model parameters:
    -----------------
    params: dictionary containing all model parameters. For which the params_to_draw are included and their value is an initial guess for model initialization.

    tau: time-step of simulation. 

    time_unit: unit of time, can be days 'D', weeks 'W', months 'M'. Keep in mind that this must be the same as what was done for the calibration process!

    n: number of repeat simulations. Default = 1

    Output
    ------
    xarray containing model simulations
    """

    # set the start and end-date to correct format: 
    #.strftime("%Y-%m-%d") 
    # extract the mcmc results from the folder path
    mcmc_xarray = xr.open_dataset(mcmc_file)
    start_date, end_date = mcmc_xarray.attrs.get("start_calibration"), mcmc_xarray.attrs.get("end_calibration")
    start_date, end_date = datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")

    # extract the parameter values
    parameters_to_draw_from = {}
    for param in params_to_draw:
        # extract the posterior samples for that parameter:
        param_samples = mcmc_xarray[param].values # shape: (iterations, chains)
        # flatten the samples over iterations and chains
        flattened_samples = param_samples.flatten()    
        # store the mcmc results for that parameter in params_to_draw_from
        parameters_to_draw_from[param] = flattened_samples

    # Set the scaling_factor in params to the scaling factor defined above:
    params["sf"] = scaling_factors_filtered[scaling_factor_name].to_frame()

    if "rho" in parameters_to_draw_from:
        # select a random rho from the posterior
        rho = np.mean(parameters_to_draw_from["rho"])
    else:
        rho = params["rho"]

    #########################
    # set initial conditions
    #########################
    initial_infected = round((1/rho)*counts_filtered[start_date], 0)

    # Equal probabilities for 4 infectious compartments
    probabilities = [0.25, 0.25, 0.25, 0.25]

    # Stochastic division using multinomial
    [init_I1, init_I2, init_I12, init_I21] = np.random.multinomial(initial_infected, probabilities)
    
    demo = pd.read_excel("/home/rita/PyProjects/DI-MOB-BionamiX/data/WP1_ Demographic_Data/Population counts and proportions Municipio Cfgos_v17092024.xlsx")

    # extract the number of people during the year of start_date
    initN = demo.loc[demo['Year'] == start_date.year, 'Population_Muni'].item()
    initS = initN - (init_I1 + init_I2 + init_I12 + init_I21)

    initial_states = {
        'S': np.array([initS]),
        'I1': np.array([init_I1]),
        'I2': np.array([init_I2]),
        'I12': np.array([init_I12]),
        'I21': np.array([init_I21])
    }

    ###################
    # initialize model 
    ###################
    time_dep_beta = time_dependent_beta(sf=params["sf"])
    model = SEIR2_v2(initial_states=initial_states, parameters=params,
        time_dependent_parameters={"beta_0": time_dep_beta})
    
    ###################
    # Simulate the model with parameters drawn from MCMC
    ###################
    # simulate the model + draw_fnc
    draw_fnc = create_draw_fnc(posterior_samples = parameters_to_draw_from)    

    print(f"Starting simulation for {scaling_factor_name} with {n} repeats")

    out = model.sim(time=[start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")], N =n , draw_function=draw_fnc, tau=tau, time_unit=time_unit)

    return out

def plot_simulations_vs_real_data(results, var_name, real_data, title, path_figure, scaling_factor_name):
    """
    Plots individual model simulations and compares them with real data.

    Parameters:
    - results (xarray.Dataset): Dataset containing model results.
    - var_name (str): The variable (e.g., "I_rep") to plot from the model results.
    - real_data (pandas.Series): Real data with dates as index.
    - title (str): Title of the plot.
    - scaling_factor_name (str): to use in the name to store the plot

    """
    # Extract the dates and the number of draws
    dates = results["date"].values  # Convert xarray DataArray to NumPy array
    num_draws = results[var_name].shape[0]  # Number of draws

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual trajectories for the selected variable (e.g., I_rep)
    for i in range(num_draws):
        ax.plot(dates, results[var_name][i], color="red", alpha=0.1)

    # Overlay mean trend for the selected variable
    mean_var = np.mean(results[var_name], axis=0)
    ax.plot(dates, mean_var, color="red", linewidth=2, label=f"Mean {var_name}")

    # Overlay real data (counts_filtered)
    common_dates = real_data.index.intersection(dates)  # Ensure matching dates
    ax.scatter(common_dates, real_data.loc[common_dates], 
            color="black", label="Real Data", zorder=3, marker="o", s=20)

    # Add labels, legend, and title
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{var_name} (Simulated vs Real Data)")
    ax.set_title(title)
    ax.legend()

    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()  # Ensure everything fits nicely
    
    # Save the plot in the same directory as the mcmc_file
    save_path = os.path.join(path_figure, f"Simulated_vs_counts_for_{scaling_factor_name}_{run_date}.pdf")

    plt.savefig(save_path)
    print("saving as", save_path)

    plt.show()
    # plt.close()

if __name__ == "__main__":

    #########################################
    # LOAD THE CALIBRATION RESULTS AS XARRAY
    #########################################
    ds = xr.open_dataset(mcmc_file)

    ################################
    # Load the epidemiological data
    ################################
    start_date = pd.to_datetime(ds.attrs["start_calibration"])
    end_date = pd.to_datetime(ds.attrs["end_calibration"])

    ## Reading the dataframe
    epi_data = pd.read_excel("/home/rita/PyProjects/DI-MOB-BionamiX/data/WP1_Epidemiological_Data/DENV/DENV_12-23_cleaned_Area_CP_Mz_IDCODE_v18112024_forsharing.xlsx")
    # Create a counts dataframe per date
    counts = epi_data['XDate_Positivity'].value_counts()
    # rename XDate_Positivity to date
    counts.rename_axis('date', inplace=True)
    # sort by date: 
    counts = counts.sort_index()
    # create weekly counts
    counts_weekly = counts.resample('W-SAT').sum()
    # select only data between start - end calibration
    counts_weekly = counts_weekly[start_date:end_date]

    ##########################
    ## Define scaling factors
    ##########################

    # LOAD THE METEO DATA FOR SCALING FACTORS
    file_path = "/home/rita/PyProjects/DI-MOB-BionamiX/data/WP2_Meteorological_Data"
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
        scaling_methods= ["seasonal_forcing", "mordecai_aeg", "lambrechts", "perkins"]
    )

    for key, df in scaling_factors.items():
        df.index = meteo.index

    # ##############################################
    # Get overlap between dates sf and epidemiology
    # ##############################################

    demo = pd.read_excel("/home/rita/PyProjects/DI-MOB-BionamiX/data/WP1_ Demographic_Data/Population counts and proportions Municipio Cfgos_v17092024.xlsx")

    initN = np.rint(demo["Population_Muni"][0])

    # Sort each DataFrame in scaling_factors_filtered by date and reset index
    scaling_factors_filtered = {
        key: df.sort_index()[df.index >= start_date]
        for key, df in scaling_factors.items()
    }

    # Resample from Saturday to Saturday and take the mean
    scaling_factors_weekly = {
        model: df['sf'].resample('W-SAT').mean()  
        for model, df in scaling_factors_filtered.items()
    }

    # Find the intersection of indices
    common_dates = counts_weekly.index.intersection(scaling_factors_weekly['seasonal_forcing'].index)
    counts_filtered = counts_weekly.loc[common_dates]
    # Filter each DataFrame in scaling_factors to only common dates
    scaling_factors_filtered = {key: df.loc[common_dates] for key, df in scaling_factors_weekly.items()}

    #################################
    # INITIATE AND SIMULATE THE MODEL 
    #################################

    # it doesn't matter what scaling_factor you plug-in at the moment, the function will correct it 
    params={'alpha':182.5, 'b':2.77e-05, 'd':2.45e-05, 'sigma':6, 'gamma':7, 'psi': 1.5, 'beta_0' : 0.3, 'sf' : scaling_factors_filtered['seasonal_forcing'], 'rho' : 0.10} 

    # convert daily rates into weekly rates
    if time_unit == "W": 
        params_weekly = convert_params_to_weeks(params)
        params = params_weekly


    out = sim_with_draws(scaling_factor_name= scaling_factor_name,
                         scaling_factors_filtered = scaling_factors_filtered,
                         counts_filtered = counts_filtered, 
                            params_to_draw= ["alpha", "beta_0", "sigma", "gamma", "rho"], 
                            mcmc_file = mcmc_file,
                            params = params, tau = tau, time_unit=time_unit, n=n)

    ##############################
    # plot the results
    ############################

    plot_simulations_vs_real_data(out, var_name = "I_rep", real_data= counts_filtered, title = f"Model simulation vs counts for {scaling_factor_name} {run_date}", scaling_factor_name=scaling_factor_name, path_figure= path_figure)