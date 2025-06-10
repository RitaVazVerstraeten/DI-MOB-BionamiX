# utility functions

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

from Scaling_functions_beta_t import generate_scaling_factors

def pso_to_dictionary(samples_path, identifier, run_date, g, fg, history=None):
    """
    Store the results of Particle Swarm Optimization (PSO) in a dictionary format.

    Parameters:
        samples_path (str): Path where results will be stored.
        identifier (str): A unique name for the results.
        g (array): The swarm's best known position (global best).
        fg (scalar): The objective function value at g.
        history (list, optional): A list of best values over iterations.
        run_date (str): The date of the run (default: today's date).

    Returns:
        dict: A dictionary containing the PSO results.
    """
    # Convert NumPy arrays to lists for JSON serialization
    samples_dict = {
        "global_best_position": g.tolist() if isinstance(g, np.ndarray) else g,
        "global_best_value": fg,
        "history": history if history is not None else [],
        "run_date": run_date,
    }

    # Save dictionary to JSON
    filename = f"{identifier}_PSO_{run_date}.json"
    filepath = os.path.join(samples_path, filename)

    with open(filepath, "w") as file:
        json.dump(samples_dict, file)

    print(f"PSO results saved at: {filepath}")
    return samples_dict

# to store the nelder_mead_output 
def nelder_mead_to_dictionary(samples_path, identifier, run_date, theta, ftheta, history=None):
    """
    Store the results of Nelder-Mead optimization in a dictionary format.

    Parameters:
        samples_path (str): Path where results will be stored.
        identifier (str): A unique name for the results.
        theta (list): The estimated parameters from the optimization.
        ftheta (scalar): The objective function value at theta.
        history (list, optional): A list of best values over iterations.
        run_date (str): The date of the run (default: today's date).

    Returns:
        dict: A dictionary containing the Nelder-Mead optimization results.
    """
    # Convert results to a dictionary
    results_dict = {
        "estimated_parameters": theta.tolist() if isinstance(theta, np.ndarray) else theta,
        "objective_value": ftheta,
        "history": history if history is not None else [],
        "run_date": run_date,
    }

    # Save dictionary to JSON
    filename = f"{identifier}_NelderMead_{run_date}.json"
    filepath = os.path.join(samples_path, filename)

    with open(filepath, "w") as file:
        json.dump(results_dict, file)

    print(f"Nelder-Mead results saved at: {filepath}")
    return results_dict


# function to convert params from days to weeks
def convert_params_to_weeks(params):
    """
    Convert SEIR2 model parameters alpha, birthrate, deathrate, beta_0, incubation period and infectious period from days to weeks.
    
    Parameters:
        params (dict): Dictionary containing SEIR parameters with time in days.
        Can contain any parameter, but only the rate-related parameters "alpha", "b", "d", "beta_0", "sigma", and "gamma" will be altered.  
    
    Returns:
        dict: Dictionary with parameters adjusted to weeks.
    """
    params_week = params.copy()

    # convert time-dependent parameters: 
    if "alpha" in params:
        params_week["alpha"] /= 7 # TCI period (days -> weeks)
    if "b" in params:
        params_week["b"] *= 7 # birth rate (per day -> per week)
    if "d" in params:
        params_week["d"] *= 7 # death rate (per day -> per week)
    if "beta_0" in params:
        params_week["beta_0"] *= 7 # Transmission rate (per day -> per week)
    if "sigma" in params:
        params_week["sigma"] /= 7  # Incubation period (days -> weeks)
    if "gamma" in params:
        params_week["gamma"] /= 7  # Infectious period (days -> weeks)
    return params_week

def load_epi_and_scaling_factors(start_date, end_date, time_unit):
    """
    Function to load the epi count data and create the scaling factors for defined start- and end-dates

    input
    -----
    start_date and end_date (string or pd.datetime): dates between which you would like extract count data
    time_unit (string): days, weeks, months (examples: "D", "W"). The default is "D"

    output
    ------
    counts_filtered: pd.Series containing counts by unit of time (days, weeks, months) with date as index. 
    scaling_factors_filtered: dictionary containing scaling factors for each scaling_factor model (lambrecht, mordecai, perkins, seasonal_forcing), per unit of time, with dates as index. 

    """
    # Function to lead epi data and scaling factors for pre-defined start- and end-date and time_unit
    ## Reading the dataframe
    epi_data = pd.read_excel("/home/rita/PyProjects/DI-MOB-BionamiX/data/WP1_Epidemiological_Data/DENV/DENV_12-23_cleaned_Area_CP_Mz_IDCODE_v18112024_forsharing.xlsx")
    # Create a counts dataframe per date
    counts = epi_data['XDate_Positivity'].value_counts()
    # rename XDate_Positivity to date
    counts.rename_axis('date', inplace=True)
    # sort by date: 
    counts = counts.sort_index()

    if time_unit == "W":
        # create weekly counts
        counts_weekly = counts.resample('W-SAT').sum()
        counts = counts_weekly

    # select only data between start - end calibration
    counts_filtered = counts[start_date:end_date]

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

    # Sort each DataFrame in scaling_factors_filtered by date and reset index
    scaling_factors_filtered = {
        key: df.sort_index()[df.index >= start_date]
        for key, df in scaling_factors.items()
    }

    if time_unit == "W":
        # Resample from Saturday to Saturday and take the mean
        scaling_factors_weekly = {
            model: df['sf'].resample('W-SAT').mean()  
            for model, df in scaling_factors_filtered.items()
        }
        scaling_factors_filtered = scaling_factors_weekly

    # Find the intersection of indices
    common_dates = counts_filtered.index.intersection(scaling_factors_filtered['seasonal_forcing'].index)

    counts_filtered = counts_filtered.loc[common_dates]
    # Filter each DataFrame in scaling_factors to only common dates
    scaling_factors_filtered = {key: df.loc[common_dates] for key, df in scaling_factors_filtered.items()}

    return counts_filtered, scaling_factors_filtered