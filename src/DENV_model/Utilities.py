# utility functions

import json
import os
from datetime import datetime
import numpy as np

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