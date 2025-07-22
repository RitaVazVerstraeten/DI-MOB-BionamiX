# Scaling factors s(T,P) for beta(t) based on A. Andronico et al. Journal of Infectious Diseases (2024)

import math
import numpy as np
import pandas as pd

def briere(temperature, c, t_min, t_max):
    res = 0.0
    if t_min < temperature < t_max:
        res = c * temperature * (temperature - t_min) * math.sqrt(t_max - temperature)
    return res

def quadratic(temperature, c, t_min, t_max):
    res = 0.0
    if t_min < temperature < t_max:
        res = c * (temperature - t_min) * (t_max - temperature)
    return res

def lambrechts_scaling(temperature):
    # Taken from: L. Lambrechts et al, PNAS 108 (2011)
    res = 0.0
    if 12.286 <= temperature <= 32.461:
        res = 0.001044 * temperature * (temperature - 12.286) * math.sqrt(32.461 - temperature)
    return res ** 2

def perkins_scaling(temperature, rainfall):
    # Taken from: T. A. Perkins et al, PLOS Currents Outbreaks (2015)
    log_res = (
        -25.66 + 2.121 * temperature + 1.188e-2 * rainfall
        - 4.231e-2 * temperature ** 2 - 2.882e-5 * rainfall ** 2
    )
    return math.exp(log_res)

def perkins_scaling_norm(temperature, rainfall):
    # Adapted from: T. A. Perkins et al, PLOS Currents Outbreaks (2015)
    # if temperature or rainfall is missing: 
    if math.isnan(temperature) or math.isnan(rainfall):
        return 0.0
    
    log_res = (
        -25.66 + 2.121 * temperature + 1.188e-2 * rainfall
        - 4.231e-2 * temperature ** 2 - 2.882e-5 * rainfall ** 2
    )
    # Maximum Value: 8.5481 at Temperature=25.07, Rainfall=206.11
    # Minimum Value: 0.0000 at Temperature=7.51, Rainfall=0.00
    raw_value = math.exp(log_res)
    # the normalization is in fact (raw_value - min_value) / (max_value - min_value), but the min_value is simply = 0 
    return raw_value/8.5481 # So that max scaling = 1 (for T = 25.07°C and precipitation = 206.11 mm)


def mordecai_scaling_albopictus(temperature):
    # Taken from: E. A. Mordecai et al, PLoS Negl Trop Dis 11 (2017)
    # Parameters are those for Aedes albopictus for DENV
    a = briere(temperature, 1.93e-4, 10.25, 38.32) # per-mosquito biting rate
    efd = briere(temperature, 4.88e-2, 8.02, 35.65) # Eggs per Female per Day
    pea = quadratic(temperature, 3.61e-3, 9.04, 39.33) # mosquito egg-to-adult survival probability
    mdr = briere(temperature, 6.38e-5, 8.60, 39.66) # mosquito immature development rate
    lf = quadratic(temperature, 0.15, 9.16, 37.73) # mosquito lifespan + DENV 
    b = briere(temperature, 7.35e-4, 15.84, 36.40) # proportion of infectious bites infecting S humans (mosquito - to - human)
    c = briere(temperature, 4.39e-4, 3.62, 36.82) # proportion of bites on I humans infecting S mosquitoes (human - to - mosquito)
    pdr = briere(temperature, 1.09e-4, 10.39, 43.05) # parasite development rate → 1/EIP

    res = a * a * b * c
    if lf > 0.0 and pdr > 0.0:
        res *= math.exp(-1.0 / (lf * pdr))
    res *= efd * pea * mdr * lf ** 3
    return math.sqrt(res) / 86.5331  # So that max scaling = 1 (at T = 28.40) --> this is specific for DENV + Aedes albopictus 


def mordecai_scaling_aegypti(temperature):
    # Taken from: E. A. Mordecai et al, PLoS Negl Trop Dis 11 (2017)
    # Parameters are those for Aedes aegypti for DENV
    a = briere(temperature, 2.02e-4, 13.35, 40.08) # per-mosquito biting rate
    efd = briere(temperature, 8.66e-3, 14.58, 34.61) # Eggs per Female per Day
    pea = quadratic(temperature, 5.99e-3, 13.56, 38.29) # mosquito egg-to-adult survival probability
    mdr = briere(temperature, 7.86e-5, 11.36, 39.17) # mosquito immature development rate
    lf = quadratic(temperature, 0.15, 9.16, 37.73) # mosquito lifespan + DENV
    b = briere(temperature, 8.49e-4, 17.05, 35.83) # proportion of infectious bites infecting S humans (mosquito - to - human)
    c = briere(temperature, 4.91e-4, 12.22, 37.46) # proportion of bites on I humans infecting S mosquitoes (human - to - mosquito)
    pdr = briere(temperature, 6.65e-5, 10.68, 45.90) # parasite development rate → 1/EIP

    res = a * a * b * c
    if lf > 0.0 and pdr > 0.0:
        res *= math.exp(-1.0 / (lf * pdr))
    res *= efd * pea * mdr * lf ** 3

    return math.sqrt(res) / 25.1551  # So that max scaling = 1 (for T = 29.04) --> this is specific for DENV + Aedes aegypti 




# Standard seasonal forcing equation
def seasonal_forcing_scaling(time, params, is_week=False):
    """
    Compute the seasonal forcing scaling.
    
    Parameters:
    - params: dictionary containing the following parameters
        - base: estimation of average scaling factor at baseline (when no seasonal forcing is present)
        - beta_1: Amplitude of the sinusoid.
        - omega: Frequency of the sinusoid (should be close to 2π for annual cycles).
        - ph: Phase shift in days.
    - time: Scalar or array-like time variable (days or weeks).
    - week (array-like): Week of the year (1 to 52).
    - is_week (bool): If True, treats `x` as weeks (1 to 52). Otherwise, treats `x` as days (1 to 365).

    Returns:
    - The scaling factor for beta(t) in seasonal forcing. 
    """
    if is_week: 
        period = 52
    else: 
        period = 365
    # period = 52 if is_week else 365
    # print("period = ", period)
    return params["base"] + params["beta_1"] * np.sin((params["omega"] * (time / period)) + (2 * np.pi * params["ph"] / period))

# Standard seasonal forcing equation - fitted to Mordecai_aegypti scaling factor series 
def seasonal_forcing_fitted(time, is_week=False):
    """
    Compute the seasonal forcing scaling.
    
    Parameters:
    - params that are now fixed values for WEEKLY data: 
        - base: estimation of average scaling factor at baseline (when no seasonal forcing is present) = 0.628149
        - beta_1: Amplitude of the sinusoid. = -0.308845
        - omega: Frequency of the sinusoid (should be close to 2π for annual cycles). = 6.28313
        - ph: Phase shift in days. = 9.45066

    - params that are now fixed values for DAILY data: 
        - base: = 0.596389
        - beta_1 = -0.309048
        - omega = 5.63878
        - ph = 91.4211

    - time: Scalar or array-like time variable (days or weeks). 
    - is_week (bool): If True, treats `x` as weeks (1 to 52). Otherwise, treats `x` as days (1 to 365).

    Returns:
    - The scaling factor for beta(t) in seasonal forcing. 
    """
    if is_week: 
        period = 52
        # print("is_week = True")
        # res = params["base"] + params["beta_1"] * np.sin((params["omega"] * (time / period)) + (2 * np.pi * params["ph"] / period))
        res = 0.628149 - 0.308845 * np.sin((6.28313 * (time / period)) + (2 * np.pi * 9.45066/ period))
    else: 
        # print("is_week = False", "time = ", time, str(time))
        time = time.dayofyear
        period = 365
        res = 0.596389 - 0.309048 * np.sin((5.63878 * (time / period)) + (2 * np.pi * 91.4211/ period))
    return res

# Function to extract the beta(t) parameter as beta_0 * s(T,P) 
def get_beta(curr_time, temperature_series, rainfall_series, transmission_model, params, is_week): 
    """
    Compute the beta(t) as beta_0 * s(T,P).
    
    Parameters:
    - curr_time: timepoint, either a date or a week number (if is_week = True)
    - temperature_series: series of temperature values with time (dates or weeks) as index
    - rainfall_series: series of rainfall values with time (dates or weeks) as index
    - transmission_model: selection of model for scaling (lambrechts, perkins, mordecai, perk_norm, seasonal_forcing)
    - is_week (bool): If True, treats curr_time as weeks (1 to 52*years). Otherwise, treats curr_time as days (1 to 365*years).

    Returns:
    - The scaled transmission factor (beta(t)) value.
    """
    # get the current temperature and rainfall
    temp = temperature_series(curr_time)
    rain = rainfall_series(curr_time)

    # determine what the scaling factor will be 
    if transmission_model == "lambrechts":
        scaling = lambrechts_scaling(temp)
    elif transmission_model == "perkins":
        scaling = perkins_scaling(temp, rain)
    elif transmission_model == "mordecai":
        scaling = mordecai_scaling(temp)
    elif transmission_model == "perk_norm":
        scaling = perkins_scaling_norm(temp, rain)
    elif transmission_model == "seasonal_forcing": 
        scaling = seasonal_forcing_scaling(curr_time, params=params, is_week = is_week)
    else:
        scaling = 1.0
    
    # calculate beta(t)
    return params["beta_0"] * scaling



def generate_scaling_factors(temperature_series, rainfall_series, dates, scaling_methods=None):
    """
    Generate scaling factors based on temperature, rainfall, and dates.
    
    Parameters:
        temperature_series (pd.Series): Time series of temperatures.
        rainfall_series (pd.Series): Time series of rainfall values.
        dates (pd.DatetimeIndex): Date indices for the time series.
        scaling_methods (list): List of scaling methods to use. Options:
                                ['mordecai_aeg', 'mordecai_albo', 'lambrechts', 'perkins', 'seasonal_forcing'].
                                If None, all methods are calculated.
                                
    Returns:
        dict: Dictionary of scaling factors, with each scaling factor formatted as a DataFrame
              with a single column named 'sf'.
    """
    available_methods = {
        'mordecai_aeg': lambda: temperature_series.apply(mordecai_scaling_aegypti),
        'mordecai_albo': lambda: temperature_series.apply(mordecai_scaling_albopictus),
        'lambrechts': lambda: temperature_series.apply(lambrechts_scaling),
        'perkins': lambda: pd.Series(
            [perkins_scaling_norm(temp, rain) for temp, rain in zip(temperature_series, rainfall_series)],
            index=temperature_series.index
        ),
        'seasonal_forcing': lambda: pd.Series(
            [seasonal_forcing_fitted(date, is_week=False) for date in dates],
            index=dates
        )
    }
    
    # get the time variable from the meteo data 
    time = temperature_series.index

    # Validate methods and calculate scaling factors
    scaling_methods = scaling_methods or available_methods.keys()
    invalid_methods = set(scaling_methods) - available_methods.keys()
    if invalid_methods:
        raise ValueError(f"Invalid scaling methods: {invalid_methods}. Available methods are: {list(available_methods.keys())}")
    
    # Calculate and format scaling factors
    scaling_factors = {
        method: available_methods[method]().to_frame(name="sf").reset_index(drop=True)
        for method in scaling_methods
    }
    
    return scaling_factors, time

