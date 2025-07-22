import numpy as np
import pandas as pd
import xarray as xr
import os 
import importlib
from datetime import datetime
import matplotlib.pyplot as plt

from DENV_models_pySODM import JumpProcess_SEIR2_beta_by_Temp_sf_BirthDeath_reported_v2 as SEIR2_v2
from Scaling_functions_beta_t import generate_scaling_factors
from time_dep_parameters import time_dependent_beta
import Utilities
importlib.reload(Utilities)  # Reload to ensure latest changes are applied
from Utilities import load_epi_and_scaling_factors, convert_params_to_weeks, aggregate_meteo_daily_to_weekly


# load the epi data 
###################
start_date = '2012-01-01'
end_date = '2020-12-31'
counts_filtered, scaling_factors_filtered = load_epi_and_scaling_factors(start_date=start_date, end_date=end_date, time_unit= "W")

epi_weekly = counts_filtered.copy()

# load the meteo data
###################
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

meteo_weekly = aggregate_meteo_daily_to_weekly(meteo, group_cols = "stat_week")

# Extract temperature and rainfall series
temperature_series = meteo["temp_med"]
rainfall_series = meteo["precip_ponderada"]

# Handle missing values
if temperature_series.isna().any() or rainfall_series.isna().any():
    #print("Warning: Missing values detected. Filling with interpolation.")
    temperature_series = temperature_series.interpolate()
    rainfall_series = rainfall_series.interpolate()

print(f"Minimal temperature in Cienfuegos: {meteo['temp_min'].min()}")
print(f"Maximal temperature in Cienfuegos: {meteo['temp_max'].max()}")




def plot_time_dep_beta_per_sf(scaling_factors, params, scaling_factor_names, start_date, end_date):
    """
    Plot the time-dependent beta(t) for each scaling factor.
    """
    plt.figure(figsize=(14, 5))
    for name in scaling_factor_names:
        # Use the correct scaling factor
        sf = scaling_factors[name]
        # Use the beta_0 from params
        beta_0 = params["beta_0"] if isinstance(params["beta_0"], float) else params["beta_0"][0]
        # Compute beta(t)
        beta_t = beta_0 * sf
        plt.plot(sf.index, beta_t, label=name)
    plt.xlabel("Date")
    plt.ylabel("beta(t)")
    plt.title("Time-dependent beta(t) per scaling factor for Cienfuegos")
    plt.legend()
    plt.tight_layout()
    plt.show()

####################################

params={'alpha':182.5, 'b':2.77e-05, 'd':2.45e-05, 'sigma':6, 'gamma':7, 'psi': 1.5, 'beta_0' : 0.3, 'sf' : scaling_factors_filtered['seasonal_forcing'], 'rho' : 0.10} 

# Plot the time-dependent beta for each scaling factor
plot_time_dep_beta_per_sf(scaling_factors_filtered, params, scaling_factors_filtered.keys(), start_date, end_date)


##############################################
# plot meteo variables and scaling factors
##############################################


def plot_meteo_scaling_epi(meteo_weekly, scaling_factors, epi_weekly, save_path=None):
    """
    Plots temperature, precipitation, all scaling factors (overlaid), and DENV_total for a given UF.
    """
    meteo_weekly['date'] = pd.to_datetime(meteo_weekly['date'], errors='coerce')
    epi_weekly['date'] = pd.to_datetime(epi_weekly['date'], errors='coerce')
    scaling_names = list(scaling_factors.keys())

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    # 1. Temperature
    axes[0].plot(meteo_weekly['date'], meteo_weekly['temp_med'], color='tab:orange')
    axes[0].set_ylabel('Temperature (°C)')

    # 2. Precipitation
    axes[1].bar(meteo_weekly['date'], meteo_weekly['precip_ponderada'], color='tab:blue', width = 7)
    axes[1].set_ylabel('Precipitation (mm)')

    # 3. Overlay of all scaling factors
    exlude = ['seasonal_forcing']
    for name in scaling_names:
        if name not in exlude:
            sf = scaling_factors[name]
            axes[2].plot(sf.index, sf.values, label=name)
    axes[2].set_ylabel('Scaling Factor')
    axes[2].legend()

    # 4. DENV_total
    axes[3].plot(epi_weekly.index, epi_weekly, color='tab:red')
    axes[3].set_ylabel('DENV_total')
    axes[3].set_xlabel('Date')

    # Add vertical lines at the start of each year on all plots
    years = epi_weekly.index.dt.year.dropna().unique()
    for ax in axes:
        for year in years:
            first_of_year = epi_weekly[epi_weekly.index.dt.year == year].index.min()
            ax.axvline(first_of_year, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi = 300)
    plt.show()

# Usage:
plot_meteo_scaling_epi(
    meteo_weekly,
    scaling_factors_filtered,
    epi_weekly,
)

# Additional plotting functions

def plot_cuban_scaling_factors(scaling_factors, start_date=None, end_date=None, save_path=None):
    """
    Simple function to plot just the scaling factors for Cuban data.
    
    Parameters:
    - scaling_factors: Dictionary of scaling factors
    - start_date, end_date: Optional date range
    - save_path: Optional path to save the figure
    """
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(12, 6))
    
    # Color palette for different scaling factors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (name, sf_data) in enumerate(scaling_factors.items()):
        color = colors[i % len(colors)]
        
        # Handle both DataFrame and Series
        if isinstance(sf_data, pd.DataFrame) and 'sf' in sf_data.columns:
            data_to_plot = sf_data['sf']
            index_to_plot = sf_data.index
        elif isinstance(sf_data, pd.Series):
            data_to_plot = sf_data
            index_to_plot = sf_data.index
        else:
            continue
            
        # Filter by date range if specified
        if start_date and end_date:
            mask = (index_to_plot >= start_date) & (index_to_plot <= end_date)
            data_to_plot = data_to_plot[mask]
            index_to_plot = index_to_plot[mask]
        
        plt.plot(index_to_plot, data_to_plot, linewidth=2.5, label=name.replace('_', ' ').title(), color=color)
    
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Scaling Factor', fontsize=12, fontweight='bold')
    plt.title('Temperature-Dependent Scaling Factors for Dengue Transmission in Cuba', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator([1, 7]))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return plt.gcf()


def plot_comprehensive_meteo_scaling_epi(meteo_data, scaling_factors, epi_data, start_date=None, end_date=None, figsize=(16, 12)):
    """
    Plot meteorological data, scaling factors, and epidemiological data in a comprehensive view.
    
    Parameters:
    - meteo_data: DataFrame with meteorological data (temp_med, precip_ponderada, etc.)
    - scaling_factors: Dictionary of scaling factors with DataFrames containing 'sf' column
    - epi_data: Series with epidemiological case counts
    - start_date, end_date: Optional date range for filtering
    - figsize: Figure size tuple
    """
    import matplotlib.dates as mdates
    
    # Filter data by date range if specified
    if start_date or end_date:
        if hasattr(meteo_data, 'index') and hasattr(meteo_data.index, 'slice_locs'):
            meteo_filtered = meteo_data.loc[start_date:end_date] if start_date and end_date else meteo_data
        else:
            meteo_filtered = meteo_data
        epi_filtered = epi_data.loc[start_date:end_date] if start_date and end_date else epi_data
        scaling_filtered = {name: sf.loc[start_date:end_date] if start_date and end_date else sf 
                          for name, sf in scaling_factors.items()}
    else:
        meteo_filtered = meteo_data
        epi_filtered = epi_data
        scaling_filtered = scaling_factors
    
    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle('Meteorological Data, Scaling Factors, and Epidemiological Cases', fontsize=16, fontweight='bold')
    
    # 1. Temperature plot
    ax1 = axes[0]
    if 'temp_med' in meteo_filtered.columns:
        ax1.plot(meteo_filtered.index, meteo_filtered['temp_med'], 'r-', linewidth=1.5, label='Mean Temperature')
    if 'temp_min' in meteo_filtered.columns and 'temp_max' in meteo_filtered.columns:
        ax1.fill_between(meteo_filtered.index, meteo_filtered['temp_min'], meteo_filtered['temp_max'], 
                        alpha=0.3, color='red', label='Min-Max Range')
    ax1.set_ylabel('Temperature (°C)', fontweight='bold')
    ax1.set_title('Temperature Over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Precipitation plot
    ax2 = axes[1]
    if 'precip_ponderada' in meteo_filtered.columns:
        ax2.bar(meteo_filtered.index, meteo_filtered['precip_ponderada'], 
               alpha=0.7, color='blue', width=7 if len(meteo_filtered) < 200 else 1)
    ax2.set_ylabel('Precipitation (mm)', fontweight='bold')
    ax2.set_title('Precipitation Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Scaling factors plot
    ax3 = axes[2]
    colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, (name, sf_data) in enumerate(scaling_filtered.items()):
        color = colors[i % len(colors)]
        if isinstance(sf_data, pd.DataFrame) and 'sf' in sf_data.columns:
            ax3.plot(sf_data.index, sf_data['sf'], linewidth=2, label=name, color=color)
        elif isinstance(sf_data, pd.Series):
            ax3.plot(sf_data.index, sf_data, linewidth=2, label=name, color=color)
    ax3.set_ylabel('Scaling Factor', fontweight='bold')
    ax3.set_title('Scaling Factors Over Time')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Epidemiological cases plot
    ax4 = axes[3]
    ax4.plot(epi_filtered.index, epi_filtered.values, 'k-', linewidth=2, marker='o', markersize=3, label='Cases')
    ax4.fill_between(epi_filtered.index, 0, epi_filtered.values, alpha=0.3, color='red')
    ax4.set_ylabel('Cases', fontweight='bold')
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.set_title('Epidemiological Cases Over Time')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis dates
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 7]))  # Jan and July
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.xticks(rotation=45)
    
    return fig

##############################################
# Execute the plots
##############################################

# Create the simple scaling factors plot
print("Creating scaling factors plot...")
fig_sf = plot_cuban_scaling_factors(
    scaling_factors=scaling_factors_filtered,
    start_date=start_date,
    end_date=end_date,
    save_path=None
)

# Create comprehensive plot
print("Creating comprehensive plot...")
fig_comprehensive = plot_comprehensive_meteo_scaling_epi(
    meteo_data=meteo, 
    scaling_factors=scaling_factors_filtered, 
    epi_data=epi_weekly,
    start_date=start_date,
    end_date=end_date
)