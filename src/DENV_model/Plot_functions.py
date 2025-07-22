# Compilation of the plotting functions for my epidemic curves
import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results(out_temp, params, variables_to_plot=[('I1', 'I21'), ('I2', 'I12')],
                            colors=['red', 'yellow'], mean_variables=['R', 'S'], mean_colors=['blue', 'green'], add_to_title = '', save_path = None):
    """
    Function to create a 5-subplot figure with specific variables and their means.
    
    Arguments:
    out_temp : xarray.DataArray
        The dataset containing the simulation results.
    params : dict
        A dictionary containing model parameters (e.g., 'beta_0').
    variables_to_plot : list of tuples
        List of pairs of variables (e.g., [('I1', 'I21'), ('I2', 'I12')]).
    colors : list
        Colors to use for each variable pair.
    mean_variables : list
        Variables (like 'R' and 'S') to plot along with the main variables.
    mean_colors : list
        Colors for the mean variables.
    """
    
    # Create a 5x1 subplot layout for the two serotype subplots plus I_new and I_cum
    fig, axs = plt.subplots(5, figsize=(10, 12))
    
    # Helper function to plot variables
    def plot_variable(ax, data, color, label, linestyle='-', plot_draws=True, log=False):
        mean_data = data.mean(dim='draws')
        if log:
            mean_data = np.log(mean_data)
        if plot_draws:
            for draw in data['draws']:
                draw_data = data.sel(draws=draw)
                if log:
                    draw_data = np.log(draw_data)
                ax.plot(data['date'], draw_data, color=color, alpha=0.1, linewidth=0.5)
        ax.plot(data['date'], mean_data, color=color, linestyle=linestyle, linewidth=2, label=label)

    #######################################################
    # Plot the two serotype subplots for I1/I21 and I2/I12
    #######################################################
    for i, (firstI, secondI) in enumerate(variables_to_plot):
        plot_variable(axs[i], out_temp[firstI], colors[i], f'Mean {firstI}', log=False)
        plot_variable(axs[i], out_temp[secondI], colors[i], f'Mean {secondI}', linestyle='--', log=False)
        
        # Plot R and S in the first two subplots
        for j, mean_var in enumerate(mean_variables):
            plot_variable(axs[i], out_temp[mean_var], mean_colors[j], f'Mean {mean_var}', plot_draws=False, log=False)
        
        axs[i].set_title(f'{firstI} and {secondI} with Mean R and S')

    ######################################
    # Third subplot for I_new, R, and S
    ######################################
    plot_variable(axs[2], out_temp["I_new"], 'purple', 'Mean I_new', log=False)
    for j, mean_var in enumerate(mean_variables):
        plot_variable(axs[2], out_temp[mean_var], mean_colors[j], f'Mean {mean_var}', plot_draws=False, log=False)
    axs[2].set_title('Mean I_new with Mean R and S')

    ######################################
    # Fourth subplot for I_cum, R, and S
    ######################################
    plot_variable(axs[3], out_temp["I_cum"], 'red', 'Mean I_cum', log=False)
    for j, mean_var in enumerate(mean_variables):
        plot_variable(axs[3], out_temp[mean_var], mean_colors[j], f'Mean {mean_var}', plot_draws=False, log=False)
    axs[3].set_title('Mean I_cum with Mean R and S')

    ######################################
    # Fifth subplot for I1, I2, R1 and R2
    ######################################
    plot_variable(axs[4], out_temp["I1"], "red", 'Mean I1', log=False)
    plot_variable(axs[4], out_temp["I2"], "yellow", 'Mean I2', log=False)
    plot_variable(axs[4], out_temp["R1"], "lightblue", linestyle='--', label='Mean R1', log=False)
    plot_variable(axs[4], out_temp["R2"], "lightblue", linestyle=':', label='Mean R2', log=False)

    for j, mean_var in enumerate(mean_variables):
        plot_variable(axs[4], out_temp[mean_var], mean_colors[j], f'Mean {mean_var}', plot_draws=False, log=False)
    
    axs[4].set_title('I1, I2, R1, and R2 with Mean R and S')
    axs[4].set_xlabel('Time (days)')
    axs[4].set_ylabel('Population')
    axs[4].legend()

    # Set the title for the entire figure
    fig.suptitle(f'Simulation with $\\beta_0 = {params["beta_0"]}$, {add_to_title}', fontsize=16)

    # Adjust the layout
    for i, ax in enumerate(axs):
        if i < 4: 
            ax.set_ylabel('Population')
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.legend()

            # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Adjust the layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to avoid overlap with suptitle
    plt.show()


################################################################################################################################
# Plot from above, but this time with a wrapper to plot the Mordecai, Lambrechts, and Perkins Scaling Factors for beta(T, P)
################################################################################################################################
def plot_combined_scaling_factors(out_temp_combined, params, scaling_factors=["mordecai", "lambrechts", "perkins"], save_path = None):
    """
    Wrapper function to overlay results from different scaling factors on the same plots 
    using different line styles.
    
    Arguments:
    out_temp_combined : xarray.Dataset
        Dataset with a `scaling_factor` dimension containing the simulation results.
    params : dict
        Model parameters for annotation in the plots.
    scaling_factors : list
        List of scaling factor labels to iterate over (e.g., ["mord", "lamb", "perk"]).
    """
    for sf in scaling_factors:
        data = out_temp_combined.sel(scaling_factor=sf)
        
        # Call the original plotting function but pass the linestyle
        plot_simulation_results(
            out_temp=data,
            params=params,
            variables_to_plot=[('I1', 'I21'), ('I2', 'I12')],
            colors=['red', 'yellow'],  # Use consistent colors
            mean_variables=['R', 'S'],
            mean_colors=["blue", "green"],  # Optional: customize mean variable colors
            add_to_title= f"and Scaling Factor: {sf}",
            save_path = save_path
        )
        
        # # Set the title for the entire figure
        # fig.suptitle(f'Simulation with $\\beta_0 = {params["beta_0"]}$ and Scaling Factor: {sf}', fontsize=16)

    
    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()

################################################################################################################################
# Plot the Mordecai, Lambrechts, and Perkins Scaling Factors for beta(T, P) - I, R, and S
###############################################################################################################################
import matplotlib.lines as mlines


def single_plot_scaling_factors(out_temp_combined, scaling_factors=['mordecai', 'lambrechts', 'perkins'],
                                variables_to_plot=['I_new', 'R', 'S'], line_styles=['-', '--', ':'],
                                params=None, save_path=None, log=False):
    """
    Function to plot selected variables for different scaling factors with individual draws and means.
    
    Arguments:
    out_temp_combined : xarray.Dataset
        The dataset with `scaling_factor`, `date`, and `draws` coordinates.
    scaling_factors : list, optional
        List of scaling factors to iterate over. Default is ['mordecai', 'lambrechts', 'perkins'].
    variables_to_plot : list, optional
        List of variables to plot (e.g., ['I_new', 'R', 'S']). Default is ['I_new', 'R', 'S'].
    line_styles : list, optional
        List of line styles for each scaling factor. Default is ['-', '--', ':'].
    params : dict, optional
        Dictionary of model parameters (e.g., for the title). Default is None.
    save_path : str, optional
        Path to save the plot (if provided). Default is None.
    log : bool, optional
        Whether to apply a logarithmic transformation to the y-axis. Default is False.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 6))  # Single plot for all datasets

    # Loop through each scaling factor and plot it
    for i, sf in enumerate(scaling_factors):
        # Extract data for the current scaling factor
        data = out_temp_combined.sel(scaling_factor=sf)
        
        # Loop through the selected variables
        for var, color, label in zip(variables_to_plot, ['red', 'blue', 'green'], ['I_new', 'Mean R', 'Mean S']):
            # Plot the individual draws for the variable
            for draw in data['draws']:
                draw_data = data.sel(draws=draw)
                if log:
                    draw_data[var] = np.log(draw_data[var])  # Apply log transformation
                ax.plot(draw_data['date'], draw_data[var], color=color, alpha=0.2, linewidth=0.5)
            
            # Plot the mean of the variable
            mean_var = data[var].mean(dim='draws')
            if log:
                mean_var = np.log(mean_var)  # Apply log transformation
            ax.plot(data['date'], mean_var, color=color, linestyle=line_styles[i], linewidth=2)

    # Create a legend for line styles
    style_legend = [mlines.Line2D([], [], color='black', linestyle=ls, linewidth=2, label=sf) 
                    for ls, sf in zip(line_styles, scaling_factors)]
    
    # Create a legend for colors
    color_legend = [mlines.Line2D([], [], color=col, linestyle='-', linewidth=2, label=label) 
                    for col, label in zip(['red', 'blue', 'green'], ['I_new', 'Mean R', 'Mean S'])]
    
    # Combine both legends into one location
    ax.legend(handles=style_legend + color_legend, 
              title="Legend", 
              bbox_to_anchor=(1.05, 0.5), 
              loc='center left', 
              fontsize=12, 
              title_fontsize=13)

    # Set title and labels
    title = f"Comparison of Scaling Factors"
    if params:
        title += f" for $\\beta_0 = {params['beta_0']}$"
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')

    # Set logarithmic y-axis if requested
    if log:
        ax.set_yscale('log')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Show the plot
    plt.show()

################################################################################################################################
# Plot the Mordecai, Lambrechts, and Perkins Scaling Factors for beta(T, P) - one color per scaling factor
###############################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

def single_plot_scaling_factors_ColorPerSF(out_temp_combined, 
                                scaling_factors=['mordecai', 'lambrechts', 'perkins'],
                                variables_to_plot=['I_new', 'R', 'S'], 
                                colors=['red', 'blue', 'green'], 
                                line_styles=['-', '--', ':'], 
                                params=None, save_path=None, log=False):
    """
    Function to plot selected variables for different scaling factors with individual draws and means.

    Arguments:
    out_temp_combined : xarray.Dataset
        The dataset with `scaling_factor`, `date`, and `draws` coordinates.
    scaling_factors : list, optional
        List of scaling factors to iterate over. Default is ['mordecai', 'lambrechts', 'perkins'].
    variables_to_plot : list, optional
        List of variables to plot (e.g., ['I_new', 'R', 'S']). Default is ['I_new', 'R', 'S'].
    colors : list, optional
        List of colors for each scaling factor. Default is ['red', 'blue', 'green'].
    line_styles : list, optional
        List of line styles for each variable. Default is ['-', '--', ':'].
    params : dict, optional
        Dictionary of model parameters (e.g., for the title). Default is None.
    save_path : str, optional
        Path to save the plot (if provided). Default is None.
    log : bool, optional
        Whether to apply a logarithmic transformation to the y-axis. Default is False.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 6))  # Single plot for all datasets

    # Loop through each variable
    for var_idx, (var, linestyle) in enumerate(zip(variables_to_plot, line_styles)):
        # Loop through each scaling factor
        for sf_idx, (sf, color) in enumerate(zip(scaling_factors, colors)):
            # Extract data for the current scaling factor
            data = out_temp_combined.sel(scaling_factor=sf)

            # Plot the individual draws for the variable
            for draw in data['draws']:
                draw_data = data.sel(draws=draw)
                if log:
                    draw_data[var] = np.log(draw_data[var])  # Apply log transformation
                ax.plot(draw_data['date'], draw_data[var], color=color, alpha=0.2, linewidth=0.5)
            
            # Plot the mean of the variable
            mean_var = data[var].mean(dim='draws')
            if log:
                mean_var = np.log(mean_var)  # Apply log transformation
            ax.plot(data['date'], mean_var, color=color, linestyle=linestyle, linewidth=2,
                    label=f'{var} (scale={sf})' if var_idx == 0 else None)  # Avoid duplicate legend entries

    # Create a legend for line styles
    style_legend = [mlines.Line2D([], [], color='black', linestyle=ls, linewidth=2, label=var) 
                    for ls, var in zip(line_styles, variables_to_plot)]
    
    # Create a legend for colors
    color_legend = [mlines.Line2D([], [], color=col, linestyle='-', linewidth=2, label=sf) 
                    for col, sf in zip(colors, scaling_factors)]
    
    # Combine both legends into one location
    ax.legend(handles=style_legend + color_legend, 
              title="Legend", 
              bbox_to_anchor=(1.05, 0.5), 
              loc='center left', 
              fontsize=12, 
              title_fontsize=13)

    # Set title and labels
    title = f"Comparison of Scaling Factors"
    if params:
        title += f" for $\\beta_0 = {params['beta_0']}$"
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')

    # Set logarithmic y-axis if requested
    if log:
        ax.set_yscale('log')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Show the plot
    plt.show()


def add_vertical_lines(ax, intervals, color='gray', linestyle='--', linewidth=0.8):
    """Add vertical lines to the given axes at the specified intervals."""
    for x in intervals:
        ax.axvline(x=x, color=color, linestyle=linestyle, linewidth=linewidth)

###################################################
# Function to plot mean and individual draws
###################################################
def plot_variable(ax, data, color, label, linestyle='-', plot_draws=True, log=False):
    mean_data = data.mean(dim='draws')
    
    if log:
        # Apply log transformation
        mean_data = np.log(mean_data)
    
    # Plot individual simulations if plot_draws is True
    if plot_draws:
        for draw in data['draws']:
            draw_data = data.sel(draws=draw)
            if log:
                draw_data = np.log(draw_data)
            ax.plot(data['time'], draw_data, color=color, alpha=0.1, linewidth=0.5)
    
    # Plot the mean
    ax.plot(data['time'], mean_data, color=color, linestyle=linestyle, linewidth=2, label=label)