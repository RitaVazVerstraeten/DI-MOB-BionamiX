# SENSITIVITY ANALYSIS FILE
# the steps in the following file were informed by: https://salib.readthedocs.io/en/latest/user_guide/basics.html


##################
# importing SALib
##################
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib import ProblemSpec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import datetime

###########################
# adjustable parameters
###########################
save=True
results_folder= '/home/rita/PyProjects/DI-MOB-BionamiX/results/DENV_model/Temporal_SEIR2_SF_birthdeath_test/sobol_sensitivity/'
results_name='sobol_temp_SEIR2'
calc_second_order = True

##############################
# setting up the ProblemSpec
##############################
# ProblemSpec gives me a nicer overview of the results at the end 

problem_definition = {
    'num_vars': 10,
    'names': ['alpha','b', 'd', 'sigma', 'gamma', 'omega', 'psi', 'beta_0', 'beta_1', 'ph'],
    'bounds': [
        [7, 200],  # days of temporary cross-immunity
        [2.3835616438356162e-05, 2.9863013698630136e-05], # births from the CF data in Parameter_ranges_simulations.ipynb
        [2.1643835616438358e-05, 2.6849315068493153e-05], # deaths
        [4.00, 7.00],  # sigma 
        [4.00, 10.00],  # gamma 
        [np.pi, 4*np.pi ], #omega seasonal recurrence from 1/2 year to 2 yearly 
        [1, 2.4],  # psi - Recher 2009 and Ten Bosch 2016
        [0.1, 0.4],  # beta_0
        [0.1, 0.4],  # beta_1
        [0, 2 * np.pi]  # phi -> phase 
    ],
    'outputs': ['Y']
}

sp = ProblemSpec(problem_definition)

##############################
# sampling the parameters 
##############################
sp.sample_sobol(1000)

##############################
# Running the model 
##############################

# load the model 
from DENV_models_pySODM import JumpProcess_SEIR2_SeasonalForcing_BirthDeath as temporal_SEIR2

# try this with a evaluate_model function instead of a loop: 
initN = 200000
initI1 = 20
initI2 = 20
initS = initN - (initI1 + initI2)

init_states = {'S': initS,
               'I1': initI1,
               'I2': initI2}

def evaluate_model(X):
    # X is a single sample from the Sobol sampling
    params_dict = dict(zip(problem_definition['names'], X))  # Map names to parameter values
    
    model_temp = temporal_SEIR2(initial_states=init_states, parameters=params_dict)
    output_all = model_temp.sim(time=365, tau=1.0, output_timestep=1)
    output_I_cum = output_all["I_cum"]
    return output_I_cum.values[-1]  # Return the last value of I_cum

Y = np.zeros(sp.samples.shape[0])  # Store the results

## RUNNING PARALLEL ON ALL CORES TOOK 32 MINUTES
with ProcessPoolExecutor() as executor:
    Y = list(executor.map(evaluate_model, sp.samples))


# Set the results for Sobol analysis
print(np.array(Y).shape)

sp.set_results(np.array(Y)) # needs to be a np.array instead of a list

##############################
# calculate the sobol indices
##############################
Si = sobol.analyze(problem = problem_definition, Y=np.array(Y))

# Si = sp.analyze_sobol()
print("Sobol indices", Si)

# Si is a Python dict with the keys "S1", "S2", "ST", "S1_conf", "S2_conf", and "ST_conf". 
# The _conf keys store the corresponding confidence intervals, typically with a confidence level of 95%. Use the keyword argument print_to_console=True to print all indices.

print("Si length", len(Si)) #6 --> S1, S2, ST, along with their 95% CIs

######################
# plotting the results
######################
sp.analyze_sobol()
axes = sp.plot()
# axes[0].set_yscale('log')
fig = plt.gcf()  # get current figure
fig.set_size_inches(20, 8)
plt.tight_layout()

# Date at which script is started
run_date = str(datetime.date.today())

# Save the figure
file_path = results_folder + results_name + '_' + run_date + '_overviewplot.png'
plt.savefig(file_path)

#####################
# Storing the results for later use
#####################

# Save the results to the specified path
if calc_second_order:
    total_Si, first_Si, second_Si = Si.to_df()
    if save:
        # Use the context manager to handle the ExcelWriter
        with pd.ExcelWriter(results_folder + results_name + '_' + run_date + '.xlsx', engine='openpyxl') as writer:
            # Concatenate and write the S1 and S1ST data
            S1ST = pd.concat([total_Si, first_Si], axis=1)
            S1ST.to_excel(writer, sheet_name='S1ST')
            
            # Write second-order sensitivity results
            S2 = pd.DataFrame(Si['S2'], index=problem_definition['names'], columns=problem_definition['names'])
            S2.to_excel(writer, sheet_name='S2')
            
            S2_conf = pd.DataFrame(Si['S2_conf'], index=problem_definition['names'], columns=problem_definition['names'])
            S2_conf.to_excel(writer, sheet_name='S2_conf')
else:
    total_Si, first_Si = Si.to_df()
    if save:
        # Use the context manager to handle the ExcelWriter
        with pd.ExcelWriter(results_folder + results_name + '_' + run_date + '.xlsx', engine='openpyxl') as writer:
            # Concatenate and write the S1ST data
            S1ST = pd.concat([total_Si, first_Si], axis=1)
            S1ST.to_excel(writer, sheet_name='S1ST')