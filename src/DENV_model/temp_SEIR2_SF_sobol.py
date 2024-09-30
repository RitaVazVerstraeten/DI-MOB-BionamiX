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


##############################
# setting up the ProblemSpec
##############################
# ProblemSpec gives me a nicer overview of the results at the end 

problem_definition = {
    'num_vars': 7,
    'names': ['alpha', 'beta_0', 'beta_1', 'sigma', 'gamma', 'psi', 'ph'],
    'bounds': [
        [7, 200],  # days of temporary cross-immunity
        [0.3, 0.5],  # beta_0
        [0, 1],  # beta_1
        [3.00, 7.00],  # sigma
        [3.00, 7.00],  # gamma
        [0, 1],  # psi
        [0, 2 * np.pi]  # ph
    ],
    'outputs': ['Y']
}

sp = ProblemSpec(problem_definition)

##############################
# sampling the parameters 
##############################
sp.sample_sobol(100)

##############################
# Running the model 
##############################

# load the model 
from DENV_models_pySODM import JumpProcess_SEIR2_SeasonalForcing as temporal_SEIR2

# try this with a evaluate_model function instead of a loop: 
initN = 100000
initI1 = 5
initI2 = 5
initS = initN - (initI1 + initI2)

init_states = {'S': initS,
               'I1': initI1,
               'I2': initI2}

def evaluate_model(X):
    print("X",X)
    # X is a single sample from the Sobol sampling
    params_dict = dict(zip(problem_definition['names'], X))  # Map names to parameter values
    
    model_temp = temporal_SEIR2(states=init_states, parameters=params_dict)
    output_all = model_temp.sim(time=365, samples={}, tau=1.0, output_timestep=1)
    output_I_cum = output_all["I_cum"]
    return output_I_cum.values[-1]  # Return the last value of I_cum

Y = np.zeros(sp.samples.shape[0])  # Store the results
# for i in range(sp.samples.shape[0]):
#     Y[i] = evaluate_model(sp.samples[i])

## RUNNING PARALLEL ON ALL CORES TOOK 4 MINUTES
with ProcessPoolExecutor() as executor:
    Y = list(executor.map(evaluate_model, sp.samples))

# Set the results for Sobol analysis
print(np.array(Y).shape)
sp.set_results(np.array(Y)) # needs to be a np.array instead of a list

##############################
# calculate the sobol indices
##############################
Si = sp.analyze_sobol()

# Si is a Python dict with the keys "S1", "S2", "ST", "S1_conf", "S2_conf", and "ST_conf". 
# The _conf keys store the corresponding confidence intervals, typically with a confidence level of 95%. Use the keyword argument print_to_console=True to print all indices.

print("first-order Si", Si["S1"]) # first order indices
print("Total order Si", Si["ST"]) # total order indices
print("Second-order Si", Si["S2"]) # second-order indices


# Total order indices are much higher than first order --> it's likely that there are higher-orer interactions between my parameters

print("Si length", len(Si)) #6 --> S1, S2, ST, along with their 95% CIs

print("param1-param7:", Si['S2'])
print("Interaction parameter 1 with parameter 3:", Si['S2'][0,2])
print("x2-x3:", Si['S2'][1,2])

# store results to pd.dataframe
total_Si, first_Si, second_Si = Si.to_df()

print(sp.analyze_sobol())


######################
# plotting the results
######################

axes = sp.plot()
# axes[0].set_yscale('log')
fig = plt.gcf()  # get current figure
fig.set_size_inches(20, 8)
plt.tight_layout()



