# DENV model with 4 serotypes
#############################################################
# DELTA_1_3 MODEL ADJUSTED FOR USE INITIAL_CONDITION_FUNCTION
#############################################################
import pandas as pd
import numpy as np
from pySODM.models.base import ODE, JumpProcess

from time_dep_parameters import time_dependent_beta

from itertools import permutations



################################
# TEMPORAL MODEL + 4 SEROTYPES AND 4 POSSIBLE INFECTIONS IN SERIES
################################

class JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF(JumpProcess):
    """
    Stochastic SEIR4 model for DENV with 4 serotypes
    No inclusion of age. All births go into susceptible compartment. All deaths leave from every possible compartment (they are natural deaths, unrelated to DENV). A fraction of all infected is considered "reported". 

    All the S, E, I, R compartments are available 4 x in the model to represent each serotypes - except for the very first S (susceptible to all) and the very last R (recovered from all serotypes).

    -- Parameters -- 
    alpha : temporary cross-immunity
    b : birth rate - constant for now
    d : death rate - constant for now and not the same as b
    sigma : incubation period
    gamma : infectious period
    psi : enhanced / inhibited infectiousness of secondary infection
    rho:  proportion; reported cases / all infected (for secondary infections) (based on Imai et al. (2016))
    delta:  proportion; probability of detecting a primary/tertiary infection relative to a secondary infection (based on Imai et al. (2016))
    ---------------

    The temperature depedence is implemented through an equations that represents the temperature-dependence of mosquito variables. These equations have also been published in Andronico et al. (2024)
    -- Parameters --
        beta_0 : the baseline beta 
    -- Temperature dependence functions ---
        scaling factor -> time series added to model
    ---------------

    """

    ################################
    # helper function to generate all possible states 
    ################################
    # All state prefixes
    prefixes = ["S", "E", "I", "R"]
    # All serotypes
    serotypes = ["1", "2", "3", "4"]

    states_list = []
    for prefix in prefixes:
        for length in range(1, 5):  # lengths 1 to 3
            for combo in permutations(serotypes, length):
                states_list.append(prefix + ''.join(combo))

    # print(states_list)

    # Add the single 'S' string
    states_list.insert(0, "S")  
    # Replace any 'RXXXX' (strings starting with 'R' and 5 digits) by just 'R'
    states_list = [
        state for state in states_list
        if not (
            (state.startswith('R') and len(state) == 5) or
            (state.startswith('S') and len(state) == 5)
        )
    ]

    # Then add the single 'R' string
    states_list.append('R')
    # print("after replacement RXXXX length", len(states_list))
    
    # append the "special" states for new infections, cumulative infections and reported infections
    states_list.extend(['I_new', 'I_cum', 'I_rep'])
    
    # add an extra I_rep per serotype
    states_list.extend(['I_1_rep', 'I_2_rep', 'I_3_rep', 'I_4_rep'])

    # LENGTH OF THE STATES LIST
    # print("Length of states_list:", len(states_list))
    # print(states_list)

    ####################################
    # Create a mapping from state names to indices
    ####################################

    # This will be used to quickly access the index of a state in the states_list
    state_name_to_idx = {state: i for i, state in enumerate(states_list)}

    states = ['S', 'S1', 'S2', 'S3', 'S4', 'S12', 'S13', 'S14', 'S21', 'S23', 'S24', 'S31', 'S32', 'S34', 'S41', 'S42', 'S43', 'S123', 'S124', 'S132', 'S134', 'S142', 'S143', 'S213', 'S214', 'S231', 'S234', 'S241', 'S243', 'S312', 'S314', 'S321', 'S324', 'S341', 'S342', 'S412', 'S413', 'S421', 'S423', 'S431', 'S432', 'E1', 'E2', 'E3', 'E4', 'E12', 'E13', 'E14', 'E21', 'E23', 'E24', 'E31', 'E32', 'E34', 'E41', 'E42', 'E43', 'E123', 'E124', 'E132', 'E134', 'E142', 'E143', 'E213', 'E214', 'E231', 'E234', 'E241', 'E243', 'E312', 'E314', 'E321', 'E324', 'E341', 'E342', 'E412', 'E413', 'E421', 'E423', 'E431', 'E432', 'E1234', 'E1243', 'E1324', 'E1342', 'E1423', 'E1432', 'E2134', 'E2143', 'E2314', 'E2341', 'E2413', 'E2431', 'E3124', 'E3142', 'E3214', 'E3241', 'E3412', 'E3421', 'E4123', 'E4132', 'E4213', 'E4231', 'E4312', 'E4321', 'I1', 'I2', 'I3', 'I4', 'I12', 'I13', 'I14', 'I21', 'I23', 'I24', 'I31', 'I32', 'I34', 'I41', 'I42', 'I43', 'I123', 'I124', 'I132', 'I134', 'I142', 'I143', 'I213', 'I214', 'I231', 'I234', 'I241', 'I243', 'I312', 'I314', 'I321', 'I324', 'I341', 'I342', 'I412', 'I413', 'I421', 'I423', 'I431', 'I432', 'I1234', 'I1243', 'I1324', 'I1342', 'I1423', 'I1432', 'I2134', 'I2143', 'I2314', 'I2341', 'I2413', 'I2431', 'I3124', 'I3142', 'I3214', 'I3241', 'I3412', 'I3421', 'I4123', 'I4132', 'I4213', 'I4231', 'I4312', 'I4321', 'R1', 'R2', 'R3', 'R4', 'R12', 'R13', 'R14', 'R21', 'R23', 'R24', 'R31', 'R32', 'R34', 'R41', 'R42', 'R43', 'R123', 'R124', 'R132', 'R134', 'R142', 'R143', 'R213', 'R214', 'R231', 'R234', 'R241', 'R243', 'R312', 'R314', 'R321', 'R324', 'R341', 'R342', 'R412', 'R413', 'R421', 'R423', 'R431', 'R432', 'R', 'I_new', 'I_cum', 'I_rep', 'I_1_rep', 'I_2_rep', 'I_3_rep', 'I_4_rep']
    
    parameters = ['alpha','b', 'd', 'beta_0', 'sigma', 'gamma', 'psi', 'rho', 'delta_1', 'delta_3', 'sf', 'epi_uf', 'f_imm_1', 'f_imm_2', 'f_imm_3', 'f_imm_4', 'pop_uf', 'compartment_fractions', 'states_list']

    ##########################################    
    # parameters = ['alpha','b', 'd', 'beta_0', 'sigma', 'gamma', 'psi', 'rho', 'sf'] 

    @staticmethod
    def compute_rates(t, 
                      S , S1 , S2 , S3 , S4 , S12 , S13 , S14 , S21 , S23 , S24 , S31 , S32 , S34 , S41 , S42 , S43 , S123 , S124 , S132 , S134 , S142 , S143 , S213 , S214 , S231 , S234 , S241 , S243 , S312 , S314 , S321 , S324 , S341 , S342 , S412 , S413 , S421 , S423 , S431 , S432 , 
                      
                      E1 , E2 , E3 , E4 , E12 , E13 , E14 , E21 , E23 , E24 , E31 , E32 , E34 , E41 , E42 , E43 , E123 , E124 , E132 , E134 , E142 , E143 , E213 , E214 , E231 , E234 , E241 , E243 , E312 , E314 , E321 , E324 , E341 , E342 , E412 , E413 , E421 , E423 , E431 , E432 , E1234 , E1243 , E1324 , E1342 , E1423 , E1432 , E2134 , E2143 , E2314 , E2341 , E2413 , E2431 , E3124 , E3142 , E3214 , E3241 , E3412 , E3421 , E4123 , E4132 , E4213 , E4231 , E4312 , E4321 , 
                      
                      I1 , I2 , I3 , I4 , I12 , I13 , I14 , I21 , I23 , I24 , I31 , I32 , I34 , I41 , I42 , I43 , I123 , I124 , I132 , I134 , I142 , I143 , I213 , I214 , I231 , I234 , I241 , I243 , I312 , I314 , I321 , I324 , I341 , I342 , I412 , I413 , I421 , I423 , I431 , I432 , I1234 , I1243 , I1324 , I1342 , I1423 , I1432 , I2134 , I2143 , I2314 , I2341 , I2413 , I2431 , I3124 , I3142 , I3214 , I3241 , I3412 , I3421 , I4123 , I4132 , I4213 , I4231 , I4312 , I4321 , 
                      
                      R1 , R2 , R3 , R4 , R12 , R13 , R14 , R21 , R23 , R24 , R31 , R32 , R34 , R41 , R42 , R43 , R123 , R124 , R132 , R134 , R142 , R143 , R213 , R214 , R231 , R234 , R241 , R243 , R312 , R314 , R321 , R324 , R341 , R342 , R412 , R413 , R421 , R423 , R431 , R432 , R , 

                      I_new , I_cum , I_rep , I_1_rep, I_2_rep, I_3_rep, I_4_rep,
                      
                      b, d, beta_0, alpha, sigma, gamma, psi,rho, delta_1,delta_3, sf, 
                      epi_uf, f_imm_1, f_imm_2, f_imm_3, f_imm_4, pop_uf, compartment_fractions, states_list):
        
        """
        Compute the rates for each state in the SEIR4 model with 4 serotypes.
        Parameters
        ----------
        t : int or datetime
            The current time step in the simulation.    
            
        states: 
            S, S1, S2, S3, S4, S12, S13, S14, S21, S23, S24, S31, S32, S34, S41, S42, S43, S123, S124, S132, S134, S142, S143, S213, S214, S231, S234, S241, S243, S312, S314, S321, S324, S341, S342, S412, S413, S421, S423, S431, S432,
            E1, E2, E3, E4, E12, E13, E14, E21, E23, E24, E31, E32, E34, E41, E42, E43, E123, E124, E132, E134, E142, E143, E213, E214, E231, E234, E241, E243, E312, E314, E321, E324, E341, E342, E412, E413, E421, E423,
            E431, E432, E1234, E1243, E1324, E1342, E1423, E1432, E2134, E2143, E2314, E2341, E2413, E2431, E3124, E3142, E3214, E3241, E3412, E3421, E4123, E4132, E4213, E4231 E4312,  
            I1 , I2 , I3 , I4 , I12 , I13 , I14 , I21 , I23 , I24 , I31 , I32 , I34 , I41 , I42 , I43 , I123 , I124 , I132 , I134 , I142 , I143 , I213 , I214 , I231 , I234 , I241 , I243 , I312 , I314 , I321 , I324 , I341 , I342 , I412 , I413 , I421 , I423 , I431 , I432 , I1234 , I1243 , I1324 , I1342 , I1423 , I1432 , I2134 , I2143 , I2314 , I2341 , I2413 , I2431 , I3124 , I3142 , I3214 , I3241 , I3412 , I3421 , I4123 , I4132 , I4213 , I4231 , I4312 , I4321 , 
            R1 , R2 , R3 , R4 , R12 , R13 , R14 , R21 , R23 , R24 , R31 , R32 , R34 , R41 , R42 , R43 , R123 , R124 , R132 , R134 , R142 , R143 , R213 , R214 , R231 , R234 , R241 , R243 , R312 , R314 , R321 , R324 , R341 , R342 , R412 , R413 , R421 , R423 , R431 , R432 , R , 
            I_new , I_cum , I_rep , I_1_rep, I_2_rep, I_3_rep, I_4_rep
        model parameters:
        b, d, beta_0, alpha, sigma, gamma, psi,rho, delta_1,delta_3, sf
        
        Returns
        -------
        rates : dict
            A dictionary where keys are state names and values are lists of rates for each state."""
        
        # Extract the states list
        # states_list = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported.states_list        # Add these lines after the class definition
        
        # state_name_to_idx = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported.state_name_to_idx
        
        # infectious_states_by_sero = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported.infectious_states_by_sero
        serotypes = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.serotypes
        
        states_list = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.states_list
        
        state_name_to_idx = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.state_name_to_idx
        
        infectious_states_by_sero = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.infectious_states_by_sero
        
        primary_idx_by_sero = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.primary_idx_by_sero

        # All state arguments in order (excluding t and parameters at the end)
        state_values = [
            S, S1, S2, S3, S4, S12, S13, S14, S21, S23, S24, S31, S32, S34, S41, S42, S43, S123, S124, S132, S134, S142, S143, S213, S214, S231, S234, S241, S243, S312, S314, S321, S324, S341, S342, S412, S413, S421, S423, S431, S432, 
            
            E1, E2, E3, E4, E12, E13, E14, E21, E23, E24, E31, E32, E34, E41, E42, E43, E123, E124, E132, E134, E142, E143, E213, E214, E231, E234, E241, E243, E312, E314, E321, E324, E341, E342, E412, E413, E421, E423, E431, E432, E1234, E1243, E1324, E1342, E1423, E1432, E2134, E2143, E2314, E2341, E2413, E2431, E3124, E3142, E3214, E3241, E3412, E3421, E4123, E4132, E4213, E4231, E4312, E4321,
            
            I1, I2, I3, I4, I12, I13, I14, I21, I23, I24, I31, I32, I34, I41, I42, I43, I123, I124, I132, I134, I142, I143, I213, I214, I231, I234, I241, I243, I312, I314, I321, I324, I341, I342, I412, I413, I421, I423, I431, I432, I1234, I1243, I1324, I1342, I1423, I1432, I2134, I2143, I2314, I2341, I2413, I2431, I3124, I3142, I3214, I3241, I3412, I3421, I4123, I4132, I4213, I4231, I4312, I4321,
            
            R1, R2, R3, R4, R12, R13, R14, R21, R23, R24, R31, R32, R34, R41, R42, R43, R123, R124, R132, R134, R142, R143, R213, R214, R231, R234, R241, R243, R312, R314, R321, R324, R341, R342, R412, R413, R421, R423, R431, R432, R,
            I_new, I_cum, I_rep, I_1_rep, I_2_rep, I_3_rep, I_4_rep]

        # use np array for each state value
        # Reconstruct the state array from the individual arguments
        state_array = np.array(state_values)    
        
        # Calculate total population as the sum of all states except the special states 
        exclude_states = {'I_new', 'I_cum', 'I_rep', 'I_1_rep', 'I_2_rep', 'I_3_rep', 'I_4_rep'}
        exclude_idxs = [state_name_to_idx[s] for s in exclude_states if s in state_name_to_idx]

        T = np.sum(np.delete(state_array, exclude_idxs))

        # set beta_t based on the time-dependent beta function
        beta_t = beta_0

        # CALCULATE THE FOI FOR EACH SEROTYPE
        forces = {}
        for sero in serotypes:
            # extract the indices of infectious states for this serotype
            idxs = infectious_states_by_sero[sero]
            # separate the primary infectious state from the others
            primary_idx = primary_idx_by_sero[sero]
            primary_val = state_array[primary_idx]
            higher_idxs = [i for i in idxs if i != primary_idx]
            sum_higher = np.sum(state_array[higher_idxs]) if higher_idxs else 0
            # calculate the force of infection for this serotype
            if np.isnan(T) or T == 0:
                print(f"[DEBUG] T is invalid at t={t}: T={T}, state_array={state_array[:5]}, beta_t={beta_t}")
            else:
                forces[sero] = beta_t * (primary_val + psi * sum_higher) / T


        #######################
        # CALCULATE THE RATES FOR EACH STATE
        ####################### 
        # Initialize rates dictionary
        rates = {}  

        for state in states_list:
            # print(f"\nProcessing state: {state}")

            idx = state_name_to_idx[state]
        # suscptibles
            if state.startswith("S"):
                # Find which serotypes this state has already experienced
                experienced = set(state[1:])  # skip the "S"
                # For "S" (fully susceptible), experienced will be empty
                # For "S12", experienced = {"1", "2"}, etc.

                # List of serotypes this state is still susceptible to
                susceptible_to = [sero for sero in serotypes if sero not in experienced]
                # print(f"Experienced serotypes: {experienced} and susceptible to: {susceptible_to}")

                # Build the rates list: FOI for each susceptible serotype
                rate_list = [forces[sero] for sero in susceptible_to]
                
                # Add DEATH rate for all S* states
                rate_list.append(d * np.ones(state_array[idx].shape))
                
                # For "S" (fully susceptible), add BIRTHS TO S
                if state == "S":
                    rate_list.append(b * np.ones(state_array[idx].shape))
                
                rates[state] = rate_list
                # print(f"Rates for state {state}: {rates[state]}")
                
            elif state.startswith("E"):
                rates[state] = [
                (1 / sigma) * np.ones(state_array[idx].shape),
                d * np.ones(state_array[idx].shape)]
                
            elif state.startswith("I") and state not in ["I_new", "I_cum", "I_rep", "I_1_rep", "I_2_rep", "I_3_rep", "I_4_rep"]:
                rates[state] = [
                    (1 / gamma) * np.ones(state_array[idx].shape),
                    d * np.ones(state_array[idx].shape)]
                
            elif state.startswith("R"):
                if len(state) > 1:  # e.g., "R12", "R123", etc.
                    rates[state] = [
                        (1 / alpha) * np.ones(state_array[idx].shape),  # TCI
                        d * np.ones(state_array[idx].shape)]
                else: # just "R"
                    rates[state] = [d * np.ones(state_array[idx].shape)] # death rate
        # # TRYING TO RESOLVE ValueError: p < 0, p > 1 or p contains NaNs
        # === DEBUG: Print rates that are invalid ===
        
        for state, rate_list in rates.items():
            for i, rate in enumerate(rate_list):
                arr = np.array(rate)
                if np.any(arr < 0)  or np.any(np.isnan(arr)):
                    print(f"[DEBUG] Invalid rate in state '{state}' (rate idx {i}), NaNs={np.isnan(arr).sum()}")
                    print(f"  Rate values (first 5): {arr[:5]}")
                    print(f"  State value: {state_array[idx]}")
                    print(f"  Time t: {t}")
                    # print(f"  Parameters: b={b}, d={d}, beta_0={beta_0}, alpha={alpha}, sigma={sigma}, gamma={gamma}, psi={psi}, rho={rho}, delta_1={delta_1}, delta_3={delta_3}, sf={sf}")

        return rates
    

           
               
    @staticmethod
    def apply_transitionings(t, tau, transitionings, 
                            S , S1 , S2 , S3 , S4 , S12 , S13 , S14 , S21 , S23 , S24 , S31 , S32 , S34 , S41 , S42 , S43 , S123 , S124 , S132 , S134 , S142 , S143 , S213 , S214 , S231 , S234 , S241 , S243 , S312 , S314 , S321 , S324 , S341 , S342 , S412 , S413 , S421 , S423 , S431 , S432 , 
                      
                            E1 , E2 , E3 , E4 , E12 , E13 , E14 , E21 , E23 , E24 , E31 , E32 , E34 , E41 , E42 , E43 , E123 , E124 , E132 , E134 , E142 , E143 , E213 , E214 , E231 , E234 , E241 , E243 , E312 , E314 , E321 , E324 , E341 , E342 , E412 , E413 , E421 , E423 , E431 , E432 , E1234 , E1243 , E1324 , E1342 , E1423 , E1432 , E2134 , E2143 , E2314 , E2341 , E2413 , E2431 , E3124 , E3142 , E3214 , E3241 , E3412 , E3421 , E4123 , E4132 , E4213 , E4231 , E4312 , E4321 , 
                            
                            I1 , I2 , I3 , I4 , I12 , I13 , I14 , I21 , I23 , I24 , I31 , I32 , I34 , I41 , I42 , I43 , I123 , I124 , I132 , I134 , I142 , I143 , I213 , I214 , I231 , I234 , I241 , I243 , I312 , I314 , I321 , I324 , I341 , I342 , I412 , I413 , I421 , I423 , I431 , I432 , I1234 , I1243 , I1324 , I1342 , I1423 , I1432 , I2134 , I2143 , I2314 , I2341 , I2413 , I2431 , I3124 , I3142 , I3214 , I3241 , I3412 , I3421 , I4123 , I4132 , I4213 , I4231 , I4312 , I4321 , 
                            
                            R1 , R2 , R3 , R4 , R12 , R13 , R14 , R21 , R23 , R24 , R31 , R32 , R34 , R41 , R42 , R43 , R123 , R124 , R132 , R134 , R142 , R143 , R213 , R214 , R231 , R234 , R241 , R243 , R312 , R314 , R321 , R324 , R341 , R342 , R412 , R413 , R421 , R423 , R431 , R432 , R , 

                            I_new , I_cum , I_rep, I_1_rep, I_2_rep, I_3_rep, I_4_rep,
                            
                            b, d, beta_0, alpha, sigma, gamma, psi, rho, delta_1, delta_3, sf, epi_uf, f_imm_1, f_imm_2, f_imm_3, f_imm_4, pop_uf, compartment_fractions, states_list):
        
        """
        Apply the transitionings to the current states and return the new states.
        Parameters
        ----------
        t : int or datetime
            The current time step in the simulation.
        tau : int or datetime
            The time step at which the transitionings were calculated.
        transitionings : dict
            A dictionary where keys are state names and values are lists of transitionings for each state.
        states: values 
            S , S1 , S2 , S3 , S4 , S12 , S13 , S14 , S21 , S23 , S24 , S31 , S32 , S34 , S41 , S42 , S43 , S123 , S124 , S132 , S134 , S142 , S143 , S213 , S214 , S231 , S234 , S241 , S243 , S312 , S314 , S321 , S324 , S341 , S342 , S412 , S413 , S421 , S423 , S431 , S432 , 
        
            E1 , E2 , E3 , E4 , E12 , E13 , E14 , E21 , E23 , E24 , E31 , E32 , E34 , E41 , E42 , E43 , E123 , E124 , E132 , E134 , E142 , E143 , E213 , E214 , E231 , E234 , E241 , E243 , E312 , E314 , E321 , E324 , E341 , E342 , E412 , E413 , E421 , E423 , E431 , E432 , E1234 , E1243 , E1324 , E1342 , E1423 , E1432 , E2134 , E2143 , E2314 , E2341 , E2413 , E2431 , E3124 , E3142 , E3214 , E3241 , E3412 , E3421 , E4123 , E4132 , E4213 , E4231 , E4312 , E4321 , 
            
            I1 , I2 , I3 , I4 , I12 , I13 , I14 , I21 , I23 , I24 , I31 , I32 , I34 , I41 , I42 , I43 , I123 , I124 , I132 , I134 , I142 , I143 , I213 , I214 , I231 , I234 , I241 , I243 , I312 , I314 , I321 , I324 , I341 , I342 , I412 , I413 , I421 , I423 , I431 , I432 , I1234 , I1243 , I1324 , I1342 , I1423 , I1432 , I2134 , I2143 , I2314 , I2341 , I2413 , I2431 , I3124 , I3142 , I3214 , I3241 , I3412 , I3421 , I4123 , I4132 , I4213 , I4231 , I4312 , I4321 , 
            
            R1 , R2 , R3 , R4 , R12 , R13 , R14 , R21 , R23 , R24 , R31 , R32 , R34 , R41 , R42 , R43 , R123 , R124 , R132 , R134 , R142 , R143 , R213 , R214 , R231 , R234 , R241 , R243 , R312 , R314 , R321 , R324 , R341 , R342 , R412 , R413 , R421 , R423 , R431 , R432 , R , 

            I_new , I_cum , I_rep, I_1_rep, I_2_rep, I_3_rep, I_4_rep
            
        model parameters:
            b, d, beta_0, alpha, sigma, gamma, psi, rho, delta_1, delta_3, sf
            
        Returns
        -------                                                                          The individual states after applying the transitionings.
        """                                                                                    

        # print("\n t in apply_transitionings", t)
        # transitionings = {k: [int(np.round(v, 0)) for v in (v if isinstance(v, list) else [v])] for k, v in transitionings.items()} # to avoid that the states become non-integegers and therefore go below 0
        # transitionings = {k: [int(np.round(v, 0))] for k, v in transitionings.items()}
        
        # clip negative values to zero
        transitionings = {k: [max(0, int(np.round(arr))) for arr in v] for k, v in transitionings.items()}

        
        states_list = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.states_list
        
        state_name_to_idx = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.state_name_to_idx
        
        serotypes = JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.serotypes
        
        # Build state_array from arguments (order must match states_list)
        state_values = [
            S, S1, S2, S3, S4, S12, S13, S14, S21, S23, S24, S31, S32, S34, S41, S42, S43, S123, S124, S132, S134, S142, S143, S213, S214, S231, S234, S241, S243, S312, S314, S321, S324, S341, S342, S412, S413, S421, S423, S431, S432,
            E1, E2, E3, E4, E12, E13, E14, E21, E23, E24, E31, E32, E34, E41, E42, E43, E123, E124, E132, E134, E142, E143, E213, E214, E231, E234, E241, E243, E312, E314, E321, E324, E341, E342, E412, E413, E421, E423, E431, E432, E1234, E1243, E1324, E1342, E1423, E1432, E2134, E2143, E2314, E2341, E2413, E2431, E3124, E3142, E3214, E3241, E3412, E3421, E4123, E4132, E4213, E4231, E4312, E4321,
            I1, I2, I3, I4, I12, I13, I14, I21, I23, I24, I31, I32, I34, I41, I42, I43, I123, I124, I132, I134, I142, I143, I213, I214, I231, I234, I241, I243, I312, I314, I321, I324, I341, I342, I412, I413, I421, I423, I431, I432, I1234, I1243, I1324, I1342, I1423, I1432, I2134, I2143, I2314, I2341, I2413, I2431, I3124, I3142, I3214, I3241, I3412, I3421, I4123, I4132, I4213, I4231, I4312, I4321,
            R1, R2, R3, R4, R12, R13, R14, R21, R23, R24, R31, R32, R34, R41, R42, R43, R123, R124, R132, R134, R142, R143, R213, R214, R231, R234, R241, R243, R312, R314, R321, R324, R341, R342, R412, R413, R421, R423, R431, R432, R,
            I_new, I_cum, I_rep, I_1_rep, I_2_rep, I_3_rep, I_4_rep
        ]
        state_array = np.array(state_values)

        # Precompute mapping from E state to (S_state, index)
        E_to_S_and_idx = {}
        for state in states_list:
            if state.startswith("E"):
                subscript = state[1:]
                S_state = "S" + subscript[:-1] if subscript[:-1] else "S"
                new_sero = subscript[-1]
                experienced = set(S_state[1:])
                susceptible_to = [sero for sero in serotypes if sero not in experienced]
                idx = susceptible_to.index(new_sero)
                E_to_S_and_idx[state] = (S_state, idx)
                
        # Prepare new state array
        new_state_array = np.copy(state_array)
        
        for state in states_list:
            idx = state_name_to_idx[state]

        # Susceptibles 
            if state.startswith("S"):
                if state == "S":
                    new_state_array[idx] = (
                        state_array[idx]  # the prior number in S
                        - sum(transitionings[state][i] for i in range(5))  # S -> E transitions & deaths
                        + transitionings[state][5]  # + births
                    )
                    new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))
                
                else: # for S1, S2, S3, S4, S12, S13, etc.
                    prior_R = "R" + state[1:]
                    new_state_array[idx] = state_array[idx] 
                    + transitionings[prior_R][0] # R -> S transitions 
                    - sum(transitionings[state]) # S -> E transitions and deaths
                    new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))

        # Exposed                    
            elif state.startswith("E"):
                prior_S, s_idx = E_to_S_and_idx[state] # Get the corresponding S state and index (e.g. ("S", 0) for "E1")
                new_state_array[idx] = (
                    state_array[idx]
                    + transitionings[prior_S][s_idx]  # S -> E transition
                    - transitionings[state][0] # E -> I transition
                    - transitionings[state][1] # deaths
                )
                new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))

        # infectious                    
            elif state.startswith("I") and state not in ["I_new", "I_cum", "I_rep", "I_1_rep", "I_2_rep", "I_3_rep", "I_4_rep"]:
                prior_E = "E" + state[1:]
                # For regular I states, use the transitionings formula
                new_state_array[idx] = (
                    state_array[idx] # the prior number in I
                    + transitionings[prior_E][0]  # E -> I transition
                    - transitionings[state][0]  # I -> R transition
                    - transitionings[state][1]  # deaths
                )
                new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))
                
        # recovered                
            elif state.startswith("R"):
                if len(state) > 1:
                    # For R1, R21, R431, etc.
                    prior_I = "I" + state[1:]
                    new_state_array[idx] = (
                        state_array[idx]
                        + transitionings[prior_I][0] # I -> R transition
                        - transitionings[state][0] # R -> S
                        - transitionings[state][1] # deaths
                    )
                    new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))
                else:
                    # For the final R (just "R")
                    new_state_array[idx] = (
                        state_array[idx] # the prior number in R
                        + sum(transitionings[s][0] for s in states_list if s.startswith("I") and len(s) > 4 and s not in ["I_new", "I_cum", "I_rep","I_1_rep", "I_2_rep", "I_3_rep", "I_4_rep"]) # sum of all I states with 4 serotype exposure
                        - transitionings[state][0] # deaths
                    )
                    new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))
                    
        # Derivative states
            elif state == "I_new":
                # I_new is the sum of all transitions from any E state to I at timestep t
                new_state_array[idx] = sum(transitionings[s][0] for s in states_list if s.startswith("E"))
                new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))

                
            elif state == "I_rep":
                # I_rep is rho * I_new -> reported cases at timestep t
                transitions = 0
                for s in states_list:
                    if s.startswith("E"):
                        # The corresponding I state is "I" + s[1:]
                        I_state = "I" + s[1:]
                        if len(I_state) == 2:  # Primary infection (e.g., I1, I3)
                            transitions += rho *delta_1* transitionings[s][0]
                        # Secondary infection: length 3 (e.g., I21, I32)
                        elif len(I_state) == 3:
                            transitions += rho * transitionings[s][0]
                        # tertiary and quaternary infection
                        else:
                            transitions += rho* delta_3 * transitionings[s][0]
                new_state_array[idx] = transitions
                new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))
            
            # get the number of reported cases per serotype (I_1_rep, I_2_rep, etc.)
            elif state.startswith("I_") and state.endswith("_rep"):
                # e.g., state == "I_1_rep"
                sero = state[2]  # '1', '2', '3', or '4'
                # Find all E-states ending with this serotype
                E_states = [s for s in states_list if s.startswith("E") and s[-1] == sero]
                transitions = 0
                for E_state in E_states:
                    # The corresponding I state is "I" + E_state[1:]
                    I_state = "I" + E_state[1:]
                    if len(I_state) == 2:  # Primary infection (e.g., I1, I2)
                        transitions += rho * delta_1 * transitionings[E_state][0]
                    elif len(I_state) == 3: # Secondary infections (e.g., I21, I32)
                        transitions += rho * transitionings[E_state][0]
                    else: # Tertiary and quaternary infections (e.g., I231, I1234)
                        transitions += rho * delta_3 * transitionings[E_state][0]
                new_state_array[idx] = transitions
                new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))

                
            elif state == "I_cum":
                # I_cum is previous I_cum plus I_new
                new_state_array[idx] = state_array[state_name_to_idx["I_cum"]] + new_state_array[state_name_to_idx["I_new"]]
                new_state_array[idx] = max(0, int(np.round(new_state_array[idx])))

        
        # return the new states as a tuple in the order of the original states_list
        return tuple(new_state_array[state_name_to_idx[state]] for state in states_list)

# Add these lines after the class definition

JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.infectious_states_by_sero = {
    sero: [i for i, state in enumerate(JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.states_list)
           if state.startswith("I") and state.endswith(sero)]
    for sero in ["1", "2", "3", "4"]
}
JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.primary_idx_by_sero = {
    sero: JumpProcess_SEIR4_beta_by_Temp_sf_BirthDeath_reported_d_1_3_ICF.state_name_to_idx[f"I{sero}"]
    for sero in ["1", "2", "3", "4"]
}