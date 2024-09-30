## DENV models file based on Infuenza example pySODM

import pandas as pd
import numpy as np
from pySODM.models.base import ODE, JumpProcess

class ODE_SEIR2_model(ODE):
    """
    SEIR model for DENV with age groups and 2 serotypes, so heterotypic infection.
    Same beta, sigma, gamma, and psi variables for both serotypes.
    """
    
    states = ['S','E1','E2','I1', 'I2', 'R1','R2','H1', 'H2','R12']
    parameters = ['beta','sigma', 'gamma', 'psi']
    dimensions = ['age_group']

    @staticmethod
    def integrate(t, S, E1, E2, I1, I2, R1, R2, H1, H2, R12, beta, sigma, gamma, psi):
        
        # Calculate total population
        T = S+E1+I1+R1+H2+E2+I2+R2+H1+R12
        # Calculate differentials
        dS = -beta*((I1+H1) +(I2+H2))* S/T 
        dE1 = beta*((I1+H1))* S/T - 1/sigma*E1
        dE2 = beta*((I2+H2))* S/T - 1/sigma*E2
        dI1 = (1/sigma)*E1 - 1/gamma*I1
        dI2 = (1/sigma)*E2 - 1/gamma*I2
        dR1 = 1/gamma*I1 - psi*beta *(I2+H2) *R1/T
        dR2 = 1/gamma*I2 - psi*beta *(I1+H1) *R2/T
        dH1 = psi*beta *(I1+H1) *R2/T - gamma*H1
        dH2 = psi*beta *(I2+H2) *R1/T - gamma*H2
        dR12 = gamma(H1+H2)
        
        return dS, dE1, dE2, dI1, dI2, dR1, dR2, dH1, dH2, dR12

class JumpProcess_SEIR2_model(JumpProcess):
    """
    Stochastic SEIR2 model for DENV with age-groups and 2 serotypes - same beta for all ages
    """
    
    states = ['S','E1','E2','I1', 'I2', 'R1','R2','H1', 'H2','R12']
    parameters = ['beta','sigma', 'gamma', 'psi']
    dimensions = ['age_group']

    @staticmethod
    def compute_rates(t, S, E1, E2, I1, I2, R1, R2, H1, H2, R12, beta, sigma, gamma, psi):
        
        # Calculate total population
        T = S+E1+I1+R1+H2+E2+I2+R2+H1+R12
        # Compute rates per model state
        rates = {
            'S': [beta*((I1+H1)/T)*np.ones(T.shape), beta*((I2+H2)/T)*np.ones(T.shape)], # I think this multiplication with np.ones(T.shape is for the age-groups) but maybe this is not needed seeing as I1 and H1 will already have the correct shapes
            'E1': [1/sigma*np.ones(T.shape),],
            'E2': [1/sigma*np.ones(T.shape),],
            'I1': [1/gamma*np.ones(T.shape),],
            'I2': [1/gamma*np.ones(T.shape),],
            'R1': [psi*beta*(I2+H2)/T*np.ones(T.shape),],
            'R2': [psi*beta*(I1+H1)/T*np.ones(T.shape),],
            'H1': [1/gamma*np.ones(T.shape),],
            'H2': [1/gamma*np.ones(T.shape),]
        }
        
        return rates
    
    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, E1, E2, I1, I2, R1, R2, H1, H2, R12, beta, sigma, gamma, psi):

        S_new  = S - transitionings['S'][0] - transitionings['S'][1]
        E1_new = E1 + transitionings['S'][0] - transitionings['E1'][0]
        E2_new = E2 + transitionings['S'][1] - transitionings['E2'][0]
        I1_new = I1 + transitionings['E1'][0] - transitionings['I1'][0]
        I2_new = I2 + transitionings['E2'][0] - transitionings['I2'][0]
        R1_new = R1 + transitionings['I1'][0] - transitionings['R1'][0]
        R2_new = R2 + transitionings['I2'][0] - transitionings['R2'][0]
        H1_new = H1 + transitionings['R2'][0] - transitionings['H1'][0]
        H2_new = H2 + transitionings['R1'][0] - transitionings['H2'][0]
        R12_new = R12 + transitionings['H1'][0] + transitionings['H2'][0]
        
        return S_new, E1_new, E2_new, I1_new, I2_new, R1_new, R2_new, H1_new, H2_new, R12_new

class JumpProcess_SEIRH_BetaPerAge_model(JumpProcess):
    """
    Stochastic SEIR2 model for DENV with age-groups and 2 serotypes AND DIFFERENT BETA PARAMETERS PER AGE-GROUP
    """
    
    states = ['S','E1','E2','I1', 'I2', 'R1','R2','H1', 'H2','R12']
    parameters = ['sigma', 'gamma', 'psi']
    stratified_parameters = ['beta'] ## this is new
    dimensions = ['age_group']

    @staticmethod
    def compute_rates(t, S, E1, E2, I1, I2, R1, R2, H1, H2, R12, beta, sigma, gamma, psi):
        
        # Calculate total population
        T = S+E1+I1+R1+H2+E2+I2+R2+H1+R12
        # Compute rates per model state
        rates = {
            'S': [beta*((I1+H1)/T)*np.ones(T.shape), beta*((I2+H2)/T)*np.ones(T.shape)], # I think this multiplication with np.ones(T.shape is for the age-groups) but maybe this is not needed seeing as I1 and H1 will already have the correct shapes
            # does nothing change in the rate calculation for S? Does the stratified parameter automatically follow the age-groups?
            'E1': [1/sigma*np.ones(T.shape),],
            'E2': [1/sigma*np.ones(T.shape),],
            'I1': [1/gamma*np.ones(T.shape),],
            'I2': [1/gamma*np.ones(T.shape),],
            'R1': [psi*beta*(I2+H2)/T*np.ones(T.shape),],
            'R2': [psi*beta*(I1+H1)/T*np.ones(T.shape),],
            'H1': [1/gamma*np.ones(T.shape),],
            'H2': [1/gamma*np.ones(T.shape),]
        }
        
        return rates
    
    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, E1, E2, I1, I2, R1, R2, H1, H2, R12, beta, sigma, gamma, psi):

        S_new  = S - transitionings['S'][0] - transitionings['S'][1]
        E1_new = E1 + transitionings['S'][0] - transitionings['E1'][0]
        E2_new = E2 + transitionings['S'][1] - transitionings['E2'][0]
        I1_new = I1 + transitionings['E1'][0] - transitionings['I1'][0]
        I2_new = I2 + transitionings['E2'][0] - transitionings['I2'][0]
        R1_new = R1 + transitionings['I1'][0] - transitionings['R1'][0]
        R2_new = R2 + transitionings['I2'][0] - transitionings['R2'][0]
        H1_new = H1 + transitionings['R2'][0] - transitionings['H1'][0]
        H2_new = H2 + transitionings['R1'][0] - transitionings['H2'][0]
        R12_new = R12 + transitionings['H1'][0] + transitionings['H2'][0]
        
        return S_new, E1_new, E2_new, I1_new, I2_new, R1_new, R2_new, H1_new, H2_new, R12_new   

################################
# the TEMPORAL SEIR2 model + SF
################################

class JumpProcess_SEIR2_SeasonalForcing(JumpProcess):
    """
    Stochastic "real" SEIR2 model for DENV with 2 serotypes, no age-groups and + seasonal forcing.

    All the E, I, R compartments are available twice in the model to represent both serotypes
    The S and R12 are independent of serotype

    -- Parameters -- 
    alpha : temporary cross-immunity
    beta_0 : the baseline beta 
    beta_1 : additional infectivity due to seasonal forcing, which is dependent on time 
    sigma : incubation period
    gamma : infectious period
    psi : enhanced / inhibited infectiousness of secondary infection
    ph : the phase of the cos forcing curve, this is adjusted shift the curve to fit the DENV temporal patterns in Cuba 
    ---------------
    """
    
    states = ['S','S1', 'S2', 'E1', 'E2', 'E12', 'E21', 'I1', 'I2', 'I12', 'I21', 'R1', 'R2', 'R', 'I_new', 'I_cum']
    parameters = ['alpha', 'beta_0', 'beta_1', 'sigma', 'gamma', 'psi', 'ph'] # the beta_1 parameter is the same over all age-groups seeing as seasonality affects everyone the same

    @staticmethod
    def compute_rates(t, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, I_new, I_cum, beta_0, beta_1, alpha, sigma, gamma, psi, ph):
        
        # Calculate total population
        T = S+E1+I1+R1+S1+E12+I12+E2+I2+R2+S2+E21+I21+R

        # calculate the Beta_t
        #beta_t = beta_0 * (1 - beta_1 * np.cos(2*np.pi *(t/365) + ph)) # this should result in 4 values per timepoint (one for each age-group)
        beta_t = beta_0 * (1 - beta_1 * np.sin(2*np.pi *(t/365) + ph))

        # Compute rates per model state
        rates = {

            'S': [beta_t*((I1+ psi*I21)/T), beta_t*((I2+psi*I12)/T)], # I think this multiplication with np.ones(T.shape is for the age-groups) but maybe this is not needed seeing as I1 and H1 will already have the correct shapes
            'S1': [beta_t*((I2+psi*I12)/T),],
            'S2': [beta_t*((I1+ psi*I21)/T),],

            'E1': [1/sigma*np.ones(T.shape),],
            'E2': [1/sigma*np.ones(T.shape),],
            'E12': [1/sigma*np.ones(T.shape),],
            'E21': [1/sigma*np.ones(T.shape),], 

            'I1': [1/gamma*np.ones(T.shape),],
            'I2': [1/gamma*np.ones(T.shape),],
            'I12': [1/gamma*np.ones(T.shape),],
            'I21': [1/gamma*np.ones(T.shape),],

            'R1': [1/alpha*np.ones(T.shape),],
            'R2': [1/alpha*np.ones(T.shape),]
            }
        
        return rates
    
    @staticmethod
    def apply_transitionings(t, tau, transitionings, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, I_new, I_cum, beta_0, beta_1, alpha, sigma, gamma, psi, ph):

        S_new  = S - transitionings['S'][0] - transitionings['S'][1]
        E1_new = E1 + transitionings['S'][0] - transitionings['E1'][0]
        E2_new = E2 + transitionings['S'][1] - transitionings['E2'][0]
        I1_new = I1 + transitionings['E1'][0] - transitionings['I1'][0]
        I2_new = I2 + transitionings['E2'][0] - transitionings['I2'][0]
        R1_new = R1 + transitionings['I1'][0] - transitionings['R1'][0]
        R2_new = R2 + transitionings['I2'][0] - transitionings['R2'][0]

        S1_new = S1 + transitionings['R1'][0] - transitionings['S1'][0]
        S2_new = S2 + transitionings['R2'][0] - transitionings['S2'][0]       
        E12_new = E12 + transitionings['S1'][0] - transitionings['E12'][0]
        E21_new = E21 + transitionings['S2'][0] - transitionings['E21'][0]
        I12_new = I12 + transitionings['E12'][0] - transitionings['I12'][0]
        I21_new = I21 + transitionings['E21'][0] - transitionings['I21'][0]        
        R_new = R + transitionings['I12'][0] + transitionings['I21'][0]

        # derivative state: 
        I_new_new = transitionings['E1'][0] + transitionings['E2'][0] + transitionings['E12'][0] + transitionings['E12'][0]
        I_cum_new = I_cum + I_new
        
        return S_new, S1_new, S2_new, E1_new, E2_new, E12_new, E21_new, I1_new, I2_new, I12_new,I21_new,  R1_new, R2_new,   R_new,I_new_new, I_cum_new
    

class JumpProcess_SEIRH_BetaPerAge_SeasonalForcing(JumpProcess):
    """
    Stochastic SEIR2 model for DENV with 2 serotypes, baseline beta values per age-group and seasonal forcing.

    Beta_0 represents the baseline beta value with is specific to the age-group
    Beta_1 represents the additional infectivity due to seasonal forcing, which is dependent on time 
    The ph parameter represents the phase of the cos forcing curve, this is adjusted shift the curve to fit the DENV temporal patterns in Cuba
    All the E, I, R, and H compartments are available twice in the model to represent both serotypes
    The S and R12 are independent of serotype
    """
    
    states = ['S','E1','E2','I1', 'I2', 'R1','R2','H1', 'H2','R12']
    parameters = ['beta_1', 'sigma', 'gamma', 'psi', 'ph'] # the beta_1 parameter is the same over all age-groups seeing as seasonality affects everyone the same
    stratified_parameters = ['beta_0'] ## this parameter is stratified per age group
    dimensions = ['age_group']

    @staticmethod
    def compute_rates(t, S, E1, E2, I1, I2, R1, R2, H1, H2, R12, beta_0, beta_1, sigma, gamma, psi, ph):
        
        # Calculate total population
        T = S+E1+I1+R1+H2+E2+I2+R2+H1+R12

        # calculate the Beta_t
        #beta_t = beta_0 * (1 - beta_1 * np.cos(2*np.pi *(t/365) + ph)) # this should result in 4 values per timepoint (one for each age-group)
        beta_t = beta_0 * (1 - beta_1 * np.sin(2*np.pi *(t/365) + ph))

        # Compute rates per model state
        rates = {
            'S': [beta_t*((I1+H1)/T)*np.ones(T.shape), beta_t*((I2+H2)/T)*np.ones(T.shape)], # I think this multiplication with np.ones(T.shape is for the age-groups) but maybe this is not needed seeing as I1 and H1 will already have the correct shapes
            # does nothing change in the rate calculation for S? Does the stratified parameter automatically follow the age-groups?
            'E1': [1/sigma*np.ones(T.shape),],
            'E2': [1/sigma*np.ones(T.shape),],
            'I1': [1/gamma*np.ones(T.shape),],
            'I2': [1/gamma*np.ones(T.shape),],
            'R1': [psi*beta_t*(I2+H2)/T*np.ones(T.shape),],
            'R2': [psi*beta_t*(I1+H1)/T*np.ones(T.shape),],
            'H1': [1/gamma*np.ones(T.shape),],
            'H2': [1/gamma*np.ones(T.shape),]
        }
        
        return rates
    
    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, E1, E2, I1, I2, R1, R2, H1, H2, R12, beta_0, beta_1, sigma, gamma, psi, ph):

        S_new  = S - transitionings['S'][0] - transitionings['S'][1]
        E1_new = E1 + transitionings['S'][0] - transitionings['E1'][0]
        E2_new = E2 + transitionings['S'][1] - transitionings['E2'][0]
        I1_new = I1 + transitionings['E1'][0] - transitionings['I1'][0]
        I2_new = I2 + transitionings['E2'][0] - transitionings['I2'][0]
        R1_new = R1 + transitionings['I1'][0] - transitionings['R1'][0]
        R2_new = R2 + transitionings['I2'][0] - transitionings['R2'][0]
        H1_new = H1 + transitionings['R2'][0] - transitionings['H1'][0]
        H2_new = H2 + transitionings['R1'][0] - transitionings['H2'][0]
        R12_new = R12 + transitionings['H1'][0] + transitionings['H2'][0]
        
        return S_new, E1_new, E2_new, I1_new, I2_new, R1_new, R2_new, H1_new, H2_new, R12_new
    

    ################################
    # the real SEIR2 model 
    ###############################

class JumpProcess_SEIR2_BetaPerAge_SeasonalForcing(JumpProcess):
    """
    Stochastic "real" SEIR2 model for DENV with 2 serotypes, baseline beta values per age-group and seasonal forcing.

    Beta_0 represents the baseline beta value with is specific to the age-group
    Beta_1 represents the additional infectivity due to seasonal forcing, which is dependent on time 
    The ph parameter represents the phase of the cos forcing curve, this is adjusted shift the curve to fit the DENV temporal patterns in Cuba
    All the E, I, R, and H compartments are available twice in the model to represent both serotypes
    The S and R12 are independent of serotype

    Instead of an H compartment, we make a new SEIR after first infection 
    """
    
    states = ['S','S1', 'S2', 'E1', 'E2', 'E12', 'E21', 'I1', 'I2', 'I12', 'I21', 'R1', 'R2', 'R']
    parameters = ['alpha', 'beta_1', 'sigma', 'gamma', 'psi', 'ph'] # the beta_1 parameter is the same over all age-groups seeing as seasonality affects everyone the same
    stratified_parameters = ['beta_0'] ## this parameter is stratified per age group
    dimensions = ['age_group']

    @staticmethod
    def compute_rates(t, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, beta_0, beta_1, alpha, sigma, gamma, psi, ph):
        
        # Calculate total population
        T = S+E1+I1+R1+S1+E12+I12+E2+I2+R2+S2+E21+I21+R

        # calculate the Beta_t
        #beta_t = beta_0 * (1 - beta_1 * np.cos(2*np.pi *(t/365) + ph)) # this should result in 4 values per timepoint (one for each age-group)
        beta_t = beta_0 * (1 - beta_1 * np.sin(2*np.pi *(t/365) + ph))

        # Compute rates per model state
        rates = {

            'S': [beta_t*((I1+ psi*I21)/T)*np.ones(T.shape), beta_t*((I2+psi*I12)/T)*np.ones(T.shape)], # I think this multiplication with np.ones(T.shape is for the age-groups) but maybe this is not needed seeing as I1 and H1 will already have the correct shapes
            # does nothing change in the rate calculation for S? Does the stratified parameter automatically follow the age-groups?
            'S1': [beta_t*((I2+psi*I12)/T)*np.ones(T.shape),],
            'S2': [beta_t*((I1+ psi*I21)/T)*np.ones(T.shape),],

            'E1': [1/sigma*np.ones(T.shape),],
            'E2': [1/sigma*np.ones(T.shape),],
            'E12': [1/sigma*np.ones(T.shape),],
            'E21': [1/sigma*np.ones(T.shape),],     

            'I1': [1/gamma*np.ones(T.shape),],
            'I2': [1/gamma*np.ones(T.shape),],
            'I12': [1/gamma*np.ones(T.shape),],
            'I21': [1/gamma*np.ones(T.shape),],

            'R1': [1/alpha*np.ones(T.shape),],
            'R2': [1/alpha*np.ones(T.shape),]
            }
        
        return rates
    
    @staticmethod
    def apply_transitionings(t, tau, transitionings, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, beta_0, beta_1, alpha, sigma, gamma, psi, ph):

        S_new  = S - transitionings['S'][0] - transitionings['S'][1]
        E1_new = E1 + transitionings['S'][0] - transitionings['E1'][0]
        E2_new = E2 + transitionings['S'][1] - transitionings['E2'][0]
        I1_new = I1 + transitionings['E1'][0] - transitionings['I1'][0]
        I2_new = I2 + transitionings['E2'][0] - transitionings['I2'][0]
        R1_new = R1 + transitionings['I1'][0] - transitionings['R1'][0]
        R2_new = R2 + transitionings['I2'][0] - transitionings['R2'][0]

        S1_new = S1 + transitionings['R1'][0] - transitionings['S1'][0]
        S2_new = S2 + transitionings['R2'][0] - transitionings['S2'][0]       
        E12_new = E12 + transitionings['S1'][0] - transitionings['E12'][0]
        E21_new = E21 + transitionings['S2'][0] - transitionings['E21'][0]
        I12_new = I12 + transitionings['E12'][0] - transitionings['I12'][0]
        I21_new = I21 + transitionings['E21'][0] - transitionings['I21'][0]        
        R_new = R + transitionings['I12'][0] + transitionings['I21'][0]
        
        return S_new, S1_new, S2_new, E1_new, E2_new, E12_new, E21_new, I1_new, I2_new, I12_new,I21_new,  R1_new, R2_new,   R_new


    ################################
    # the SPATIAL SEIR2 model 
    ###############################

class JumpProcess_SEIR2_spatial_stochastic(JumpProcess):
    """
    Stochastic "real" SEIR2 model for DENV with 2 serotypes, baseline beta values per age-group and seasonal forcing.

    Beta_0 represents the baseline beta value with is specific to the age-group
    Beta_1 represents the additional infectivity due to seasonal forcing, which is dependent on time 
    The ph parameter represents the phase of the cos forcing curve, this is adjusted shift the curve to fit the DENV temporal patterns in Cuba
    All the E, I, R, and H compartments are available twice in the model to represent both serotypes
    The S and R12 are independent of serotype
    I_cum represents the cummulate amount of infectious people (I1, I2, I12, and I21) over up untill timepoint t

    Instead of an H compartment, we make a new SEIR after first infection 
    """
    
    states = ['S','S1', 'S2', 'E1', 'E2', 'E12', 'E21', 'I1', 'I2', 'I12', 'I21', 'R1', 'R2', 'R', 'I_cum']
    parameters = ['alpha', 'beta_1', 'sigma', 'gamma', 'psi', 'ph', 'ODmatrix'] # the beta_1 parameter is the same over all age-groups seeing as seasonality affects everyone the same
    stratified_parameters = [[],['beta_0']] ## beta_0 is stratified per age group
    dimensions = ['NIS','age_group']

    @staticmethod
    def compute_rates(t, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, I_cum, #time + SEIR2 states
                      beta_1, alpha, sigma, gamma, psi, ph, #SEIR2 parameters
                      beta_0, ODmatrix): # age-stratified parameter (OD matrix is not age-stratified yet)
        
        # calculate the Beta_t
        #beta_t = beta_0 * (1 - beta_1 * np.cos(2*np.pi *(t/365) + ph)) # this should result in 4 values per timepoint (one for each age-group)
        beta_t = beta_0 * (1 - beta_1 * np.sin(2*np.pi *(t/365) + ph))

        #################################
        # Calculate total population
        #################################

        T = S+E1+I1+R1+S1+E12+I12+E2+I2+R2+S2+E21+I21+R # this should result in a total population per location, haven't checked this though

        #################################
        # COMPUTE INFECIOUS PRESSURE ####
        #################################

        # for the total population and for the relevant compartments I1, I2, I12, I21
        G = S.shape[0] #spatial stratification
        N = S.shape[1] #age stratification

        # compute populations after travel (ODmatrix)
        ODmatrixT_ndarray = np.transpose(ODmatrix).values
        T_mob = ODmatrixT_ndarray @ T
        I1_mob = ODmatrixT_ndarray @ I1
        I2_mob = ODmatrixT_ndarray @ I2
        I12_mob = ODmatrixT_ndarray @ I12
        I21_mob = ODmatrixT_ndarray @ I21

        # Compute the infectious population: 
        infpop_mob = (I1_mob + I2_mob +I12_mob + I21_mob)/T_mob


        # Compute rates per model state
        size_dummy =  np.ones([G,N], np.float64)
        rates = {

            'S': [beta_t*((I1_mob+ psi*I12_mob)/T_mob)*np.ones(T.shape[1]), beta_t*((I2_mob+psi*I12_mob)/T_mob)*np.ones(T.shape[1])], # 
            'S1': [beta_t*((I2+psi*I12)/T_mob)*np.ones(T.shape[1]),],
            'S2': [beta_t*((I1+ psi*I21)/T_mob)*np.ones(T.shape[1]),],

            'E1': [(1/sigma)*size_dummy,],
            'E2': [(1/sigma)*size_dummy,],
            'E12': [(1/sigma)*size_dummy,],
            'E21': [(1/sigma)*size_dummy,],     

            'I1': [(1/gamma)*size_dummy,],
            'I2': [(1/gamma)*size_dummy,],
            'I12': [(1/gamma)*size_dummy,],
            'I21': [(1/gamma)*size_dummy,],

            'R1': [(1/alpha)*size_dummy,],
            'R2': [(1/alpha)*size_dummy,]
            }
        
        return rates
    
    @staticmethod
    def apply_transitionings(t, tau, transitionings, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, I_cum,  #time + SEIR2 states
                      beta_1, alpha, sigma, gamma, psi, ph, #SEIR2 parameters
                      beta_0, ODmatrix): # age-stratified parameter
        
        S_new  = S - transitionings['S'][0] - transitionings['S'][1]
        E1_new = E1 + transitionings['S'][0] - transitionings['E1'][0]
        E2_new = E2 + transitionings['S'][1] - transitionings['E2'][0]
        I1_new = I1 + transitionings['E1'][0] - transitionings['I1'][0]
        I2_new = I2 + transitionings['E2'][0] - transitionings['I2'][0]
        R1_new = R1 + transitionings['I1'][0] - transitionings['R1'][0]
        R2_new = R2 + transitionings['I2'][0] - transitionings['R2'][0]

        S1_new = S1 + transitionings['R1'][0] - transitionings['S1'][0]
        S2_new = S2 + transitionings['R2'][0] - transitionings['S2'][0]       
        E12_new = E12 + transitionings['S1'][0] - transitionings['E12'][0]
        E21_new = E21 + transitionings['S2'][0] - transitionings['E21'][0]
        I12_new = I12 + transitionings['E12'][0] - transitionings['I12'][0]
        I21_new = I21 + transitionings['E21'][0] - transitionings['I21'][0]        
        R_new = R + transitionings['I12'][0] + transitionings['I21'][0]

        # derivative state: 
        I_cum_new = I_cum + transitionings['E1'][0] + transitionings['E2'][0] + transitionings['E12'][0] + transitionings['E12'][0]
        
        return S_new, S1_new, S2_new, E1_new, E2_new, E12_new, E21_new, I1_new, I2_new, I12_new,I21_new,  R1_new, R2_new,   R_new, I_cum_new
    

################################################################
# the SPATIAL SEIR2 model - NO AGES - not finished yet - OLD/ MANUAL APPROACH
###############################################################

class JumpProcess_SEIR2_spatial_stochastic_noAges(JumpProcess):
    """
    Stochastic "real" SEIR2 model for DENV with 2 serotypes, no ages and with seasonal forcing.

    Beta_0 represents the baseline beta value
    Beta_1 represents the additional infectivity due to seasonal forcing, which is dependent on time 
    The ph parameter represents the phase of the cos forcing curve, this is adjusted shift the curve to fit the DENV temporal patterns in Cuba
    f_h represents the fraction of contacts made at home (vs non-home)
    All the E, I, R compartments are available twice in the model to represent both serotypes
    The S and R12 are independent of serotype
    I_cum represents the cummulate amount of infectious people (I1, I2, I12, and I21) over up untill timepoint t

    This function has an attempt at sending people back home post infection. NEEDS TO BE FINALIZED
    """
        
    states = ['S','S1', 'S2', 'E1', 'E2', 'E12', 'E21', 'I1', 'I2', 'I12', 'I21', 'R1', 'R2', 'R', 'I_cum']
    parameters = ['alpha', 'beta_0', 'beta_1', 'sigma', 'gamma', 'psi', 'ph', 'ODmatrix', 'f_h'] 
    dimensions = ['NIS']

    @staticmethod
    def compute_rates(t, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, I_cum, #time + SEIR2 states
                      beta_0, beta_1, alpha, sigma, gamma, psi, ph, #SEIR2 parameters
                      ODmatrix, f_h): # dimensions
        
        # calculate the Beta_t
        beta_t = beta_0 * (1 - beta_1 * np.sin(2*np.pi *(t/365) + ph))

        #################################
        # Calculate total population
        #################################

        T = S+E1+I1+R1+S1+E12+I12+E2+I2+R2+S2+E21+I21+R # this should result in a total population per location, haven't checked this though

        #################################
        # COMPUTE INFECIOUS PRESSURE ####
        #################################

        # for the total population and for the relevant compartments I1, I2, I12, I21

        # compute populations after travel (ODmatrix)
        ODmatrixT_ndarray = np.transpose(ODmatrix).values
        T_mob = ODmatrixT_ndarray @ T
        I1_mob = ODmatrixT_ndarray @ I1
        I2_mob = ODmatrixT_ndarray @ I2
        I12_mob = ODmatrixT_ndarray @ I12
        I21_mob = ODmatrixT_ndarray @ I21

        #####################################################
        # Compute the infectious presssure at home and at work: 
        #####################################################
        IP_home = [f_h*beta_t * ((I1 + psi*I12)/T)*np.ones(T.shape), f_h*beta_t * ((I2 + psi*I21)/T)*np.ones(T.shape)]
        IP_nonhome = [(1-f_h)*beta_t*((I1_mob + psi*I21_mob)/T_mob)*np.ones(T.shape), (1-f_h)*beta_t*((I2_mob+psi*I12_mob)/T_mob)*np.ones(T.shape)]

        #####################################################
        # Compute the fraction of susceptibles from each home location: 
        #####################################################
        S_mob = ODmatrixT_ndarray @ S

        # this represents the fraction of susceptibles from each home location (i) at each destination (j). It's calculated by OD * S / S_mob
        # the where(S_mob>0) is added to avoid dividing by 0
        S_fraction = np.divide(ODmatrix * S[:, np.newaxis], S_mob, where=(S_mob > 0))  # gives a locations x locations matrix 

        #####################################################
        # Redistribute infections from non-home locations back to home locations- this is not yet in the correct location of the code...
        #####################################################
        infections_to_home = S_fraction @ np.array(IP_nonhome).sum(axis=0)  # Total infections at non-home locations, summed over home locations


        # Compute rates per model state
        rates = {

            # 'S': [beta_t*((I1_mob + psi*I12_mob)/T_mob)*np.ones(T.shape), beta_t*((I2_mob+psi*I12_mob)/T_mob)*np.ones(T.shape)], # I removed the T.shape[1], because there is only one dimension now
            'S': [IP_home[0] + IP_nonhome[0],  # Add home and non-home infectious pressure 
              IP_home[1] + IP_nonhome[1]],  # Same for the second serotype
            
            'S1': [IP_home[0] + IP_nonhome[0],],
            'S2': [IP_home[1] + IP_nonhome[1],],

            'E1': [(1/sigma)*np.ones(T.shape),],
            'E2': [(1/sigma)*np.ones(T.shape),],
            'E12': [(1/sigma)*np.ones(T.shape),],
            'E21': [(1/sigma)*np.ones(T.shape),],     

            'I1': [(1/gamma)*np.ones(T.shape),],
            'I2': [(1/gamma)*np.ones(T.shape),],
            'I12': [(1/gamma)*np.ones(T.shape),],
            'I21': [(1/gamma)*np.ones(T.shape),],

            'R1': [(1/alpha)*np.ones(T.shape),],
            'R2': [(1/alpha)*np.ones(T.shape),]
            }
        
        ##################################################### 
        # redistribute those infected at the nonhome location 
        #####################################################

        # Calculate infections_home based on S_fraction and IP_nonhome - this is a 1 x locations that shows the number of infections in the nonhome location redistributed to their home locations based on the fraction of S present from each home at each nonhome location
        infections_home = S_fraction @ np.array(IP_nonhome).sum(axis=0)

        return rates
    
    @staticmethod
    def apply_transitionings(t, tau, transitionings, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, I_cum,  #time + SEIR2 states
                             beta_0, beta_1, alpha, sigma, gamma, psi, ph, #SEIR2 parameters
                             ODmatrix, f_h): # dimensions 
        
        S_new  = S - transitionings['S'][0] - transitionings['S'][1]
        E1_new = E1 + transitionings['S'][0] - transitionings['E1'][0]
        E2_new = E2 + transitionings['S'][1] - transitionings['E2'][0]
        I1_new = I1 + transitionings['E1'][0] - transitionings['I1'][0]
        I2_new = I2 + transitionings['E2'][0] - transitionings['I2'][0]
        R1_new = R1 + transitionings['I1'][0] - transitionings['R1'][0]
        R2_new = R2 + transitionings['I2'][0] - transitionings['R2'][0]

        S1_new = S1 + transitionings['R1'][0] - transitionings['S1'][0]
        S2_new = S2 + transitionings['R2'][0] - transitionings['S2'][0]       
        E12_new = E12 + transitionings['S1'][0] - transitionings['E12'][0]
        E21_new = E21 + transitionings['S2'][0] - transitionings['E21'][0]
        I12_new = I12 + transitionings['E12'][0] - transitionings['I12'][0]
        I21_new = I21 + transitionings['E21'][0] - transitionings['I21'][0]        
        R_new = R + transitionings['I12'][0] + transitionings['I21'][0]

        # derivative state: 
        I_cum_new = I_cum + transitionings['E1'][0] + transitionings['E2'][0] + transitionings['E12'][0] + transitionings['E12'][0]
        
        return S_new, S1_new, S2_new, E1_new, E2_new, E12_new, E21_new, I1_new, I2_new, I12_new,I21_new,  R1_new, R2_new,   R_new, I_cum_new
    

################################################################
# the SPATIAL SEIR2 model - WITH AGES - not finished yet - new EINSTEIN APPROACH
###############################################################

class JumpProcess_SEIR2_spatial_age_sto_einstein(JumpProcess):
    """
    Stochastic "real" SEIR2 model for DENV with 2 serotypes, ages, and seasonal forcing.

    The idea of this model is that we can test whether the dimensions are expandable - so whether the model will work with 1 or 5 age groups, 1 location or 50, and so on. 

    Beta_0 represents the baseline beta value with is specific to the age-group
    Beta_1 represents the additional infectivity due to seasonal forcing, which is dependent on time 
    The ph parameter represents the phase of the cos forcing curve, this is adjusted shift the curve to fit the DENV temporal patterns in Cuba
    All the E, I, and R compartments are available twice in the model to represent both serotypes
    The S and R12 are independent of serotype
    I_cum represents the cummulate amount of infectious people (I1, I2, I12, and I21) over up untill timepoint t

    The N parameter is the contact matrix (which could potentially also vary per location) - but not in this situation
    f_h represents the fraction of daily contacts made at home. The reverse (1-f_h) is the fraction of nonhome contacts. 

    The dimensions of the initial conditions should be 
    - for states: age x locations
    - for the ODmatrix: age x departure location x destination location
    - for the contact matrix N : age x age x location
    IF MOBILITY OR CONTACT DON'T DIFFER PER LOCATION OR AGE, JUST REPLICATE YOUR UNIQUE MATRIX TO ALIGN TO THESE DIMENSIONS
    """
    
    states = ['S','S1', 'S2', 'E1', 'E2', 'E12', 'E21', 'I1', 'I2', 'I12', 'I21', 'R1', 'R2', 'R', 'I_cum']
    parameters = ['alpha', 'beta_1', 'sigma', 'gamma', 'psi', 'ph', 'f_h', 'beta_0', 'ODmatrix', 'N'] # the beta_1 parameter is the same over all age-groups seeing as seasonality affects everyone the same
    # stratified_parameters = [['beta_0', 'ODmatrix'], ['N']] ## beta_0 is stratified per age group, along with ODmatrix. N is stratified by location. The parameter in the stratified_parameters don't need to be mentioned in parameters = 
    dimensions = ['age_group','location']

    @staticmethod
    def compute_rates(t, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, I_cum, #time + SEIR2 states
                      beta_1, alpha, sigma, gamma, psi, ph, #SEIR2 parameters
                      beta_0, ODmatrix, f_h, N): # 
        
        # calculate the Beta_t
        #beta_t = beta_0 * (1 - beta_1 * np.cos(2*np.pi *(t/365) + ph)) # this should result in 4 values per timepoint (one for each age-group)
        beta_t = beta_0 * (1 - beta_1 * np.sin(2*np.pi *(t/365) + ph))

        print("ODmatrix", ODmatrix)
        #################################
        # Calculate total population
        #################################

        T = S+E1+I1+R1+S1+E12+I12+E2+I2+R2+S2+E21+I21+R # this should result in a total population per location, haven't checked this though

        # # for the total population and for the relevant compartments I1, I2, I12, I21
        NS = S.shape[0] #age stratification (number of agegroups) 
        NA = S.shape[1] #spatial stratification (number of shapes)

        #################################
        # Calculate mobile population
        #################################

        # Perform the multiplication: summing over the departure locations (axis 1)
        # 'ajk,aj->ak' means:
        # - 'a' is the age group,
        # - 'j' is the departure location we sum over,
        # - 'k' is the destination location.
        I1_mob = np.einsum('jla,aj->al', ODmatrix, I1) # resulting is ages x destination location
        I2_mob = np.einsum('jla,aj->al', ODmatrix, I2)
        I12_mob = np.einsum('jla,aj->al', ODmatrix, I12)
        I21_mob = np.einsum('jla,aj->al', ODmatrix, I21)
        T_mob = np.einsum('jla,aj->al', ODmatrix, T)

        #################################
        # COMPUTE Force of Infection #### This is written to have different OD matrix per age-group
        #################################

        # Perform the multiplication: summing over the age groups from the infectious - result is number of interactions with susceptible age groups x locations
        # 'kj,ikj->ij' means:
        # - 'a' is the age group (infectious) we sum over,
        # - 'i' is the age group (susceptibles) 
        # - 'l' is the destination location 
        lambda_home_sero1 = beta_t[:, np.newaxis] * (f_h) * np.einsum('il,ail->al', (I1+ psi*I21)/T, N)  ## we need the [:, np.newaxis] because otherwise we are multiplying a beta_1 size (2,) with (2,3) 
        lambda_home_sero2 = beta_t[:, np.newaxis]  * (f_h) * np.einsum('il,ail-> al', (I2+ psi*I12)/T, N)

        # Perform the multiplication: summing over the age groups from the infectious and the departure locations - result is number of interactions with suscpetible age groups x locations - 
        # 'kj,ikj->ij' means:
        # - 'a' is the age group (susceptibles)
        # - 'i' is the age group (infectious) we sum over
        # - 'j' is the origin location 
        # - 'l' is destination locations we sum over,
        lambda_visit_sero1 = beta_t[:, np.newaxis] * (1-f_h) * np.einsum ('jla, il, ail -> aj' , ODmatrix , (I1_mob + psi*I21_mob)/T_mob , N)
        lambda_visit_sero2 = beta_t[:, np.newaxis] * (1-f_h) * np.einsum ('jla, il, ail -> aj' , ODmatrix , (I2_mob + psi*I12_mob)/T_mob , N)


        #################################
        # Compute rates per model state
        #################################
        size_dummy =  np.ones([NS,NA], np.float64) # age x space
        rates = {

            'S': [lambda_home_sero1 + lambda_visit_sero1, lambda_home_sero2 + lambda_visit_sero2], #combo of the FOI of home and visit per serotype

            'S1': [lambda_home_sero1 + lambda_visit_sero1,],
            'S2': [lambda_home_sero2 + lambda_visit_sero2,],

            'E1': [(1/sigma)*size_dummy,],
            'E2': [(1/sigma)*size_dummy,],
            'E12': [(1/sigma)*size_dummy,],
            'E21': [(1/sigma)*size_dummy,],     

            'I1': [(1/gamma)*size_dummy,],
            'I2': [(1/gamma)*size_dummy,],
            'I12': [(1/gamma)*size_dummy,],
            'I21': [(1/gamma)*size_dummy,],

            'R1': [(1/alpha)*size_dummy,],
            'R2': [(1/alpha)*size_dummy,]
            }
        
        return rates
    
    @staticmethod
    def apply_transitionings(t, tau, transitionings, S,S1, S2, E1, E2, E12, E21, I1, I2, I12, I21, R1, R2, R, I_cum, #time + SEIR2 states
                      beta_1, alpha, sigma, gamma, psi, ph, #SEIR2 parameters
                      beta_0, ODmatrix, f_h, N): # 
        
        S_new  = S - transitionings['S'][0] - transitionings['S'][1]
        E1_new = E1 + transitionings['S'][0] - transitionings['E1'][0]
        E2_new = E2 + transitionings['S'][1] - transitionings['E2'][0]
        I1_new = I1 + transitionings['E1'][0] - transitionings['I1'][0]
        I2_new = I2 + transitionings['E2'][0] - transitionings['I2'][0]
        R1_new = R1 + transitionings['I1'][0] - transitionings['R1'][0]
        R2_new = R2 + transitionings['I2'][0] - transitionings['R2'][0]

        S1_new = S1 + transitionings['R1'][0] - transitionings['S1'][0]
        S2_new = S2 + transitionings['R2'][0] - transitionings['S2'][0]       
        E12_new = E12 + transitionings['S1'][0] - transitionings['E12'][0]
        E21_new = E21 + transitionings['S2'][0] - transitionings['E21'][0]
        I12_new = I12 + transitionings['E12'][0] - transitionings['I12'][0]
        I21_new = I21 + transitionings['E21'][0] - transitionings['I21'][0]        
        R_new = R + transitionings['I12'][0] + transitionings['I21'][0]

        # derivative state: 
        I_cum_new = I_cum + transitionings['E1'][0] + transitionings['E2'][0] + transitionings['E12'][0] + transitionings['E12'][0]
        
        return S_new, S1_new, S2_new, E1_new, E2_new, E12_new, E21_new, I1_new, I2_new, I12_new,I21_new,  R1_new, R2_new,   R_new, I_cum_new