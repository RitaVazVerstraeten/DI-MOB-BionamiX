# file to create time-dependent parameters pySODM
import pandas as pd

class time_dependent_beta:
    def __init__(self, sf): 
        """ 
        Initializing with a scaling factor dataframe
        """
        self.sf = sf
        # print(f"[DEBUG] Intialized time_dependent_beta with sf:\n{self.sf}") # debuggin 

    def __call__(self, t, states, param):
        """
        Re-defines beta_0 as the scaled beta(t) = beta_0 * scaling_factor, where the scaling factor varies according to temperature and precipitation. 

        input
        ------
        t: datetime.datetime or integer
            time in the simulation

        states: dict
            model states on time 't'

        param: dict
            original value of model parameter to modify
        
        sf: pandas.core.dataframe
            date-indexed dataframe of scaling factors 

            
        output
        ------
        beta(t): np.ndarray
            time-varying transmission rate     
        """
        # print(f"[DEBUG] __call__ triggered! with t={t}, states={states}, param={param}")

        if t in self.sf.index:
            sf_t = self.sf.loc[t, 'sf']
        else:
            sf_t = 1  # Default if t is out of bounds
            print(f"[WARNING] t={t} is not in sf index. Using default sf=1.")


        # beta_0 = param
        # print("\ninside time_dependent_beta function")
        # beta_t = beta_0 * self.sf.loc[t, 'sf']
        # print("t:", t, "sf:", self.sf.loc[t, 'sf'])
        # print("inside time_dependent_beta function, beta_t: ", beta_t)
        beta_t = param*sf_t
        
        # print(f"[DEBUG] Computed beta_t: beta_0 ({param}) * sf ({sf_t}) = {beta_t}")

        return beta_t