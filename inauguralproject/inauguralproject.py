from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self, alpha=0.5, sigma=1, wM=1.0, wF=1.0, wF_vec=np.linspace(0.8,1.2,5)):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = alpha
        par.sigma = sigma

        # d. wages
        par.wM = wM
        par.wF = wF
        par.wF_vec = wF_vec

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility returns net utility. (utility - disutility) for given amounts of time spend working."""

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
#            print(f"sigma 0: {par.sigma}")
            H = np.min(HM, HF)
        if par.sigma == 1:
#            print(f"sigma 1: {par.sigma}")
            H = HM**(1-par.alpha)*HF**par.alpha
        if par.sigma!=0 and par.sigma!=1:
#            print(f"sigma ikke 0 eller 1: {par.sigma}")
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # defines the utility function as the objective fuction to be minimized
        obj = lambda x: -self.calc_utility(LM=x[0],HM=x[1],LF=x[2],HF=x[3])
        bounds      = ((0,24),(0,24),(0,24),(0,24))
        constraints = ({'type':'ineq','fun':lambda x: 24-x[0]-x[1]}, {'type':'ineq','fun':lambda x: 24-x[2]-x[3]})
        res = optimize.minimize(obj, x0= [1,1,1,1], method='SLSQP', 
                                bounds=bounds, constraints=constraints)


        return res    

    def solve_wF_vec(self, alpha=0.5,sigma=1):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol

        #apply the alpha and sigma parameters
        par.alpha = alpha
        par.sigma = sigma

        # defines the utility function as the objective fuction to be minimized
        bounds      = ((0,24),(0,24),(0,24),(0,24))
        constraints = ({'type':'ineq','fun':lambda x: 24-x[0]-x[1]}, {'type':'ineq','fun':lambda x: 24-x[2]-x[3]})
        j = 0

        for wf in par.wF_vec:
                par.wF = wf
                obj = lambda x: -self.calc_utility(LM=x[0],HM=x[1],LF=x[2],HF=x[3])

                res = optimize.minimize(obj, x0= [1,1,1,1], method='SLSQP', 
                                        bounds=bounds, constraints=constraints)
                
                sol.LM_vec[j] = res.x[0]
                sol.HM_vec[j] = res.x[1]
                sol.LF_vec[j] = res.x[2]
                sol.HF_vec[j] = res.x[3]
                j = j +1

        # function only returns values of log(Hf/Hm) and log(wf)
        l_hf_hm = np.log(sol.HF_vec/sol.HM_vec)
        l_wf_wm = np.log(par.wF_vec)
        return l_hf_hm, l_wf_wm

    def run_regression(self, a=0.5,s=1):
        """ run regression """

        par = self.par
        sol = self.sol

        # the model is solved for different values of wf
        self.solve_wF_vec(alpha=a,sigma=s)

        # the model betas are calculated
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol.beta0, sol.beta1
    
    def estimate(self):
        """ estimate alpha and sigma """
        # i write the complicated lambda function using the run regression function.
        obj = lambda x: (0.4-self.run_regression(a=x[0],s=x[1])[0])**2+(-0.1-self.run_regression(a=x[0],s=x[1])[1])**2
        res = optimize.minimize(obj, x0= [0.5,0.5], method='Nelder-Mead')
        
        return res








class NewModelClass:

    def __init__(self, alpha=0.5, sigma=1, wM=1.0, wF=1.0, wF_vec=np.linspace(0.8,1.2,5)):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = alpha
        par.sigma = sigma

        # d. wages
        par.wM = wM
        par.wF = wF
        par.wF_vec = wF_vec

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility returns net utility. (utility - disutility) for given amounts of time spend working."""

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
#            print(f"sigma 0: {par.sigma}")
            H = np.min(HM, HF)
        if par.sigma == 1:
#            print(f"sigma 1: {par.sigma}")
            H = HM**(1-par.alpha)*HF**par.alpha
        if par.sigma!=0.0000 and par.sigma!=1.000:
#            print(f"sigma ikke 0 eller 1: {par.sigma}")
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt


    def solve(self):
        """ solve model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # defines the utility function as the objective fuction to be minimized
        obj = lambda x: -self.calc_utility(LM=x[0],HM=x[1],LF=x[2],HF=x[3])
        bounds      = ((0,24),(0,24),(0,24),(0,24))
        constraints = ({'type':'ineq','fun':lambda x: 24-x[0]-x[1]}, {'type':'ineq','fun':lambda x: 24-x[2]-x[3]}, {'type':'ineq','fun':lambda x: 8-x[2]})
        res = optimize.minimize(obj, x0= [1,1,1,1], method='SLSQP', 
                                bounds=bounds, constraints=constraints)


        return res    

    def solve_wF_vec(self, alpha=0.5,sigma=1):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol

        #apply the alpha and sigma parameters
        par.alpha = alpha
        par.sigma = sigma

        # defines the utility function as the objective fuction to be minimized
        bounds      = ((0,24),(0,24),(0,24),(0,24))
        constraints = ({'type':'ineq','fun':lambda x: 24-x[0]-x[1]}, {'type':'ineq','fun':lambda x: 24-x[2]-x[3]}, {'type':'ineq','fun':lambda x: 4-x[2]})
        j = 0

        for wf in par.wF_vec:
                par.wF = wf
                obj = lambda x: -self.calc_utility(LM=x[0],HM=x[1],LF=x[2],HF=x[3])

                res = optimize.minimize(obj, x0= [1,1,1,1], method='SLSQP', 
                                        bounds=bounds, constraints=constraints)
                
                sol.LM_vec[j] = res.x[0]
                sol.HM_vec[j] = res.x[1]
                sol.LF_vec[j] = res.x[2]
                sol.HF_vec[j] = res.x[3]
                j = j +1

        # function only returns values of log(Hf/Hm) and log(wf)
        l_hf_hm = np.log(sol.HF_vec/sol.HM_vec)
        l_wf_wm = np.log(par.wF_vec)
        return l_hf_hm, l_wf_wm

    def run_regression(self, a=0.5,s=1):
        """ run regression """

        par = self.par
        sol = self.sol

        # the model is solved for different values of wf
        self.solve_wF_vec(alpha=a,sigma=s)

        # the model betas are calculated
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
        return sol.beta0, sol.beta1
    
    def estimate(self):
        """ estimate alpha and sigma """
        # i write the complicated lambda function using the run regression function.
        obj = lambda x: (0.4-self.run_regression(a=x[0],s=x[1])[0])**2+(-0.1-self.run_regression(a=x[0],s=x[1])[1])**2
        res = optimize.minimize(obj, x0= [0.5,0.5], method='Nelder-Mead')
        
        return res