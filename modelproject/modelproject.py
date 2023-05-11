from types import SimpleNamespace
import numpy as np
from scipy import optimize


def solve_ss(alpha, c):
    """ Example function. Solve for steady state k. 

    Args:
        c (float): costs
        alpha (float): parameter

    Returns:
        result (RootResults): the solution represented as a RootResults object.

    """ 
    
    # a. Objective function, depends on k (endogenous) and c (exogenous).
    f = lambda k: k**alpha - c
    obj = lambda kss: kss - f(kss)

    #. b. call root finder to find kss.
    result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
    
    return result


class BasicRamsey():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
        
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor

        # b. firms
        par.A = np.nan
        par.production_function = 'ces'
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # substitution parameter        
        par.delta = 0.05 # depreciation rate

        # c. initial
        par.K_lag_ini = 1.0

        # d. misc
        par.solver = 'scipy' # solver for the equation syste, 'broyden' or 'scipy'
        par.Tpath = 500 # length of transition path, "truncation horizon"

    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['A','K','C','rk','w','r','Y','K_lag']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

    # Minder om at kalibrere modellen, man antager at vi ender i en steady state værdi
    # Det er det første man gør efter man har defineret modellen
    def find_steady_state(self,KY_ss,do_print=True):
        """ find steady state """

        par = self.par
        ss = self.ss

        # a. find A
        ss.K = KY_ss
        Y,_,_ = production(par,1.0,ss.K)
        ss.A = 1/Y

        # b. factor prices
        ss.Y,ss.rk,ss.w = production(par,ss.A,ss.K)
        assert np.isclose(ss.Y,1.0)

        ss.r = ss.rk-par.delta
        
        # c. implied discount factor
        par.beta = 1/(1+ss.r)

        # d. consumption
        ss.C = ss.Y - par.delta*ss.K

        if do_print:

            print(f'Y_ss = {ss.Y:.4f}')
            print(f'K_ss/Y_ss = {ss.K/ss.Y:.4f}')
            print(f'rk_ss = {ss.rk:.4f}')
            print(f'r_ss = {ss.r:.4f}')
            print(f'w_ss = {ss.w:.4f}')
            print(f'A = {ss.A:.4f}')
            print(f'beta = {par.beta:.4f}')

    # Her opskrives det non-linear equation system H.
    # linings systemet indeholder alle perioders transitions ligninger For C og K
    def evaluate_path_errors(self):
        """ evaluate errors along transition path """

        par = self.par
        ss = self.ss
        path = self.path

        # a. consumption        
        C = path.C
        C_plus = np.append(path.C[1:],ss.C)
        
        # b. capital
        K = path.K
        K_lag = path.K_lag = np.insert(K[:-1],0,par.K_lag_ini)
        
        # Hvis der skal laves en ny ligning skal de rogså defineres nogle nye variable
        # i capital defineres K og K_lag fordi den er bagudskuende
        # i consumptions defineres C og C_plus fordi den er fremadskuende
        

        # c. production and factor prices
        path.Y,path.rk,path.w = production(par,path.A,K_lag)
        path.r = path.rk-par.delta
        r_plus = np.append(path.r[1:],ss.r)


        # super vigtigt. Her kan man rette og tilføje ligninger der skal gælde i alle perioder.
        # d. errors (also called H)
        errors = np.nan*np.ones((2,par.Tpath))
        errors[0,:] = C**(-par.sigma) - par.beta*(1+r_plus)*C_plus**(-par.sigma)
        errors[1,:] = K - ((1-par.delta)*K_lag + (path.Y - C))
        
        return errors.ravel()
  
    def solve(self,do_print=True):
        """ solve for the transition path """

        par = self.par
        ss = self.ss
        path = self.path
        
        # a. equation system
        def eq_sys(x):
            
            # i. update
            x = x.reshape((2,par.Tpath))
            path.C[:] = x[0,:]
            path.K[:] = x[1,:]
            
            # ii. return errors
            return self.evaluate_path_errors()

        # her skal der også tilføjes flere guesses til ligningssystemet hvis man udvider med flere ligninger.
        # b. initial guess
        x0 = np.nan*np.ones((2,par.Tpath))
        x0[0,:] = ss.C
        x0[1,:] = ss.K
        x0 = x0.ravel()

        # c. call solver
        root = optimize.root(eq_sys,x0,method='hybr',options={'factor':1.0})
        # the factor determines the size of the initial step
        #  too low: slow
        #  too high: prone to errors

        x = root.x    
        # d. final evaluation
        eq_sys(x)

def production(par,A,K_lag):
    """ production and factor prices """

    # a. production and factor prices
    if par.production_function == 'ces':

        # a. production
        Y = A*( par.alpha*K_lag**(-par.theta) + (1-par.alpha)*(1.0)**(-par.theta) )**(-1.0/par.theta)

        # b. factor prices
        rk = A*par.alpha*K_lag**(-par.theta-1) * (Y/A)**(1.0+par.theta)
        w = A*(1-par.alpha)*(1.0)**(-par.theta-1) * (Y/A)**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # a. production
        Y = A*K_lag**par.alpha * (1.0)**(1-par.alpha)

        # b. factor prices
        rk = A*par.alpha * K_lag**(par.alpha-1) * (1.0)**(1-par.alpha)
        w = A*(1-par.alpha) * K_lag**(par.alpha) * (1.0)**(-par.alpha)

    else:

        raise Exception('unknown type of production function')

    return Y,rk,w    



class GovernmentRamsey():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
        
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor

        # b. firms
        par.A = np.nan
        par.production_function = 'ces'
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # substitution parameter        
        par.delta = 0.05 # depreciation rate

        # c. initial
        par.K_lag_ini = 1.0

        # d. misc
        par.solver = 'scipy' # solver for the equation syste, 'broyden' or 'scipy'
        par.Tpath = 500 # length of transition path, "truncation horizon"

        # e. Taxes
        par.tau_w = 0.01
        par.tau_k = 0.01


    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['A','K','C','rk','w','r','Y','K_lag','G']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

    # Minder om at kalibrere modellen, man antager at vi ender i en steady state værdi
    # Det er det første man gør efter man har defineret modellen
    def find_steady_state(self,KY_ss,do_print=True):
        """ find steady state """

        par = self.par
        ss = self.ss

        # a. find A
        ss.K = KY_ss*(1-par.tau_k)
        Y,_,_ = production(par,1.0,ss.K)
        ss.A = 1/Y

        # b. factor prices
        ss.Y,ss.rk,ss.w = production(par,ss.A,ss.K)
        assert np.isclose(ss.Y,1.0)

        ss.r = (ss.rk-par.delta)
        
        # c. implied discount factor
        par.beta = 1/((1+ss.r)*(1-par.tau_k))

        # d. consumption
        ss.G = ss.K*(1+ss.r)*par.tau_k + ss.w * par.tau_w
        ss.C = ss.Y - par.delta*ss.K-ss.G

        if do_print:

            print(f'Y_ss = {ss.Y:.4f}')
            print(f'K_ss/Y_ss = {ss.K/ss.Y:.4f}')
            print(f'rk_ss = {ss.rk:.4f}')
            print(f'r_ss = {ss.r:.4f}')
            print(f'w_ss = {ss.w:.4f}')
            print(f'A = {ss.A:.4f}')
            print(f'beta = {par.beta:.4f}')
            print(f'G = {ss.G:.4f}')
            print(f'C = {ss.C:.4f}')

    # Her opskrives det non-linear equation system H.
    # linings systemet indeholder alle perioders transitions ligninger For C og K
    def evaluate_path_errors(self):
        """ evaluate errors along transition path """

        par = self.par
        ss = self.ss
        path = self.path

        # a. consumption        
        C = path.C
        C_plus = np.append(path.C[1:],ss.C)
        
        # b. capital
        K = path.K
        K_lag = path.K_lag = np.insert(K[:-1],0,par.K_lag_ini)
        
        # Hvis der skal laves en ny ligning skal de rogså defineres nogle nye variable
        # i capital defineres K og K_lag fordi den er bagudskuende
        # i consumptions defineres C og C_plus fordi den er fremadskuende
        

        # c. production and factor prices
        path.Y,path.rk,path.w = production(par,path.A,K_lag)
        path.r = path.rk-par.delta
        r_plus = np.append(path.r[1:],ss.r)

        # goverment
#        path.G = K_lag*(1+path.r)*par.tau_k + path.w * par.tau_w
        path.G = K*(1+path.r)*par.tau_k + path.w * par.tau_w

        # super vigtigt. Her kan man rette og tilføje ligninger der skal gælde i alle perioder.
        # d. errors (also called H)
        errors = np.nan*np.ones((2,par.Tpath))
        errors[0,:] = C**(-par.sigma) - par.beta*(1+r_plus)*(1-par.tau_k)*C_plus**(-par.sigma)
        errors[1,:] = K - ((1-par.delta)*K_lag + (path.Y - C - path.G))
        
        return errors.ravel()
  
    def solve(self,do_print=True):
        """ solve for the transition path """

        par = self.par
        ss = self.ss
        path = self.path
        
        # a. equation system
        def eq_sys(x):
            
            # i. update
            x = x.reshape((2,par.Tpath))
            path.C[:] = x[0,:]
            path.K[:] = x[1,:]
            
            # ii. return errors
            return self.evaluate_path_errors()

        # her skal der også tilføjes flere guesses til ligningssystemet hvis man udvider med flere ligninger.
        # b. initial guess
        x0 = np.nan*np.ones((2,par.Tpath))
        x0[0,:] = ss.C
        x0[1,:] = ss.K
        x0 = x0.ravel()

        # c. call solver
        root = optimize.root(eq_sys,x0,method='hybr',options={'factor':1.0})
        # the factor determines the size of the initial step
        #  too low: slow
        #  too high: prone to errors
            
        x = root.x

        # d. final evaluation
        eq_sys(x)















class test2():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
        
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor

        # b. firms
        par.A = np.nan
        par.production_function = 'ces'
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # substitution parameter        
        par.delta = 0.05 # depreciation rate

        # c. initial
        par.K_lag_ini = 1.0

        # d. misc
        par.solver = 'scipy' # solver for the equation syste, 'broyden' or 'scipy'
        par.Tpath = 500 # length of transition path, "truncation horizon"

        # e. Taxes
        par.tau_w = 0.001
        par.tau_k = 0.005


    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['A','K','C','rk','w','r','Y','K_lag','G']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

    # Minder om at kalibrere modellen, man antager at vi ender i en steady state værdi
    # Det er det første man gør efter man har defineret modellen
    def find_steady_state(self,KY_ss,do_print=True):
        """ find steady state """

        par = self.par
        ss = self.ss

        # a. find A
        ss.K = KY_ss*(1-par.tau_k)
        Y,_,_ = production(par,1.0,ss.K)
        ss.A = 1/Y

        # b. factor prices
        ss.Y,ss.rk,ss.w = production(par,ss.A,ss.K)
        assert np.isclose(ss.Y,1.0)

        ss.r = (ss.rk-par.delta)
        
        # c. implied discount factor
        par.beta = 1/(1+ss.r)

        # d. consumption
        ss.G = ss.K*(1+ss.r)*par.tau_k + ss.w * par.tau_w
        ss.C = ss.Y - par.delta*ss.K-ss.G

        if do_print:

            print(f'Y_ss = {ss.Y:.4f}')
            print(f'K_ss/Y_ss = {ss.K/ss.Y:.4f}')
            print(f'rk_ss = {ss.rk:.4f}')
            print(f'r_ss = {ss.r:.4f}')
            print(f'w_ss = {ss.w:.4f}')
            print(f'A = {ss.A:.4f}')
            print(f'beta = {par.beta:.4f}')
            print(f'G = {ss.G:.4f}')
            print(f'C = {ss.C:.4f}')

    # Her opskrives det non-linear equation system H.
    # linings systemet indeholder alle perioders transitions ligninger For C og K
    def evaluate_path_errors(self):
        """ evaluate errors along transition path """

        par = self.par
        ss = self.ss
        path = self.path

        # a. consumption        
        C = path.C
        C_plus = np.append(path.C[1:],ss.C)
        
        # b. capital
        K = path.K
        K_lag = path.K_lag = np.insert(K[:-1],0,par.K_lag_ini)
        
        # Hvis der skal laves en ny ligning skal de rogså defineres nogle nye variable
        # i capital defineres K og K_lag fordi den er bagudskuende
        # i consumptions defineres C og C_plus fordi den er fremadskuende
        

        # c. production and factor prices
        path.Y,path.rk,path.w = production(par,path.A,K_lag)
        path.r = path.rk-par.delta
        r_plus = np.append(path.r[1:],ss.r)

        # goverment
        path.G = K_lag*(1+path.r)*par.tau_k + path.w * par.tau_w


        # super vigtigt. Her kan man rette og tilføje ligninger der skal gælde i alle perioder.
        # d. errors (also called H)
        errors = np.nan*np.ones((2,par.Tpath))
        errors[0,:] = C**(-par.sigma) - par.beta*(1+r_plus)*(1-par.tau_k)*C_plus**(-par.sigma)
        errors[1,:] = K - ((1-par.delta)*K_lag + (path.Y - C - path.G))
        
        return errors.ravel()
  
    def solve(self,do_print=True):
        """ solve for the transition path """

        par = self.par
        ss = self.ss
        path = self.path
        
        # a. equation system
        def eq_sys(x):
            
            # i. update
            x = x.reshape((2,par.Tpath))
            path.C[:] = x[0,:]
            path.K[:] = x[1,:]
            
            # ii. return errors
            return self.evaluate_path_errors()

        # her skal der også tilføjes flere guesses til ligningssystemet hvis man udvider med flere ligninger.
        # b. initial guess
        x0 = np.nan*np.ones((2,par.Tpath))
        x0[0,:] = ss.C
        x0[1,:] = ss.K
        x0 = x0.ravel()

        # c. call solver
        root = optimize.root(eq_sys,x0,method='hybr',options={'factor':1.0})
        # the factor determines the size of the initial step
        #  too low: slow
        #  too high: prone to errors
            
        x = root.x

        # d. final evaluation
        eq_sys(x)




