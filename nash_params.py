from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.integrate import solve_ivp
import utils as utils
import simulations_nash as simulations

class nash_params:
    """
    Simulation of the informed trader and broker's beliefs on the parameters
    """
    def __init__(self, env, inf, uninf, broker,
                 theta_B, mu_B, sigma_B, 
                 theta_B_br, mu_B_br, sigma_B_br, kappa_alpha_br, sigma_alpha_br,
                 strategy_type, mispecify, nsims):
        
        self.env = env
        
        params_env_br = {'b': env.b, 'alpha0': env.alpha0, 'kappa_alpha': kappa_alpha_br, 'sigma_alpha': sigma_alpha_br, 
                         'S0': env.S0, 'sigma_s': env.sigma_s, 'corr': env.corr, 'T': env.T, 'Nt': env.Nt}
        
        self.env_br = utils.environment(**params_env_br)
        
        params_inf = {'env': env, 'Q0' : inf.Q0, 'k' : inf.k, 'beta0' : inf.beta0, 'beta1' : inf.beta1, 'rho0' : inf.rho0,
                      'rho1' : inf.rho1, 'sigma_B' : sigma_B, 'mu_B' : mu_B, 'theta_B' : theta_B, 'sigma_0' : inf.sigma_0}
        
        params_inf_br = {'env': env, 'Q0' : inf.Q0, 'k' : inf.k, 'beta0' : inf.beta0, 'beta1' : inf.beta1, 'rho0' : inf.rho0,
                         'rho1' : inf.rho1, 'sigma_B' : sigma_B_br, 'mu_B' : mu_B_br, 'theta_B' : theta_B_br, 'sigma_0' : inf.sigma_0}
        
        self.inf = utils.informed(**params_inf)
        
        self.inf_br = utils.informed(**params_inf_br)
        
        self.uninf = uninf
        
        params_broker = {'env': self.env_br, 'inf': self.inf_br, 'uninf': self.uninf,
                         'Q0': broker.Q0, 'k': broker.k, 'beta0': broker.beta0, 'beta1': broker.beta1, 'rho0': broker.rho0,
                         'rho1': broker.rho1, 'sigma_B': self.inf_br.sigma_B, 'mu_B': self.inf_br.mu_B, 'theta_B': self.inf_br.theta_B, 
                         'sigma_0': broker.sigma_0, 'scale_ivp': broker.scale_ivp}
        
        self.broker = utils.broker(**params_broker)
        
        self.nsims = nsims
        
        np.random.seed(100)
        self.W = self.env.simulate_BM_with_drift(x0=0, mu=0, sigma=1, nsims=self.nsims)
        self.alpha = self.env.simulate_alpha(nsims = self.nsims)
        self.nu_U = self.uninf.simulate_uninformed_flow(env, nsims=self.nsims)
        self.strategy_type = strategy_type
        self.mispecify = mispecify
        
        self.model = simulations.simulation(self.env, self.inf, self.uninf, self.broker, 
                                            self.W, self.alpha, self.nu_U, 
                                            strategy = self.strategy_type, mispecify = self.mispecify, mispecify_scale = 1, nsims = self.nsims)
        self.J_I = self.model.H_I.mean() 
        self.J_B = self.model.H_B.mean() 