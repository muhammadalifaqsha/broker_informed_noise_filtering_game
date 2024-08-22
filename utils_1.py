from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.integrate import solve_ivp

def dM(t, T, M, U_eval, V_eval, B_eval):
    M_matrix = M.reshape((4,4))
    diff  = (M_matrix.T@U_eval(T-t)@M_matrix + M_matrix@V_eval(T-t) + (M_matrix@V_eval(T-t)).T + B_eval(T-t)).flatten()
    return diff

def solve_riccati_implicit_timde_dep(
        T, nb_t, d,
        Q, Y, U, PT
):
    dt = T / (nb_t - 1)

    Pts = np.empty([nb_t, d, d])
    Pts[-1, ...] = PT

    # print(np.shape(Q))
    for t in range(nb_t - 2, -1, -1):
        if d == 1:
            Ptplusdt = np.array([np.squeeze(Pts[t + 1, ...])])
        else:
            Ptplusdt = np.squeeze(Pts[t + 1, ...])

        P = Ptplusdt.copy()
        for i in range(10):
            diff_big_mat = np.eye(d * d) + dt * (np.kron(Y[t].T, np.eye(d)) + np.kron(np.eye(d), Y[t].T) + np.kron(
                np.eye(d),
                np.transpose(U[t] @ P)
            ) + np.kron(P @ U[t], np.eye(d)))
            err = P + dt * (Q[t] + Y[t].T @ P + P @ Y[t] + P @ U[t] @ P) - Ptplusdt
            err = np.linalg.inv(diff_big_mat) @ err.flatten()
            P = P - err.reshape(d, d)
            
        P1 = Ptplusdt - dt * (Q[t+1] + Y[t+1].T @ Ptplusdt + Ptplusdt @ Y[t+1] + Ptplusdt @ U[t+1] @ Ptplusdt)
        P_avg = 0.5*(P1 + P)
        Pts[t, ...] = P_avg.copy()

    return Pts



class environment:
    """
    Model parameters for the environment.
    """
    def __init__(self, b = 1e-5, alpha0 = 0., kappa_alpha = 1., sigma_alpha = 1., S0 =100., sigma_s = 1., corr = 0, T = 1., Nt =1_000):
        self.b = b
        self.alpha0 = alpha0
        self.kappa_alpha = kappa_alpha
        self.sigma_alpha = sigma_alpha
        self.T = T
        self.S0 = S0
        self.sigma_s = sigma_s
        self.Nt = Nt
        self.corr = corr
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))

    def simulate_BM_with_drift(self, x0=0, mu=0, sigma=1, nsims=1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = x0
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] + dt * mu + np.sqrt(dt) * sigma * errs[t,:]
        return x

    def simulate_alpha(self, nsims = 1):
        dt = self.T/(self.Nt)
        alpha = np.zeros((self.Nt+1, nsims))
        alpha[0,:] = self.alpha0
        for it in range(self.Nt):
            alpha[it+1,:] = alpha[it,:] - self.kappa_alpha*alpha[it,:]*dt + self.sigma_alpha*np.random.randn(1, nsims)*np.sqrt(dt)
        return alpha

    def simulate_stock(self, alpha):
        nsteps, nsims = alpha.shape
        dt = self.T/(nsteps-1)
        S = np.zeros_like(alpha)
        S[0,:] = self.S0
        for it in range(nsteps-1):
            S[it+1,:] = S[it,:] + alpha[it,:]*dt + self.sigma_s* np.random.randn(1, nsims)*np.sqrt(dt) 
        return S   
    


    
class informed:
    """
    Informed trader's parameters and functions.
    """
    def __init__(self, env, Q0 = 0., k = 1e-3, beta0 = 0, beta1 = 0, rho0 = 0, rho1 = 0,
                  sigma_B = 60., mu_B = 0., theta_B = 10., sigma_0 = 0., c_importance = 1):
        self.Q0 = Q0
        self.k = k
        self.beta0 = beta0
        self.beta1 = beta1
        self.rho0 = rho0
        self.rho1 = rho1
        self.sigma_B = sigma_B
        self.mu_B = mu_B
        self.theta_B = theta_B
        self.sigma_0 = sigma_0
        self.c_importance = c_importance
        self.timesteps = env.timesteps
        self.N = len(self.timesteps)

        # computing the conditional variance of the filter
        
        dP = lambda t, p: self.sigma_B**2 - 2*self.theta_B*p - (env.b**2)*(p**2)/(env.sigma_s**2)
        p_init = np.array([self.sigma_0])
        sol = solve_ivp(fun = dP, 
                        t_span = [0,env.T], 
                        y0 = p_init,
                        t_eval = self.timesteps,
                        method = 'DOP853'
                        )
        self.V_I = sol.y.reshape(-1,)
        self.V_I_eval = lambda t: CubicSpline(self.timesteps, self.V_I)(t)
        
        _ts = self.timesteps
        _Gt = lambda t, z: np.array([
                                        # ODE z's
                                        (z[9]*z[0])/(2*self.k) + self.mu_B*self.theta_B*z[2],
                                        1 - env.kappa_alpha*z[1] + (z[9]*z[1])/(2*self.k),
                                        env.b - self.theta_B*z[2] + (z[9]*z[2])/(2*self.k),
                                        (z[9]*z[0])/(2*self.k) + (z[0]**2)/(4*self.k) + self.mu_B*self.theta_B*z[5] + env.b* env.corr *env.sigma_alpha*(1/env.sigma_s)*self.V_I_eval(env.T-t)*z[6] + (env.sigma_alpha**2)*z[7] + (env.b*self.V_I_eval(env.T-t)/env.sigma_s)**2*z[8],
                                        (z[9]*z[1])/(2*self.k) + (z[0]*z[1])/(2*self.k) - env.kappa_alpha*z[4] + self.mu_B*self.theta_B*z[6],
                                        (z[9]*z[2])/(2*self.k) + (z[0]*z[2])/(2*self.k) - self.theta_B*z[5] + 2*self.mu_B*self.theta_B*z[8],
                                        (z[1]*z[2])/(2*self.k) - env.kappa_alpha*z[6] - self.theta_B*z[6],
                                        (z[1]**2)/(4*self.k) - 2*env.kappa_alpha*z[7],
                                        (z[2]**2)/(4*self.k) - 2*self.theta_B*z[8],
                                        # ODE for g2
                                        (z[9]**2/self.k) - (self.rho0 + self.rho1*self.V_I_eval(env.T-t))
                                       ] )
        init_cond = np.zeros((10,))
        init_cond[-1] = -self.beta0 - self.beta1*self.V_I_eval(env.T)
        _sol        = solve_ivp(fun = _Gt, 
                           t_span = [0, env.T], 
                           y0 = init_cond, 
                           t_eval = self.timesteps,
                           method = 'DOP853')
        self.z        = _sol.y[:,::-1]
        self.z_eval = lambda t: CubicSpline(self.timesteps, self.z, axis=1)(t)
        

    def informed_speed(self, t, alpha, hat_nu, Q):
        z_eval_t = self.z_eval(t)
        z0 = z_eval_t[0]
        z1 = z_eval_t[1]
        z2 = z_eval_t[2]
        g2 = z_eval_t[9]
        informed_optimal = (z0 + z1*alpha + z2*hat_nu + 2*Q *g2)/(2*self.k)
        return informed_optimal
    
  
class uninformed:
    """
    Uninformed trader: parameters
    """
    def __init__(self, kappa, nu0, sigma, k):
        self.kappa = kappa
        self.nu0 = nu0
        self.sigma = sigma
        self.k = k

    def simulate_uninformed_flow(self, env, nsims = 1):
        dt = env.T/(env.Nt)
        nu = np.zeros((env.Nt+1, nsims))
        nu[0,:] = self.nu0
        for it in range(env.Nt):
            nu[it+1,:] = nu[it,:] - self.kappa*nu[it,:]*dt + self.sigma*np.random.randn(1, nsims)*np.sqrt(dt)
        return nu



class broker:
    """
    Broker's parameters and functions.
    """
    def __init__(self, env, inf, uninf, Q0 = 0, k = 1e-3, beta0 = 0, beta1 = 0, rho0 = 0, rho1 = 0, sigma_B = 60, mu_B = 0, theta_B = 10, sigma_0 = 0, scale_ivp = 1):
        self.Q0 = Q0
        self.k = k
        self.beta0 = beta0
        self.beta1 = beta1
        self.rho0 = rho0
        self.rho1 = rho1
        self.sigma_B = sigma_B
        self.mu_B = mu_B
        self.theta_B = theta_B
        self.sigma_0 = sigma_0
        self.timesteps = env.timesteps
        self.N = len(self.timesteps)
        self.scale_ivp = scale_ivp
        
        # computing the matrices in the HJB Riccatti equation
        
        z = inf.z
        f0 = z[0] / 2 / inf.k
        f1 = z[1] / 2 / inf.k
        f2 = inf.c_importance * z[2] / 2 / inf.k
        f3 = z[9] / inf.k
        V_I = inf.V_I
        
        f2_eval = lambda t: CubicSpline(self.timesteps, f2)(t)
        self.denom_constr = lambda t: np.sqrt(self.k - inf.k * f2_eval(t)**2)
        
        # calculating the conditional variance of the filter
        
        dP = lambda t, p: (1 - env.corr**2) * env.sigma_alpha**2 + 2 * (-env.kappa_alpha - env.corr*env.sigma_alpha / env.sigma_s)*p - (p**2)/env.sigma_s**2
        p_init = np.array([self.sigma_0])
        sol = solve_ivp(fun = dP, 
                        t_span = [0,env.T], 
                        y0 = p_init,
                        t_eval = self.timesteps,
                        method = 'DOP853'
                        )          
        self.V_B = sol.y.reshape(-1,)
        self.V_B_eval = lambda t: CubicSpline(self.timesteps, self.V_B)(t)
        
        self.P1 = np.zeros((self.N, 1, 4))
        self.P2 = np.zeros((self.N, 4 , 4))
        self.P3 = np.zeros((self.N, 1, 1))
        self.P4 = np.zeros((self.N, 1 , 4))
        self.P5 = np.zeros((self.N, 4 , 4))
        self.P6 = np.zeros((self.N, 1, 1))
        self.P7 = np.zeros((self.N, 1 , 4))
        self.P8 = np.zeros((self.N, 1, 4))
        
        self.U = np.zeros((self.N, 4 , 4))
        self.V = np.zeros((self.N, 4 , 4))
        self.B = np.zeros((self.N, 4 , 4))
        
        for t in range(self.N):
            self.P1[t, :, :] = np.array([[-f0[t], 0, 0, f0[t]]])
            
            self.P2[t, :, :] = np.array([[0, 0, 0, 0],
                           [-f1[t], -env.kappa_alpha, 0, f1[t]],
                           [-1, 0, -uninf.kappa, 0],
                           [-f3[t], 0, 0, f3[t]]])
            
            self.P3[t, :, :] = np.array([[inf.k*f0[t]**2]])
            self.P4[t, :, :] = np.array([[0, f0[t]*f1[t]*inf.k, 0, f0[t]*f3[t]*inf.k]])

            self.P5[t, :, :] = np.array([[-(self.rho0 + self.V_B[t]*self.rho1), .5, 0, 0],
                           [.5, inf.k*f1[t]**2, 0, f1[t]*f3[t]*inf.k],
                           [0, 0, uninf.k, 0],
                           [0, f1[t]*f3[t]*inf.k, 0, inf.k*f3[t]**2]])
            
            self.P6[t, :, :] = np.array([[f0[t]*f2[t]*inf.k]]) / np.sqrt(self.k - inf.k*f2[t]**2)

            self.P7[t,:, :] = np.array([[env.b/2, f1[t]*f2[t]*inf.k, 0, f2[t]*f3[t]*inf.k]])/(np.sqrt(self.k - inf.k*f2[t]**2))
    
            self.P8[t, :, :] = np.array([[1-f2[t], 0, 0, f2[t]]])/ np.sqrt(self.k - inf.k*f2[t]**2) / 2
        
            self.U[t, :, :] = 4*(self.P8[t, :, :]).T@(self.P8[t, :, :])
    
            self.V[t, :, :] = 2*(self.P8[t, :, :]).T@(self.P7[t, :, :]) + (self.P2[t, :, :]).T
    
            self.B[t, :, :] = (self.P7[t, :, :]).T@(self.P7[t, :, :]) + self.P5[t, :, :]
        
        # numerically solve the matrix ODEs
        
        self.U_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.U[:, i, j])(t).item() for j in range(4)] for i in range(4)])
        self.V_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.V[:, i, j])(t).item() for j in range(4)] for i in range(4)])
        self.B_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.B[:, i, j])(t).item() for j in range(4)] for i in range(4)])
        
        self.dG2B = lambda t, G2: dM(t, env.T, G2, self.U_eval, self.V_eval, self.B_eval)        
        self.G2B_terminal = np.zeros((4, 4))
        self.G2B_terminal[0, 0] = -(self.beta0 + self.beta1 * self.V_B[-1])
        self.G2B = solve_ivp(fun = self.dG2B, 
                        t_span = [0,env.T], 
                        y0 = (self.G2B_terminal).flatten(),
                        t_eval = self.timesteps,
                        method = 'DOP853').y[:,::-1]
        
#         self.G2B_test = solve_ivp(fun = self.dG2B, 
#                         t_span = [0,env.T], 
#                         t_eval = self.timesteps,
#                         y0 = (self.G2B_terminal).flatten(),
#                         method = 'DOP853', dense_output = True)
        
        G2B_copy = np.array([((self.G2B)[:,it]).reshape((4,4)) for it in range(self.N)])
        
        self.G2B = G2B_copy
        
#         self.G2B = solve_riccati_implicit_timde_dep(env.T, self.N, 4,
#                                               -self.B, -self.V, -self.U, self.G2B_terminal)
        
        self.G2B_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.G2B[:, i, j])(t).item() for j in range(4)] for i in range(4)])
        
        self.P1_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.P1[:, i, j])(t).item() for j in range(4)] for i in range(1)])
        self.P2_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.P2[:, i, j])(t).item() for j in range(4)] for i in range(4)])
        self.P3_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.P3[:, i, j])(t).item() for j in range(1)] for i in range(1)])
        self.P4_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.P4[:, i, j])(t).item() for j in range(4)] for i in range(1)])
        self.P5_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.P5[:, i, j])(t).item() for j in range(4)] for i in range(4)])
        self.P6_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.P6[:, i, j])(t).item() for j in range(1)] for i in range(1)])
        self.P7_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.P7[:, i, j])(t).item() for j in range(4)] for i in range(1)])
        self.P8_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.P8[:, i, j])(t).item() for j in range(4)] for i in range(1)])

        dG1B = lambda t, G1: 2*(self.P1_eval(env.T-t)@G1 + G1@(self.P2_eval(env.T-t)).T + (self.P6_eval(env.T-t)).T@(self.P7_eval(env.T-t)) + 2*G1@(self.P8_eval(env.T-t)).T@(self.P7_eval(env.T-t))
                                + 2*(self.P6_eval(env.T-t)).T@(self.P8_eval(env.T-t))@(self.G2B_eval(env.T-t)) + 4*G1@(self.P8_eval(env.T-t)).T@(self.P8_eval(env.T-t))@(self.G2B_eval(env.T-t)) + (self.P4_eval(env.T-t))).flatten()
        G1B_init = np.zeros((4,))
        self.G1B = solve_ivp(fun = dG1B, 
                        t_span = [0,env.T], 
                        y0 = G1B_init,
                        t_eval = self.timesteps
                        ).y[:,::-1]
        self.G1B_eval = lambda t: np.array([CubicSpline(self.timesteps, self.G1B[i,:])(t) for i in range(4)])

        dG0B = lambda t, G0: (2*(self.P1_eval(env.T-t))@(self.G1B_eval(env.T-t)).T + self.P3_eval(env.T-t) + (self.P6_eval(env.T-t)).T@(self.P6_eval(env.T-t))
                             + 4*(self.P6_eval(env.T-t)).T@(self.P8_eval(env.T-t))@(self.G1B_eval(env.T-t)).T
                             + 4*((self.P8_eval(env.T-t))@(self.G1B_eval(env.T-t)).T).T@(self.P8_eval(env.T-t))@(self.G1B_eval(env.T-t)).T
                             + ((self.V_B_eval(env.T-t) + env.corr * env.sigma_s * env.sigma_alpha)/env.sigma_s)**2 * (self.G2B_eval(env.T-t))[1, 1]
                             + uninf.sigma**2 * (self.G2B_eval(env.T-t))[2,2]).item()
        G0B_init = np.array([0])
        self.G0B = solve_ivp(fun = dG0B, 
                        t_span = [0,env.T], 
                        y0 = G0B_init,
                        t_eval = self.timesteps
                        ).y.reshape(-1,)[::-1]
        
        self.G0B_eval = lambda t: CubicSpline(self.timesteps, self.G0B)(t)
        
        # compute the coefficients of control
        
        self.coef_speed_B = np.zeros((self.N, 1, 5))
        for j in range(self.N):
            self.coef_speed_B[j, :, :1] = self.P6[j, :, :] + 2*self.P8[j, :, :]@(self.G1B[:, j]).reshape((4,1))
            self.coef_speed_B[j, :, 1:] = self.P7[j, :, :] + 2*self.P8[j, :, :]@self.G2B[j, :, :]
            self.coef_speed_B[j, :, :] /= self.denom_constr(self.timesteps[j])
        
    def calculate_r(self, inf, t, qB, alphaH, nuU, qI, nuI):
        z_eval_t = inf.z_eval(t)
        f0 = z_eval_t[0] / 2 / inf.k
        f1 = z_eval_t[1] / 2 / inf.k
        f2 = inf.c_importance * z_eval_t[2] / 2 / inf.k
        f3 = z_eval_t[9] / inf.k
        
        X = np.array([[qB, alphaH, nuU, qI]])
        
        nuB = self.P6_eval(t) + 2*self.P8_eval(t)@(self.G1B_eval(t)).T + (self.P7_eval(t) + 2*self.P8_eval(t)@self.G2B_eval(t))@X.T
            
        nuB /= self.denom_constr(t)
        
        nuB = nuB.reshape((-1,))
        
        nuI_est = f0 + f1*alphaH + f2*nuB + f3*qI
        
        r = nuI - nuI_est     
        
        return r
    
    def optimal_trading_rate(self, t, qB, alphaH, nuU, qI):
        
        X = np.array([[qB, alphaH, nuU, qI]])
        
        nuB = self.P6_eval(t) + 2*self.P8_eval(t)@(self.G1B_eval(t)).T + (self.P7_eval(t) + 2*self.P8_eval(t)@self.G2B_eval(t))@X.T
            
        nuB /= self.denom_constr(t)
        
        return nuB.reshape((-1,))
    
    def benchmark_1(self, inf, t, qB, alphaH, qI):
        z_eval_t = inf.z_eval(t)
        f0 = z_eval_t[0] / 2 / inf.k
        f1 = z_eval_t[1] / 2 / inf.k
        f2 = inf.c_importance * z_eval_t[2] / 2 / inf.k
        f3 = z_eval_t[9] / inf.k
        
        X = np.array([[qB, alphaH, nuU, qI]])
