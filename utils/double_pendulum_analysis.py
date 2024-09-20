import os
import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# physical parameters
fps = 60 # frames per second
L1, L2 = 0.205, 0.179  # pendulum rod lengths (m)
w1, w2 = 0.038, 0.038  # pendulum rod widths (m)
m1, m2 = 0.262, 0.110  # bob masses (kg)
g = 9.81  # gravitational acceleration (m/s^2)

I1 = m1 * (L1**2 + w1**2) / 12.0
I2 = m2 * (L2**2 + w2**2) / 12.0

def f_ode(t, y):
    theta1, theta2, omega1, omega2 = y

    # Define the coefficients of the mass matrix
    M11 = (1/4)*m1*L1**2 + m2*L1**2 + I1
    M12 = (1/2)*m2*L1*L2*np.cos(theta1 - theta2)
    M22 = (1/4)*m2*L2**2 + I2
    M = np.array([[M11, M12], [M12, M22]])

    # Define the forcing terms
    F1 = (1/2)*m2*L1*L2*omega2**2*np.sin(theta1 - theta2) - ((1/2)*m1 + m2)*g*L1*np.sin(theta1)
    F2 = (1/2)*m2*L1*L2*omega1**2*np.sin(theta1 - theta2) - (1/2)*m2*g*L2*np.sin(theta2)

    omega_dot = np.dot(np.linalg.inv(M), np.array([F1, F2]))

    return [omega1, omega2, omega_dot[0], omega_dot[1]]


def periodic_extension(th):
    len_seq = th.shape[0]
    # preprocessing: periodic extension of angles
    for i in range(1, len_seq):
        if th[i] - th[i-1] > np.pi:
            th[i:] -= 2*np.pi
        elif th[i] - th[i-1] < -np.pi:
            th[i:] += 2*np.pi

    return th


def fit(th1, th2, vel_th1, vel_th2, t_grid):
    len_seq = th1.shape[0]

    # isolated data
    if len_seq == 1:
        return np.nan, np.nan
    
    # objective function
    def f_obj(x):
        th1_ini, th2_ini, vel_th1_ini, vel_th2_ini = x
        # solve dynamic equations
        y0 = [th1_ini, th2_ini, vel_th1_ini, vel_th2_ini]
        t_span = (t_grid[0], t_grid[-1])
        sol = solve_ivp(f_ode, t_span, y0, t_eval=t_grid, method='RK45')
        # compute error
        th1_f, th2_f = sol.y[0], sol.y[1]
        error = np.mean((th1 - th1_f)**2 + (th2 - th2_f)**2)
        return error

    initial_guess = [th1[0], th2[0], vel_th1[0], vel_th2[0]]
    result = minimize(f_obj, initial_guess)
    y0 = result.x
    t_span = (t_grid[0], t_grid[-1])
    sol = solve_ivp(f_ode, t_span, y0, t_eval=t_grid, method='RK45')
    th1_f, th2_f = sol.y[0], sol.y[1]

    return th1_f, th2_f


def fit_all(data_filepath, save_path):
    phys_all = np.load(os.path.join(data_filepath, 'phys_vars.npy'), allow_pickle=True).item()
    theta_1 = phys_all['theta_1']
    theta_2 = phys_all['theta_2']
    vel_theta_1 = phys_all['vel_theta_1']
    vel_theta_2 = phys_all['vel_theta_2']
    reject = phys_all['reject']
    t_grid = np.arange(fps) / fps
    
    fitted_theta_1 = np.zeros(theta_1.shape)
    fitted_theta_2 = np.zeros(theta_2.shape)
    for n in range(2, theta_1.shape[0]):
        print(n)
        sub_ids = np.ma.clump_unmasked(np.ma.masked_array(theta_1[n], reject[n]))
        plt.figure()
        
        for k, ids in enumerate(sub_ids):
            th1_p = periodic_extension(theta_1[n, ids].copy())
            th2_p = periodic_extension(theta_2[n, ids].copy())
            vel_th1_p = vel_theta_1[n, ids].copy()
            vel_th2_p = vel_theta_2[n, ids].copy()
            th1_f, th2_f = fit(th1_p, th2_p, vel_th1_p, vel_th2_p, t_grid[ids])
            fitted_theta_1[n, ids] = th1_f
            fitted_theta_2[n, ids] = th2_f

            plt.plot(t_grid[ids], th1_p, color='b', label='theta1 ground truth', lw=2.5)
            plt.plot(t_grid[ids], th2_p, color='r', label='theta2 ground truth', lw=2.5)
            plt.plot(t_grid[ids], th1_f, '--', color='b', label='theta1 fitted', lw=2.5)
            plt.plot(t_grid[ids], th2_f, '--', color='r', label='theta2 fitted', lw=2.5)
            break

        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.legend()
        plt.title(f'Video no. {n}')
        plt.savefig(os.path.join(save_path, str(n)+'.png'))
        plt.close()

if __name__ == "__main__":
    # mkdir analysis_save first
    fit_all('./data/double_pendulum', 'analysis_save')