'''
This script provides utility functions estimating velocities
and other physical quantities from positions of a spring mass.
'''

import numpy as np
from scipy.interpolate import CubicSpline

# physical parameters
fps = 60  # frames per second
k = 80    # spring constant
m = 1.0   # bob mass (kg)




'''
Calculate velocities from a sequence of positions
using numerical differentiation.
method='fd': finite difference;
method='spline': cubic spline fitting.
'''
def calc_velocity(pos, method='spline'):
    len_seq = pos.shape[0]

    # isolated data
    if len_seq == 1:
        return np.nan
    
    vel = np.zeros(len_seq)

    # finite difference
    if method == 'fd':
        for i in range(1, len_seq):
            vel[i] = (pos[i] - pos[i-1]) * fps
        vel[0] = (pos[1] - pos[0]) * fps
    
    # cubic spline fitting
    elif method == 'spline':
        t = np.arange(len_seq) / fps
        cs = CubicSpline(t, pos)
        vel= cs(t, 1)
        # use finite difference at boundary points to improve accuracy
        vel[0] = (pos[1] - pos[0]) * fps
        vel[-1] = (pos[-1] - pos[-2]) * fps
    
    else:
        assert False, 'Unrecognizable differentiation method!'
    
    return vel

'''
Calculate energies from position and angular velocities
'''
def calc_energy(pos, vel):
    T = 0.5 * m * vel**2
    V = 0.5 * k * pos**2
    E = T + V
    return T, V, E