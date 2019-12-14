import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

def sci_sol(radius=10, velocity=20e3, density=3000, strength=10e5, angle=45, init_altitude=100e3, distance=0, dt=0.05, tmax=1e4, fragmentation=True, num_scheme='RK45', radians=False, Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, Rp=6371e3, g=9.81, rho0=1.2, H=8000, alpha=0.3):
    '''
    Solves analytical solution for meteroid impact

    Parameters
    ----------

    radius : float
        The radius of the asteroid in meters

    velocity : float
        The entery speed of the asteroid in meters/second

    density : float
        The density of the asteroid in kg/m^3

    strength : float
        The strength of the asteroid (i.e., the ram pressure above which
        fragmentation and spreading occurs) in N/m^2 (Pa)

    angle : float
           The initial trajectory angle of the asteroid to the horizontal
           By default, input is in degrees. If 'radians' is set to True, the
           input should be in radians

    init_altitude : float, optional
        Initial altitude in m

    radians : logical, optional
        Whether angles should be given in degrees or radians. Default=False
        Angles returned in the DataFrame will have the same units as the
        input


   Returns
   -------
   Result : DataFrame
           pandas dataFrame with collumns:
           altitude, velocity, dedz

   '''
    if radians is False:  # converts degrees to radians
        angle = angle * (np.pi) / 180

    mass = 4 / 3 * np.pi * (radius ** 3) * density
    y = np.array([velocity, mass, angle, init_altitude, distance, radius])

    rhoa = lambda x: rho0 * np.exp(-x/H)

    def f(self, y):
        '''
        0: velocity
        1: mass
        2: angle
        3: altitude
        4: distance
        5: radius
        '''
        f = np.zeros_like(y)
        f[0] = - (Cd * rhoa(y[3]) * y[0]**2 * np.pi * y[5]**2) / (2 * y[1]) + (g * np.sin(y[2]))
        f[1] = - (C_H * rhoa(y[3]) * np.pi * y[5]**2 * y[0]**3) / (2 * Q)
        f[2] = g * np.cos(y[2]) / y[0]  - (Cl * rhoa(y[3]) * np.pi * y[5]**2 * y[0]) / (2 * y[1]) - (y[0] * np.cos(y[2])) / (Rp + y[3])
        f[3] = - y[0] * np.sin(y[2])
        f[4] = (y[0] * np.cos(y[2])) / (1 + y[3] / Rp)
        if fragmentation == True and strength <= (rhoa(y[3]) * y[0]**2):
            f[5] = np.sqrt(7/2 * alpha * rhoa(y[3]) / density) * y[0]
        else:
            f[5] = 0
        return f
    
    t = np.arange(0, tmax, dt)
    result = solve_ivp(f, [0, tmax], y, method=num_scheme, t_eval=t)
    result = result.y

    dedz = np.zeros(len(result[0]))
    ke = ((1/2 * result[1, 1:] * result[0, 1:]**2) - (1 / 2 * result[1, :-1] * result[0, :-1]**2)) / 4.184e12
    alt = (result[3, 1:] - result[3, :-1]) / 1e3
    dedz[1:] = ke / alt
    i = np.where(dedz < 0)
    dedz[i] = 0

    result = pd.DataFrame({'velocity': result[0], 'mass': result[1], 'angle': result[2], 'altitude': result[3], 'distance': result[4], 'radius': result[5], 'time': t, 'dedz': dedz})

    return result
