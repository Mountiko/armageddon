import numpy as np
from armageddon.solver import Planet
import matplotlib.pyplot as plt
from analytical import anal_sol
import time
import scipy_test
from asteroid_par import parse_data
import dask

earth = Planet()

df, out = earth.impact(10, 20e3, 3000, 1e5, angle=0)

print(df)
earth.plot_results(df)



'''
fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.scatter(df.velocity, df.altitude, label='numeric', marker='.', color='r')
ax1.plot(sci_res.velocity, sci_res.altitude, label='scipy', color='b')
ax1.plot(anal_res.velocity, anal_res.altitude, label='anal', color='g')
ax1.set_ylabel('velocity')
ax1.set_xlabel('altitude')
ax1.set_ylim(0,1e5)
ax1.grid()
ax1.legend()

ax2.scatter(df.dedz, df.altitude, label='numeric', marker='.', color='r')
ax2.plot(sci_res.dedz, sci_res.altitude, label='scipy', color='b')
ax2.plot(anal_res.dedz, anal_res.altitude, label='anal', color='g')
ax2.set_ylabel('dedz')
ax2.set_xlabel('altitude')
ax2.set_ylim(0,1e5)
ax2.grid()
ax2.legend()

plt.show()'''