import matplotlib.pyplot as plt
import frontend as sir
import numpy as np


# Creates an atmosphere having constant parameters with HSRA as its base and plots it
HSRA = sir.Atmosphere('AtmosphereModels/hsra11.mod')

HSRA.set_atmospheric_parameters(method='first', fill=0.8)
HSRA.set_atmospheric_parameters(method='scalar', incl=60, magf=500, azim=30, vlos=5e5)

fig, axs = HSRA.plot_atmosphere()
fig.tight_layout()
plt.show()

HSRA.write_atm('test.mod')

# Creates an object for the Si line, modifies its index, writes the malla.grid file and plots it
Si = sir.Profiles('Inversion_Si/SiI_avg_QS.per')
Si.modify_index(1)   # Assuming the Si line is set to 1 in the LINES file
Si.write_wavelength_file(resolution=18.3)

fig, axs = Si.plot_profiles(index=1)
fig.tight_layout()
plt.show()

# Create the requisite files for the inversion and run it
trol_file  = 'SIR_TEST/sir.trol'
lines_file = 'SIR_TEST/LINES'
abundances = 'SIR_TEST/ASPLUND'

# Corresponds to Stokes I, Q, U, V respectively
weights = [1, 5, 5, 5]
# Corresponds to temperature, electron pressure, microturbulence, magnetic field, 
# LOS velocity, inclination and azimuthal angle respectively
nodes1  = ['2, 5', '', '1', '1, 2', '1, 2', '1, 2', '1, 2']

I1 = sir.Inversion(atmosphere=HSRA, profiles=Si, trol_file=trol_file, lines_file=lines_file, abundances=abundances, ncycles=2, weights=weights, nodes1=nodes1)
I1.run_inversion(suppress=False)

# Plot and analyse the results
fig, axs = I1.op_profiles[0].plot_profiles(1, color='red')
fig, axs = I1.op_profiles[1].plot_profiles(1, fig, axs, color='blue')
fig.tight_layout()
plt.show()

fig, axs = I1.op_atmos[0].plot_atmosphere(color='red')
fig, axs = I1.op_atmos[1].plot_atmosphere(fig, axs, color='blue')
fig.tight_layout()
plt.show()
