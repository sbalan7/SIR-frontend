import matplotlib.pyplot as plt
from astropy.io import fits
import frontend as sir
import numpy as np
import os


# Creates an atmosphere having constant parameters with HSRA as its base
atm0, atm1 = sir.set_const_atm('xAtmosphereModels/hsra11.mod', micr=1e5, magf=250, vlos=0.5e5, incl=60)

# Writes the atmosphere to a file and runs SIR for the Si line (assuming all parameters are set)
sir.write_atm('Inversion_Files_Si/modelg.mod', atm0, atm1)
sir.run_SIR('Inversion_Files_Si', 'sir.trol')

# Plotting the profiles after the inversion
fig, axs = plt.subplots(2, 2)

fig, axs = sir.plot.plot_spectra('Inversion_Files_Si/GRIS_SiI_avg_QS.per', 'black', fig, axs)
fig, axs = sir.plot.plot_spectra('Inversion_Files_Si/modelg_1.per', 'blue', fig, axs)
fig, axs = sir.plot.plot_spectra('Inversion_Files_Si/modelg_2.per', 'red', fig, axs)

fig.suptitle('Average Quiet Sun Pixel')
fig.tight_layout()
plt.show()

# Plotting the atmospheric parameters after the inversion
fig, axs = plt.subplots(2, 3, figsize=(10, 6))

fig, axs = sir.plot.add_atmosphere('Inversion_Files_Si/modelg.mod', 'black', fig, axs)
fig, axs = sir.plot.add_atmosphere('Inversion_Files_Si/modelg_1.mod', 'blue', fig, axs)
fig, axs = sir.plot.add_atmosphere('Inversion_Files_Si/modelg_2.mod', 'red', fig, axs)

fig.suptitle('Average Quiet Sun Pixel')
fig.tight_layout()
plt.show()