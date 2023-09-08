# GREGOR Data Analysis and SIR Python Frontend

SIR Frontend v0.10a - Minimal Functional Inversion Module Completed, electron pressure defaults fixed

A Python based frontend to support and implement a variety of functions to interface the running of SIR, the FORTRAN based package to invert solar atmospheres. Requires matplotlib, numpy, mplcursors and os. 

TODO:
1. Need to add support for synthesis and response functions 
2. Need to complete support for lesser used aspects of the inversion process
3. Need to improve robustness out of user input and capture more errors
4. Add comments and clean up code


## Usage

To analyse data from GREGOR, first extract information from the `\level1` fits files using standard IO operations. Once the Stokes parameters are extracted from these files, the `\frontend` package can be used to analyse this information. It is necessary to have this folder in the root directory of where the other code will be run to avoid a `ModuleNotFound` error. A suggestion is to create a different folder within this directory to allow for an easy manipulation of the SIR input and output files. Additionally, before using the package, make sure to edit the variable in the `__init__.py` file to reflect the path to the SIR FORTRAN code on your device.

```
.
├── frontend
│   ├── __init__.py
│   ├── readme.md
│   └── plot.py
├── inversion_files
│   ├── ASPLUND
│   ├── LINES
│   ├── modelg.mod
│   ├── profiles.per
│   ├── sir.trol
│   └── wavelength.grid
├── inversion.py
├── synthesis.py
└── analysis.ipynb
```

To run an inversion, a guess atmosphere model is required. For this, create an object in the atmosphere class by initializing it with an existing reference atmosphere (examples can be found at the original SIR repository linked [here](https://github.com/BasilioRuiz/SIR-code/tree/master/models)). You can also provide a path to save the resulting atmosphere with the parameter `path_to_file`. Additionally, to restrict the range of $\log{\tau}$ values covered by the atmosphere model, you can set the `taumin` or `taumax` variables to the **LOG TAU** values you want to restrict the model to. Only the rows within these bounds will be used if these parameters are set. Atmospheric parameters can be edited using four methods provided by the `set_atmospheric_parameters` method of the class. A scalar trend sets the provided value constantly for all optical depths and is indicated by using the 'scalar' parameter. To set a linear dependence on log tau, use the 'lineartau' method, the parameter will be set as (val[0] + val[1] * log tau). A linear dependence can be set on log tau and the existing parameter value using the 'linear' method, the parameter will be set as (val[0] + val[1] * log tau + val[2] * param). One can also set a custom dependence on log tau by passing the value set they want to the parameter to have over the domain by using the method 'custom'. To edit the first three parameters on the top of the atmosphere file, use method 'first' and edit them. The atmospheric parameters can be written to a file using the `write_atm` method by passing the file name and the parameters can be plotted by using `plot_atmosphere` method. Refer to the minimal working example below.

```python
import matplotlib.pyplot as plt
import frontend as sir
import numpy as np

HSRA = sir.Atmosphere('AtmosphereModels/hsra11.mod')

HSRA.set_atmospheric_parameters(method='first', fill=0.8)
HSRA.set_atmospheric_parameters(method='scalar', incl=60, magf=500, azim=30)
HSRA.set_atmospheric_parameters(method='lineartau', vlos=[5e5, 200])
HSRA.set_atmospheric_parameters(method='linear', micr=[1e4, -160, -0.5])

fig, axs = HSRA.plot_atmosphere()
fig.tight_layout()
plt.show()

HSRA.write_atm('test.mod')
```

After preparing the atmosphere files, extract the spectral lines from the level 2 data and save it to a profiles file. Use the Profiles class to read this file. Modify the index of the profiles by using `modify_index(index)` method of the class. To stack together multiple profiles for inverting different lines simultaneously, use the `add_profiles(profile2)` method to add an extra profile. Make sure to select the right index for the new profile before adding it. Just like the atmosphere method, the features of the profiles can be written to a file and plotted by using the `write_profile` and `plot_profiles`. However, the `plot_profiles` method additionally requires the index parameter and it plots the corresponding line alone. Additionally, the wavelength file can be written to the `write_wavelength_file` along with the `resolution` parameter provided to it. Refer to the minimal working example below.

```python
# Assuming the imports are the same as before
Si = sir.Profiles('Inversion_Si/SiI_avg_QS.per')
Ca = sir.Profiles('Inversion_Ca/CaI_avg_QS.per')

joint = Si
joint.modify_index(1)
Ca.modify_index(2)
joint.add_profile(Ca)

fig, axs = joint.plot_profiles(index=1)
fig.tight_layout()
plt.show()

fig, axs = joint.plot_profiles(index=2)
fig.tight_layout()
plt.show()

joint.write_wavelength_file(resolution=18.3)
```

To begin the inversion process, first move the lines file and abundances file to the same directory as the trol file. Make sure that the trol file in this directory has the default template for SIR (this requirement will be lifted in future versions of the code). Create an instance of the `Inversion` class and give it the atmosphere and profile objects which have been prepared until now. Additionally, it requires the path to the trol file, lines file and abundances file as well. The number of cycles, weights of Stokes parameters, and nodes for inversion can also be passed as parameters to the constructor, or set by editing the attributes of the class. To run the inversion, simply use `run_inversion` which has a single parameter to optionally suppress the output generated by SIR. A list of size ncycles will get populated with objects of the output profiles and atmospheres, which can be accessed using the `op_profiles` and `op_atmos` attributes. Support to stray light files or PSF files will be added soon.

```python
trol_file  = 'SIR_TEST/sir.trol'
lines_file = 'SIR_TEST/LINES'
abundances = 'SIR_TEST/ASPLUND'

# Corresponds to Stokes I, Q, U, V respectively
weights = [1, 5, 5, 5]
# Corresponds to temperature, electron pressure, microturbulence, magnetic field, 
# LOS velocity, inclination and azimuthal angle respectively
nodes1  = ['2, 5', '', '1', '1, 2', '1, 2', '1, 2', '1, 2']

I1 = sir.Inversion(atmosphere=HSRA, profiles=joint, trol_file=trol_file, lines_file=lines_file, abundances=abundances, ncycles=2, weights=weights, nodes1=nodes1)
I1.run_inversion(suppress=False)

fig, axs = I1.op_profiles[0].plot_profiles(1, color='red')
fig, axs = I1.op_profiles[1].plot_profiles(1, fig, axs, color='blue')
fig.tight_layout()
plt.show()

fig, axs = I1.op_atmos[0].plot_atmosphere(color='red')
fig, axs = I1.op_atmos[1].plot_atmosphere(fig, axs, color='blue')
fig.tight_layout()
plt.show()
```

An interactive line plot can be made by using the `frontend.plot.interactive_line_plot(x, y, xlabel, ylabel, plot_color='blue', line_color='#505050', box_color='white', box_alpha=0.8)` method (work in progress, currently only supports individual plots, updating to be compatible with subplots).
