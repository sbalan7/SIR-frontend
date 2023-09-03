import matplotlib.pyplot as plt
from astropy.io import fits
import frontend as sir
import numpy as np
import os


# Creating and writing the atmosphere file based off the hot11 model
atm0, atm1 = sir.set_const_atm('xAtmosphereModels/hot11.mod')
sir.write_atm('Joint_Inversion/modelg.mod', atm0, atm1)

'''
File manipulation trick for simultaneous inversions, assuming the individual profiles are present
'''

# First, read the individual profile files
[_, lambdas1, linei1, lineq1, lineu1, linev1] = sir.read_existing_profile('Inversion_Files_Ca/GRIS_CaI_DP.per')
[_, lambdas2, linei2, lineq2, lineu2, linev2] = sir.read_existing_profile('Inversion_Files_Si/GRIS_SiI_DP.per')

# Create a path for the combined profiles and provide a new index for the lines
# Make sure to set add=True to append instead of overwrite
path = 'Joint_Inversion/GRIS_CaI_SiI_joint_DP.per'
sir.write_profile(path, 1, lambdas1, linei1, lineq1, lineu1, linev1, add=False)
sir.write_profile(path, 2, lambdas1, linei1, lineq1, lineu1, linev1, add=True)
sir.write_profile(path, 3, lambdas1, linei1, lineq1, lineu1, linev1, add=True)
sir.write_profile(path, 4, lambdas1, linei1, lineq1, lineu1, linev1, add=True)
sir.write_profile(path, 5, lambdas2, linei2, lineq2, lineu2, linev2, add=True)

# Provide the new profiles file for the wavelength file to get autocalculated.
sir.wavelength_file(path, 'Joint_Inversion/malla.grid', 18.3)

# Example of editing the trol file parameters
sir.edit_trol_file('Joint_Inversion/sir.trol', ['profiles'], ['GRIS_CaI_SiI_joint_DP.per'])

# Running SIR and reading the obtained model, it can now be plotted
sir.run_SIR('Joint_Inversion', 'sir.trol')
[_, l1, i1, q1, u1, v1] = sir.read_existing_profile('Joint_Inversion/modelg_2.per')


