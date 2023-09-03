import matplotlib.pyplot as plt
from astropy.io import fits
from frontend import plot
import numpy as np
import os


path_to_sir = '/home/sbalan7/Downloads/Code/Sunspots/SIR/sir2020_04new/sir.x'

def run_SIR(path, trol, suppress=True):
    os.chdir(path)
    if suppress:
        os.system('echo ' + trol + ' | '+path_to_sir+' >/dev/null 2>&1')
    else:
        os.system('echo ' + trol + ' | '+path_to_sir)
    os.chdir('..')

def set_const_atm(model, temp=None, pres=None, micr=None, magf=None, vlos=None, incl=None, azim=None):
    atm0 = np.loadtxt(model, skiprows=0, max_rows=1)
    atm1 = np.loadtxt(model, skiprows=1)

    if temp is not None:
        atm1[:, 1] = atm1[:, 1] * 0 + temp     # Temperature
    
    if pres is not None:
        atm1[:, 2] = atm1[:, 2] * 0 + pres     # Pressure
    
    if micr is not None:
        atm1[:, 3] = atm1[:, 3] * 0 + micr     # Microturbulence
    
    if magf is not None:
        atm1[:, 4] = atm1[:, 4] * 0 + magf     # Mag Field
    
    if vlos is not None:
        atm1[:, 5] = atm1[:, 5] * 0 + vlos     # LOS Velocity
    
    if incl is not None:
        atm1[:, 6] = atm1[:, 6] * 0 + incl     # Inclination
    
    if azim is not None:
        atm1[:, 7] = atm1[:, 7] * 0 + azim     # Azimuth

    return atm0, atm1

def set_complex_atm(model, macr, fill, strl, logt, temp, pres, micr, magf, vlos, incl, azim):
    atm0 = np.loadtxt(model, skiprows=0, max_rows=1)
    atm1 = np.loadtxt(model, skiprows=1)

    atm0 = np.array([macr, fill, strl])

    atm1[:, 0] = logt     # log_tau
    atm1[:, 1] = temp     # Temperature
    atm1[:, 2] = pres     # Pressure
    atm1[:, 3] = micr     # Microturbulence
    atm1[:, 4] = magf     # Mag Field
    atm1[:, 5] = vlos     # LOS Velocity
    atm1[:, 6] = incl     # Inclination
    atm1[:, 7] = azim     # Azimuth

    return atm0, atm1

def write_atm(path, atm0, atm1):
    with open(path, 'wb') as f:
        np.savetxt(f, atm0.reshape(1, 3), delimiter='    ', fmt='%.5f')
        np.savetxt(f, atm1, delimiter='    ', fmt=['%.3f','%.4f','%.6e','%.6e','%.4f','%.4e','%.4f','%.4f','%.6f','%.6e','%.9e'])

def read_existing_profile(path):
    # Returns in the form of [indices, lambdas, linei, lineq, lineu, linev]
    return np.loadtxt(path).T

def write_profile(path, index, lambdas, linei, lineq, lineu, linev, add=False):
    if add:
        with open(path, 'ab') as f:
            np.savetxt(f, np.transpose([index*np.ones(len(lambdas)), lambdas, linei, lineq, lineu, linev]), delimiter='    ',fmt=['%.1f','%.8f','%.8e','%.8e','%.8e','%.8e'])
    else:
        with open(path, 'wb') as f:
            np.savetxt(f, np.transpose([index*np.ones(len(lambdas)), lambdas, linei, lineq, lineu, linev]), delimiter='    ',fmt=['%.1f','%.8f','%.8e','%.8e','%.8e','%.8e'])

def atomic_parameters_file(path, index, ion, wave, E, pot, loggf, transition, alpha, sigma):
    pass

def wavelength_file(profile_path, grid_path, resolution):
    data = read_existing_profile(profile_path)
    idxs = np.unique(data[0])
    L = []

    for idx in idxs:
        line = data[:,np.where(data[0] == idx)][:, 0, :][1]
        line = f"{int(idx):<10}:{line[0]:15.4f},{resolution:7.1f},{line[-1]:15.4f}\n"
        L.append(line)

    with open(grid_path, 'w') as f:
        f.writelines(L)

def edit_trol_file(path, param, value):
    with open(path, 'r+') as f:
        L = f.readlines()
    
    params = {'ncycles':0, 'profiles':1, 'stray':2, 'psf':3, 'grid':4, 'lines':5, 'abundances':6, 'atmguess1':7, 'atmguess2':8, 'wstoki':9, 'wstokq':10, 
    'wstoku':11, 'wstokv':12, 'autonode':13, 'tempnodes1':14, 'presnodes1':15, 'micrnodes1':16, 'magfnodes1':17, 'vlosnodes1':18, 'inclnodes1':19, 
    'azimnodes1':20, 'invmacro1':21, 'tempnodes2':22, 'presnodes2':23, 'micrnodes2':24, 'magfnodes2':25, 'vlosnodes2':26, 'inclnodes2':27, 
    'azimnodes2':28, 'invmacro2':29, 'invfill':30, 'invstray':31, 'mu':32, 'snr':33, 'contcont':34, 'svdtol':35, 'initdiag':36, 'interpstrat':37, 
    'gaspres1':38, 'gaspres2':39, 'magpres':40, 'nltedep':41}
    
    for (p, v) in zip(param, value):
        idx = params[p]
        line = L[idx]
        L[idx] = line.split(":")[0] + ':' + str(v) + '!' + line.split("!")[1]
    
    with open(path, 'w') as f:
        f.writelines(L)

def set_trol_file(path, ncycles, profiles, tempnodes1, presnodes1, micrnodes1, magfnodes1, vlosnodes1, inclnodes1, azimnodes1,
                   stray="", psf="", grid='malla.grid', lines='LINES', abundances='ASPLUND', atmguess1='modelg.mod', atmguess2='',
                   wstoki=1, wstokq=1, wstoku=1, wstokv=1, autonode="", invmacro1="", invmacro2="", invfill="", invstray="",
                   tempnodes2="", presnodes2="", micrnodes2="", magfnodes2="", vlosnodes2="", inclnodes2="", azimnodes2="",
                   mu="", snr="", contcont="", svdtol="", initdiag="", interpstrat="", gaspres1="", gaspres2="", magpres="", nltedep=""):
    L = []
    L.append(f"Number of cycles          (*):{ncycles}              ! (0=synthesis)\n")
    L.append(f"Observed profiles         (*):{profiles}             ! target.mod\n")
    L.append(f"Stray light file             :{stray}                ! (none=no stray light contam)\n")
    L.append(f"PSF file                     :{psf}                  ! (none=no convolution with PSF)\n")
    L.append(f"Wavelength grid file      (s):{grid}                 ! (none=automatic selection)\n")
    L.append(f"Atomic parameters file       :{lines}                ! (none=DEFAULT LINES file)\n")
    L.append(f"Abundances file              :{abundances}           ! (none=DEFAULT ABUNDANCES file)\n")
    L.append(f"Initial guess model 1     (*):{atmguess1}            ! target.mod\n")
    L.append(f"Initial guess model 2        :{atmguess2}            ! \n")
    L.append(f"Weight for Stokes I          :{str(wstoki)}          ! (DEFAULT=1; 0=not inverted)\n")
    L.append(f"Weight for Stokes Q          :{str(wstokq)}          ! (DEFAULT=1; 0=not inverted)\n")
    L.append(f"Weight for Stokes U          :{str(wstoku)}          ! (DEFAULT=1; 0=not inverted)\n")
    L.append(f"Weight for Stokes V          :{str(wstokv)}          ! (DEFAULT=1; 0=not inverted)\n")
    L.append(f"AUTOMATIC SELECT. OF NODES?  :{str(autonode)}        ! (DEFAULT=0=no; 1=yes)\n")
    L.append(f"Nodes for temperature 1      :{tempnodes1}           ! \n")
    L.append(f"Nodes for electr. press. 1   :{presnodes1}           ! \n")
    L.append(f"Nodes for microturb. 1       :{micrnodes1}           ! \n")
    L.append(f"Nodes for magnetic field 1   :{magfnodes1}           ! \n")
    L.append(f"Nodes for LOS velocity 1     :{vlosnodes1}           ! \n")
    L.append(f"Nodes for gamma 1            :{inclnodes1}           ! \n")
    L.append(f"Nodes for phi 1              :{azimnodes1}           ! \n")
    L.append(f"Invert macroturbulence 1?    :{invmacro1}            ! (0 or blank=no, 1=yes)\n")
    L.append(f"Nodes for temperature 2      :{tempnodes2}           ! \n")
    L.append(f"Nodes for electr. press. 2   :{presnodes2}           ! \n")
    L.append(f"Nodes for microturb. 2       :{micrnodes2}           ! \n")
    L.append(f"Nodes for magnetic field 2   :{magfnodes2}           ! \n")
    L.append(f"Nodes for LOS velocity 2     :{vlosnodes2}           ! \n")
    L.append(f"Nodes for gamma 2            :{inclnodes2}           ! \n")
    L.append(f"Nodes for phi 2              :{azimnodes2}           ! \n")
    L.append(f"Invert macroturbulence 2?    :{invmacro2}            ! (0 or blank=no, 1=yes)\n")
    L.append(f"Invert filling factor?       :{invfill}              ! (0 or blank=no, 1=yes)\n")
    L.append(f"Invert stray light factor?   :{invstray}             ! (0 or blank=no, 1=yes)\n")
    L.append(f"mu=cos (theta)               :{mu}                   ! (DEFAULT: mu=1. mu<0 => West)\n")
    L.append(f"Estimated S/N for I          :{snr}                  ! (DEFAULT: 1000)\n")
    L.append(f"Continuum contrast           :{contcont}             ! (DEFAULT: not used)\n")
    L.append(f"Tolerance for SVD            :{svdtol}               ! (DEFAULT value: 1e-4)\n")
    L.append(f"Initial diagonal element     :{initdiag}             ! (DEFAULT value: 1.e-3)\n")
    L.append(f"Splines/Linear Interpolation :{interpstrat}          ! (0 or blank=splines, 1=linear)\n")
    L.append(f"Gas pressure at surface 1    :{gaspres1}             ! (0 or blank=Pe boundary cond.\n")
    L.append(f"Gas pressure at surface 2    :{gaspres2}             ! (0 or blank=Pe boundary cond.\n")
    L.append(f"Magnetic pressure term?      :{magpres}              ! (0 or blank=no, 1=yes\n")
    L.append(f"NLTE Departures filename     :{nltedep}              ! blanck= LTE (Ej. depart_6494.dat'\n")
    
    with open(path, 'w') as f:
        f.writelines(L)
