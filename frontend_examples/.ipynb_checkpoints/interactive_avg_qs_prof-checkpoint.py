import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import mplcursors

def stokes(path, iquv):
    f = fits.open(path)
    
    data = f[0].data
    head = f[0].header
    
    return data[iquv::4,:,:]

def load_data(k):
    path1 = f"level1/28jun22.00{k}-01cc"
    path2 = f"level1/28jun22.00{k}-02cc"

    stoki = np.append(stokes(path1, 0), stokes(path2, 0), axis=0)/100000
    stokq = np.append(stokes(path1, 1), stokes(path2, 1), axis=0)/100000
    stoku = np.append(stokes(path1, 2), stokes(path2, 2), axis=0)/100000
    stokv = np.append(stokes(path1, 3), stokes(path2, 3), axis=0)/100000
    
    return (stoki, stokq, stoku, stokv)

# change this to use the right fits file
# load_data() works only for the case where
# the observation is split into two files
(stoki, stokq, stoku, stokv) = load_data("4")

stoki = stoki[40:90, 40:90].mean(axis=(0, 1))
stokq = stokq[40:90, 40:90].mean(axis=(0, 1))*100.
stoku = stoku[40:90, 40:90].mean(axis=(0, 1))*100.
stokv = stokv[40:90, 40:90].mean(axis=(0, 1))*100.

wl_offset = 10823.9395911
wl_disp = 0.0181301857924

indices = np.arange(0, len(stoki), 1)
lambdas = (wl_offset + indices * wl_disp)

def show_annotation(sel, xi):
    v = axs[0, 0].axvline(xi, color='#505050', ls=':', lw=1)
    sel.extras.append(v)
    v = axs[0, 1].axvline(xi, color='#505050', ls=':', lw=1)
    sel.extras.append(v)
    v = axs[1, 0].axvline(xi, color='#505050', ls=':', lw=1)
    sel.extras.append(v)
    v = axs[1, 1].axvline(xi, color='#505050', ls=':', lw=1)
    sel.extras.append(v)

def annot_I(sel):
    xi = sel.target[0]
    show_annotation(sel, xi)
    annotation_str = f'Wavelength: {xi:.3f} $\AA$\nStokes I/I$_c$: {np.interp(xi, lambdas, stoki):.3f}'
    sel.annotation.set_text(annotation_str)
    sel.annotation.get_bbox_patch().set(fc='white', alpha=0.8)

def annot_Q(sel):
    xi = sel.target[0]
    show_annotation(sel, xi)
    annotation_str = f'Wavelength: {xi:.3f} $\AA$\nStokes Q/I$_c$ [%]: {np.interp(xi, lambdas, stokq):.3f}'
    sel.annotation.set_text(annotation_str)
    sel.annotation.get_bbox_patch().set(fc='white', alpha=0.8)

def annot_U(sel):
    xi = sel.target[0]
    show_annotation(sel, xi)
    annotation_str = f'Wavelength: {xi:.3f} $\AA$\nStokes U/I$_c$ [%]: {np.interp(xi, lambdas, stoku):.3f}'
    sel.annotation.set_text(annotation_str)
    sel.annotation.get_bbox_patch().set(fc='white', alpha=0.8)

def annot_V(sel):
    xi = sel.target[0]
    show_annotation(sel, xi)
    annotation_str = f'Wavelength: {xi:.3f} $\AA$\nStokes V/I$_c$ [%]: {np.interp(xi, lambdas, stokv):.3f}\n'
    sel.annotation.set_text(annotation_str)
    sel.annotation.get_bbox_patch().set(fc='white', alpha=0.8)

fig, axs = plt.subplots(2, 2, figsize=(14, 8))
color = 'black'

I_plot = axs[0, 0].plot(lambdas, stoki, color=color)
axs[0, 0].set(xlabel='$\lambda$ [$\AA$]', title='Stokes I/I$_c$')

Q_plot = axs[0, 1].plot(lambdas, stokq, color=color)
axs[0, 1].set(xlabel='$\lambda$ [$\AA$]', title='Stokes Q/I$_c$ [%]')

U_plot = axs[1, 0].plot(lambdas, stoku, color=color)
axs[1, 0].set(xlabel='$\lambda$ [$\AA$]', title='Stokes U/I$_c$ [%]')

V_plot = axs[1, 1].plot(lambdas, stokv, color=color)
axs[1, 1].set(xlabel='$\lambda$ [$\AA$]', title='Stokes V/I$_c$ [%]')

mplcursors.cursor(I_plot, hover=True).connect('add', annot_I)
mplcursors.cursor(Q_plot, hover=True).connect('add', annot_Q)
mplcursors.cursor(U_plot, hover=True).connect('add', annot_U)
mplcursors.cursor(V_plot, hover=True).connect('add', annot_V)

fig.tight_layout()
plt.show()