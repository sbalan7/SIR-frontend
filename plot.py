import matplotlib.pyplot as plt
import numpy as np
import mplcursors


def plot_spectra(path, color, fig, axs):
    [indices, lambdas, linei, lineq, lineu, linev] = np.loadtxt(path).T

    axs[0, 0].plot(lambdas, linei, color=color)
    axs[0, 0].set(xlabel='$\lambda$ [m$\AA$]', title='Stokes I/I$_c$')

    axs[0, 1].plot(lambdas, lineq*100., color=color)
    axs[0, 1].set(xlabel='$\lambda$ [m$\AA$]', title='Stokes Q/I$_c$ [%]')

    axs[1, 0].plot(lambdas, lineu*100., color=color)
    axs[1, 0].set(xlabel='$\lambda$ [m$\AA$]', title='Stokes U/I$_c$ [%]')

    axs[1, 1].plot(lambdas, linev*100., color=color)
    axs[1, 1].set(xlabel='$\lambda$ [m$\AA$]', title='Stokes V/I$_c$ [%]')

    return fig, axs

def add_atmosphere(atm1, color, fig, axs):
    # Temperature
    axs[0, 0].plot(atm1[:,0], atm1[:,1], color=color, linewidth=1.0)
    axs[0, 0].set(title='Temperature [K]', xlabel='log '+r'$\tau$')

    # Microturbulence
    axs[0, 1].plot(atm1[:,0], atm1[:,3]/1e5, color=color, linewidth=1.0)
    axs[0, 1].set(title='Microturbulence [km/s]',xlabel='log '+r'$\tau$')

    # LOS velocity
    axs[0, 2].plot(atm1[:,0], atm1[:,5]/1e5, color=color, linewidth=1.0)
    axs[0, 2].set(title='LOS velocity [km/s]',xlabel='log '+r'$\tau$')

    # Magnetic field strength
    axs[1, 0].plot(atm1[:,0], atm1[:,4], color=color, linewidth=1.0)
    axs[1, 0].set(title='B [G]',xlabel='log '+r'$\tau$')

    # Magnetic field inclination
    axs[1, 1].plot(atm1[:,0], atm1[:,6], color=color, linewidth=1.0)
    axs[1, 1].set(title='Inclination [deg]',xlabel='log '+r'$\tau$')

    # Magnetic field azimuth
    axs[1, 2].plot(atm1[:,0], atm1[:,7], color=color, linewidth=1.0)
    axs[1, 2].set(title='Azimuth [deg]',xlabel='log '+r'$\tau$')
    
    return fig, axs

def interactive_line_plot(x, y, xlabel, ylabel, plot_color='blue', line_color='#505050', box_color='white', box_alpha=0.8):
    def show_annotation(sel):
        xi = sel.target[0]
        v = ax.axvline(xi, color=line_color, ls=':', lw=1)
        sel.extras.append(v)
        annotation_str = f'{xlabel}: {xi:.3f} \n{ylabel}: {np.interp(xi, x, y):.3f}'
        sel.annotation.set_text(annotation_str)
        sel.annotation.get_bbox_patch().set(fc=box_color, alpha=box_alpha)

    fig, ax = plt.subplots()
    ax.plot(x, y, color=plot_color)
    cursor = mplcursors.cursor(hover=True)
    cursor.connect('add', show_annotation)

    return fig, ax