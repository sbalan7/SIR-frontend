# GREGOR Data Analysis and SIR Frontend Testing

SIR Frontend v0.01

TODO:
1. Rewrite the functions into classes for a cleaner and more powerful library
2. Add more support to plotting with matplotlib's API properly
3. Add comments and clean up code

To analyse data from GREGOR, first extract information from the `\level1` fits files using standard IO operations. Once the Stokes parameters are extracted from these files, the `\frontend` package can be used to analyse this information. It is necessary to have this folder in the root directory of where the other code will be run to avoid a `ModuleNotFound` error. A suggestion is to create a different folder within this directory to allow for an easy manipulation of the SIR input and output files.

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

Once the data is loaded from the fits files, a line of interest can be extracted from it using numpy operations. To provide this information to SIR, save this data to a profiles file using the frontend package as `frontend.write_profile(path, index, lambdas, linei, lineq, lineu, linev, add=False)`. `index` is an integer parameter corresponding to the index of the transition as used in the wavelength file or the atomic parameters file. `lambdas` and the four `linex` parameters correspond to the wavelength of the profiles and their Stokes parameters in order. If required, multiple different lines can be written to the same file by setting the `add` parameter to `True`, which will result in the new line getting appended instead of overwritten. Similarly, a profiles file can also be read by using the `frontend.read_existing_profile(path)`. Once the profiles file is written to disk, it can be plotted by using `frontend.plot.plot_spectra(path, color, fig, axs)`. It is possible to overplot multiple profiles as shown below.

```python
import matplotlib.pyplot as plt
import frontend as sir

fig, axs = plt.subplots(2, 2)
fig, axs = sir.plot.plot_spectra('modelg_0.per', 'black', fig, axs)
fig, axs = sir.plot.plot_spectra('modelg_1.per', 'blue', fig, axs)
fig, axs = sir.plot.plot_spectra('modelg_2.per', 'red', fig, axs)
fig.tight_layout()
plt.show()
```

The next step is to prepare an inital atmosphere guess for inversions. An atmosphere with constant values can be set by using `frontend.set_const_atm(model, temp=None, pres=None, micr=None, magf=None, vlos=None, incl=None, azim=None)`. For any physical parameter passed to this method, the constant will be set for that parameter across the entire optical depth range. A more complex atmosphere consisting of variation can be set by using `frontend.set_complex_atm(model, macr, fill, strl, logt, temp, pres, micr, magf, vlos, incl, azim)` (this method is a work in progress, will add an option to have a linear stratification and to make editing parameters optional). Much like with the profiles, the atmospheric parameters can be plotted using `frontend.plot.add_atmosphere(path, color, fig, axs)`. However, the shape of subplots would have to be `(2, 3)` this time as the method plots temperature, microturbulent velocity, LOS velocity, magnetic field strength, inclination and azimuth against the logarithm of the optical depth.

The wavelength file will be automatically calculated from the profiles file for all lines in it. Calling `frontend.wavelength_file(profile_path, grid_path, resolution)` will calculate and update the file accordingly. The `frontend.atomic_parameters_file()` method is still a work in progress. 

The control file (trol file) can be written either using this package's `frontend.set_trol_file` method or by writing the text file manually. Each parameter in a trol file can also be edited individually instead of rewriting the whole trol file by using the `frontend.edit_trol_file(path, param, value)` method. A list of parameters and their corresponding values (in another list passed to the same method) can be used to selectively edit only some of the parameters. Do note that even if a single parameter is being edited, it must be passed to the function within a list as so `fronend.edit_trol_file('sir.trol', ['profiles'], ['sample.per'])`. Similarly, multiple parameters can be edited by using `fronend.edit_trol_file('sir.trol', ['ncycles', 'profiles', 'wstoki'], [5, 'sample.per', 10])`. The names of the parameters must be the same as those used in `frontend.set_trol_file`.

```python
set_trol_file(path, ncycles, profiles, tempnodes1, presnodes1, micrnodes1, magfnodes1, vlosnodes1, inclnodes1, azimnodes1,
                   stray="", psf="", grid='malla.grid', lines='LINES', abundances='ASPLUND', atmguess1='modelg.mod', atmguess2='',
                   wstoki=1, wstokq=1, wstoku=1, wstokv=1, autonode="", invmacro1="", invmacro2="", invfill="", invstray="",
                   tempnodes2="", presnodes2="", micrnodes2="", magfnodes2="", vlosnodes2="", inclnodes2="", azimnodes2="",
                   mu="", snr="", contcont="", svdtol="", initdiag="", interpstrat="", gaspres1="", gaspres2="", magpres="", nltedep="")
```

To run SIR, first set the path to SIR on your system in the `__init__.py` file. This is a one time setting which can be edited later if required. Provide the path to the directory containing the all the files required for SIR to run (the trol file and all the files mentioned within the trol file including the atmosphere, profiles, atomic parameters, wavelength, PSF, etc.) followed by the name of the trol file itself to `frontend.run_SIR(path, trol, suppress=True)`. If you want to suppress the output of running SIR, set the `supress` parameter to `True`.

An interactive line plot can be made by using the `frontend.plot.interactive_line_plot(x, y, xlabel, ylabel, plot_color='blue', line_color='#505050', box_color='white', box_alpha=0.8)` method (work in progress, currently only supports individual plots, updating to be compatible with subplots).
