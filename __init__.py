import matplotlib.pyplot as plt
from frontend import plot
import numpy as np
import os


path_to_sir = '/home/sbalan7/Downloads/Code/Sunspots/SIR/sir2020_04new/sir.x'

class Atmosphere:
    """
    Allows for easy creation and modification of the atmosphere. Send this to the 
    SIR class for running SIR with this instance of the atmosphere.
    """
    def __init__(self, path_to_model, path_to_file='modelg.mod', taumin=None, taumax=None):
        self.atm0, self.atm1 = np.loadtxt(path_to_model, skiprows=0, max_rows=1), np.loadtxt(path_to_model, skiprows=1)
        
        self.model = path_to_model
        self._file = path_to_file

        if taumin is not None:
            mask = np.where(self.atm1[:, 0] > taumin)
            self.atm1 = self.atm1[mask]

        if taumax is not None:
            mask = np.where(self.atm1[:, 0] < taumax)
            self.atm1 = self.atm1[mask]
        
    #def set_atmospheric_parameters(self, temp=None, pres=None, micr=None, magf=None, vlos=None, incl=None, azim=None):
    def set_atmospheric_parameters(self, method, **kwargs):
        """
        Set atmospheric parameters by setting the type of the input.
        Accepts 'scalar', 'lineartau', 'linear', 'custom'.
        'scalar': results in the correspinding value entered for the argument being used as is.
        'lineartau': needs a tuple (a, b), p = a + b * tau will be used to set the guess atm.
        'linear': needs a tuple (a, b, c), p = a + b * tau + c * p, the parameter gets a linear 
        dependence on both tau and itself while also getting shifted by a.
        'custom': provide the complete variation of the parameter over logtau which will be used.
        'first': allows for setting the parameters in the first line, as macr for macroturbulent
        velocity, fill for filling factor, stry for stray light in the kwargs. 
        For kwargs, use temp for temperature, pres for pressure, micr for 
        microturbulent velocity, magf for magnetic field, vlos for line
        of sight velocity, incl for inclination and azim for azimuth
        """

        if method=='first':
            params = {'macr': 0, 'fill': 1, 'stry': 2}

            vals = [(params[kw], kwargs[kw]) for kw in kwargs]
            
            for idx, val in vals:
                self.atm0[idx] = val
            
            return

        params = {'temp': 1, 'pres': 2, 'micr': 3, 'magf': 4, 'vlos': 5, 'incl': 6, 'azim': 7}
        
        vals = [(params[kw], kwargs[kw]) for kw in kwargs]
        
        if method=='scalar':
            for idx, val in vals:
                self.atm1[:, idx] = val + self.atm1[:, idx] * 0

        if method=='lineartau':
            for idx, val in vals:
                self.atm1[:, idx] = val[0] + self.atm1[:, 0] * val[1]
        
        if method=='linear':
            for idx, val in vals:
                self.atm1[:, idx] = val[0] + self.atm1[:, 0] * val[1] + self.atm1[:, idx] * val[2]
        
        if method=='custom':
            for idx, val in vals:
                self.atm1[:, idx] = val
    
    def write_atm(self, path=None):
        if path is None:
            path = self._file
        with open(path, 'wb') as f:
            np.savetxt(f, self.atm0.reshape(1, 3), delimiter='    ', fmt='%.5f')
            np.savetxt(f, self.atm1, delimiter='    ', fmt=['%.3f','%.4f','%.6e','%.6e','%.4f','%.4e','%.4f','%.4f','%.6f','%.6e','%.9e'])

    def plot_atmosphere(self, fig=None, axs=None, **kwargs):
        if fig is None:
            fig, axs = plt.subplots(2, 3, figsize=(10, 6))
        fig, axs = plot.add_atmosphere(self.atm1, fig, axs, **kwargs)
        return fig, axs

class Profiles:
    """
    For manipulation of the Stokes profiles and their corresponding parameters.
    """
    def __init__(self, path_to_file):
        [self.indices, self.lambdas, self.linei, self.lineq, self.lineu, self.linev] = np.loadtxt(path_to_file).T
        self.path_to_file = path_to_file
        self.resolution = None

    def modify_index(self, index):
        # add a modification to edit the index of an existing line if multiple are present
        self.indices = index * np.ones(len(self.indices))

    def add_profile(self, profile2): 
        # add a check to ensure that the new profile doesnt have a repeated index
        self.indices = np.concatenate((self.indices, profile2.indices))
        self.lambdas = np.concatenate((self.lambdas, profile2.lambdas))

        self.linei = np.concatenate((self.linei, profile2.linei))
        self.lineq = np.concatenate((self.lineq, profile2.lineq))
        self.lineu = np.concatenate((self.lineu, profile2.lineu))
        self.linev = np.concatenate((self.linev, profile2.linev))
    
    def write_profile(self, path=None):
        if path is None:
            path = path_to_file
        with open(path, 'wb') as f:
            np.savetxt(f, np.transpose([self.indices, self.lambdas, self.linei, self.lineq, self.lineu, self.linev]), delimiter='    ',fmt=['%.1f','%.8f','%.8e','%.8e','%.8e','%.8e'])
    
    def plot_profiles(self, index, fig=None, axs=None, **kwargs):
        if fig is None:
            fig, axs = plt.subplots(2, 2)
        data = np.array([self.indices, self.lambdas, self.linei, self.lineq, self.lineu, self.linev])
        line = data[:,np.where(data[0] == index)[0]]
        fig, axs = plot.plot_spectra(line, fig, axs, **kwargs)
        return fig, axs
    
    def write_wavelength_file(self, resolution, grid_path='malla.grid'):
        self.resolution = resolution
        data = np.array([self.indices, self.lambdas, self.linei, self.lineq, self.lineu, self.linev])
        idxs = np.unique(self.indices)
        L = []

        for idx in idxs:
            line = data[:,np.where(data[0] == idx)[0]][1]
            line = f"{int(idx):<10}:{line[0]:15.4f},{resolution:7.1f},{line[-1]:15.4f}\n"
            L.append(line)

        with open(grid_path, 'w') as f:
            f.writelines(L)

        [self.indices, self.lambdas, self.linei, self.lineq, self.lineu, self.linev] = data

class SIR:
    def __init__(self, atmosphere, profiles, trol_file, lines_file, abundances, stray_light=None, psf=None, ncycles=None):
        self.atm = atmosphere
        self.prf = profiles
        self.ncycles = ncycles

        self.pwd = os.getcwd()
        self.run_path, self.trol = os.path.split(trol_file)
        _, self.abundances = os.path.split(abundances)
        _, self.lines = os.path.split(lines_file)
        
        self.trol_path = os.path.join(self.run_path, self.trol)
        self.atm_path  = os.path.join(self.run_path, 'modelg.mod')
        self.prf_path  = os.path.join(self.run_path, 'profiles.per')
        self.grid_path = os.path.join(self.run_path, 'malla.grid')

        if not os.path.isfile(lines_file):
            raise FileNotFoundError('Lines file is not found, place it in the same directory as the trol file')

        if not os.path.isfile(abundances):
            raise FileNotFoundError('Abundances file is not found, place it in the same directory as the trol file')
        
        '''
        # To be implemented later
        if stray_light is not None:
            self.stray_light = 0
        
        if psf is not None:
            self.psf = 0
        '''
        
    def edit_trol_file(self, param, value):
        with open(self.trol_path, 'r+') as f:
            L = f.readlines()

        params = {'ncycles':0, 'profiles':1, 'stray':2, 'psf':3, 'grid':4, 'lines':5, 'abundances':6, 'atmguess1':7, 'atmguess2':8, 'wstoki':9, 'wstokq':10, 
        'wstoku':11, 'wstokv':12, 'autonode':13, 'tempnodes1':14, 'presnodes1':15, 'micrnodes1':16, 'magfnodes1':17, 'vlosnodes1':18, 'inclnodes1':19, 
        'azimnodes1':20, 'invmacro1':21, 'tempnodes2':22, 'presnodes2':23, 'micrnodes2':24, 'magfnodes2':25, 'vlosnodes2':26, 'inclnodes2':27, 
        'azimnodes2':28, 'invmacro2':29, 'invfill':30, 'invstray':31, 'mu':32, 'snr':33, 'contcont':34, 'svdtol':35, 'initdiag':36, 'interpstrat':37, 
        'gaspres1':38, 'gaspres2':39, 'magpres':40, 'nltedep':41}
        
        for (p, v) in zip(param, value):
            idx = params[p]
            line = L[idx]
            if '!' in line:
                L[idx] = line.split(":")[0] + ':' + str(v) + '!' + line.split("!")[1]
            else:
                L[idx] = line.split(":")[0] + ':' + str(v) + '\n'

        with open(self.trol_path, 'w') as f:
            f.writelines(L)

    def run_SIR(self, suppress=True):
        os.chdir(self.run_path)
        if suppress:
            os.system('echo ' + self.trol + ' | '+path_to_sir+' >/dev/null 2>&1')
        else:
            os.system('echo ' + self.trol + ' | '+path_to_sir)
        os.chdir(self.pwd)

class Inversion(SIR):
    def __init__(self, atmosphere, profiles, trol_file, lines_file, abundances, stray_light=None, psf=None, 
                 ncycles=2, autonode=0, weights=[], nodes1=[], nodes2=[]):
        '''
        # Temporarily works only with the normal operation of SIR.
        # Can not handle two atmospheres and advanced files yet.
        '''
        super().__init__(atmosphere, profiles, trol_file, lines_file, abundances, stray_light=stray_light, psf=psf, ncycles=ncycles)
        self.autonode = autonode
        self.op_atmos = None
        self.op_profiles = None

        if len(weights) == 4:
            self.weights = weights    
        elif len(weights) == 0:
            self.weights = [1, 1, 1, 1]
            print('Using default weights of [1, 1, 1, 1]')
        else:
            raise ValueError('Weights is a list of length 4')
        
        if len(nodes1) == 7:
            self.nodes1 = nodes1
        elif len(nodes1) == 0:
            self.nodes1 = ['1', '', '1', '1', '1', '1', '1']
        else:
            raise ValueError('Nodes is a list of length 7')
        
        if len(nodes2) == 7:
            self.nodes2 = nodes2
        elif len(nodes2) == 0:
            self.nodes2 = ['', '', '', '', '', '', '']
        else:
            raise ValueError('Nodes is a list of length 7')
        
    def run_inversion(self, suppress=True):
        self.atm.write_atm(path=self.atm_path)
        self.prf.write_profile(path=self.prf_path)
        self.prf.write_wavelength_file(self.prf.resolution, self.grid_path)

        self.edit_trol_file(['ncycles'], [str(self.ncycles)])
        self.edit_trol_file(['profiles', 'grid', 'lines', 'abundances', 'atmguess1'], ['profiles.per', 'malla.grid', self.lines, self.abundances, 'modelg.mod'])

        self.edit_trol_file(['wstoki', 'wstokq', 'wstoku', 'wstokv'], [str(int(w)) for w in self.weights])

        self.edit_trol_file(['tempnodes1', 'presnodes1', 'micrnodes1', 'magfnodes1', 'vlosnodes1', 'inclnodes1', 'azimnodes1'], [str(k) for k in self.nodes1])
        self.edit_trol_file(['tempnodes2', 'presnodes2', 'micrnodes2', 'magfnodes2', 'vlosnodes2', 'inclnodes2', 'azimnodes2'], [str(k) for k in self.nodes2])

        self.run_SIR(suppress)
        
        
        self.op_profiles = [Profiles(os.path.join(self.run_path, f'modelg_{str(k+1)}.per')) for k in range(self.ncycles)]
        self.op_atmos = [Atmosphere(os.path.join(self.run_path, f'modelg_{str(k+1)}.mod')) for k in range(self.ncycles)]



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
