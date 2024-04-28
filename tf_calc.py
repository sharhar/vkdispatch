import numpy as np

import pprint
from scipy.fft import rfft2, irfft2, fft2, ifft2, fftshift, ifftshift, fftfreq, rfftfreq

alpha = 0.0072973525693 # fine structure constant
c = 299792458 # speed of light [m/s]
hbar = 6.62607004e-34/(2*np.pi) # reduced planck's constant [kg*m^2/s]
e_mass = 0.511e6 # electron mass [eV/c^2]
e_compton = 2.42631023867e-2 # compton wavelength [A]
n_avogadro = 6.02214076e23 # avogadro's number [mol^-1]
a0 = 0.529 # bohr radius [A]
e_charge_VA = 14.4 # electron charge [V*A]
e_charge_SI = 1.602176634e-19 # elementary charge [C]
J_per_eV = 1.602176634e-19 # joules per electron volt (unit conversion)
rho_bulk_ice = 0.0314315 # amorphous ice number density [molecules/A^3]

def get_freqCoord(shape, pix_size, neg_flag=False):
    """
    Frequency space for a camera with given shape and pixel size.
    Units: inverse of pix_size units.
    """
    fax_r = fftshift(fftfreq(shape[0], d=pix_size))
    fax_c = fftshift(fftfreq(shape[1], d=pix_size))
    if neg_flag:
        fax_r = -fax_r
        fax_c = -fax_c
    fgrid = np.meshgrid(fax_r, fax_c, indexing='ij')
    return (fgrid, fax_r, fax_c)

def get_azimuthalCoord(row, col):
    """
    Generates azimuthal coordinate using atan2.
    Units: [rad]
    """
    phi = np.arctan2(row, col) # azimuthal coordinate [rad]
    return phi

def get_radialCoord(row, col, dep=None):
    """
    Generates radial coordinate.
    """
    if dep is None:
        rho = (row**2 + col**2)**(1/2)
    else:
        rho = (row**2 + col**2 + dep**2)**(1/2)
    return rho

def get_johnsonEnvelope(srad, johnson_var):
    """
    Calculates transfer function envelope due to Johnson noise [dimensionless]
    Units: [dimensionless]
    """
    return np.exp(-2 * np.pi**2 * johnson_var * srad**2)

def get_gammaLorentz(HT):
    """
    Calculates Lorentz factor [dimensionless] from HT [V].
    Units: [dimensionless]
    """
    # note electron mass is in [V]
    return 1 + HT / e_mass # [dimensionless]

def get_beta(HT):
    """
    Calculates electron speed [units of speed of light] from HT [V].
    """
    gamma_lorentz = get_gammaLorentz(HT) # Lorentz factor [dimensionless]
    return np.sqrt(1 - 1 / gamma_lorentz**2) # [units of c]

def get_eWlenFromHT(HT):
    """
    Calculates electron wavelength [A] from HT [V].
    Units: [A]
    """
    gamma_lorentz = get_gammaLorentz(HT) # Lorentz factor [dimensionless]
    beta = get_beta(HT) # electron speed relative to c [dimensionless]
    return e_compton / (gamma_lorentz * beta) # electron wavelength [A]

def get_circPowerFromEta0Deg(eta0_deg, HT, l_wlen, NA):
    """
    Calculates LPP circulating power [W] from maximum phase value, eta_0 [deg].
    Units [W]
    """
    e_wlen = get_eWlenFromHT(HT) # electron wavelength [A]
    eta0_rad = np.deg2rad(eta0_deg)
    
    # circulating power
    return eta0_rad * hbar * c**2 / (np.sqrt(2 / np.pi**3) * alpha *\
                                     e_wlen*1e-10 * l_wlen*1e-10 * NA) # [W]

def get_eta0DegFromCircPower(circ_power, HT, l_wlen, NA):
    """
    Calculates maximum phase  value, eta_0 [rad], from LPP circulating power [W].
    Units: [deg]
    """
    e_wlen = get_eWlenFromHT(HT) # electron wavelength [A]
    
    # max phase shift (note: this is computed in SI units)
    eta0_rad = np.sqrt(2 / np.pi**3) * alpha * circ_power * e_wlen*1e-10 *\
        l_wlen*1e-10 * NA / (hbar * c**2) # [rad]
    return np.rad2deg(eta0_rad) # [deg]

def get_reversalAngleDeg(HT):
    """
    Calculates reversal angle [deg] from HT [V].
    Units: [deg]
    """
    beta = get_beta(HT) # electron speed relative to c [dimensionless]
    return np.rad2deg(np.arccos(np.sqrt(1/2)/beta)) # [deg]

def get_sigmaE(HT):
    """
    Calculates interaction parameter for scaling between projected potential and phase.
    Units: [rad/(V*A)]
    """
    gamma_lorentz = get_gammaLorentz(HT) # Lorentz factor
    beta = get_beta(HT) # electron speed relative to c [dimensionless]
    wlen = e_compton / (gamma_lorentz * beta) # wavelength [A]
    return 2*np.pi / (wlen * HT) * ((e_mass * J_per_eV) + e_charge_SI * HT) /\
        (2 * e_mass * J_per_eV + e_charge_SI*HT) # [rad/(V*A)]

def get_sigmaFromFWHM(fwhm):
    """
    Converts full-width at half maximum to standard deviation.
    """
    # convert full width at half maximum to standard deviation
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def get_relativisticEpsilon(HT):
    """
    Calculates relativistic correction factor [dimensionless]
    Units: [dimensionless]
    """
    return (1 + HT / e_mass) / (1 + HT / (2 * e_mass)) # [dimensionless]

def get_defocusSpread(Cc, HT, energy_spread_std, accel_voltage_std, lens_current_std):
    """
    Calculates the defocus spread for evaluation of temporal coherence.
    Units: [A]
    """
    epsilon = get_relativisticEpsilon(HT) # relativistic correction
    return Cc * np.sqrt((epsilon * energy_spread_std / HT)**2 +\
                        (epsilon * accel_voltage_std)**2 +\
                            4 * lens_current_std**2) # standard deviation [A]

class PhasePlate:
    """
    PhasePlate is the parent class of phase plates for the back focal plane.
    """
    def __init__(self, **kwargs):
        # default values
        # self.eta0_deg = 90
        
        # update values from kwargs
        allowed_keys = {'eta0_deg'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
    
    def summary(self):
        summary_keys = {'eta0_deg', 'mode',
                        'wlen', 'NA', 'rc_ang_deg', 'cz_ang_deg', 'pol_ang_deg',
                        'long_offset', 'trans_offset', 'long_phi_deg', 'circ_power', 'w0', 'zR',
                        'zernike_cut_on'}
        summary = {}
        summary.update((k, v) for k, v in self.__dict__.items() if k in summary_keys)
        print('PhasePlate summary:')
        pprint.pprint(summary)

class LaserPP(PhasePlate):
    """
    Class for laser phase plate objects.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # default values
        self.mode = 'lpp'
        self.wlen = 1.064e4
        self.NA = 0.05
        self.rc_ang_deg = 0 # in-plane angle [deg]
        self.cz_ang_deg = 0 # out-of-plane angle [deg]
        self.pol_ang_deg = 90 # light polarization [deg] or 'rra' for reversal angle
        self.long_offset = 0 # longitudinal laser offset [A]
        self.trans_offset = 0 # transeverse laser offset [A]
        self.long_phi_deg = 0 # longitudinal shift [deg], complementary to long_offset (0: antinode)
        
        # update values from kwargs
        allowed_keys = {'eta0_deg', 'circ_power',
                        'wlen', 'NA', 'rc_ang_deg', 'cz_ang_deg', 'pol_ang_deg',
                        'long_offset', 'trans_offset', 'long_phi_deg'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        
        self.get_derivLaserParams()
        
        self.check_laserOn()
    
    def get_derivLaserParams(self):
        """
        Calculates higher-order parameters of the LPP.
        """
        self.w0 = self.wlen / (np.pi * self.NA) # 1/e^2 waist radius [A]
        self.zR = self.w0 / self.NA # Rayleigh range [A]
    
    def check_laserOn(self, silent=True):
        """
        Checks if the laser is 'on', meaning power or phase shift is specified.
        """
        if not any([k in ['eta0_deg', 'circ_power'] for k in self.__dict__.keys()]):
            raise Exception('Either eta0_deg or circ_power should be specified.')
        if not silent:
            print('Laser is on')
        
class ZernikePP(PhasePlate):
    """
    Class for Zernike phase plate objects.
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        # default values
        self.mode = 'zernike'
        self.eta0_deg = 90
        self.zernike_cut_on = 0.01 # cut on frequency for zernike pp [A^-1]
        
        # update values from kwargs
        allowed_keys = {'mode', 'eta0_deg', 'zernike_cut_on'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

class NoPP(PhasePlate):
    """
    Class specifying no phase plate.
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        # default values
        self.mode = 'off'
        self.eta0_deg = 0
        
        # update values from kwargs
        allowed_keys = {'mode'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

class Microscope:
    """
    Class for Microscope objects, contains details for calculation of aberrations and coherence.
    """
    def __init__(self, **kwargs):
        # default values
        self.HT = 300e3 # high tension [V]
        self.astig_z = 0 # astigmatism half-offset (extrema are +/- from defocus) [A]
        self.astig_deg = 0 # angle of astigmatism major axis [deg]
        self.Cs = 4.8e7 # spherical aberration [A]
        self.Cc = 7.6e7 # chromatic aberration [A]
        self.f_OL = 20e7 # effective focal length [A]
        self.energy_spread_fwhm = 0.9 # full-width half-max [eV]
        self.beam_semiangle = 2.5e-6 # standard deviation [rad]
        self.accel_voltage_std = 0.07e-6 # standard deviation/value (0.07 ppm)
        self.lens_current_std = 0.1e-6 # standard deviation/value (0.1 ppm)
        self.johnson_var = 0.137 # variance of Johnson noise [A^2]
        self.enable_coh_env = True # basic coherence envelope
        self.enable_coh_xterm = False # cross term in coherence envelope
        self.enable_coh_pp = False # phase plate coherence envelope
        self.enable_coh_johnson = False # enable Johnson noise envelope
        
        # update values from kwargs
        allowed_keys = {'HT', 'astig_z', 'astig_deg', 'Cs', 'Cc', 'f_OL',
                        'energy_spread_fwhm', 'beam_semiangle', 'accel_voltage_std', 'lens_current_std', 'johnson_var',
                        'enable_coh_env', 'enable_coh_xterm', 'enable_coh_pp', 'enable_coh_johnson'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        self.get_derivParams()

    def get_derivParams(self):
        """
        Calculates higher-order parameters of Microscope objects.
        """
        self.gamma_lorentz = get_gammaLorentz(self.HT) # Lorentz factor [dimensionless]
        self.beta = get_beta(self.HT) # electron speed relative to c [dimensionless]
        self.wlen = get_eWlenFromHT(self.HT) # electron wavelength [A]
        self.sigma_e = get_sigmaE(self.HT) # interaction parameter [rad/(V*A)]
        self.energy_spread_std = get_sigmaFromFWHM(self.energy_spread_fwhm) # energy spread [std]
        self.defocus_spread = get_defocusSpread(self.Cc, self.HT, self.energy_spread_std,
                                                   self.accel_voltage_std, self.lens_current_std) # defocus spread (std) [A]
    
    def summary(self):
        summary_keys = {'HT', 'astig_z', 'astig_deg', 'Cs', 'Cc', 'f_OL',
                        'energy_spread_fwhm', 'beam_semiangle', 'accel_voltage_std', 'lens_current_std', 'johnson_var',
                        'enable_coh_env', 'enable_coh_xterm', 'enable_coh_pp', 'enable_coh_johnson'}
        summary = {}
        summary.update((k, v) for k, v in self.__dict__.items() if k in summary_keys)
        print('Microscope summary:')
        pprint.pprint(summary)

def get_laserCoordsL(pp, sco: Microscope, sgrid):
    """
    Generates dimensionless coordinate system L for calculating laser pattern.
    Units: [dimensionless]
    """
    
    # calculate physical coordinates in the back focal plane
    Sc_phys = sgrid[1] * sco.wlen * sco.f_OL # [A]
    Sr_phys = sgrid[0] * sco.wlen * sco.f_OL # [A]
    
    # rotate and translate
    Rr_rot = -Sc_phys * np.sin(np.deg2rad(pp.rc_ang_deg)) +\
        Sr_phys * np.cos(np.deg2rad(pp.rc_ang_deg)) # [A]
    Rc_rot = Sc_phys * np.cos(np.deg2rad(pp.rc_ang_deg)) +\
        Sr_phys * np.sin(np.deg2rad(pp.rc_ang_deg)) # [A]
    Rr_trans = Rr_rot - pp.trans_offset # transverse offset [A]
    Rc_trans = Rc_rot - pp.long_offset # longitudinal offset [A]
    
    # make dimensionless
    Lr = Rr_trans / pp.w0 # [dimensionless]
    Lc = Rc_trans / pp.zR # [dimensionless]
    
    return (Lr, Lc)   

def get_laserPhase(pp, sco: Microscope, sgrid):
    """
    Calculates the standing wave profile of a (single) laser phase plate.
    Units: [rad]
    """
    
    # calculate dimensionless laser coordinates
    (Lr, Lc) = get_laserCoordsL(pp, sco, sgrid) # [dimensionless]
    
    # calculate phase due to laser standing wave
    return pp.eta0_rad / 2 * np.exp(-2 * Lr**2 / (1 + Lc**2)) / np.sqrt(1 +  Lc**2) *\
        (1 + (1 - 2 * sco.beta**2 * (np.cos(np.deg2rad(pp.pol_ang_deg)))**2) *\
            np.exp(-np.deg2rad(pp.cz_ang_deg)**2 * (2 / (pp.NA)**2) * (1 + Lc**2)) *\
                (1 + Lc**2)**(-1 / 4) * np.cos(2 * Lc * Lr**2 / (1 + Lc**2) +\
                                            4 * Lc / (pp.NA **2) -\
                                                1.5 * np.arctan(Lc) -\
                                                    np.deg2rad(pp.long_phi_deg))) # [rad]

def get_zernikePhase(pp: ZernikePP, srad):
    """
    Calculates phase profile for a Zernike phase plate.
    Units: [rad]
    """
    
    return np.where(srad < pp.zernike_cut_on, pp.eta0_rad, 0) # [rad]

def get_phasePlatePhase(pp, sco: Microscope, srad, sgrid):
    """
    Determines total contribution of all phase plates to eta.
    Units: [rad]
    """
    
    # values to be incremented
    eta_tot = np.zeros_like(srad) # cumulative phase pattern [rad]
    eta0_tot_deg = 0 # cumulative eta0 [deg]
    
    if pp.mode == 'off':
        pass
    elif pp.mode == 'zernike':
        eta0_tot_deg += pp.eta0_deg # [deg]
        pp.eta0_rad = np.deg2rad(pp.eta0_deg) # max phase shift [rad]
        
        # phase plate phase
        eta_tot += get_zernikePhase(pp, srad) 
    elif pp.mode == 'lpp': # LPP handling: power, phase shift, polarization
        # ensure power and phase shift are both calculated
        if 'eta0_deg' in pp.__dict__.keys():
            pp.circ_power = get_circPowerFromEta0Deg(pp.eta0_deg, sco.HT,
                                                        pp.wlen, pp.NA) # [W]
            pp.eta0_rad = np.deg2rad(pp.eta0_deg) # [rad]
            eta0_tot_deg += pp.eta0_deg # [deg]
        elif 'circ_power' in pp.__dict__.keys():
            eta0_deg = get_eta0DegFromCircPower(pp.circ_power, sco.HT,
                                                        pp.wlen, pp.NA) # [deg]
            pp.eta0_rad = np.deg2rad(eta0_deg) # [rad]
            eta0_tot_deg += eta0_deg # [deg]
            
        else:
            raise Exception('Either eta0_deg or circ_power should be specified.')
        
        # laser polarization handling
        if pp.pol_ang_deg == 'rra':
            pp.pol_ang_deg = get_reversalAngleDeg(sco.HT) # [deg]
            print('Set laser polarization to reversal angle ({:.2f} deg)'.format(pp.pol_ang_deg))
        
        # phase plate phase
        eta_tot += get_laserPhase(pp, sco, sgrid) # [rad]
            
    # store eta0 in radians also
    eta0_tot_rad = np.deg2rad(eta0_tot_deg) # [rad]
    
    return (eta_tot, eta0_tot_rad) # ([rad], [rad])

def prepareTF(shape, pix_size, mode, upsample_factor=2) -> None:
    rows = shape[0] * upsample_factor
    cols = shape[1] * upsample_factor

    pix_size = pix_size / upsample_factor

    sco = Microscope(enable_coh_env=True, enable_coh_johnson=False, Cs=2.7e7, Cc=2.7e7, f_OL=3.5e7)

    (sgrid_unshift, sax_r, sax_c) = get_freqCoord((rows, cols), pix_size) # [A^-1]

    sgrid = [np.fft.ifftshift(sgrid_unshift[0]), np.fft.ifftshift(sgrid_unshift[1])]

    srad = get_radialCoord(sgrid[0], sgrid[1])
    sphi = get_azimuthalCoord(sgrid[0], sgrid[1])

    zernike_cut_on = 0.000675 # [1/A]
    oa = NoPP() # "open aperture"
    zpp = ZernikePP(eta0_deg=90, zernike_cut_on=zernike_cut_on) # zernike phase plate
    lpp = LaserPP(eta0_deg=90) # laser phase plate

    modes = [oa, lpp, zpp]

    pp = modes[mode] # specify phase plate to be used

    (eta_tot, _) = get_phasePlatePhase(pp, sco, srad, sgrid) # [rad]
    
    sigma_f = sco.defocus_spread # [A]
    sigma_s = sco.beam_semiangle / sco.wlen # [A^-1] from radians
    
    V2 = -np.pi * sco.wlen * srad**2 # [A]

    V_scaler = np.ones(shape=srad.shape) * (-sigma_s**2 / 2)

    mag_pre = np.exp(- sigma_f**2 * V2**2 / 2) * get_johnsonEnvelope(srad, sco.johnson_var)

    gamma_pre_adder = np.pi * 0.5 * sco.Cs * sco.wlen**3 * srad**4
    gamma_pre_scaler = -np.pi * sco.wlen * srad**2

    V1_r_adder = 2 * np.pi * (sco.astig_z * sco.wlen * srad * np.sin(sphi - 2 * np.deg2rad(sco.astig_deg)) + sco.Cs * sco.wlen**3 * srad**3 * np.sin(sphi))
    V1_r_scaler = -2 * np.pi * sco.wlen * srad * np.sin(sphi)

    V1_c_adder = 2 * np.pi * (sco.Cs * sco.wlen**3 * srad**3 * np.cos(sphi) - sco.astig_z * sco.wlen * srad * np.cos(sphi - 2 * np.deg2rad(sco.astig_deg)))
    V1_c_scaler = -2 * np.pi * sco.wlen * srad * np.cos(sphi)

    return (V1_r_scaler, V1_c_scaler, 
                        V1_r_adder, V1_c_adder,
                        mag_pre, V_scaler, 
                        gamma_pre_scaler, gamma_pre_adder, eta_tot)
