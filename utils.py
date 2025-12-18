# Miscellations classes and functions for utility.

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import curve_fit


class get_phase_mu:
    """Collect orbital parameters and information, including orbital phases, mu, transit duration, duration between ingress and egress, array indices of in-transit and out-of-transit, 

    Parameters
    ----------
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    time : `numpy array`
        time of observations in BJD.

    Methods
    -------
    get_phase(planet_params, time)
        computes orbital phases, transit duration, time between ingress and egress, array indices in-transit and out-of-transit.
    mu(phases, planet_params)
        computes mu.
    """
    def __init__(self, planet_params:dict, time:np.array):

        phases, tr_dur, tr_ingress_egress, in_indices, out_indices = self.get_phase(planet_params, time)
        
        self.phases = phases

        self.tr_dur = tr_dur
        self.tr_ingress_egress = tr_ingress_egress

        self.in_indices = in_indices
        self.out_indices = out_indices

        mu_values = self.mu(phases, planet_params)
        self.mu_values = mu_values

    def get_phase(self, planet_params:dict, time:np.array):

        t0         = planet_params["t0"]
        dfp        = planet_params["dfp"]
        P_orb      = planet_params["P_orb"]
        inc_planet = np.radians(planet_params["inc_planet"])
        Rp_Rs      = planet_params["Rp_Rs"]
        a_R        = planet_params["a_R"]

        t_epoch = t0 + 0.5+2.4e6 - dfp*P_orb  #MBJD
        norb = (time-t_epoch)/P_orb
        nforb = [round(x) for x in norb]
        phases = norb-nforb

        tr_dur = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1+Rp_Rs)**2 - a_R**2 * np.cos(inc_planet)**2))
        tr_ingress_egress = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1-Rp_Rs)**2 -a_R**2 * np.cos(inc_planet)**2))

        in_indices  = np.where(np.abs(phases) <= tr_dur/2)[0]
        out_indices = np.where(np.abs(phases) >  tr_dur/2)[0]

        return phases, tr_dur, tr_ingress_egress, in_indices, out_indices

    @staticmethod
    def mu(phases:np.array, planet_params:dict):

        inc_planet = planet_params["inc_planet"]
        a_R        = planet_params["a_R"]

        b = a_R*np.cos(inc_planet*np.pi/180) # impact parameter

        return np.sqrt(1 - b**2 - (a_R*np.sin(2*np.pi*np.abs(phases)))**2)
    

# linear interpolation taking into account covariances
def linear_interpolation_matrix(x_old, x_new):
    """Builds a sparse matrix W that linearly interpolates data from x_old → x_new.
    Each row i corresponds to interpolation weights for x_new[i].

    Parameters
    ----------
    x_old, x_new : `numpy array`
        original and interpolated arrays.

    Returns
    -------
    W : `numpy array`
        linear interpolation matrix.
    """
    W = lil_matrix((len(x_new), len(x_old)))

    for i, xv in enumerate(x_new):
        if xv <= x_old[0]: #extrapolate using first two points
            j = 0
            x0, x1 = x_old[j], x_old[j+1]
            w1 = (x1 - xv) / (x1 - x0)
            w2 = (xv - x0) / (x1 - x0)
        elif xv >= x_old[-1]: #extrapolate using last two points
            j = len(x_old) - 2
            x0, x1 = x_old[j], x_old[j+1]
            w1 = (x1 - xv) / (x1 - x0)
            w2 = (xv - x0) / (x1 - x0)
        else:
            j = np.searchsorted(x_old, xv) - 1
            x0, x1 = x_old[j], x_old[j+1]
            w1 = (x1 - xv) / (x1 - x0)
            w2 = (xv - x0) / (x1 - x0)

        W[i, j]   = w1
        W[i, j+1] = w2

    return W.tocsr()



class profile_models:
    """Spectral line/CCF profile models. Models available: modified Gaussian, Gaussian and Lorentzian.

    Parameters
    ----------
    model : `str`
        type of profile model to fit.

    Methods
    -------
    modified_gaussian(x, *params)
        modified Gaussian model as described in Dravins, D. et al. (2017).
    gaussian(x, *params)
        Gaussian model.
    lorentzian(x, *params)
        Lorentzian model.
    r2(y, yfit)
        compute coefficient of determination.
    """
    def __init__(self, model:str):

        if model == "modified Gaussian":
            self.model = self.modified_gaussian
        elif model == "Gaussian":
            self.model = self.gaussian
        elif model == "Lorentzian":
            self.model = self.lorentzian

    def modified_gaussian(self, x, *params):
        y0,x0,sigma,a,c = params
        return y0-a*np.exp(-0.5*(np.abs(x-x0)/sigma)**c)

    def gaussian(self, x, *params):
        y0, x0, sigma, a = params
        return y0-a*np.exp(-0.5*(np.abs(x-x0)/sigma)**2)

    def lorentzian(self, x,*params):
        y0,x0,gamma,a = params
        return y0 - a * (gamma**2 / ((x - x0)**2 + gamma**2))

    @staticmethod
    def r2(y, yfit):
        """Compute coefficient of determination.

        Parameters
        ----------
        y : `numpy array`
            observed CCF flux.
        yfit : `numpy array`
            fitted CCF flux.

        Returns
        -------
        r : float
            coefficient of determination.
        """
        ssres = np.sum((y-yfit)**2)
        sstot = np.sum((y-np.mean(y))**2)
        r = 1 - ssres/sstot
        return r


def fit_CCF(phase:float, CCF:np.array, ccf_type:str, model:str, print_output:bool):
    """Fit a CCF profile to observed CCF.

    Parameters
    ----------
    phase : `float`
        orbital phase.
    CCF : `numpy array`
        CCF profile (RV, flux and flux error).
    ccf_type : `str`
        whether it's a local, average out-of-transit or raw CCF.
    model : `str`
        type of profile model to fit.
    print_output : `bool` 
        whether to print the output.

    Returns
    -------
    central_rv : `numpy array` 
        fitted central RV of CCF.
    continuum : `numpy array` 
        fitted continuum level of CCF.
    intensity : `numpy array` 
        fitted line-center intensity of CCF.
    width : `numpy array` 
        fitted line-width measure of CCF.
    R2 : `numpy array` 
        coefficient of determination of fit.
    rv : `numpy array` 
        radial velocities array.
    d : `numpy array` 
        CCF flux array.
    de : `numpy array`
        CCF flux uncertainty array.
    y_fit : `numpy array`
        fitted CCF flux array.
    """
    d = CCF[0]
    de = CCF[1]
    rv = CCF[2]

    model_fit = profile_models(model).model
        
    if model == "modified Gaussian":
        parameters = ["y0","x0","sigma","a","c"]
        p0          = [np.max(d),           0,        1, np.max(d)-np.min(d),       1] #y0, x0, sigma, a, c
        upper_bound = [   np.inf,  np.max(rv),   np.inf,              np.inf,  np.inf]
        lower_bound = [        0,  np.min(rv),        0,                   0,       0]
        width_multiplier = 1
    
    elif model == "Gaussian":
        parameters = ["y0","x0","sigma","a"]
        p0          = [np.max(d),           0,        1, np.max(d)-np.min(d)] #y0, x0, sigma, a
        upper_bound = [   np.inf,  np.max(rv),   np.inf,              np.inf]
        lower_bound = [        0,  np.min(rv),        0,                   0]
        width_multiplier = 2*np.sqrt(2*np.log(2)) #FWHM of Gaussian
    
    elif model == "Lorentzian":
        parameters = ["y0","x0","gamma","a"]
        p0          = [np.max(d),           0,        1, np.max(d)-np.min(d)] #y0, x0, gamma, a
        upper_bound = [   np.inf,  np.max(rv),   np.inf,              np.inf]
        lower_bound = [        0,  np.min(rv),        0,                   0]
        width_multiplier = 2 #FWHM of Lorentzian

    popt, pcov = curve_fit(f=model_fit, xdata=rv, ydata=d, sigma=de, bounds=(lower_bound,upper_bound), absolute_sigma=True, p0=p0)
    y_fit = model_fit(rv, *popt)

    central_rv = [popt[1], np.sqrt(pcov[1,1])]
    continuum = [popt[0], np.sqrt(pcov[0,0])]
    intensity = [(1-(popt[3]/popt[0]))*100, ((popt[3]/popt[0])*np.sqrt(np.abs(pcov[3,3])/(popt[3]**2)+np.abs(pcov[0,0])/(popt[0]**2)))*100]
    width = [width_multiplier*popt[2],  width_multiplier*np.sqrt(pcov[2,2])] 

    R2 = np.around(profile_models.r2(d,y_fit),4)

    if print_output:

        print("#"*30)
        print(f"Fitting {model} model to {ccf_type} CCF")
        if ccf_type == "local":
            print(f"Phase: {str(phase)[:6]}")
        print("-"*30)
        print("Fit parameters:")
        for j,param in enumerate(parameters):
            print(f"{param} = {popt[j]:.06f} ± {np.sqrt(pcov[j,j]):.06f}")
        print("R^2: ", R2)
        print("-"*30)
        print("CCF parameters:")
        print(f"Central RV [km/s]: {central_rv[0]:.06f} ± {central_rv[1]:.06f}")
        print(f"Continuum: {continuum[0]:.06f} ± {continuum[1]:.06f}")
        print(f"Line-center intensity [%]: {intensity[0]:.06f} ± {intensity[1]:.06f}")
        print(f"Line-width measure [km/s]: {width[0]:.06f} ± {width[1]:.06f}")

    return central_rv, continuum, intensity, width, R2, rv, d, de, y_fit