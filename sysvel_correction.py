from utils import *
from plots import *

import numpy as np

class sysvel_correction:
    """
    Extract the RV component due to the star's motion around the barycentre, excluding the component due to the stellar systemic velocity.
    Fits a choosen profile to the CCF, then a linear model to the out-of-transit central RVs and subtracts it to all CCFs.
    
    Args:
        planet_params (dict): dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
        time (numpy array): time of observations in BJD.
        CCFs (numpy array): matrix with the CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points)
        model (str): type of profile model to fit (modified Gaussian from Dravins works, Gaussian or Lorentzian).
        plot_fits (bool): whether to plot the fit for each CCF.
        plot_sys_vel (bool): whether to plot the central RV (systemic velocity) in function of orbital phase. 
        save (str, optional): path to save the plots.

    Returns:
        CCFs_sysvel_corr (numpy array): CCFs corrected by the systemic velocity.
        x0_corr (numpy array): central RVs corrected by the systemic velocity.
    """

    def __init__(self, planet_params:dict, time:np.array, CCFs:np.array, model:str, print_output:bool, plot_fits:bool, plot_sys_vel:bool, save:str=None):

        phase_mu = get_phase_mu(planet_params, time)
        phases, tr_dur, tr_ingress_egress, in_indices, out_indices = phase_mu.phases, phase_mu.tr_dur, phase_mu.tr_ingress_egress, phase_mu.in_indices, phase_mu.out_indices

        y0 = np.zeros((CCFs.shape[0],2))
        x0 = np.zeros((CCFs.shape[0],2))

        for i in range(CCFs.shape[0]):
            fit_CCF(phases[i], CCFs[i], ccf_type="raw", model=model, print_output=print_output, plot_fit=plot_fits, save=save)

        poly_coefs, poly_cov = np.polyfit(phases[out_indices], x0[:,0][out_indices], w=1/x0[:,1][out_indices],deg=1,cov=True)

        x0_corr = np.zeros_like(x0)
        CCFs_corr = np.zeros_like(CCFs)

        for i in range(CCFs.shape[0]):
            d = CCFs[i,0]
            de = CCFs[i,1]
            
            d_corr = d/y0[i,0]

            CCFs_corr[i,0] = d_corr
            CCFs_corr[i,1] = d_corr * np.sqrt( (y0[i,1]/y0[i,0])**2 + (de/d)**2 )
            CCFs_corr[i,2] = CCFs[i,2] - (poly_coefs[0]*phases[i] + poly_coefs[1])
            
            x0_corr[i,0] = x0[i,0] - (poly_coefs[0]*phases[i] + poly_coefs[1])
            x0_corr[i,1] = np.sqrt( x0[i,1]**2 + poly_cov[0,0]*phases[i]**2 + poly_cov[1,1])

        sysvel_correction.CCFs_sysvel_corr = CCFs_corr
        sysvel_correction.x0_corr = x0_corr

        if plot_sys_vel:
            plot_sysvel_corr(phases, tr_dur, tr_ingress_egress, in_indices, out_indices, x0, poly_coefs, x0_corr, save)