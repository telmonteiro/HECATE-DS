from utils import *
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

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
        plot_rm (bool): whether to plot the central RV in function of orbital phase, representing the R-M effect. 
        save (str, optional): path to save the plots.

    Returns:
        CCFs_RM_corr (numpy array): CCFs corrected by the RM effect.
        x0_corr (numpy array): central RVs corrected by the RM effect.
    """

    def __init__(self, planet_params:dict, time:np.array, CCFs:np.array, model:str, plot_fits:bool, plot_rm:bool, save=None):

        phase_mu = get_phase_mu(planet_params, time)
        phases, tr_dur, tr_ingress_egress, in_indices, out_indices = phase_mu.phases, phase_mu.tr_dur, phase_mu.tr_ingress_egress, phase_mu.in_indices, phase_mu.out_indices

        y0 = np.zeros((CCFs.shape[0],2))
        x0 = np.zeros((CCFs.shape[0],2))

        for i in range(CCFs.shape[0]):
            phase = phases[i]
            d = CCFs[i,0]
            de = CCFs[i,1]
            rv = CCFs[i,2]

            if model == "modified Gaussian":
                model_fit = profile_models(model).model
                p0          = [np.max(d),           0,        1, np.max(d)-np.min(d),       1] #y0, x0, sigma, a, c
                upper_bound = [   np.inf,  np.max(rv),   np.inf,              np.inf,  np.inf]
                lower_bound = [        0,  np.min(rv),        0,                   0,       0]
            elif model == "Gaussian":
                model_fit = profile_models(model).model
                p0          = [np.max(d),           0,        1, np.max(d)-np.min(d)] #y0, x0, sigma, a
                upper_bound = [   np.inf,  np.max(rv),   np.inf,              np.inf]
                lower_bound = [        0,  np.min(rv),        0,                   0]
            elif model == "Lorentzian":
                model_fit = profile_models(model).model
                p0          = [np.max(d),           0,        1, np.max(d)-np.min(d)] #y0, x0, gamma, a
                upper_bound = [   np.inf,  np.max(rv),   np.inf,              np.inf]
                lower_bound = [        0,  np.min(rv),        0,                   0]

            popt, pcov = curve_fit(f=model_fit, xdata=rv, ydata=d, sigma=de, bounds=(lower_bound,upper_bound), absolute_sigma=True, p0=p0)

            y0[i,0], y0[i,1] = popt[0], np.sqrt(pcov[0,0])
            x0[i,0], x0[i,1] = popt[1], np.sqrt(pcov[1,1])

            if plot_fits:
                y_fit = model_fit(rv, *popt)
                self._plot_fits(rv, d, de, y_fit, model, phase, save)

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

        sysvel_correction.CCFs_RM_corr = CCFs_corr
        sysvel_correction.x0_corr = x0_corr

        if plot_rm:
            self._plot_sysvel_corr(phases, tr_dur, tr_ingress_egress, in_indices, out_indices, x0, poly_coefs, x0_corr, save)


    def _plot_sysvel_corr(self, phases, tr_dur, tr_ingress_egress, in_indices, out_indices, x0, poly_coefs, x0_corr, save=None):

        fig, axes = plt.subplots(ncols=2, figsize=(13,5))

        l0 = axes[0].axvspan(-tr_dur/2., tr_dur/2., alpha=0.3, color='orange')
        l1 = axes[0].axvspan(tr_ingress_egress/2.-tr_dur/2, -tr_ingress_egress/2.+tr_dur/2, alpha=0.4, color='orange')
        l2 = axes[0].errorbar(phases[in_indices], x0[:,0][in_indices], x0[:,1][in_indices], fmt="r.", markersize=10, elinewidth=10)
        l3 = axes[0].errorbar(phases[out_indices], x0[:,0][out_indices], x0[:,1][out_indices], fmt="k.", markersize=10, elinewidth=10)
        l4 = axes[0].plot(phases[out_indices], poly_coefs[0]*phases[out_indices]+poly_coefs[1], color="black", lw=1)

        axes[0].set_ylabel('Radial Velocities [km/s]')
        axes[0].set_xlabel('Orbital Phases')
        axes[0].grid()
        axes[0].set_axisbelow(True)

        axes[1].axvspan(-tr_dur/2, tr_dur/2, alpha=0.3, color="orange")
        axes[1].axvspan(tr_ingress_egress/2.-tr_dur/2, -tr_ingress_egress/2.+tr_dur/2, alpha=0.4, color='orange')
        axes[1].errorbar(phases[in_indices], x0_corr[:,0][in_indices], x0_corr[:,1][in_indices], fmt="r.", markersize=10, elinewidth=10)
        axes[1].errorbar(phases[out_indices], x0_corr[:,0][out_indices], x0_corr[:,1][out_indices], fmt="k.", markersize=10, elinewidth=10)
        axes[1].axhline(0, lw=1, ls= "--", color="k")
        
        axes[1].set_ylabel('Radial Velocities [km/s]')
        axes[1].set_xlabel('Orbital Phases')
        axes[1].grid()
        axes[1].set_axisbelow(True)

        labels = ['Partially in transit','Fully in transit','out of transit linear fit']
        fig.legend([l0,l1,l4], labels=labels, loc='lower center',ncol=3, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle('Central values of CCFs',fontsize=19)

        plt.tight_layout()

        if save:
            plt.savefig(save+"RM_correction.pdf", dpi=300, bbox_inches="tight")

        plt.show()


    def _plot_fits(self, rv, d, de, y_fit, model, phase, save):

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,7), gridspec_kw={'height_ratios': [1.7, 1]})
        fig.suptitle(f'Model: {model}, Phase: {str(phase)[:6]}')

        axes[0,0].scatter(rv, d, color="k")
        axes[0,0].errorbar(rv, d, yerr=de, color='black', capsize=5, linewidth=0, elinewidth=1)
        axes[0,0].plot(rv, y_fit, label='fit', color="r", lw=2)
        axes[0,0].set_xlabel('Radial Velocities [km/s]'); axes[0,0].set_ylabel('CCF')
        axes[0,0].grid(); axes[0,0].set_axisbelow(True)
        axes[0,0].legend()

        axes[0,1].scatter(rv, d-y_fit, color="k")
        axes[0,1].set_xlabel('Radial Velocities [km/s]'); axes[0,1].set_ylabel('Residuals')
        axes[0,1].grid(); axes[0,1].set_axisbelow(True)

        axes[1,0].hist(d-y_fit, bins=10, edgecolor='k', color="k")
        axes[1,0].set_xlabel('Residuals'); axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(); axes[1,0].set_axisbelow(True)

        axes[1,1].hist(de, bins=10, edgecolor='k', color="k")
        axes[1,1].set_xlabel('Uncertainties'); axes[1,1].set_ylabel('Frequency')
        axes[1,1].grid(); axes[1,1].set_axisbelow(True)
        axes[1,1].tick_params(axis='x', which='major', labelsize=12)

        plt.tight_layout()

        if save:
            plt.savefig(save+f"CCF_fit_{str(phase)[:6]}.pdf", dpi=200, bbox_inches="tight")

        plt.show()