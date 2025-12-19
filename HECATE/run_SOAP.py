# Wrapper class to use SOAPv4.

import SOAP # type: ignore
from ldtk import LDPSetCreator, BoxcarFilter
import matplotlib.pyplot as plt
import numpy as np
from . import utils

class run_SOAP:
    """Wrapper around SOAPv4.
    Simulate transit light curve using SOAPv4 (Cristo, E., et al. 2025) to calibrate the local spectra flux.
    Uses Limb darkening toolkit (ldtk) to compute the limb-darkening coefficients.
    
    Parameters
    ----------
        time : `numpy array` 
            time of observations in BJD.
        stellar_params : `dict` 
            dictionary containing the following stellar parameters: effective temperature and error, superficial gravity and error, metallicity and error, rotation period, radius and stellar inclination.
        planet_params : `dict`
            dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
        min_wave : `int` 
            minimum wavelength [nm] of spectrograph.
        max_wave : `int` 
            maximum wavelength [nm] of spectrograph.
        plot : `bool` 
            whether to plot the simulated transit light curve.
        save
            path to save the plot.

    Returns
    -------
        Flux_SOAP : `numpy array` 
            simulated flux from SOAP.
    """
    def __init__(self, time:np.array, stellar_params:dict, planet_params:dict, min_wav:int=380, max_wav:int=788, plot:bool=True, save=None):
        
        Teff, Teff_err = stellar_params["Teff"], stellar_params["Teff_err"]
        logg, logg_err = stellar_params["logg"], stellar_params["logg_err"]
        FeH, FeH_err   = stellar_params["FeH"], stellar_params["FeH_err"]
        P_rot          = stellar_params["P_rot"]
        R_star         = stellar_params["R_star"]
        inc_star       = stellar_params["inc_star"]

        a_R        = planet_params["a_R"]
        Rp_Rs      = planet_params["Rp_Rs"]
        e          = planet_params["e"]
        P_orb      = planet_params["P_orb"]
        w          = planet_params["w"]
        inc_planet = planet_params["inc_planet"]
        lbda       = planet_params["lbda"]

        phase_mu = utils.get_phase_mu(planet_params, time)
        phases, tr_dur, tr_ingress_egress = phase_mu.phases, phase_mu.tr_dur, phase_mu.tr_ingress_egress

        filters = [BoxcarFilter('filter', min_wav, max_wav)] 
        sc = LDPSetCreator(teff=(Teff, Teff_err), logg = (logg, logg_err), z = (FeH, FeH_err), filters=filters)
        ps = sc.create_profiles() #create the limb darkening profiles
        ldcn, _ = ps.coeffs_qd(do_mc=True) #coefficients and quadratic profile errors

        sim = SOAP.Simulation(active_regions=[]) #light curve
        sim.star.set(prot=P_rot, incl=inc_star, radius=R_star, teff=Teff, u1=ldcn[0,0], u2=ldcn[0,1])
        sim.planet.set(P=P_orb, t0=0, e=e, w=w, ip=inc_planet, lbda=lbda, a=a_R, Rp=Rp_Rs)

        output = sim.calculate_signal(psi=phases/P_rot*P_orb, skip_rv=True)
        Flux_SOAP = output.flux

        if plot == True:
            #self._plot(phases, tr_dur, tr_ingress_egress, Flux_SOAP, save)
            sim.visualize(output=output, plot_type="flux")
            
            if save:
                plt.savefig(save+"SOAP_light_curve.pdf", dpi=200, bbox_inches="tight")

        self.flux = Flux_SOAP


    def _plot(self, phases:np.array, tr_dur:float, tr_ingress_egress:float, Flux_SOAP:np.array, save=None):
        """
        Simple plot of simulated transit light curve.

        Parameters
        ----------
        phases : `numpy array`
            orbital phases.
        tr_dur : `float`
            transit duration.
        tr_ingress_egress : `float`
            duration between ingress and egress of transit.
        Flux_SOAP : `numpy array`
            simulated stellar flux from SOAP.
        save
            whether to save the plot and where.
        """

        plt.figure(figsize=(8,5))

        plt.axvspan(-tr_dur/2, tr_dur/2, alpha=0.4, color='orange', label="middle of transit")
        plt.axvspan(-tr_ingress_egress/2., tr_ingress_egress/2., alpha=0.3, color='orange', label="ingress+egress")
        plt.scatter(phases, Flux_SOAP, color="k")

        plt.title('SOAP transit light curve')
        plt.ylabel("Normalized Flux")
        plt.xlabel("Orbital Phases")
        plt.legend(fontsize=14)

        if save:
            plt.savefig(save+"SOAP_light_curve.pdf", dpi=200, bbox_inches="tight")

        plt.show()