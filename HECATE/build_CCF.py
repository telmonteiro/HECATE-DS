# Class to build CCF from spectra.

from bisect import bisect_left
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from . import utils

class build_CCF:
    """Class to build CCFs from spectra and a spectral line-list. Method adapted from iCCF.meta.espdr_compute_CCF_fast of iCCF package.
        
    Parameters
    ----------
    time : `numpy array`
        time of observations in BJD.
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    spectra : `numpy array`
        matrix with the spectra (wavelenth in air, flux, flux error and quality flag), with shape (N_spectra, 4, N_pixels).
    RV_reference : `numpy array`
        RV grid for interpolation.
    mask : `numpy array`
        spectral line-list to perform cross-correlation.
    berv : `numpy array`
        BERV at time of observation.
    bervmax : `numpy array`
        maximum BERV.
    plot : `bool`
        whether to plot CCFs in function of 
    
    Methods
    -------
    compute_CCF(ll, flux, error, quality, RV_reference, mask, berv, bervmax, mask_width=0.5, plot=True)
        performs cross-correlation between a given spectrum and the line-list following the procedure of iCCF.
    """
    def __init__(self, time:np.array, planet_params:dict, spectra:np.array, RV_reference:np.array, mask:np.array, berv:np.array, bervmax:np.array, plot:bool=False):

        CCFs = np.zeros((spectra.shape[0], 3, RV_reference.shape[0]))

        for i in range(spectra.shape[0]):

            ccf = self.compute_CCF(ll = spectra[i,0], flux = spectra[i,1], error = spectra[i,2], quality = spectra[i,3],
                            RV_reference = RV_reference, mask = mask, berv = berv[i], bervmax = bervmax[i], mask_width = 0.5, plot=False)
            
            CCFs[i,0] = ccf[0]
            CCFs[i,1] = ccf[1]
            CCFs[i,2] = RV_reference
        
        self.CCFs = CCFs

        if plot:
            
            phases = utils.get_phase_mu(planet_params, time).phases
            norm = Normalize(vmin=phases.min(), vmax=phases.max())
            cmap = plt.get_cmap('coolwarm_r')

            fig, ax = plt.subplots(figsize=(7,4))

            for i in range(CCFs.shape[0]):
                color = cmap(norm(phases[i]))
                ax.plot(CCFs[i,2], CCFs[i,0], c=color)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Orbital Phase')
            ax.set_xlabel("RV [km/s]")
            ax.set_ylabel("CCF Flux")
            ax.grid()
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.show()

    # adapted from iCCF.meta.espdr_compute_CCF_fast
    @staticmethod
    def compute_CCF(ll:np.array, flux:np.array, error:np.array, quality:np.array, RV_reference:np.array, mask:np.array, berv:float, bervmax:float, mask_width:float=0.5, plot:bool=True):
        """Performs cross-correlation between a given spectrum and the line-list following the procedure of meta.espdr_compute_CCF_fast of iCCF.
        
        Parameters
        ----------
        ll : `numpy array`
            wavelength array in air.
        flux : `numpy array`
            flux array.
        error : `numpy array`
            flux error array in air.
        quality : `numpy array`
            quality flag array in air.
        RV_reference : `numpy array`
            RV grid.
        mask : `numpy array`
            spectral line-list.
        berv : `float`
            BERV value.
        bervmax : `float`
            maximum BERV value.
        mask_width : `float`
            width of pixel, predefined to 0.5 km/s (ESPRESSO).
        plot : `bool`
            wether to plot the computed CCF.

        Returns
        -------
        ccf_flux : `numpy array`
            computed CCF flux.
        ccf_error : `numpy array` 
            computed CCF flux error.
        ccf_quality : `numpy array`
            quality flags of used spectra.
        """
        c = 299792.458 # km/s
        nx_s2d = flux.size
        n_mask = mask.shape[0]
        contrast = np.ones_like(mask) # box window
        nx_ccf = RV_reference.shape[0]

        ccf_flux = np.zeros_like(RV_reference)
        ccf_error = np.zeros_like(RV_reference)
        ccf_quality = np.zeros_like(RV_reference)

        dll= np.gradient(ll)
        ll2 = ll - dll / 2.0 

        imin, imax = 1, nx_s2d
        while(imin < nx_s2d and quality[imin-1] != 0):
            imin += 1
        while(imax > 1 and quality[imax-1] != 0):
            imax -= 1

        if imin >= imax:
            return
    
        llmin = ll[imin + 1 - 1] / (1. + berv / c) * (1. + bervmax / c) / (1. + RV_reference[0] / c)
        llmax = ll[imax - 1 - 1] / (1. + berv / c) * (1. - bervmax / c) / (1. + RV_reference[nx_ccf - 1] / c)

        imin, imax = 0, n_mask - 1

        while (imin < n_mask and mask[imin] < (llmin + 0.5 * mask_width / c * llmin)):
            imin += 1
        while (imax >= 0     and mask[imax] > (llmax - 0.5 * mask_width / c * llmax)):
            imax -= 1

        for i in range(imin, imax + 1):

            llcenter = mask[i] * (1. + RV_reference[nx_ccf // 2] / c)
            w = contrast[i] * contrast[i]

            for j in range(0, nx_ccf):
                llcenter = mask[i] * (1. + RV_reference[j] / c)
                llstart = llcenter - 0.5 * mask_width / c * llcenter
                llstop = llcenter + 0.5 * mask_width / c * llcenter

                index1 = bisect_left(ll2, llstart) + 1
                index2 = bisect_left(ll2, llcenter) + 1
                index3 = bisect_left(ll2, llstop) + 1

                k = j
                for index in range(index1, index3):
                    ccf_flux[k] += w * flux[index-1]

                ccf_flux[k] += w * flux[index1-1-1] * (ll2[index1-1]-llstart) / dll[index1-1-1] 
                ccf_flux[k] -= w * flux[index3-1-1] * (ll2[index3-1]-llstop) / dll[index3-1-1] 

                ccf_error[k] += w * w * error[index2 - 1 - 1] * error[index2 - 1 - 1]
                ccf_quality[k] += quality[index2 - 1 - 1]

        ccf_error = np.sqrt(ccf_error)

        if plot:
            plt.figure(figsize=(6,4))
            plt.errorbar(x=RV_reference, y=ccf_flux, yerr=ccf_error, fmt="k.")
            plt.xlabel("RV [km/s]")
            plt.ylabel("CCF Flux")
            plt.grid()
            plt.show()

        return ccf_flux, ccf_error, ccf_quality