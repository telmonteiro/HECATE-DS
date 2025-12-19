# Personal functions to fetch data.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.io import fits
import os
import glob
from . import utils

def get_CCFs(planet_params:dict, directory_path:str='Eduardos_code/white_light_ccfs/', day:str='2021-08-11', index_to_remove:str="last", plot:bool=True):
    """Fetch ESPRESSO white-light CCFs data.

    Parameters
    ----------
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    directory_path : `str`
        path where the white light CCFs are stored.
    day : `str`
        in case where the target was observed in more than one night, choose the one to use.
    index_to_remove : `str`
        in case the user wants to remove a given CCF a priori.
    plot : `bool` 
        whether to plot the CCF profiles colored by orbital phase.

    Returns
    -------
    CCFs : `numpy array` 
        matrix with the CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
    time : `numpy array` 
        time of observations in BJD.
    airmass : `numpy array` 
        airmass at the time of observation.
    berv : `numpy array` 
        BERV at time of observation.
    bervmax : `numpy array` 
        maximum BERV.
    snr : `numpy array` 
        signal-to-noise ratio of observation.
    list_ccfs : `str` 
        list of CCFs file path and names.
    """
    listfiles = glob.glob(os.path.join(directory_path, '*.fits'))

    list_ccfs = [name for name in listfiles if day in name and 'SKY' in name]
    list_ccfs = sorted(list_ccfs)

    #removing low SNR observations
    if index_to_remove == "last":
        list_ccfs = list_ccfs[:-1]
    elif index_to_remove == None:
        list_ccfs = list_ccfs
    else:
        list_ccfs = [x for i,x in enumerate(list_ccfs) if i not in index_to_remove]

    n_points = fits.getdata(list_ccfs[0],1)[-1].shape[0]

    CCFs = np.zeros((len(list_ccfs), 3, n_points))
    time = np.zeros(len(list_ccfs))
    airmass = np.zeros(len(list_ccfs))
    berv = np.zeros(len(list_ccfs))
    bervmax = np.zeros(len(list_ccfs))
    snr = np.zeros(len(list_ccfs))

    for i,name in enumerate(list_ccfs):

        d = fits.getdata(name,1)
        de = fits.getdata(name,2)
        h = fits.getheader(name)

        N = len(d[-1])
        a = h['HIERARCH ESO RV START']
        step = h['HIERARCH ESO RV STEP']
        X = np.arange(a,a+step*N,step)

        CCFs[i,0,:] = d[-1]
        CCFs[i,1,:] = de[-1]
        CCFs[i,2,:] = X 

        time[i] = h['HIERARCH ESO QC BJD']
        airmass[i] = h['HIERARCH ESO TEL1 AIRM START'] 
        berv[i] = h['HIERARCH ESO QC BERV']
        bervmax[i] = h['HIERARCH ESO QC BERVMAX']
        snr[i] = h['HIERARCH ESO QC ORDER111 SNR'] #order 567.76 nm to 576.42 nm

    if plot:
        phases = utils.get_phase_mu(planet_params, time).phases
        norm = Normalize(vmin=phases.min(), vmax=phases.max())
        cmap = plt.get_cmap('coolwarm_r')

        fig, ax = plt.subplots(figsize=(6,3.5))

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

    return CCFs, time, airmass, berv, bervmax, snr, list_ccfs


def get_spectra(directory_path:str='Eduardos_code/telluric_corrected_spectra/', day:str='2021-08-11', index_to_remove:str="last"):
    """Fetch ESPRESSO spectra.

    Parameters
    ----------
    directory_path : `str`
        path where the spectra are stored.
    day : `str`
        in case where the target was observed in more than one night, choose the one to use.
    index_to_remove : `str`
        in case the user wants to remove a given spectra a priori.

    Returns
    -------
    spectra : `numpy array` 
        matrix with the spectra (wavelenth in air, flux, flux error and quality flag), with shape (N_spectra, 4, N_pixels).
    time : `numpy array` 
        time of observations in BJD.
    airmass : `numpy array` 
        airmass at the time of observation.
    berv : `numpy array` 
        BERV at time of observation.
    bervmax : `numpy array` 
        maximum BERV.
    snr : `numpy array` 
        signal-to-noise ratio of observation.
    list_spectra : `str` 
        list of spectra file path and names.
    """
    listfiles = glob.glob(os.path.join(directory_path, '*.fits'))

    list_spectra = [name for name in listfiles if day in name and 'SKY' in name]
    list_spectra = sorted(list_spectra)

    #removing low SNR observation
    if index_to_remove == "last":
        list_spectra = list_spectra[:-1]
    else:
        list_spectra = [x for i,x in enumerate(list_spectra) if i not in index_to_remove]

    hdul0 = fits.open(list_spectra[0])
    tbl = hdul0[1].data   
    hdul0.close()
    npix = np.array(tbl.field(1)).shape[0]

    spectra = np.zeros((len(list_spectra),4,npix))
    time = np.zeros(len(list_spectra))
    airmass = np.zeros(len(list_spectra))
    berv = np.zeros(len(list_spectra))
    bervmax = np.zeros(len(list_spectra))
    snr = np.zeros(len(list_spectra))
    
    for i in range(len(list_spectra)):

        hdul = fits.open(list_spectra[i])
        hdr = hdul[0].header
        tbl = hdul[1].data   
        hdul.close()

        spectra[i,0] = np.array(tbl.field(1)) #wavelength in air
        spectra[i,1] = np.array(tbl.field(2)) #flux
        spectra[i,2] = np.array(tbl.field(3)) #flux error
        spectra[i,3] = np.array(tbl.field(4)) #quality

        time[i] = hdr['HIERARCH ESO QC BJD']
        airmass[i] = hdr['HIERARCH ESO TEL1 AIRM START'] 
        berv[i] = hdr['HIERARCH ESO QC BERV']
        bervmax[i] = hdr['HIERARCH ESO QC BERVMAX']
        snr[i] = hdr['HIERARCH ESO QC ORDER111 SNR'] #order 567.76 nm to 576.42 nm

    return spectra, time, airmass, berv, bervmax, snr, list_spectra