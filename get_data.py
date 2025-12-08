import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import glob
from utils import get_phase_mu
from matplotlib.colors import Normalize

def get_CCFs(planet_params, directory_path='Eduardos_code/white_light_ccfs/', day='2021-08-11', index_to_remove="last", plot=True):

    listfiles = glob.glob(os.path.join(directory_path, '*.fits'))

    list_ccfs = [name for name in listfiles if day in name and 'SKY' in name]
    list_ccfs = sorted(list_ccfs)

    #removing low SNR observations
    if index_to_remove == "last":
        list_ccfs = list_ccfs[:-1]
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
        phases = get_phase_mu(planet_params, time).phases
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


def get_spectra(directory_path='Eduardos_code/telluric_corrected_spectra/', day='2021-08-11', index_to_remove="last"):

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



def plot_air_snr(planet_params, time, airmass, snr, save=None):

    phase_mu = get_phase_mu(planet_params, time)
    phases, tr_dur, tr_ingress_egress = phase_mu.phases, phase_mu.tr_dur, phase_mu.tr_ingress_egress

    fig, ax0 = plt.subplots(figsize=(7,4.5))

    l0 = ax0.axvspan(-tr_dur/2., tr_dur/2., alpha=0.3, color='orange')
    l1 = ax0.axvspan(tr_ingress_egress/2.-tr_dur/2, -tr_ingress_egress/2.+tr_dur/2, alpha=0.4, color='orange')
    l2 = ax0.scatter(phases, airmass, color='black')

    ax0.set_xlabel('Orbital Phase', fontsize=14)
    ax0.set_ylabel('Airmass', fontsize=14)
    ax0.tick_params(axis="y")

    ax1 = ax0.twinx()
    l3 = ax1.scatter(phases, snr, color='black', marker="x")
    ax1.set_ylabel('SNR order 111', fontsize=14)
    ax1.tick_params(axis="y")

    labels = ['Partially in-transit','Fully in-transit', 'Airmass', 'SNR']
    fig.legend([l0, l1, l2, l3], labels=labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=12)

    plt.tight_layout()

    if save:
        plt.savefig(save+"airmass_snr.pdf", dpi=300, bbox_inches="tight")

    plt.show()