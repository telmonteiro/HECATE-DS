# File with plotting functions.

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from . import utils


def plot_air_snr(planet_params:dict, time:np.array, airmass:np.array, snr:np.array, save=None):
    """Plot airmass and SNR at spectral order 111 (midpoint in the selected Fe I spectral lines) of spectra used.

    Parameters
    ----------
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    time : `numpy array`
        time of observations in BJD.
    airmass : `numpy array`
        airmass at the time of observation.
    snr : `numpy array`
        signal-to-noise ratio (SNR) at spectral order 111.
    save
        path to save plot. 
    """
    phase_mu = utils.get_phase_mu(planet_params, time)
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


def plot_sysvel_corr(phases:np.array, tr_dur:float, tr_ingress_egress:float, in_indices:np.array, out_indices:np.array, x0:np.array, poly_coefs:np.array, x0_corr:np.array, save=None):
    """Plot stellar systemic velocity showing the R-M effect and it's correction.

    Parameters
    ----------
    phases : `numpy array``
        orbital phases.
    tr_dur : `float`
        transit duration.
    tr_ingress_egress : `float`
        duration between ingress and egress of transit.
    in_indices : `numpy array`
        array indices where planet is in transit.
    out_indices : `numpy array`
        array indices where planet is not in transit.
    x0 : `numpy array`
        non-corrected central RVs.
    poly_coefs : `numpy array`
        linear polynomial fit coefficients.
    x0_corr : `numpy array`
        corrected central RVs.
    save
        path to save plot. 
    """
    fig, axes = plt.subplots(ncols=2, figsize=(13,5))

    l0 = axes[0].axvspan(-tr_dur/2., tr_dur/2., alpha=0.3, color='orange')
    l1 = axes[0].axvspan(tr_ingress_egress/2.-tr_dur/2, -tr_ingress_egress/2.+tr_dur/2, alpha=0.4, color='orange')
    axes[0].errorbar(phases[in_indices], x0[:,0][in_indices], x0[:,1][in_indices], fmt="r.", markersize=10, elinewidth=10)
    axes[0].errorbar(phases[out_indices], x0[:,0][out_indices], x0[:,1][out_indices], fmt="k.", markersize=10, elinewidth=10)
    l2 = axes[0].plot(phases[out_indices], poly_coefs[0]*phases[out_indices]+poly_coefs[1], color="black", lw=1)

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
    fig.legend([l0,l1,l2], labels=labels, loc='lower center',ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle('Central values of CCFs',fontsize=19)

    plt.tight_layout()

    if save:
        plt.savefig(save+"RM_correction.pdf", dpi=300, bbox_inches="tight")

    plt.show()


def plot_local_CCFs(hecate, local_CCFs:np.array, CCFs_sub_all:np.array, RV_reference:np.array, ccf_type:str, save):
    """Plot local CCFs and tomography in function of orbital phases.

    Parameters
    ----------
    hecate
        HECATE class object.
    local_CCFs : `numpy array`
        local CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
    CCFs_sub_all : `numpy array`
        all subtracted CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
    RV_reference : `numpy array`
        RV grid for plotting.
    ccf_type : `str`
        whether it's a local, average out-of-transit or raw CCF.
    save
        path to save plot. 
    """
    phases = hecate.phases
    in_indices = hecate.phases_in_indices

    fig, axes = plt.subplots(nrows=2, figsize=(12,9.5), gridspec_kw={'height_ratios': [1.5, 1]})
    norm = Normalize(vmin=phases[in_indices].min(), vmax=phases[in_indices].max())
    cmap = plt.get_cmap('coolwarm_r')

    axes[0].set_title(f'Local {ccf_type} CCFs (Out-of-transit - In-transit)')

    for k, idx in enumerate(in_indices):
        sub     = local_CCFs[k,0]
        de_corr = local_CCFs[k,1]

        color = cmap(norm(phases[idx]))

        axes[0].scatter(RV_reference, sub, color=color, s=50, label=str(phases[idx])[:6])
        axes[0].errorbar(RV_reference, sub, yerr=de_corr, color='black', capsize=5, linewidth=0, elinewidth=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    cbar1 = fig.colorbar(sm, ax=axes[0])
    cbar1.set_label('Orbital Phase')

    axes[0].set_ylabel('Residual flux [total stellar flux]')
    axes[0].grid()
    axes[0].set_axisbelow(True)
    axes[0].set_xlim([RV_reference.min(),RV_reference.max()])
    axes[0].set_xticklabels([])

    im = axes[1].imshow(CCFs_sub_all[:,0], cmap='jet', extent=[RV_reference.min(), RV_reference.max(), phases.min(), phases.max()], aspect='auto', origin='lower')

    axes[1].axhline(-hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(hecate.tr_ingress_egress/2 - hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    axes[1].axhline(-hecate.tr_ingress_egress/2 + hecate.tr_dur/2, lw=1.5, ls="--", color="white")
    
    axes[1].set_xlabel('Radial Velocities [km/s]')
    axes[1].set_ylabel('Orbital Phase')

    cbar2 = fig.colorbar(im, ax=axes[1])
    cbar2.set_label('Residual flux [total stellar flux]')

    plt.tight_layout()

    if save: 
        plt.savefig(save+"local_CCFs.pdf", dpi=300, bbox_inches="tight")

    plt.show()


def plot_avg_oot_ccf(RV_reference:np.array, avg_out_of_transit_CCF:np.array, save):
    """Plot average out-of-transit CCF.

    Parameters
    ----------
    RV_reference : `numpy array`
        RV grid for plotting.
    avg_out_of_transit_CCF : `numpy array`
        average out-of-transit CCF (RV, flux and flux error).
    save
        path to save plot. 
    """
    _, ax = plt.subplots(figsize=(7,4))

    ax.scatter(RV_reference, avg_out_of_transit_CCF[0])
    ax.errorbar(x=RV_reference, y=avg_out_of_transit_CCF[0],yerr =avg_out_of_transit_CCF[1],capsize=7,capthick=1,color='black',linewidth=0,elinewidth=1)

    ax.set_title('Averaged out of transit CCF')
    ax.set_xlabel('Radial Velocities [km/s]')
    ax.set_ylabel('Normalized Flux')
    ax.grid()
    ax.set_axisbelow(True)

    if save: 
        plt.savefig(save+"avg_out_of_transit_ccf.pdf", dpi=200, bbox_inches="tight")

    plt.show()


def plot_R2(hecate, R2_array:np.array, save):
    """Plot coefficient of determination R^2 of fit(s).

    Parameters
    ----------
    hecate
        HECATE class object.
    R2_array : `numpy array`
        coefficient of determination scores.
    save
        path to save plot. 
    """
    _, ax = plt.subplots(figsize=(6,4))
    ax.scatter(hecate.in_phases, R2_array, color="k")
    ax.axhline(y=0.8, color='black',linestyle='-')

    ax.set_title('R² of fits to CCFs', fontsize=14)
    ax.set_xlabel('Orbital Phase', fontsize=15)
    ax.set_ylabel('R²', fontsize=15)
    ax.grid()
    ax.set_axisbelow(True)

    if save: 
        plt.savefig(save+"R2_fits.pdf", dpi=200, bbox_inches="tight")

    plt.show()


def plot_ccf_fit(rv:np.array, d:np.array, de:np.array, y_fit:np.array, phase:float, ccf_type:str, model:str, save):
    """Plot fit of CCF. Four subplots: (1) observed and fitted CCF profile; (2) residuals; (3) distribution of residuals; (4) distribution of data uncertainties.

    Parameters
    ----------
    rv : `numpy array`
        radial velocities.
    d : `numpy array`
        observed CCF flux.
    de : `numpy array`
        observed CCF flux uncertainty.
    y_fit : `numpy array`
        fitted CCF flux.
    phase : `float`
        orbital phase of observation.
    ccf_type : `str`
        whether it's a local, average out-of-transit or raw CCF.
    model : `str`
        type of profile model to fit.
    save
        path to save plot. 
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,7), gridspec_kw={'height_ratios': [1.7, 1]})
    
    if ccf_type == "local":
        title = f'Local CCF, Model: {model}, Phase: {str(phase)[:6]}'
    elif ccf_type == "master":
        title = f'Master out-of-transit CCF, Model: {model}'
    elif ccf_type == "raw":
        title = f'Model: {model}, Phase: {str(phase)[:6]}'

    fig.suptitle(title)

    axes[0,0].scatter(rv, d, color="k")
    axes[0,0].errorbar(rv, d, yerr=de, color='black', capsize=5, linewidth=0, elinewidth=1)
    axes[0,0].plot(rv, y_fit, label='fit', color="r", lw=2)
    axes[0,0].set_xlabel('Radial Velocities [km/s]')
    axes[0,0].set_ylabel('CCF')
    axes[0,0].grid()
    axes[0,0].set_axisbelow(True)
    axes[0,0].legend()

    axes[0,1].scatter(rv, d-y_fit, color="k")
    axes[0,1].set_xlabel('Radial Velocities [km/s]')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].grid(); axes[0,1].set_axisbelow(True)

    axes[1,0].hist(d-y_fit, bins=10, edgecolor='k', color="k")
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(); axes[1,0].set_axisbelow(True)

    axes[1,1].hist(de, bins=10, edgecolor='k', color="k")
    axes[1,1].set_xlabel('Uncertainties')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(); axes[1,1].set_axisbelow(True)
    axes[1,1].tick_params(axis='x', which='major', labelsize=12)

    plt.tight_layout()

    if save:
        plt.savefig(save+f"CCF_fit_{str(phase)[:6]}.pdf", dpi=200, bbox_inches="tight")

    plt.show()