# HECATE-DS -- HarvEsting loCAl specTra with Exoplanets (Doppler Shadow)
# Hecate is a goddess in ancient Greek religion and mythology, most often shown holding a pair of torches,
# a key, or snakes, or accompanied by dogs, and in later periods depicted as three-formed or triple-bodied. 
# Hecate is often associated with illuminating what is hidden and find your way in cross-roads.

import numpy as np
import matplotlib.pyplot as plt

from run_SOAP import run_SOAP
from nested_sampling import run_nestedsampler

from utils import *
from plots import *


class HECATE:

    """Main class for HECATE operations, allowing for the easy application of the Doppler Shadow technique to high-resolution data.

    This class encapsulates the extraction of local spectra/CCFs, as well as the analysis of CCF shapes (width, intensity, RV) and their behavior (linearity).

    Parameters
    ----------
    planet_params : `dict`
        dictionary containing the following planetary parameters: orbital period, system scale, planet-to-star radius ratio, mid-transit time, eccentricity, argument of periastron, planetary inclination and spin-orbit angle.
    stellar_params : `dict`
        dictionary containing the following stellar parameters: effective temperature and error, superficial gravity and error, metallicity and error, rotation period, radius and stellar inclination.
    time : `numpy array` 
        time of observations in BJD.
    CCFs : `numpy array` 
        matrix with the CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).

    Methods
    -------
    extract_local_CCF(model_fit, ccf_type, plot, save)
        run all steps of the Doppler Shadow extraction and returns the local CCFs. Ideal for quick extraction.
    avg_out_of_transit_CCF(CCFs, RV_reference, plot, save)
        computes the average out-of-transit CCF.
    CCF_parameters(CCFs, ccf_type, model, print_output, plot_fit, save)
        runs _fit_CCF for all local CCFs.
    sysvel_correction(CCFs, model, print_output, plot_fits, plot_sys_vel, save=None)
        removes the stellar systemic velocity from the initial CCFs RVs.
    _local_params_linear_fit(local_param, indices_final, title, priors, plot_nested)
        tests the linearity of local CCF parameters in function of orbital parameters or mu via nested sampling from Dynesty.
    plot_local_params(indices_final, local_params, master_params, linear_fit=False, plot_nested=False, save=None)
        plots the local CCF parameters in function of orbital phases and mu.

    Notes
    -----
    This tool was based on the work of Gonçalves, E. et al. (2026) and contains a wrapper of SOAPv4 (Cristo, E. et al., 2025).

    References
    ----------
    [1] Gonçalves, E. et al., "Exploring the surface of HD 189733 via Doppler Shadow Analysis of Planetary Transits," Astronomy & Astrophysics, 2026

    [2] Cristo, E. et al., "SOAPv4: A new step toward modeling stellar signatures in exoplanet research", Astronomy & Astrophysics, Vol. 702, A84, 17pp., 2025
    """

    def __init__(self, planet_params:dict, stellar_params:dict, time:np.array, CCFs:np.array):

        # Get orbital phases and mu
        phase_mu = get_phase_mu(planet_params, time)

        self.phases = phase_mu.phases
        self.phases_in_indices = phase_mu.in_indices # indices of in-transit phases
        self.phases_out_indices = phase_mu.out_indices # indices of out-of-transit phases

        self.tr_dur = phase_mu.tr_dur # transit duration
        self.tr_ingress_egress = phase_mu.tr_ingress_egress # times of transit ingress and egress

        self.in_phases = self.phases[self.phases_in_indices] # in-transit phases
        
        self.mu = phase_mu.mu_values
        self.mu_in = self.mu[self.phases_in_indices]
        
        self.mu_min = get_phase_mu.mu(self.tr_dur/2-self.tr_ingress_egress/2, planet_params)
        self.mu_max = get_phase_mu.mu(0, planet_params)

        self.planet_params = planet_params
        self.stellar_params = stellar_params
        
        self.time = time
        self.CCFs = CCFs
            

    def extract_local_CCF(self, model_fit:str, ccf_type:str, plot:dict, save=None):
        """Run all steps of the Doppler Shadow extraction (simulated light curve, systemic velocity correction, compute average out-of-transit CCF and subtraction) and returns the local CCFs. 
        Ideal for quick extraction of local CCFs.

        Parameters
        ----------
        model_fit : `str`
            profile model to fit to CCFs.
        ccf_type : `str`
            whether it's a local, average out-of-transit or raw CCF.
        plot : `dict` 
            dictionary including boolean value for each type of plot available (SOAP, fits_initial_CCF, sys_vel, avg_out_of_transit_CCF, local_CCFs).
        save 
            path to save plots.

        Returns
        -------
        local_CCFs : `numpy array`
            matrix with the local (in-transit) CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
        CCFs_flux_corr : `numpy array`
            matrix with all CCF profiles, only flux corrected, with shape (N_CCFs, 3, N_points)
        CCFs_sub_all : `numpy array` 
            matrix with all CCF profiles, flux corrected and subtracted from average out-of-transit, with shape (N_CCFs, 3, N_points).
        """

        # simulated light curve
        Flux_SOAP = run_SOAP(self.time, self.stellar_params, self.planet_params, plot=plot["SOAP"]).flux

        # systemic velocity correction
        CCFs_sysvel_corr, _ = self.sysvel_correction(self.CCFs, model=model_fit, print_output=False, plot_fits=plot["fits_initial_CCF"], plot_sys_vel=plot["sys_vel"], save=save)

        # RV grid as the maximum minimum to minimum maximum of sys. velocity corrected CCF with 0.5 km/s step (ESPRESSO pixel size)
        RV_reference = np.arange(round(np.max(CCFs_sysvel_corr[:,2,0])), round(np.min(CCFs_sysvel_corr[:,2,-1]))+0.5, 0.5) 

        # average out-of-transit CCF
        CCF_interp, avg_out_of_transit_CCF = self.avg_out_of_transit_CCF(CCFs_sysvel_corr, RV_reference, plot=plot["avg_out_of_transit_CCF"], save=save)

        CCFs_flux_corr = np.zeros_like(CCF_interp) # only flux corrected
        CCFs_sub_all = np.zeros_like(CCF_interp) # flux corrected and subtracted
        local_CCFs = np.zeros((len(self.phases_in_indices), 3, CCF_interp.shape[2])) # same as above but only in transit (local)

        l = 0
        for i in range(CCFs_flux_corr.shape[0]):
            d = CCF_interp[i,0,:]
            de = CCF_interp[i,1,:]
            
            # performing the subtraction
            sub = avg_out_of_transit_CCF[0] - d*Flux_SOAP[i]

            d_corr = d*Flux_SOAP[i]
            de_corr = np.sqrt(avg_out_of_transit_CCF[1]**2 + (de*Flux_SOAP[i])**2)

            CCFs_sub_all[i,0] = sub
            CCFs_sub_all[i,1] = de_corr
            CCFs_sub_all[i,2] = RV_reference

            if i in self.phases_in_indices:
                
                CCFs_flux_corr[i,0] = d_corr
                CCFs_flux_corr[i,1] = de*Flux_SOAP[i]
                CCFs_flux_corr[i,2] = RV_reference

                local_CCFs[l,0] = sub
                local_CCFs[l,1] = de_corr
                local_CCFs[l,2] = RV_reference

                l += 1

        if plot["local_CCFs"] == True: #local CCFs + tomography
            plot_local_CCFs(self, local_CCFs, CCFs_sub_all, RV_reference, ccf_type, save)

        return local_CCFs, CCFs_flux_corr, CCFs_sub_all
    

    def sysvel_correction(self, CCFs:np.array, model:str, print_output:bool, plot_fits:bool, plot_sys_vel:bool, save:str=None):
        """Extract the RV component due to the star's motion around the barycentre, excluding the stellar systemic velocity.
        Fits a chosen profile to the CCF, then a linear model to the out-of-transit central RVs and subtracts it to all CCFs RVs.
        
        Parameters
        ----------
            CCFs : `numpy array`
                matrix with the CCF profiles (RV, flux and flux error), with shape (N_CCFs, 3, N_points).
            model : `str` 
                type of profile model to fit.
            plot_fits : `bool`
                whether to plot the fit for each CCF.
            plot_sys_vel: `bool` 
                whether to plot the central RV (systemic velocity) in function of orbital phase. 
            save : `str`, optional
                path to save the plots.

        Returns
        -------
            CCFs_corr : `numpy array`
                CCFs corrected by the systemic velocity.
            x0_corr : `numpy array`
                central RVs corrected by the systemic velocity.
        """
        phases = self.phases
        tr_dur = self.tr_dur 
        tr_ingress_egress = self.tr_ingress_egress 
        in_indices = self.phases_in_indices
        out_indices = self.phases_out_indices

        y0 = np.zeros((CCFs.shape[0],2))
        x0 = np.zeros((CCFs.shape[0],2))

        for i in range(CCFs.shape[0]):

            central_rv, continuum, _, _, _, rv, d, de, y_fit = fit_CCF(phases[i], CCFs[i], "raw", model, print_output)

            if plot_fits:
                plot_ccf_fit(rv, d, de, y_fit, phases[i], "raw", model, save)
            
            y0[i,0] = continuum[0]
            y0[i,1] = continuum[1]

            x0[i,0] = central_rv[0]
            x0[i,1] = central_rv[1]

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

        if plot_sys_vel:
            plot_sysvel_corr(phases, tr_dur, tr_ingress_egress, in_indices, out_indices, x0, poly_coefs, x0_corr, save)

        return CCFs_corr, x0_corr


    def avg_out_of_transit_CCF(self, CCFs:np.array, RV_reference:np.array, plot:bool, save=None):
        """Computes the average out-of-transit CCF by linearly interpolating the (systemic velocity corrected) CCFs into a common grid.
        The interpolated CCF uncertainties are propagated tooking the covariances into account.

        Parameters
        ----------
        CCFs : `numpy array`
            matrix with the out-of-transit CCF profiles, with shape (N_CCFs, 3, N_points).
        RV_reference : `numpy array`
            RV grid for interpolation.
        plot : `bool` 
            whether to plot the average out of transit CCF.
        save 
            path to save plot.

        Returns
        -------
        CCF_interp : `numpy array`
            matrix with interpolated CCF profiles.
        avg_out_of_transit_CCF : `numpy array`
            matrix with the average out-of-transit CCF profile, with shape (3, N_points).
        """

        M = CCFs.shape[0]
        K = CCFs.shape[2]
        cov_matrix = np.zeros((M, K, K))
        N = 10000

        # covariance matrix obtained by sampling the CCFs 10 000 times
        for i in range(M):
            samples = np.zeros((K, N))

            for j in range(K):
                ymean = CCFs[i,0,j]
                ysigma = CCFs[i,1,j]
                samples[j,:] = np.random.normal(ymean, ysigma, N)

            cov_matrix[i,:,:] = np.cov(samples)

        out_of_transit_CCFs = np.zeros([len(self.phases_out_indices), 3, len(RV_reference)])
        CCF_interp = np.zeros([CCFs.shape[0], 3, CCFs.shape[2]])

        k, M = 0, 0
        for l in range(CCF_interp.shape[0]):
            ccf_f = CCFs[l,0]
            ccf_f_e = cov_matrix[l]  # full covariance matrix
            ccf_rv = CCFs[l,2]

            # build interpolation matrix for this CCF → target grid
            W = linear_interpolation_matrix(ccf_rv, RV_reference) 

            y_i = W @ ccf_f # interpolated flux
            cov_new = W @ ccf_f_e @ W.T # propagated covariance
            y_i_e = np.sqrt(cov_new.diagonal()) # propagated uncertainty

            CCF_interp[l,0,:] = y_i
            CCF_interp[l,1,:] = y_i_e
            CCF_interp[l,2,:] = RV_reference

            if l in self.phases_out_indices:

                out_of_transit_CCFs[k,0,:] = y_i
                out_of_transit_CCFs[k,1,:] = y_i_e
                out_of_transit_CCFs[k,2,:] = RV_reference

                k += 1
            else:
                M += 1

        average_out_of_transit_CCF = np.mean(out_of_transit_CCFs[:,0,:], axis=0)

        A_e = np.sum(out_of_transit_CCFs[:,1,:]**2, axis=0) # propagation of uncertainty into the average CCF
        average_out_of_transit_CCF_e = np.sqrt(A_e) / len(self.phases_out_indices)

        avg_out_of_transit_CCF = np.array([average_out_of_transit_CCF, average_out_of_transit_CCF_e, RV_reference])
        
        if plot:
            plot_avg_oot_ccf(RV_reference, avg_out_of_transit_CCF, save)

        return CCF_interp, avg_out_of_transit_CCF
    

    def CCF_parameters(self, CCFs:np.array, ccf_type:str, model:str, print_output:bool, plot_fit:bool, save):
        """Computes the profile parameters of an array of CCFs.

        Parameters
        ----------
        CCFs : `numpy array`
            matrix with CCF profiles, with shape (N_CCFs, 3, N_points).
        ccf_type : `str`
            whether it's a local, average out-of-transit or raw CCF. 
        model : `str`
            type of profile model to fit.
        plot_fit : `bool` 
            whether to plot the fit.
        save 
            path to save plot.

        Returns
        -------
        central_rv_array : `numpy array`
            central RV of the input CCFs.
        continuum_array : `numpy array`
            continuum level of the input CCFs.
        intensity_array : `numpy array`
            intensity of the input CCFs.
        width_array : `numpy array`
            width measure of the input CCFs.
        R2_array : `numpy array`
            coefficient of determination of all fits.
        """

        if ccf_type == "local":
            N = CCFs.shape[0]
        elif ccf_type == "master":
            N = 1

        intensity_array = np.zeros((N,2))
        central_rv_array = np.zeros((N,2))
        width_array = np.zeros((N,2))
        continuum_array = np.zeros((N,2))

        R2_array = np.zeros(N)

        for i in range(N):

            if ccf_type == "local":
                phase = self.in_phases[i]
                CCF = CCFs[i]
            elif ccf_type == "master":
                phase = None
                CCF = CCFs

            try:
                central_rv, continuum, intensity, width, R2, rv, d, de, y_fit = fit_CCF(phase, CCF, ccf_type, model, print_output)
                if plot_fit:
                    plot_ccf_fit(rv, d, de, y_fit, phase, ccf_type, model, save)
            except: # if no fit is achieved
                central_rv, continuum, intensity, width, R2 = [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], np.nan
            
            central_rv_array[i,0], central_rv_array[i,1] = central_rv[0], central_rv[1]
            continuum_array[i,0], continuum_array[i,1] = continuum[0], continuum[1]
            intensity_array[i,0], intensity_array[i,1] = intensity[0], intensity[1]
            width_array[i,0], width_array[i,1] = width[0], width[1]

            R2_array[i] = R2

        if plot_fit and ccf_type == "local":
            plot_R2(self, R2_array, save)
        
        return central_rv_array, continuum_array, intensity_array, width_array, R2_array
    

    def _local_params_linear_fit(self, local_param:np.array, indices_final:np.array, title:str, priors:list, plot_nested:bool):
        """Performs a tentative linear fit by applying nested sampling through `dynesty, comparing between a constant and unconstrained models, and then between a linear model with a positive slope and one with a negative slope.
        Useful for a first approximation analysis of the local CCF parameters.

        Parameters
        ----------
        local_param : `numpy array`
            array of a given local CCFs parameter (central RV, width, intensity).
        indices_final : `numpy array`
            indices of local CCFs to use (to discard bad data).
        title : `str` 
            CCF parameter to use as title in the plot.
        priors : `list`
            half of range of linear fit parameters (m, b) to use as priors.
        plot_nested : `bool`
            whether to plot the trace and corner plots from the `dynesty` packages.

        Returns
        -------
        phases_data : `dict`
            contains phases ('x'), the label of phases ('label'), grid of phases for plotting ('x_grid'), fitted CCF parameter ('y_fit') as an array (value, error), grid of fitted CCF parameter ('y_grid') as an array (value, error) and 'residual' between y and y_fit as an array (value, error). 
        mu_data : `dict` 
            contains mu (x), the label of mu ('label'), grid of phases for plotting ('x_grid'), fitted CCF parameter ('y_fit') as an array (value, error), grid of fitted CCF parameter ('y_grid') as an array (value, error) and 'residual' between y and y_fit as an array (value, error).
        """
        phases_data = {"x":self.in_phases[indices_final], "label":"Orbital phases"}
        mu_data = {"x":self.mu_in[indices_final], "label":r"$\mu$"}
        
        m_span, b_span = priors[0], priors[1]

        print("="*50)
        print(title)
        
        for data in [phases_data, mu_data]:

            x = phases_data["x"]

            print("-"*40)
            print(data["label"])
        
            results_nested = run_nestedsampler(x, local_param[:,0][indices_final], local_param[:,1][indices_final], m_span, b_span, plot=plot_nested).results
            lin_params, model = results_nested[0], results_nested[1]

            x_grid = np.linspace(x.min(), x.max(), 100)

            if model == "zero":
                y_fit = lin_params["b"][0] * np.ones_like(x)
                dy_fit = np.sqrt(lin_params["b"][1]**2) * np.ones_like(x)
                y_grid = lin_params["b"][0] * np.ones_like(x_grid)
                dy_grid = np.sqrt(lin_params["b"][1]**2) * np.ones_like(x_grid)
            else:
                y_fit = x*lin_params["m"][0] + lin_params["b"][0]
                dy_fit = np.sqrt((x*lin_params["m"][1])**2 + lin_params["b"][1]**2)
                y_grid = x_grid*lin_params["m"][0] + lin_params["b"][0]
                dy_grid = np.sqrt((x_grid*lin_params["m"][1])**2 + lin_params["b"][1]**2)

            residual = local_param[:,0][indices_final] - y_fit
            residual_err = np.sqrt(local_param[:,1][indices_final]**2 + dy_fit**2)

            data["x_grid"] = x_grid
            data["y_fit"] = np.array([y_fit, dy_fit])
            data["y_grid"] = np.array([y_grid, dy_grid])
            data["residual"] = np.array([residual, residual_err])
        
            print(data)

        return phases_data, mu_data


    def plot_local_params(self, indices_final:np.array, local_params:np.array, master_params:np.array, linear_fit:bool=False, plot_nested:bool=False, save=None):


    

        if linear_fit:
            fig_ph, axes_ph = plt.subplots(nrows=2, ncols=3, figsize=(16,6.2), gridspec_kw={'height_ratios': [1.5, 1]})
            fig_mu, axes_mu = plt.subplots(nrows=2, ncols=3, figsize=(16,6.2), gridspec_kw={'height_ratios': [1.5, 1]})
        else: 
            fig_ph, axes_ph = plt.subplots(nrows=1, ncols=3, figsize=(16,4.2))
            fig_mu, axes_mu = plt.subplots(nrows=1, ncols=3, figsize=(16,4.2))

        titles = ['Central Radial Velocity [km/s]', 'Line-width measure [km/s]', 'Line-center intensity [%]']
        ylabels = ["[km/s]", "[km/s]", "[%]"]

        ph_range = [-self.tr_dur/2, self.tr_dur/2]
        ph_range_inner = [self.tr_ingress_egress/2-self.tr_dur/2, self.tr_dur/2-self.tr_ingress_egress/2]
        mu_range = [0, self.mu_max]
        mu_range_inner = [self.mu_min, self.mu_max]

        plot_data = {"phases":[axes_ph, ph_range, ph_range_inner, self.in_phases[indices_final]], "mu":[axes_mu, mu_range, mu_range_inner, self.mu_in[indices_final]]}

        for i in range(len(ylabels)):

            if linear_fit: 
                plot_index = (0,i)
            else: 
                plot_index = (i)
                axes_ph[plot_index].set_xlabel("Orbital phases")
                axes_mu[plot_index].set_xlabel(r"$\mu$")

            for key in plot_data.keys():
                ax = plot_data[key][0]
                x_range = plot_data[key][1]
                x_range_inner = plot_data[key][2]
                x = plot_data[key][3]

                ax[plot_index].set_title(titles[i], fontsize=17)

                l0=ax[plot_index].axvspan(x_range[0], x_range[1], alpha=0.3, color='orange')
                l1=ax[plot_index].axvspan(x_range_inner[0], x_range_inner[1], alpha=0.4, color='orange')
                l2=ax[plot_index].axhline(y=master_params[i][:,0], color='blue', linestyle='-', lw=2, zorder=1)
                ax[plot_index].scatter(x, local_params[i][:,0][indices_final],color='blue',s=60)
                ax[plot_index].errorbar(x=x, y=local_params[i][:,0][indices_final], yerr=local_params[i][:,1][indices_final], capsize=6, capthick=0.5, color='black', linewidth=0, elinewidth=2)
                
                ax[plot_index].set_ylabel("Value " + ylabels[i], fontsize=15)
                ax[plot_index].grid()
                ax[plot_index].set_axisbelow(True)
                ax[plot_index].set_xlim(x_range)

            if linear_fit:

                if i == 0: 
                    priors = [1000, 10]
                elif i == 1: 
                    priors = [100, 100]
                elif i == 2: 
                    priors = [100, 100]

                phases_data, mu_data = self._local_params_linear_fit(local_params[i], indices_final, titles[i], priors, plot_nested)

                plot_data["phases"].append(phases_data)
                plot_data["mu"].append(mu_data)

                for key in plot_data.keys():

                    ax = plot_data[key][0]
                    x_range = plot_data[key][1]
                    x_range_inner = plot_data[key][2]
                    data = plot_data[key][-1]

                    ax[0,i].plot(data["x"], data["y_fit"][0], color='blue', linestyle='--')
                    ax[0,i].fill_between(data["x_grid"], data["y_grid"][0]-data["y_grid"][1], data["y_grid"][0]+data["y_grid"][1], color='gray', alpha=0.2, zorder=1)
                    ax[0,i].set_xticklabels([])

                    ax[1,i].axvspan(x_range[0], x_range[1], alpha=0.3, color='orange')
                    ax[1,i].axvspan(x_range_inner[0], x_range_inner[1], alpha=0.4, color='orange')
                    ax[1,i].scatter(data["x"], data["residual"][0], color='blue', s=50)
                    ax[1,i].errorbar(x=data["x"], y=data["residual"][0], yerr=data["residual"][1], capsize=5, capthick=0.5, color='black', linewidth=0, elinewidth=2)
                    ax[1,i].set_xlabel(data["label"])
                    ax[1,i].grid()
                    ax[1,i].set_axisbelow(True)
                    ax[1,i].set_xlim(x_range)
                    ax[1,i].set_ylim([-2*np.max(np.abs(data["residual"][0])), 2*np.max(np.abs(data["residual"][0]))])
                    ax[1,i].axhline(0, lw=1, ls="--", color="black")
                    ax[1,i].set_ylabel("Residuals " + ylabels[i])

        labels = ['Partially in-transit','Fully in-transit','Master out of transit']
        for fig in [fig_ph, fig_mu]:
            fig.legend([l0,l1,l2], labels=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=15)
            fig.tight_layout()

        if save:
            fig_ph.savefig(save+"local_parameters_phases.pdf", dpi=400)
            fig_mu.savefig(save+"local_parameters_mu.pdf", dpi=400)

        plt.show()