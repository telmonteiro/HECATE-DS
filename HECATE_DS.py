#HECATE-DS -- Harvesting loCAl specTra with Exoplanets (Doppler Shadow)
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from RM_correction import RM_correction
from run_SOAP import run_SOAP
from scipy.optimize import curve_fit

class HECATE:

    def __init__(self, planet_params:dict, stellar_params:dict, time:np.array, CCFs:np.array):

        phase_mu = get_phase_mu(planet_params, time)
        self.phases, self.phases_in_indices, self.phases_out_indices = phase_mu.phases, phase_mu.in_indices, phase_mu.out_indices
        self.in_phases = self.phases[np.isin(np.arange(len(self.phases)), self.phases_in_indices)]
        
        self.planet_params = planet_params
        self.stellar_params = stellar_params
        
        self.time = time
        self.CCFs = CCFs
        


    def _extract_local_CCF(self, RV_reference:np.array, model_fit:str, ccf_type:str, plot:dict, save):
        """
        """

        phases, in_indices, in_phases = self.phases, self.phases_in_indices, self.in_phases
        
        #simulated light curve
        Flux_SOAP = run_SOAP(self.time, self.stellar_params, self.planet_params, plot=plot["SOAP"]).flux

        #Rossiter-McLaughlin effect correction
        rm_corr = RM_correction(self.planet_params, self.time, self.CCFs, model=model_fit, plot_fits=plot["fits_initial_CCF"], plot_rm=plot["RM"])
        CCFs_RM_corr = rm_corr.CCFs_RM_corr

        #average out-of-transit CCF
        CCF_interp, avg_out_of_transit_CCF = self._avg_out_of_transit_CCF(CCFs_RM_corr, RV_reference, plot=plot["avg_out_of_transit_CCF"], save=save)

        CCFs_flux_corr = np.zeros_like(CCF_interp) #only flux corrected
        CCFs_sub_all = np.zeros_like(CCF_interp) #flux corrected and subtracted
        local_CCFs = np.zeros((len(in_indices), 3, CCF_interp.shape[2])) #same as above but only in transit

        l = 0
        for i in range(CCFs_flux_corr.shape[0]):
            d = CCF_interp[i,0,:]
            de = CCF_interp[i,1,:]
            
            #performing the Doppler shadow technique
            sub = avg_out_of_transit_CCF[0] - d*Flux_SOAP[i]

            d_corr = d/Flux_SOAP[i]
            de_corr = np.sqrt(avg_out_of_transit_CCF[1]**2 + (de*Flux_SOAP[i])**2)

            CCFs_sub_all[i,0] = sub
            CCFs_sub_all[i,1] = de_corr
            CCFs_sub_all[i,2] = RV_reference

            if i in in_indices:
                
                CCFs_flux_corr[i,0] = d_corr
                CCFs_flux_corr[i,1] = de*Flux_SOAP[i]
                CCFs_flux_corr[i,2] = RV_reference

                local_CCFs[l,0] = sub
                local_CCFs[l,1] = de_corr
                local_CCFs[l,2] = RV_reference
                l += 1

        if plot["local_CCFs"] == True: #local CCFs + tomography

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
            
            axes[1].set_xlabel('Radial Velocities [km/s]')
            axes[1].set_ylabel('Orbital Phase')

            cbar2 = fig.colorbar(im, ax=axes[1])
            cbar2.set_label('Residual flux [total stellar flux]')

            plt.tight_layout()

            if save: 
                plt.savefig(save+"local_CCFs.pdf", dpi=300, bbox_inches="tight")

            plt.show()

        return local_CCFs, CCFs_sub_all, CCFs_flux_corr, phases, in_phases
    

    def _avg_out_of_transit_CCF(self, CCFs, RV_reference:np.array, plot, save):
        """
        Docstring for _avg_out_of_transit_CCF
        
        :param RV_reference: Description
        :type RV_reference: np.array
        :param plot: Description
        :param save: Description
        """
        M = CCFs.shape[0]
        K = CCFs.shape[2]
        cov_matrix = np.zeros((M, K, K))
        N = 10000

        #
        for i in range(M):
            samples = np.zeros((K, N))
            for j in range(K):
                ymean = CCFs[i,0,j]
                ysigma = CCFs[i,1,j]
                samples[j,:] = np.random.normal(ymean, ysigma, N)
            cov_ccf = np.cov(samples)
            cov_matrix[i,:,:] = cov_ccf

        out_of_transit_CCFs = np.zeros([len(self.phases_out_indices), 3, len(RV_reference)])
        CCF_interp = np.zeros([CCFs.shape[0], 3, CCFs.shape[2]])

        k, M = 0, 0
        for l in range(CCF_interp.shape[0]):
            ccf_f = CCFs[l,0]
            ccf_f_e = cov_matrix[l]  #full covariance matrix
            ccf_rv = CCFs[l,2]

            W = linear_interpolation_matrix(ccf_rv, RV_reference) #build interpolation matrix for this CCF → target grid

            y_i = W @ ccf_f #interpolated flux
            #propagated covariance and uncertainty
            cov_new = W @ ccf_f_e @ W.T
            y_i_e = np.sqrt(cov_new.diagonal())

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
        A_e = np.sum(out_of_transit_CCFs[:,1,:]**2, axis=0)
        average_out_of_transit_CCF_e = np.sqrt(A_e) / len(self.phases_out_indices)

        avg_out_of_transit_CCF = np.array([average_out_of_transit_CCF, average_out_of_transit_CCF_e, RV_reference])
        
        if plot:
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

        return CCF_interp, avg_out_of_transit_CCF
    

    def _CCF_parameters(self, CCFs, ccf_type:str, model:str, print_output:bool, plot_fit:bool, save):

        if ccf_type == "local":
            N = CCFs.shape[0]
        elif ccf_type == "master":
            N = 1

        depth_array = np.zeros((N,2))
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

            central_rv, continuum, depth, width, R2 = self._fit_CCF(phase, CCF, ccf_type, model, print_output, plot_fit, save)
            
            central_rv_array[i,0], central_rv_array[i,1] = central_rv[0], central_rv[1]
            continuum_array[i,0], continuum_array[i,1] = continuum[0], continuum[1]
            depth_array[i,0], depth_array[i,1] = depth[0], depth[1]
            width_array[i,0], width_array[i,1] = width[0], width[1]

            R2_array[i] = R2

        if plot_fit and ccf_type == "local":
            _, ax = plt.subplots(figsize=(6,4))
            ax.scatter(self.in_phases, R2_array, color="k")
            ax.axhline(y=0.8, color='black',linestyle='-')

            ax.set_title('R² of fits to CCFs', fontsize=14)
            ax.set_xlabel('Orbital Phase', fontsize=15)
            ax.set_ylabel('R²', fontsize=15)
            ax.grid()
            ax.set_axisbelow(True)

            plt.show()
        
        return central_rv_array, continuum_array, depth_array, width_array, R2_array
        

    def _fit_CCF(self, phase, CCF, ccf_type, model, print_output, plot_fit, save):

        d = CCF[0]
        de = CCF[1]
        rv = CCF[2]

        model_fit = profile_models(model).model
            
        if model == "modified Gaussian":
            parameters = ["y0","x0","sigma","a","c"]
            p0          = [np.max(d),           0,        1, np.max(d)-np.min(d),       1] #y0, x0, sigma, a, c
            upper_bound = [   np.inf,  np.max(rv),   np.inf,              np.inf,  np.inf]
            lower_bound = [        0,  np.min(rv),        0,                   0,       0]
        
        elif model == "Gaussian":
            parameters = ["y0","x0","sigma","a"]
            p0          = [np.max(d),           0,        1, np.max(d)-np.min(d)] #y0, x0, sigma, a
            upper_bound = [   np.inf,  np.max(rv),   np.inf,              np.inf]
            lower_bound = [        0,  np.min(rv),        0,                   0]
        
        elif model == "Lorentzian":
            parameters = ["y0","x0","gamma","a"]
            p0          = [np.max(d),           0,        1, np.max(d)-np.min(d)] #y0, x0, gamma, a
            upper_bound = [   np.inf,  np.max(rv),   np.inf,              np.inf]
            lower_bound = [        0,  np.min(rv),        0,                   0]

        popt, pcov = curve_fit(f=model_fit, xdata=rv, ydata=d, sigma=de, bounds=(lower_bound,upper_bound), absolute_sigma=True, p0=p0)
        y_fit = model_fit(rv, *popt)

        central_rv = [popt[1], np.sqrt(pcov[1,1])]
        continuum = [popt[0], np.sqrt(pcov[0,0])]
        depth = [(1-(popt[3]/popt[0]))*100, ((popt[3]/popt[0])*np.sqrt(np.abs(pcov[3,3])/(popt[3]**2)+np.abs(pcov[0,0])/(popt[0]**2)))*100]
        width = [popt[2],  np.sqrt(pcov[2,2])] 
        R2 = np.around(profile_models.r2(d,y_fit),4)

        if print_output:
            print("#"*30)
            print(f"FITTING {model} model TO {ccf_type} CCF")
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
            print(f"Deph [%]: {depth[0]:.06f} ± {depth[1]:.06f}")
            print(f"Width [km/s]: {width[0]:.06f} ± {width[1]:.06f}")

        if plot_fit:
            
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,7), gridspec_kw={'height_ratios': [1.7, 1]})
            
            if ccf_type == "local":
                title = f'Local CCF, Model: {model}, Phase: {str(phase)[:6]}'
            elif ccf_type == "master":
                title = f'Master out-of-transit CCF, Model: {model}'

            fig.suptitle(title)

            axes[0,0].scatter(rv, d, color="k")
            axes[0,0].errorbar(rv, d, yerr=de, color='black', capsize=5, linewidth=0, elinewidth=1)
            axes[0,0].plot(rv, y_fit, label='fit', color="r", lw=2)
            axes[0,0].set_xlabel('Radial Velocities [km/s]')
            axes[0,0].set_ylabel('CCF')
            axes[0,0].grid(); axes[0,0].set_axisbelow(True)
            axes[0,0].legend()

            axes[0,1].scatter(rv, d-y_fit, color="k")
            axes[0,1].set_xlabel('Radial Velocities [km/s]'); axes[0,1].set_ylabel('Residuals')
            axes[0,1].grid(); axes[0,1].set_axisbelow(True)

            axes[1,0].hist(d-y_fit, bins=10, edgecolor='k', color="k")
            axes[1,0].set_xlabel('Residuals')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(); axes[1,0].set_axisbelow(True)

            axes[1,1].hist(de, bins=10, edgecolor='k', color="k")
            axes[1,1].set_xlabel('Uncertainties'); axes[1,1].set_ylabel('Frequency')
            axes[1,1].grid(); axes[1,1].set_axisbelow(True)
            axes[1,1].tick_params(axis='x', which='major', labelsize=12)

            plt.tight_layout()

            if save:
                plt.savefig(save+f"CCF_fit_{str(phase)[:6]}.pdf", dpi=200, bbox_inches="tight")

            plt.show()

        return central_rv, continuum, depth, width, R2