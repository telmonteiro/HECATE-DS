#HECATE-DS -- Harvesting loCAl specTra with Exoplanets (Doppler Shadow)
from auxiliar_functions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from RM_correction import RM_correction
from run_SOAP import run_SOAP

class HECATE:

    def __init__(self):
        a=2

    def _extract_local_CCF(self, planet_params, stellar_params, RV_reference, time, CCFs, ccf_type, plot, save):
        
        phases, _, _, in_indices, _ = get_phase(planet_params, time)

        Flux_SOAP = run_SOAP(time, stellar_params, planet_params, plot=plot["SOAP"]).flux

        rm_corr = RM_correction(planet_params, time, CCFs, model="Dravins", plot_fits=plot["fits_initial_CCF"], plot_rm=plot["RM"])
        CCFs_RM_corr = rm_corr.CCFs_RM_corr

        CCF_interp, avg_out_of_transit_CCF = self._avg_out_of_transit_CCF(planet_params, time, CCFs_RM_corr, RV_reference, plot=plot["avg_out_of_transit_CCF"], save=save)

        CCFs_flux_corr = np.zeros_like(CCF_interp) #only flux corrected
        CCFs_sub_all = np.zeros_like(CCF_interp) #flux corrected and subtracted
        local_CCFs = np.zeros((len(in_indices), 2, CCF_interp.shape[2])) #same as above but only in transit

        l = 0
        for i in range(CCFs_flux_corr.shape[0]):
            d = CCF_interp[i,0,:]
            de = CCF_interp[i,1,:]
            
            #performing the Doppler shadow technique
            sub = avg_out_of_transit_CCF[0] - d*Flux_SOAP[i]

            d_corr = d/Flux_SOAP[i]
            de_corr = np.sqrt(avg_out_of_transit_CCF[1]**2+(de*Flux_SOAP[i])**2)

            CCFs_sub_all[i,0] = sub
            CCFs_sub_all[i,1] = de_corr

            if i in in_indices:
                
                CCFs_flux_corr[i,0] = d_corr
                CCFs_flux_corr[i,1] = de*Flux_SOAP[i]
                local_CCFs[l,0] = sub
                local_CCFs[l,1] = de_corr
                l += 1

        if plot["local_CCFs"] == True:

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
            axes[0].set_xlim([RV_reference.min(),RV_reference.max()])
            axes[0].set_xticklabels([])

            im = axes[1].imshow(CCFs_sub_all[:,0], cmap='jet', extent=[RV_reference.min(), RV_reference.max(), phases.min(), phases.max()], aspect='auto', origin='lower')
            
            axes[1].set_xlabel('Radial Velocities [km/s]')
            axes[1].set_ylabel('Orbital Phase')

            cbar2 = fig.colorbar(im, ax=axes[1])
            cbar2.set_label('Residual flux [total stellar flux]')

            plt.tight_layout()

            if save: plt.savefig(save+"local_CCFs.pdf", dpi=300, bbox_inches="tight")

            plt.show()







    def _avg_out_of_transit_CCF(self, planet_params, time, CCFs, RV_reference, plot, save):

        _, _, _, _, out_indices = get_phase(planet_params, time)

        M = CCFs.shape[0]
        K = CCFs.shape[2]
        cov_matrix = np.zeros((M, K, K))
        N = 10000

        for i in range(M):
            samples = np.zeros((K, N))
            for j in range(K):
                ymean = CCFs[i,0,j]
                ysigma = CCFs[i,1,j]
                samples[j,:] = np.random.normal(ymean, ysigma, N)
            cov_ccf = np.cov(samples)
            cov_matrix[i,:,:] = cov_ccf

        out_of_transit_CCFs = np.zeros([len(out_indices), 3, len(RV_reference)])
        CCF_interp = np.zeros([CCFs.shape[0], 3, CCFs.shape[2]])

        k, M = 0, 0
        for l in range(CCFs.shape[0]):
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

            if l in out_indices:
                out_of_transit_CCFs[k,0,:] = y_i
                out_of_transit_CCFs[k,1,:] = y_i_e
                out_of_transit_CCFs[k,2,:] = RV_reference
                k += 1
            else:
                M += 1

        average_out_of_transit_CCF = np.mean(out_of_transit_CCFs[:,0,:], axis=0)
        A_e = np.sum(out_of_transit_CCFs[:,1,:]**2, axis=0)
        average_out_of_transit_CCF_e = np.sqrt(A_e) / len(out_indices)

        avg_out_of_transit_CCF = np.array([average_out_of_transit_CCF, average_out_of_transit_CCF_e])
        
        if plot:
            _, ax = plt.subplots(figsize=(7,4))

            ax.scatter(RV_reference, avg_out_of_transit_CCF[0])
            ax.errorbar(x=RV_reference, y=avg_out_of_transit_CCF[0],yerr =avg_out_of_transit_CCF[1],capsize=7,capthick=1,color='black',linewidth=0,elinewidth=1)
            
            ax.set_title('Averaged out of transit CCF')
            ax.set_xlabel('Radial Velocities [km/s]')
            ax.set_ylabel('Normalized Flux')
            ax.grid()
            ax.set_axisbelow(True)

            if save: plt.savefig(save+"avg_out_of_transit_ccf.pdf", dpi=200, bbox_inches="tight")

            plt.show()

        return CCF_interp, avg_out_of_transit_CCF