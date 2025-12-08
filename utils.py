import numpy as np
from scipy.sparse import lil_matrix

#profile models
class profile_models:
    def __init__(self, model):

        if model == "modified Gaussian":
            self.model = self.modified_gaussian
        elif model == "Gaussian":
            self.model = self.gaussian
        elif model == "Lorentzian":
            self.model = self.lorentzian

    def modified_gaussian(self, x, *params):
        y0,x0,sigma,a,c = params
        return y0-a*np.exp(-0.5*((np.abs(x-x0)/sigma)**c))

    def gaussian(self, x, *params):
        y0, x0, sigma, a = params
        return y0 - a * np.exp(-0.5*((x - x0)/sigma)**2)

    def lorentzian(self, x,*params):
        y0,x0,gamma,a = params
        return y0 - a * (gamma**2 / ((x - x0)**2 + gamma**2))

    @staticmethod
    def r2(y, yfit):
        ssres = np.sum((y-yfit)**2)
        sstot = np.sum((y-np.mean(y))**2)
        r = 1- ssres/sstot
        return r


class get_phase_mu:

    def __init__(self, planet_params, time):

        phases, tr_dur, tr_ingress_egress, in_indices, out_indices = self.get_phase(planet_params, time)
        mu_values = self.mu(phases, planet_params)

        self.phases = phases
        self.tr_dur = tr_dur
        self.tr_ingress_egress = tr_ingress_egress
        self.in_indices = in_indices
        self.out_indices = out_indices
        self.mu_values = mu_values

    def get_phase(self, planet_params, time):
        t0         = planet_params["t0"]
        dfp        = planet_params["dfp"]
        P_orb      = planet_params["P_orb"]
        inc_planet = np.radians(planet_params["inc_planet"])
        Rp_Rs      = planet_params["Rp_Rs"]
        a_R        = planet_params["a_R"]

        t_epoch = t0 + 0.5+2.4e6 - dfp*P_orb  #MBJD
        norb = (time-t_epoch)/P_orb
        nforb = [round(x) for x in norb]
        phases = norb-nforb

        tr_dur = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1+Rp_Rs)**2 - a_R**2 * np.cos(inc_planet)**2))
        tr_ingress_egress = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1-Rp_Rs)**2 -a_R**2 * np.cos(inc_planet)**2))

        in_indices  = np.where(np.abs(phases) <= tr_dur/2)[0]
        out_indices = np.where(np.abs(phases) >  tr_dur/2)[0]

        return phases, tr_dur, tr_ingress_egress, in_indices, out_indices

    @staticmethod
    def mu(phases, planet_params):
        inc_planet = planet_params["inc_planet"]
        a_R        = planet_params["a_R"]
        #impact parameter
        b = a_R*np.cos(inc_planet*np.pi/180)
        return np.sqrt(1 - b**2 - (a_R*np.sin(2*np.pi*np.abs(phases)))**2)
    

#linear interpolation
def linear_interpolation_matrix(x_old, x_new):
    """
    Builds a sparse matrix W that linearly interpolates data from x_old → x_new.
    Each row i corresponds to interpolation weights for x_new[i].
    """
    W = lil_matrix((len(x_new), len(x_old)))

    for i, xv in enumerate(x_new):
        if xv <= x_old[0]: #extrapolate using first two points
            j = 0
            x0, x1 = x_old[j], x_old[j+1]
            w1 = (x1 - xv) / (x1 - x0)
            w2 = (xv - x0) / (x1 - x0)
        elif xv >= x_old[-1]: #extrapolate using last two points
            j = len(x_old) - 2
            x0, x1 = x_old[j], x_old[j+1]
            w1 = (x1 - xv) / (x1 - x0)
            w2 = (xv - x0) / (x1 - x0)
        else:
            j = np.searchsorted(x_old, xv) - 1
            x0, x1 = x_old[j], x_old[j+1]
            w1 = (x1 - xv) / (x1 - x0)
            w2 = (xv - x0) / (x1 - x0)

        W[i, j]   = w1
        W[i, j+1] = w2

    return W.tocsr()