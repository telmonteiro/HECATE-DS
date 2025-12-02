import numpy as np
from scipy.sparse import lil_matrix

#profile models
def modified_gaussian(x,*params):
    y0,x0,sigma,a,c = params
    return y0-a*np.exp(-0.5*((np.abs(x-x0)/sigma)**c))

def gaussian(x,*params):
    y0, x0, sigma, a = params
    return y0 - a * np.exp(-0.5*((x - x0)/sigma)**2)

def lorentzian(x,*params):
    y0,x0,gamma,a = params
    return y0 - a * (gamma**2 / ((x - x0)**2 + gamma**2))

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


#get phase
def time_to_phase(date,tepoch, P_orb):
    norb = (date-tepoch)/P_orb
    nforb = [round(x) for x in norb]
    return norb-nforb

def get_phase(planet_params, time):
    t0         = planet_params["t0"]
    dfp        = planet_params["dfp"]
    P_orb      = planet_params["P_orb"]
    inc_planet = np.radians(planet_params["inc_planet"])
    Rp_Rs      = planet_params["Rp_Rs"]
    a_R        = planet_params["a_R"]

    t_epoch = t0 + 0.5+2.4e6 - dfp*P_orb  #MBJD
    phases = time_to_phase(time, t_epoch, P_orb)

    tr_dur = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1+Rp_Rs)**2 - a_R**2 * np.cos(inc_planet)**2))
    tr_ingress_egress = 1/np.pi * np.arcsin(1/a_R *np.sqrt((1-Rp_Rs)**2 -a_R**2 * np.cos(inc_planet)**2))

    in_indices  = np.where(np.abs(phases) <= tr_dur/2)[0]
    out_indices = np.where(np.abs(phases) >  tr_dur/2)[0]

    return phases, tr_dur, tr_ingress_egress, in_indices, out_indices