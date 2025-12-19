# Runs nested sampler on a linear regression.

import numpy as np
import matplotlib.pyplot as plt

from dynesty import DynamicNestedSampler, plotting as dyplot
from dynesty.utils import resample_equal

class run_nestedsampler:
    """Class to compare between linear models with constant slope (m=0) or unconstrained slope (m!=0) regarding CCF parameters in function of orbital phases or mu.
    Includes jittering via ln f parameter, that has a predefined prior of [-7,1] in log-scale.
    Returns a tuple with the results.

    Parameters
    ----------
    x : `numpy array`
        orbital phases or mu arrays.
    y : `numpy array`
        CCF parameter (central RV, line-width measure or line-center intensity).
    yerr : `numpy array`
        uncertainty of CCF parameter.
    m_span : `int`
        half of the range of prior on the slope.
    b_span : `int`
        half of the range of prior on the intercept.
    verbose : `bool`
        print output.
    plot : `bool`
        whether to plot the trace and corner plots from the `dynesty` packages.
    save
        path to save plots.

    Methods
    -------
    compare_models(x, y, yerr, m_span, b_span, verbose)
        compares between an unconstrained and a constant model. If the unconstrained model is preferred, it compares a model with a positive slope with one with a negative slope.
    run_dynesty(loglike, ptform, ndim)
        performs the dynamic nested sampling.
    loglike_linear(theta, x, y, yerr)
        computes likelihood for linear model (m,b, ln f).
    loglike_constant
        computes likelihood for constant model (b, ln f)
    ptform_linear(utheta, m_span, b_span)
        establishes the priors intervals.
    ptform_constant(utheta, b_span)
        establishes the priors intervals.
    ptform_positive_slope(utheta, m_span, b_span)
        establishes the priors intervals.
    ptform_negative_slope(utheta, m_span, b_span)
        establishes the priors intervals.
    """
    def __init__(self, x:np.array, y:np.array, yerr:np.array, m_span:int=100, b_span:int=100, verbose:bool=True, plot:bool=True, save=None):

        self.x = x
        self.y = y
        self.yerr = yerr
        self.m_span = m_span
        self.b_span = b_span

        resA, resB, resPos, resNeg, model, logK_linear_vs_const, logK_sign, logZ = self.compare_models(x, y, yerr, m_span, b_span)

        if model == "zero":
            result = resB
        elif model == "positive":
            result = resPos
        elif model == "negative":
            result = resNeg

        weights = np.exp(result.logwt - result.logz[-1]) # posterior samples
        samples = result.samples
        posterior_samples = resample_equal(samples, weights)
        
        linear_fit_params = {}
        if model != "zero":
            labels = ["m", "b", "ln_f"]
        else:
            labels = ["b", "ln_f"]

        if verbose: print("Linear fit parameters:")

        for i, label in enumerate(labels):
            q50 = np.percentile(posterior_samples[:, i], 50)
            q16 = np.percentile(posterior_samples[:, i], 16)
            q84 = np.percentile(posterior_samples[:, i], 84)
            linear_fit_params[label] = [q50, np.mean([q84-q50,q50-q16])]
            if verbose:
                print(f"{label} = {q50:.06f} +/- {np.mean([q84-q50,q50-q16]):.06f}")

        print("-"*30)
                
        if plot:
            fig, _ = dyplot.traceplot(result, labels=labels, fig=plt.subplots(len(labels), 2, figsize=(8, 6)))
            fig.tight_layout()
            if save: 
                fig.savefig(save+"traceplot.pdf", dpi=300)
            
            fig, _ = dyplot.cornerplot(result, show_titles=True, labels=labels, fig=plt.subplots(len(labels), len(labels), figsize=(7, 7)), quantiles=(0.16,0.5,0.84), title_quantiles=[0.16,0.5,0.84])
            if save: 
                fig.savefig(save+"cornerplot.pdf", dpi=300)
        
        self.results = (linear_fit_params, model, logK_linear_vs_const, logK_sign, logZ)


    def compare_models(self, x:np.array, y:np.array, yerr:np.array, m_span:int=100, b_span:int=100, verbose:bool=True):
        """First compares between an unconstrained and a constant model. If the unconstrained model is preferred, it compares a model with a positive slope with one with a negative slope.
        
        Parameters
        ----------
        x : `numpy array`
            orbital phases or mu arrays.
        y : `numpy array`
            CCF parameter (central RV, line-width measure or line-center intensity).
        yerr : `numpy array`
            uncertainty of CCF parameter.
        m_span : `int`
            half of the range of prior on the slope.
        b_span : `int`
            half of the range of prior on the intercept.
        verbose : `bool`
            print output.

        Returns
        -------
        resA
            dynesty results for initial linear model (m!=0).
        resB
            dynesty results for constant model (m=0).
        resPos
            dynesty results for linear model with positive slope (m>0).
        resNeg 
            dynesty results for linear model with negative slope (m<0).
        model : `str`
            type of model obtained.
        logK_linear_vs_const : `float`
            evidence difference between constant and unconstrained models.
        logK_sign : `float`
            evidence difference between positive and negative models.
        logZ : `float`
            evidence of final model.
        """
        # 1) Linear model m,b
        like_A = lambda th: self.loglike_linear(th, x, y, yerr)
        pt_A = lambda u: self.ptform_linear(u, m_span, b_span)
        resA = self.run_dynesty(like_A, pt_A, ndim=3)

        # 2) Constant model (m=0)
        like_B = lambda th: self.loglike_constant(th, x, y, yerr)
        pt_B = lambda u: self.ptform_constant(u, b_span)
        resB = self.run_dynesty(like_B, pt_B, ndim=2)

        logZ_A, logZerr_A = resA.logz[-1], resA.logzerr[-1]
        logZ_B, logZerr_B = resB.logz[-1], resB.logzerr[-1]
        logK_linear_vs_const = logZ_A - logZ_B

        if verbose:
            print("-"*30)
            print("Linear vs Constant")
            print(f"logZ(linear)   = {logZ_A:.3f} ± {logZerr_A:.3f}")
            print(f"logZ(constant) = {logZ_B:.3f} ± {logZerr_B:.3f}")
            print(f"log Bayes factor = {logK_linear_vs_const:.3f}")
        if logK_linear_vs_const > 2.3:
            if verbose:
                print("Unconstrained model favored.")
                print("-"*30)
        else:
            if verbose:
                print("Zero-slope model favored")
                print("========================")
            model = "zero"
            return resA, resB, None, None, model, logK_linear_vs_const, None, logZ_B
        
        #if linear is favored: compare sign of slope
        #positive slope
        pt_pos = lambda u: self.ptform_positive_slope(u, m_span, b_span)
        resPos = self.run_dynesty(lambda th: self.loglike_linear(th, x, y, yerr), pt_pos, ndim=3)
        #negative slope
        pt_neg = lambda u: self.ptform_negative_slope(u, m_span, b_span)
        resNeg = self.run_dynesty(lambda th: self.loglike_linear(th, x, y, yerr), pt_neg, ndim=3)

        logZ_pos, logZerr_pos = resPos.logz[-1], resPos.logzerr[-1]
        logZ_neg, logZerr_neg = resNeg.logz[-1], resNeg.logzerr[-1]
        logK_sign = logZ_neg - logZ_pos

        if logK_sign > 2.3:
            model = "negative"
            logZ = logZ_neg
        else:
            model = "positive"
            logZ = logZ_pos

        if verbose:
            print("Positive vs Negative slope")
            print("===========================")
            print(f"logZ(m>0)  = {logZ_pos:.3f} ± {logZerr_pos:.3f}")
            print(f"logZ(m<0)  = {logZ_neg:.3f} ± {logZerr_neg:.3f}")
            print(f"log Bayes factor = {logK_sign:.3f}")
            if logK_sign > 2.3:
                print("Negative slope favored.")
            else:
                print("Positive slope favored.")
            print("==========================")

        return resA, resB, resPos, resNeg, model, logK_linear_vs_const, logK_sign, logZ


    # Dynesty runner
    def run_dynesty(self, loglike, ptform, ndim):
        dsampler = DynamicNestedSampler(loglike, ptform, ndim, bound='multi', sample='rwalk')
        dsampler.run_nested()
        return dsampler.results

    # log-likelihood
    def loglike_linear(self, theta, x, y, yerr):
        m, b, lnf = theta
        model = m * x + b
        inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2 * lnf))
        return -0.5 * (np.sum((y-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

    # log-likelihood for intercept-only (b) where m ≡ 0
    def loglike_constant(self, theta, x, y, yerr):
        b, lnf = theta
        model = b
        inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2 * lnf))
        return -0.5 * (np.sum((y-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

    # prior transform
    def ptform_linear(self, utheta, m_span, b_span):
        um, ub, ulf = utheta
        m = -m_span + 2*m_span * um  
        b = -b_span + 2*b_span * ub 
        lnf = -7 + 8*ulf 
        return m, b, lnf

    def ptform_constant(self, utheta, b_span):
        ub, ulf = utheta 
        b = -b_span + 2*b_span * ub 
        lnf = -7 + 8*ulf
        return b, lnf

    def ptform_positive_slope(self, utheta, m_span, b_span):
        # positive slope: m in [0, m_max]
        um, ub, ulf = utheta
        m = 0 + m_span * um 
        b = -b_span + 2*b_span * ub
        lnf = -7 + 8*ulf 
        return m, b, lnf

    def ptform_negative_slope(self, utheta, m_span, b_span):
        # positive slope: m in [0, m_max]
        um, ub, ulf = utheta
        m = -m_span + m_span * um  # m in [-100, 0]
        b = -b_span + 2*b_span * ub # b in [-100, +100]
        lnf = -7 + 8*ulf # lnf in [-7, +1]
        return m, b, lnf