import numpy as np
import matplotlib.pyplot as plt
from dynesty import DynamicNestedSampler
from dynesty.utils import resample_equal
from dynesty import plotting as dyplot

class run_nestedsampler:
    
    def __init__(self, x, y, yerr, m_span=100, b_span=100, verbose=True, plot=True, save=None):

        self.x = x
        self.y = y
        self.yerr = yerr
        self.m_span = 100
        self.b_span = 100

        resA, resB, resPos, resNeg, model = self.compare_models(x, y, yerr, m_span, b_span)

        if model == "zero":
            result = resB
        elif model == "positive":
            result = resPos
        elif model == "negative":
            result = resNeg

        #posterior samples (weights) 
        weights = np.exp(result.logwt - result.logz[-1])
        samples = result.samples
        posterior_samples = resample_equal(samples, weights)

        labels = ["m", "b", "ln_f"]

        linear_fit_params = {}
        if model != "zero":
            print("Linear fit parameters:")
            for i, label in enumerate(labels):
                q50 = np.percentile(posterior_samples[:, i], 50)
                q16 = np.percentile(posterior_samples[:, i], 16)
                q84 = np.percentile(posterior_samples[:, i], 84)
                linear_fit_params[label] = [q50, np.mean([q84-q50,q50-q16])]
                if verbose:
                    print(f"{label} = {q50} +/- {np.mean([q84-q50,q50-q16])}")
                    
        if plot:
            fig, _ = dyplot.traceplot(result, labels=labels, fig=plt.subplots(3, 2, figsize=(8, 6)))
            fig.tight_layout()
            if save: fig.savefig(save+"traceplot.pdf", dpi=300)
            
            fig, _ = dyplot.cornerplot(result, show_titles=True, labels=labels, fig=plt.subplots(3, 3, figsize=(7, 7)), quantiles=(0.16,0.5,0.84), title_quantiles=[0.16,0.5,0.84])
            if save: fig.savefig(save+"cornerplot.pdf", dpi=300)

            plt.show()
        
        self.results = (linear_fit_params, model)


    # MASTER FUNCTION
    def compare_models(self, x, y, yerr, m_span=100, b_span=100, verbose=True):
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
            print("\n========================")
            print("Linear vs Constant")
            print("========================")
            print(f"logZ(linear)   = {logZ_A:.3f} ± {logZerr_A:.3f}")
            print(f"logZ(constant) = {logZ_B:.3f} ± {logZerr_B:.3f}")
            print(f"log Bayes factor = {logK_linear_vs_const:.3f}")
        if logK_linear_vs_const > 2.3:
            if verbose:
                print("Unconstrained model favored.")
                print("========================")
        else:
            if verbose:
                print("Zero-slope model favored")
                print("========================")
                model = "zero"
            return resA, resB, None, None, model
        
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

        if verbose:
            print("Positive vs Negative slope")
            print("========================")
            print(f"logZ(m>0)  = {logZ_pos:.3f} ± {logZerr_pos:.3f}")
            print(f"logZ(m<0)  = {logZ_neg:.3f} ± {logZerr_neg:.3f}")
            print(f"log Bayes factor = {logK_sign:.3f}")
            if logK_sign > 2.3:
                print("Negative slope favored.")
                model = "negative"
            else:
                print("Positive slope favored.")
                model = "positive"
            print("========================")

        return resA, resB, resPos, resNeg, model


    #Dynesty runner
    def run_dynesty(self, loglike, ptform, ndim):
        dsampler = DynamicNestedSampler(loglike, ptform, ndim, bound='multi', sample='rwalk')
        dsampler.run_nested()
        return dsampler.results

    #log-likelihood
    def loglike_linear(self, theta, x, y, yerr):
        m, b, lnf = theta
        model = m * x + b
        inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2 * lnf))
        return -0.5 * (np.sum((y-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

    #log-likelihood for intercept-only (b) where m ≡ 0
    def loglike_constant(self, theta, x, y, yerr):
        b, lnf = theta
        model = b
        inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2 * lnf))
        return -0.5 * (np.sum((y-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

    #prior transform
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
        #positive slope: m in [0, m_max]
        um, ub, ulf = utheta
        m = 0 + m_span * um 
        b = -b_span + 2*b_span * ub
        lnf = -7 + 8*ulf 
        return m, b, lnf

    def ptform_negative_slope(self, utheta, m_span, b_span):
        #positive slope: m in [0, m_max]
        um, ub, ulf = utheta
        m = -m_span + m_span * um  # m in [-100, 0]
        b = -b_span + 2*b_span * ub # b in [-100, +100]
        lnf = -7 + 8*ulf # lnf in [-7, +1]
        return m, b, lnf