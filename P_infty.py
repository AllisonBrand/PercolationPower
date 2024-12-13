from newman_ziff import SitePercolate
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

'''Execute a number of runs of the Newman Ziff algorithm to compute the variance 
and mean of P_infty (the fraction of occupied sites that are part of a spanning cluster), 
as a function of occupation ratio p.'''

def same(a, b, n = 0):
    '''Returns True if a and b are the same to within n digits of precision past the decimal.
    Giving a negative number for n will loosen the restriction to digits before the decimal.
    For instance, with n = 2, same(1.000, 1.001, n=2) will return True, and same(1.00, 1.01, n=2) will return False.
    With n = -1, same(1, 2, n=-1) will return True, and same(10, 20, n=-1) will return False.'''
    if abs(a - b) < 10**(-n):
        return True
    return False


def run(sp):
    '''Execute a run of the Newman Ziff algorithm using an instance of SitePercolate, 
    pull out the results on P_infty, a list where every index i has the P_infty computed
    at i+1 occupied sites. 
    
    P_infty is the total number of sites in spanning cluster(s), divided by number of occupied sites'''
    results = sp.run()

    P_infty = np.empty(len(results))
    for i in range(len(results)):
        P_infty[i] = results[i]['spanning']['P_infinity']
    
    return P_infty


# def func_P_infty(p, P_o, B):
#     '''Fitting function for P_infty as a function of '''
#     # For cubic lattice, percolation thershold p_c is 0.311
#     p_c = 0.311
#     return P_o * (p-p_c)**B



lattice_shape = (20, 20, 20)
sp = SitePercolate(lattice_shape, stats = "spanning cluster")
runs = []

# Start with two runs to get a fisrt estimate of the variance
n_runs = 20
for _ in range(n_runs):
    runs.append(run(sp))
mean = np.mean(runs, axis=0)
std_P = np.std(runs, axis=0)

# n = 2
# #  Run until the variance stabilizes to n digits of precision past the decimal.
# while not same(prev, now, n):
#     results = sp.run()
N = sp.num_sites
p = np.arange(N) / N # Occupation Ratio

# # Fit a function for P_infty:
# curve_fit(func_P_infty, p, mean, p0 = [])

# Convert variances in P_infty to variances in p that would create those variances in the mean of P_infty
smoothed_mean = np.empty(len(mean))
smoothed_mean[4:-4] = np.convolve(np.array([1/9]*9), mean, mode = 'valid')
smoothed_mean[:4] = mean[:4]
smoothed_mean[-4:] = mean[-4:]

dP_over_dp = abs(np.gradient(mean))
std_p = std_P / dP_over_dp

tp = 1.78
# volume of emulsion phase over volume of fixed bed
vol = 1.13
# void fraction of randomly poured spheres
void = 0.037
vol_frac = p*(vol-void)/vol
# delta_conductivity = tp*()


fig, axes = plt.subplots(3, 1, sharex=True)
plt.suptitle(f'Lattice Shape: {lattice_shape}')
plt.tight_layout()
for i in range(len(runs)):
    axes[0].plot(p, runs[i], 'tab:blue')
axes[0].plot(p, mean, 'r')
p_c = 0.311
B = 0.4
# P_o = (1-p_c)**-B
# axes[0].plot(p, P_o*(p-p_c)**B, '--', color='purple')
axes[0].set_title(f'Mean of {n_runs} runs')
axes[0].set_ylabel('Mean ' + r'$P_{\infty}$')
axes[1].plot(p, std_P)
axes[1].set_title('Standard Deviation')
axes[1].set_ylabel(r'std ($P_{\infty}$)')
axes[2].plot(p, dP_over_dp)
axes[2].set_title(r'$\frac{d P _{\infty}}{d p}$') # r'$\std(P_{\infty}) \cdot \frac{d p}{d P _{\infty}}$'
axes[2].set_ylabel(r'$\frac{d P _{\infty}}{d p}$')
plt.xlabel('Occupation Ratio (p)')
plt.show()