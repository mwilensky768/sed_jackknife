import pypolychord
from pypolychord.priors import UniformPrior, GaussianPrior
import numpy as np
from scipy.optimize import minimize
import argparse

def slice_setup(jk_mode=None):
    if jk_mode is None:
        slices = (slice(0,1), slice(1, 2), slice(2, 60), slice(60, 150))
    elif jk_mode == "low":
        slices = (slice(0,1), slice(1, 2), slice(2, 60))
    elif jk_mode == "high":
        slices = (slice(0,1), slice(1, 2), slice(2, 92))
    elif jk_mode == "sim":
        slices = (slice(0,1), slice(1, 2), slice(2, 60), slice(60, 150), slice(150, 151))
    else:
        raise ValueError("Invalid jk_mode")
    return slices
        

def read_dat(filedir, field, jk_mode=None):
    """
    Reads data and metadata from an npy file according to which axis is which:
    1 - frequencies
    2 - flux density in Jy/pixel
    3 - noise standard deviation in Jy/pixel
    4 - calibration error in fractional error

    Parameters:
        filedir (str): 
            Path to directory containing the files.
        field (int):
            Which field to process. Fields with designation greater than 2
            are simulation fields.
        jk_mode (None or str):
            Which PPD jackknife is being run. 

    Returns:
        data (array):
            The flux densities in Jy/pixel
        noise (array):
            The noise _variances_ in (Jy/pixel)^2
        gain_cov (array):
            The gain error covariances, in (fractional error)^2
        freqs (array):
            Frequencies in MHz
        S0_cent (float):
            The 0th entry in the flux density array, specifically used
            for centering a prior.
    """
    datarr = np.load(f"{filedir}/apdata_source{field}.npy")
    if jk_mode == "low":
        slc = slice(0, 60)
    elif jk_mode == "high":
        slc = list(range(2)) + list(range(60, 150))
    else:
        slc = slice(None)
    data = datarr[2, slc]
    noise = datarr[3, slc]**2
    gain_cov = np.diag(datarr[4, slc]**2)
    freqs = datarr[1, slc]
    S0_cent = datarr[2, 0]
    
    return data, noise, gain_cov, freqs, S0_cent

def gain_cov_process(gain_cov, bitstr, gain_std, slices=slice_setup()):
    """
    Transforms the reported gain covariance to the marginal gain covariance
    after the hyperparameters (gain offsets) are marginalized out.

    Parameters:
        gain_cov (array):
            Reported gain covariance from each experiment.
        bitstr (str):
            Binary string expressing which experiments have gain offsets.
        gain_std (float):
            The standard deviation for the prior on the gain offsets.
    Returns:
        new_gain_cov (array):
            The marginalized gain covariance.
        
    """
    bitlist = [int(bit) for bit in bitstr]
    new_gain_cov = np.copy(gain_cov)
    if np.any(bitlist):
        for bit_ind, bit in enumerate(bitlist):
            if bit:
                slice_use = slices[bit_ind]
                slice_size = slice_use.stop - slice_use.start
                ones = np.ones(slice_size)
                cov_add = gain_std**2 * np.outer(ones, ones)
                new_gain_cov[slice_use, slice_use] += cov_add
    
    return new_gain_cov

def gain_std_process(gain_std, bitstr, null_factor=1e-3):
    """
    Make the standard deviations for the gain offsets according to the
    activated entries in bitstr.

    Parameters:
        gain_std (float):
            The standard deviation of the gain offset prior.
        bitstr (str):
            Binary string expressing which experiments have gain offsets.
        null_factor (float):
            The factor by which to multiply the standard deviation
            for the deactivated entries
    Returns:
        gain_stds_use (float):
            The gain stds for activated entries, but multiplied by null_factor
            for the deactivated entries.
    """
    bitlist = [int(bit) for bit in bitstr]
    gain_stds = np.full(len(bitlist), gain_std)
    gain_stds_null = gain_stds * null_factor
    gain_stds_use = np.where(bitlist, gain_stds, gain_stds_null)
    
    return gain_stds_use
        

def get_index(alpha_0, c, freqs, ref_freq=73):
    """
    Get the (variable) spectral index.

    Parameters:
        alpha_0 (float):
            The spectral index at ref_freq
        c (float):
            The curvature parameter for the spectral index.
        freqs (float):
            Frequencies, in MHz
        ref_freq (float):
            Reference frequency, in MHz.
    Returns:
        ind (float):
            The (variable) spectral index as a function of frequency.
    """
    ind = alpha_0 + c * np.log(freqs / ref_freq)
    
    return ind

def get_model(alpha_0, S0, c, freqs, ref_freq=73):
    """
    Get the (potentially curved) power law model for the supplied frequencies and parameters.

    Parameters:
        alpha_0 (float):
            The spectral index at ref_freq
        S0 (float):
            The flux density at ref_freq
        c (float):
            The curvature parameter for the spectral index.
        freqs (float):
            Frequencies, in MHz
        ref_freq (float):
            Reference frequency, in MHz.

    Returns:
        model (float):
            The power law model with the given parameters evaluated at the
            supplied frequencies.
    """
    
    ind = get_index(alpha_0, c, freqs, ref_freq=ref_freq)
    model = S0 * (freqs / ref_freq)**ind
    
    return model

def loglike(params, freqs, data, noise, gain_cov, ref_freq=73, offset=0, low_dim=False, curv=False,
            slices=slice_setup()):
    """
    Get the log-likelihood of the parameters.

    Parameters:
        params (array_like):
            Parameters that Polychord is sampling. The first 2-3 are power law parameters,
            the rest, if any, are gain offsets.
        freqs (array):
            Frequencies, in MHz
        data (array):
            The flux densities in Jy/pixel
        noise (array):
           The noise _variances_ in (Jy/pixel)^2
        gain_cov (array):
            The processed (marginalized) gain covariances if low_dim=True, otherwise the reported
            gain variances.
        ref_freq (float):
            Reference frequency, in MHz.
        offset (float):
            Arbitrary amount to subtract from this log-likelihood. 
            Was useful for debugging in early development.
        low_dim (bool):
            Whether Polychord is sampling for a low-dimensional (pre-marginalized)
            run.
        curv (bool):
            Whether the power law is considered to be curved.
        slices (tuple):
            tuple of slices into the data
    Returns:
        logL (float):
            The log-likelihood of the parameters given the data and hyperparameters.
        (chisq, logdetcov):
            The chi-square and log|cov| at these parameter values (derived statistics)
    """
    
    if curv: 
        model_args = params[:3]
        num_plaw_params = 3
    else:
        model_args = (params[0], params[1], 0)
        num_plaw_params = 2
    model = get_model(*model_args, freqs, ref_freq=ref_freq)
 
    gained_model = np.copy(model)
    num_gains = len(params) - num_plaw_params
    if not low_dim: # apply gains, otherwise condition on gain_means=1
        for slc_ind, slc in enumerate(slices[:num_gains]):
            gained_model[slc] *= (1 + params[slc_ind + num_plaw_params])
    
    res = data - gained_model
    cov = np.outer(model, model) * gain_cov + np.diag(noise)
    
    cinv_res = np.linalg.solve(cov, res)
    
    chisq = np.sum(res * cinv_res)
    logdetcov = np.linalg.slogdet(cov)[1] + len(freqs) * np.log(2 * np.pi)
    
    logL = - 0.5 * (chisq + logdetcov) - offset
    
    return logL, (chisq, logdetcov)

def prior(cube_coords, alpha_bounds, S0_bounds, c_bounds, gain_hypermean, gain_hyperstd, low_dim=False,
          flat=False, curv=False):
    alpha_prior = UniformPrior(*alpha_bounds)(cube_coords[0])
    S0_prior = UniformPrior(*S0_bounds)(cube_coords[1])
    if curv:
        c_prior = UniformPrior(*c_bounds)(cube_coords[2])
        plaw_ret = (alpha_prior, S0_prior, c_prior)
    else:
        plaw_ret = (alpha_prior, S0_prior)
    
    # Same priors for now
    if not low_dim:
        num_gain = len(gain_hyperstd)
        gain_ret = GaussianPrior(np.full(num_gain, gain_hypermean), gain_hyperstd)(cube_coords[-num_gain:])
        gain_ret = tuple(gain_ret)
    else:
        gain_ret = ()

    return plaw_ret + gain_ret

def get_offset(init_guess, freqs, data, noise, gain_cov, alpha_bounds, S0_bounds, c_bounds, 
               gain_bounds, ref_freq=73, low_dim=False):
    """
    Does not work with jk_mode=high
    """

    def neg_logL(params):
        logL, _ = loglike(params, freqs, data, noise, gain_cov, ref_freq=ref_freq, offset=0, low_dim=low_dim)
        return -logL
    
    bounds = (alpha_bounds, S0_bounds, c_bounds, gain_bounds, gain_bounds, gain_bounds, gain_bounds)
    
    if low_dim:
        init_guess = init_guess[:3]
        bounds = bounds[:3]
    
    mini = minimize(neg_logL, init_guess, bounds=bounds)
    loc = mini["x"]
    val = -neg_logL(loc)
    print(f"Found offset {val} at location {loc}")
    
    return val
                                      
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, required=False, default="0")
    parser.add_argument("--outdir", required=True, help="Where the outputs should be stored")
    parser.add_argument("--filedir", required=False, default="./data",
                        help="Directory where the SED data live")
    parser.add_argument("--nlive-fac", dest="nlive_fac", type=int, default=2, required=False)
    parser.add_argument("--num-repeats-fac", dest="num_repeats_fac", type=int, default=1, required=False)
    parser.add_argument("--ref_freq", required=False, default=73, type=float)
    parser.add_argument("--gain-std", required=False, default=0.25, type=float, dest="gain_std")
    parser.add_argument("--low-dim", required=False, action="store_true", dest="low_dim")
    parser.add_argument("--curv", required=False, action="store_true")
    parser.add_argument("--offset", required=False, action="store_true", dest="offset")
    parser.add_argument("--flat", required=False, action="store_true",
                        help="Whether to use a flat prior on the gain offsets.")
    parser.add_argument("--bitstr", required=False, action="store", type=str)
    parser.add_argument("--jk-mode", required=False, action="store", default=None, dest="jk_mode", 
                        help="String specifying which validation jackknife is being run")
    args = parser.parse_args()

    
    """
    Constants
    """
    filedir = args.filedir
    file_root = f"MEERKLASS_field{args.field}_nlive{args.nlive_fac}_nrepeat{args.num_repeats_fac}_lowdim{args.low_dim}_curv{args.curv}_bitstr{args.bitstr}_jkmode_{args.jk_mode}_hyper"

    slices = slice_setup(args.jk_mode)
    
    data, noise, gain_cov, freqs, S0_cent = read_dat(filedir, args.field, args.jk_mode)
    if args.low_dim:
        gain_cov = gain_cov_process(gain_cov, args.bitstr, args.gain_std, slices=slices)
    
    gain_hypermean = 0
    gain_hyperstd = gain_std_process(args.gain_std, args.bitstr)
    num_gains = len(gain_hyperstd)
    
    alpha_bounds = (-1.8, 0)
    S0_bounds = (0.5 * S0_cent, 2 * S0_cent)
    c_bounds = (-0.3, 0)
    gm_bounds = (0, 3) # Just used for optimization
    
    if args.low_dim:
        nDims = 2 + int(args.curv) # curved power law
    else:
        nDims = 2 + num_gains + int(args.curv) # add gain means
    nDerived = 2
    
    # Fits from Mel's paper
    init_guesses = {"0": (-0.52, S0_cent, -0.13, 1, 1, 1, 1), 
                    "1": (-0.46, S0_cent, -0.14, 1, 1, 1, 1), 
                    "2": (-0.54, S0_cent, -0.08, 1, 1, 1, 1)}
    

    if args.offset:
        offset = get_offset(init_guesses[args.field], freqs, data, noise, gain_cov, alpha_bounds, S0_bounds,
                            c_bounds, gm_bounds, ref_freq=args.ref_freq, low_dim=args.low_dim)
    else:
        offset = 0
        
    
    settings = pypolychord.PolyChordSettings(nDims, nDerived, base_dir=f"{args.outdir}/chains", 
                                             file_root=file_root,
                                             nlive=args.nlive_fac * nDims * 25,
                                             num_repeats=args.num_repeats_fac * nDims * 5)
    
    
    """
    End constants
    """
    

    def loglikewrap(params):

        return loglike(params, freqs, data, noise, gain_cov, ref_freq=args.ref_freq, offset=offset, 
                       low_dim=args.low_dim, curv=args.curv, slices=slices)
    
    def priorwrap(cube_coords):
        return prior(cube_coords, alpha_bounds, S0_bounds, c_bounds, gain_hypermean, gain_hyperstd, 
                     low_dim=args.low_dim, flat=args.flat, curv=args.curv)


    pypolychord.run_polychord(loglikewrap, nDims, nDerived, settings, prior=priorwrap)