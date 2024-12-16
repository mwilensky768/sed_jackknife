import pypolychord
from pypolychord.priors import UniformPrior, GaussianPrior
import numpy as np
from scipy.optimize import minimize
import argparse

def slice_setup(jk_mode=None):
    """
    Get slices down the frequency axis of the input data based on which 
    jackknife is being performed.

    Parameters:
        jk_mode (str):
            Choice of jackknife. Must be one of None, 'low', 'high', 
            'sim'.

    Returns:
        slices (tuple):
            Sequence of slices where each slice corresponds to a particular
            experiment.
    """
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
        

def read_dat(filedir, fields, jk_mode=None, slices=slice_setup(), 
             single_law=False):
    """
    Reads data and metadata from an npy file according to which axis is which:
    1 - frequencies
    2 - flux density in Jy/pixel
    3 - noise standard deviation in Jy/pixel
    4 - calibration error in fractional error

    Parameters:
        filedir (str): 
            Path to directory containing the files.
        fields (int or seq):
            Which fields to process. Last field is the reference field.
            Must not jointly analyze fields with different frequencies.
        jk_mode (None or str):
            Which jackknife is being run. 

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

    Nfreqs = sum([slice.stop - slice.start for slice in slices])
    Nfields = len(fields)
    data_shape = [Nfields, Nfreqs]
    gain_cov_shape = [Nfreqs, Nfreqs]
        
    data = np.zeros(data_shape)
    noise = np.zeros(data_shape)
    gain_cov = np.zeros(gain_cov_shape)
    S0_cent = np.zeros(Nfields)

    for field_ind, field in enumerate(fields):
        datarr = np.load(f"{filedir}/apdata_source{field}.npy")
        if jk_mode == "low":
            slc = slice(0, 60)
        elif jk_mode == "high":
            slc = list(range(2)) + list(range(60, 150))
        else:
            slc = slice(None)
        data[field_ind] = datarr[2, slc]
        noise[field_ind] = datarr[3, slc]**2
        gain_cov = np.diag(datarr[4, slc]**2)
        freqs = datarr[1, slc]
        S0_cent[field_ind] = datarr[2, 0]

    
    return data, noise, gain_cov, freqs, S0_cent
        

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

def loglike(params, freqs, data, noise, gain_cov, ref_freq=73., low_dim=False, 
            curv=False, slices=slice_setup(), single_law=False):
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
        low_dim (bool):
            Whether Polychord is sampling for a low-dimensional (pre-marginalized)
            run.
        curv (bool):
            Whether the power law is considered to be curved.
        slices (tuple):
            tuple of slices into the data
        single_law (bool):
            Whether to only model a single power law.
    Returns:
        logL (float):
            The log-likelihood of the parameters given the data and hyperparameters.
        (chisq, logdetcov):
            The chi-square and log|cov| at these parameter values (derived statistics)
    """
    num_fields = data.shape[0]
    num_freqs = len(freqs)
    num_laws = num_fields + 1 - int(single_law)
    num_plaw_params_per_law = 2 + int(curv)
    num_plaw_params = num_plaw_params_per_law * num_laws

    model_params = params[:num_plaw_params].reshape(
        num_laws, 
        num_plaw_params_per_law
    )

    model = np.zeros([num_laws, num_freqs])
    for law_ind in range(num_laws):
        if curv: 
            this_model_args = model_params[law_ind]
        else:
            this_model_args = (model_params[law_ind, 0], model_params[law_ind, 1], 0)

        model[law_ind] = get_model(*this_model_args, freqs, ref_freq=ref_freq)
    if not single_law:
        model = model[:num_fields] - model[-1]

 
    gained_model = np.copy(model)
    num_gains = len(params) - num_plaw_params
    if not low_dim: # apply gains, otherwise condition on gain_means=1
        for slc_ind, slc in enumerate(slices[:num_gains]):
            gained_model[:, slc] *= (1 + params[slc_ind + num_plaw_params])

    
    res = data - gained_model
    res = res.flatten()

    cov = np.zeros([num_fields, num_freqs, num_fields, num_freqs])
    for field1 in range(num_fields):
        for field2 in range(num_fields):
            if not single_law:
                cov[field1, :, field2] += np.diag(noise[-1]) 
            if field1 == field2:
                cov[field1, :, field2] += np.diag(noise[field1])
            if low_dim:
                cov[field1, :, field2] += np.outer(model[field1], model[field2]) * gain_cov
    cov = cov.reshape(num_fields * num_freqs, num_fields * num_freqs)    
    
    cinv_res = np.linalg.solve(cov, res)
    
    chisq = np.sum(res * cinv_res)
    logdetcov = np.linalg.slogdet(cov)[1] + (num_fields * num_freqs) * np.log(2 * np.pi)
    
    logL = - 0.5 * (chisq + logdetcov)
    
    return logL, (chisq, logdetcov)

def prior(cube_coords, alpha_bounds, S0_bounds, c_bounds, gain_cov, Nfields, 
          low_dim=False, curv=False, single_law=False, enforce_min=False):
    
    nplaw_params = 2 + int(curv)
    plaw_ret = []
    for field_ind in range(Nfields + 1 - int(single_law)):
        alpha_prior = UniformPrior(*alpha_bounds)(cube_coords[field_ind * nplaw_params])
        if (field_ind > Nfields) and enforce_min: 
            # Must be less than all other S0s 
            S0_prior = UniformPrior(0, min(plaw_ret[1::nplaw_params]))(cube_coords[field_ind * nplaw_params + 1])
        else:
            S0_prior = UniformPrior(*S0_bounds[field_ind])(cube_coords[field_ind * nplaw_params + 1])
        if curv:
            c_prior = UniformPrior(*c_bounds)(cube_coords[field_ind * nplaw_params + 2])
            plaw_ret += [alpha_prior, S0_prior, c_prior]
        else:
            plaw_ret += [alpha_prior, S0_prior]
    

    if not low_dim:
        num_gain = len(gain_cov)
        gain_ret = GaussianPrior(np.full(num_gain, 0), np.sqrt(gain_cov))(cube_coords[-num_gain:])
        gain_ret = list(gain_ret)
    else:
        gain_ret = []

    return plaw_ret + gain_ret
                                      
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=int, required=False, default=0,
                        nargs="*")
    parser.add_argument("--outdir", required=True, help="Where the outputs should be stored")
    parser.add_argument("--filedir", required=False, default="./data",
                        help="Directory where the SED data live")
    parser.add_argument("--nlive-fac", dest="nlive_fac", type=int, default=1, required=False)
    parser.add_argument("--num-repeats-fac", dest="num_repeats_fac", type=int, default=1, required=False)
    parser.add_argument("--ref-freq", required=False, default=73, type=float, dest="ref_freq")
    parser.add_argument("--gain-std", required=False, default=0.25, type=float, dest="gain_std")
    parser.add_argument("--low-dim", required=False, action="store_true", dest="low_dim")
    parser.add_argument("--curv", required=False, action="store_true")
    parser.add_argument("--ref-field", required=False, action="store", type=int,
                        dest="ref_field", default=3)
    parser.add_argument("--jk-mode", required=False, action="store", default=None, dest="jk_mode", 
                        help="String specifying which validation jackknife is being run")
    parser.add_argument("--alpha-bounds", required=False, action="store", dest="alpha_bounds",
                        type=float, nargs=2, default=(-1.8, 0))
    parser.add_argument("--single-law", required=False, action="store_true",
                        dest="single_law")
    parser.add_argument("--enforce-min", action="store_true", default=False,
                        dest="enforce_min")
    parser.add_argument("--S0-bounds", required=False, default=(1, 4), action="store", nargs=2, type=float,
                        dest="S0_bounds")
    args = parser.parse_args()

    
    """
    Constants
    """
    filedir = args.filedir

    slices = slice_setup(args.jk_mode)
    Nfields = len(args.fields)
    alpha_bounds = (min(args.alpha_bounds), max(args.alpha_bounds))


    fields_as_str = [str(field) for field in args.fields]
    fieldstr = "".join(fields_as_str)
    file_root = f"MEERKLASS_fields{fieldstr}_nlive{args.nlive_fac}_nrepeat{args.num_repeats_fac}_lowdim{args.low_dim}_curv{args.curv}_jkmode_{args.jk_mode}_alpha_bounds{alpha_bounds[0]}_{alpha_bounds[1]}_ref_freq{args.ref_freq}_ref_field{args.ref_field}_enforce_min{args.enforce_min}_single_law{args.single_law}_S0_bounds_{min(args.S0_bounds)}_{max(args.S0_bounds)}"

    fields = list(args.fields) + [args.ref_field]
    data, noise, gain_cov, freqs, S0_cent = read_dat(filedir, fields, 
                                                     args.jk_mode, slices=slices)
    data = data[:Nfields] - data[-1]

    # Abbreviate gain_cov
    

    if args.low_dim:
        gain_cov[slices[2], slices[2]] = gain_cov[2, 2] # MeerKAT 1
        if jk_mode is not None:
            gain_cov[slices[3], slices[3]] = gain_cov[3, 3] # MeerKAT 2
        num_gains = 0
    else:
        num_gains = len(slices)
        gain_cov = np.array((gain_cov[0, 0], gain_cov[1, 1], gain_cov[2, 2], gain_cov[-1, -1]))[:num_gains]

    

    if args.single_law:
        S0_bounds = [(0.5 * S0_cent[field_ind], 2 * S0_cent[field_ind]) for field_ind in range(Nfields)]
    else:
        S0_bounds = (Nfields + 1) * [(min(args.S0_bounds), max(args.S0_bounds)), ] # This parameter takes on a different meaning with double_law
    c_bounds = (-0.3, 0)


    
    nplaw_params = 2 + int(args.curv)
    nDims = nplaw_params * (Nfields + 1 - int(args.single_law))
    nDims += num_gains
    nDerived = 2
        
    
    settings = pypolychord.PolyChordSettings(nDims, nDerived, 
                                             base_dir=f"{args.outdir}/chains", 
                                             file_root=file_root,
                                             nlive=args.nlive_fac * nDims * 25,
                                             num_repeats=args.num_repeats_fac * nDims * 5)
    

    """
    End constants
    """
    

    def loglikewrap(params):
        logL, (chisq, logdetcov) = loglike(
            params, 
            freqs, 
            data, 
            noise, 
            gain_cov,
            ref_freq=args.ref_freq,
            low_dim=args.low_dim,
            curv=args.curv,
            slices=slices,
            single_law=args.single_law
        )
        return logL, (chisq, logdetcov)
    
    def priorwrap(cube_coords):
        return prior(cube_coords, alpha_bounds, S0_bounds, c_bounds, 
                     gain_cov, Nfields, low_dim=args.low_dim, 
                     curv=args.curv, single_law=args.single_law, 
                     enforce_min=args.enforce_min)


    output = pypolychord.run_polychord(loglikewrap, nDims, nDerived, settings, prior=priorwrap)

"""
    param_names = []
    for field in args.fields:
        model_params_field = [
            r"$\alpha_%s(\nu_0)$" % field,
            r"$S_%s(\nu_0)$" % field,
        ]
        if args.curv:
            model_params_field.append(r"$c_%s$" % field)
        param_names.extend(model_params_field)
    
    exp_names = ["LWA", "Has", "MK1", "MK2"]
    if args.jk_mode == "high":
        exp_names.pop(2)
    for gain_ind in range(num_gains):
        param_names.append(r"$\varepsilon_%s$" % exp_names[gain_ind])

    output.make_paramnames_files(param_names)
"""
