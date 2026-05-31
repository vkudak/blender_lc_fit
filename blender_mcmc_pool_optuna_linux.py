import numpy as np
import argparse
import pathlib
import time
from multiprocessing import Pool, cpu_count
import platform
import os
import sys
from datetime import timedelta

# Disable HDF5 file locking to avoid conflicts during parallel writes
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import emcee
import optuna  # <-- Added Optuna framework
from dotenv import load_dotenv
from dime_sampler import DIMEMove
import matplotlib.pyplot as plt
import matplotlib.path
import corner
from packaging import version

from blender_support import *

def fixed_path_deepcopy(self, memo=None):
    """
    Workaround for a known deepcopy issue in older Matplotlib versions under Python 3.14.
    """
    if memo is None:
        memo = {}
    p = self.__class__(
        copy.deepcopy(self.vertices, memo),
        copy.deepcopy(self.codes, memo),
        _interpolation_steps=self._interpolation_steps,
        readonly=False
    )
    memo[id(self)] = p
    return p

def model(var_params, conf_res, delete_tmp=True, sub_name=""):
    """
    Generate synthetic LC with Blender and a Python template script defined in the config file.
    
    Args:
        var_params: Array/list of parameters being optimized (Spin period, phase, precession, etc.)
        conf_res: Dictionary containing configuration metadata
        delete_tmp: (bool) Whether to delete the generated temporary video file
        sub_name: (str) Suffix for the generated video filename
    Return:
        Synthetic LC dictionary containing keys like 'time', 'mst' (magnitude), etc.
    """
    temp_dir_name = conf_res["temp_dir_name"]
    temp_dir_path = pathlib.Path(temp_dir_name)
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    tmp_script_path = os.path.join(temp_dir_name, "temp_blender_script.py")
    rnd_gen = "_" + gen_random_str()

    video_file = make_blender_script(tmp_script_path=tmp_script_path + rnd_gen,
                                     conf_res=conf_res, var_list=var_params, sub_name=sub_name)

    if video_file is False:
        sys.exit()

    res_code = blender_render(blender_path=conf_res["blender_path"],
                              tmp_script_path=tmp_script_path + rnd_gen,
                              log_dir_path=temp_dir_path)
    if res_code != 0:
        sys.exit()

    flux_res = process_video(video_file, w=0)
    if delete_tmp:
        os.remove(video_file)
        os.remove(tmp_script_path + rnd_gen)

    synth_lc = make_lc(N=flux_res['count'], flux=flux_res['flux'],
                       s_date=conf_res['lc_start_date'], s_time=conf_res['lc_start_time'],
                       norad=conf_res['sat_norad'], fps=conf_res["fps"],
                       tle_line1=conf_res['tle_line1'], tle_line2=conf_res['tle_line2']
                       )
    return synth_lc


def lnlike(var_params, lc_time, lc_mag, lc_mag_err, conf_res):
    """
    Log-likelihood function used by MCMC. Calculates how well the synthetic model fits observed data.
    """
    synth_lc = model(var_params, conf_res)
    m_diff = model_diff(synth_lc['time'], synth_lc['mst'], lc_time, lc_mag, conf_res=conf_res,
                        norm_mag=True, norm_range=(0, 5))

    fv_filename = os.path.join(conf_res['temp_dir_name'], "var_params.txt")
    with open(fv_filename, "a") as fv:
        mlist = np.append(np.array(var_params), -0.5 * np.sum((m_diff / 1.0) ** 2))
        np.savetxt(fv, mlist, fmt='%10.2f', delimiter=" ", newline=" ")
        fv.write("\n")

    return -0.5 * np.sum((m_diff / 1.0) ** 2)


def lnprior(var_params):
    """
    Log-prior function ensuring parameters stay within bounds specified by the configuration.
    """
    for i, param in enumerate(var_params):
        g_par = g_conf_res['var_params_list'][i]
        if g_par['min_val'] > param or param > g_par['max_val']:
            return -np.inf
    return 0.0


def lnprob(var_params):
    """
    Full posterior probability computation combining prior and likelihood.
    """
    lc_time, lc_mag, lc_mag_err = g_data
    lp = lnprior(var_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(var_params, lc_time, lc_mag, lc_mag_err, g_conf_res)


def init_pool(data, conf_res):
    """
    Initializer function for multiprocess pool workers to keep global variables local to each process shell.
    """
    global g_data, g_conf_res
    g_data = data
    g_conf_res = conf_res


# =========================================================================
# NEW: OPTUNA OBJECTIVE FUNCTION
# =========================================================================
def optuna_objective(trial):
    """
    Objective function minimized by Optuna. It suggests a hyperparameter space 
    based on the configuration profile boundaries and evaluates the model error.
    """
    lc_time, lc_mag, _ = g_data
    
    # Dynamically query Optuna to suggest values within the configuration bounds
    var_params = []
    for var in g_conf_res['var_params_list']:
        val = trial.suggest_float(var['name'], var['min_val'], var['max_val'])
        var_params.append(val)
        
    try:
        # Run Blender simulation using the suggested parameter set
        synth_lc = model(var_params, g_conf_res, delete_tmp=True)
        
        # Calculate residuals against observed lightcurve
        m_diff = model_diff(synth_lc['time'], synth_lc['mst'], lc_time, lc_mag, 
                            conf_res=g_conf_res, norm_mag=True, norm_range=(0, 5))
        
        # Calculate Chi-Square error (Residual Sum of Squares)
        chi2 = np.sum((m_diff / 1.0) ** 2)
        return chi2
        
    except Exception as e:
        # If Blender crashes or encounters an I/O fault, return a high penalty cost
        return 1e10


def run_mcmc_pool(p0, nwalkers, niter, ndim, lnprob, ncpus=cpu_count()):
    """
    Executes the MCMC ensemble sampler leveraging multi-core process pools.
    """
    backend = None
    save_file = g_conf_res['save_mcmc_file']
    if save_file is not None:
        if os.path.isfile(save_file):
            if g_conf_res['rewrite_h5'] is True:
                os.remove(save_file)
        backend = emcee.backends.HDFBackend(save_file)
        backend.reset(nwalkers, ndim)

    with Pool(processes=ncpus, initializer=init_pool, initargs=(g_data, g_conf_res)) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        pool=pool,
                                        backend=backend,
                                        moves=DIMEMove())

        print("Running burn-in...")
        p0 = sampler.run_mcmc(p0, g_conf_res['niter_burn'], progress=True)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state


if __name__ == "__main__":
    load_dotenv('.env', override=True)
    parser = argparse.ArgumentParser(description='LC simulation combining Optuna global optimization with MCMC.')
    parser.add_argument('-c', '--config', help='Specify config file', required=False)
    parser.add_argument('-l', '--observed_lc', help="Path to observed LC", required=True)
    args = vars(parser.parse_args())

    os.environ.pop("XDG_RUNTIME_DIR", None)

    # Apply deepcopy monkeypatch for Python 3.14 environments if needed
    if sys.version_info >= (3, 14):
        if version.parse(matplotlib.__version__) < version.parse("3.11.0"):
            matplotlib.path.Path.__deepcopy__ = fixed_path_deepcopy

    config_name = args["config"] if args["config"] else "config.ini"
    
    if not args["observed_lc"]:
        print("Enter observed LC [-l or --observed_lc] parameter")
        sys.exit()
    obs_lc_path = args["observed_lc"]

    conf_res = read_config(conf_file=config_name)
    conf_res['st_user'] = os.getenv('ST_USER', default='None')
    conf_res['st_pass'] = os.getenv('ST_PASS', default='None')

    lc_time, lc_mag, lc_mag_err = read_original_lc(obs_lc_path)
    obs_lc_data = [lc_time, lc_mag, lc_mag_err]
    
    # Initialize the main thread global data properties
    init_pool(obs_lc_data, conf_res)

    labels = [var['name'] for var in conf_res['var_params_list']]
    ndim = len(conf_res['var_params_list'])

    # =========================================================================
    # STAGE 1: OPTUNA GLOBAL OPTIMIZATION
    # =========================================================================
    print("\n--- STARTING GLOBAL OPTIMIZATION WITH OPTUNA ---")
    
    # Lowering Optuna's verbosity to prevent standard output pollution
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Initialize optimization problem directed towards error minimization
    study = optuna.create_study(direction="minimize")
    
    ga_start_time = time.time()
    
    # Run 100 evaluation trials distributed natively over specified local CPU threads
    study.optimize(optuna_objective, n_trials=100, n_jobs=conf_res['ncpu'], show_progress_bar=True)
    
    optuna_duration = timedelta(seconds=int(time.time() - ga_start_time))
    print(f"--- Optuna optimization finished in: {optuna_duration} ---")
    print(f"Best found Chi2 value: {study.best_value}")
    
    # Restructure optimization output into an aligned ordered parameter array
    best_optuna_solution = [study.best_params[var['name']] for var in conf_res['var_params_list']]
    print(f"Optimized initial seed parameters: {best_optuna_solution}")

    # =========================================================================
    # STAGE 2: PREPARING ENSEMBLE INITIAL STATES (p0) FROM OPTUNA RESULTS
    # =========================================================================
    nwalkers = conf_res['nwalkers']
    niter = conf_res['niter']

    print(f"\nGenerating {nwalkers} MCMC walkers centered tightly around Optuna's solution...")
    
    # Create a small hyper-dimensional cloud (spread within ~2% of bounding box) around the best solution
    p0 = []
    for i in range(nwalkers):
        walker_pos = []
        for idx, var in enumerate(conf_res['var_params_list']):
            spread = (var['max_val'] - var['min_val']) * 0.02
            random_offset = np.random.uniform(-spread, spread)
            val = best_optuna_solution[idx] + random_offset
            val = np.clip(val, var['min_val'], var['max_val'])
            walker_pos.append(val)
        p0.append(np.array(walker_pos))

    np.savetxt(os.path.join(conf_res['temp_dir_name'], "p0.txt"), p0, fmt='%10.2f', header="    ".join(labels))

    fv_filename = os.path.join(conf_res['temp_dir_name'], "var_params.txt")
    with open(fv_filename, "w") as f:
        f.write("   " + "    ".join(labels) + "    resid\n")

    # =========================================================================
    # STAGE 3: FINE-TUNING AND STATISTICS VIA MCMC
    # =========================================================================
    start_mcmc_time = time.time()
    sampler, pos, prob, state = run_mcmc_pool(p0, nwalkers, niter, ndim, lnprob, ncpus=conf_res['ncpu'])

    burn_in = int(conf_res['niter_burn'])
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probabilities = sampler.get_log_prob(discard=burn_in, flat=True)

    best_index = np.argmax(log_probabilities)
    theta_max_prob = samples[best_index]
    theta_median = np.median(samples, axis=0)

    out_filename = os.path.join(conf_res['temp_dir_name'], "out_res.txt")
    with open(out_filename, "w") as f_out:
        print("\n--- PERFORMANCE AND COMPENSATED PARAMETERS ---")
        print(f"Maximum Likelihood (Max Log_prob): {theta_max_prob}")
        print(f"Median Parameters: {theta_median}")

        f_out.write("--- MCMC Results ---\n")
        f_out.write(f"Maximum Likelihood Parameters (Max Log_prob): {theta_max_prob}\n")
        f_out.write(f"Median Parameters (Cornerplot): {theta_median}\n")

        duration = timedelta(seconds=int(time.time() - start_mcmc_time))
        f_out.write(f"--- MCMC Execution Time: {duration} (Days, HH:MM:SS) ---\n")
        f_out.write(f"--- Full Application Runtime (Optuna + MCMC): {timedelta(seconds=int(time.time() - ga_start_time))} ---\n")

    # Generate residual metrics and visualization profiles for Maximum Likelihood
    print("\nGenerating LC and residuals for: MAXIMUM LIKELIHOOD...")
    best_synth_lc_max = model(theta_max_prob, conf_res, delete_tmp=False, sub_name="max")
    m_diff_max = model_diff(best_synth_lc_max['time'], best_synth_lc_max['mst'], lc_time, lc_mag,
                            norm_mag=True, save_plot=True, plot_title=f"Max_LogProb_{theta_max_prob}",
                            conf_res=conf_res, sub_name="max", norm_range=(0, 1))

    # Generate residual metrics and visualization profiles for Median parameters
    print("\nGenerating LC and residuals for: MEDIAN...")
    best_synth_lc_median = model(theta_median, conf_res, delete_tmp=False, sub_name="median")
    m_diff_median = model_diff(best_synth_lc_median['time'], best_synth_lc_median['mst'], lc_time, lc_mag,
                               norm_mag=True, save_plot=True, plot_title=f"Median_{theta_median}",
                               conf_res=conf_res, sub_name="median", norm_range=(0, 1))

    # Generate posterior parameter correlation metrics via Corner Plot
    fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    fig.tight_layout()
    plt.savefig(os.path.join(conf_res['temp_dir_name'], "corner_plot.svg"))
    print("\nAll pipeline execution stages completed successfully!")