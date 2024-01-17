import numpy as np
import argparse
import corner
import pathlib
import time
from multiprocessing import Pool, cpu_count

import emcee

from blender_support import *


# x = time
# y = mag

def model(var_params, conf_res):
    """
    Generate synthetic LC with Blender and python script defined in config file
    Args:
        conf_res: dict of config parameters
        var_params: Spin period, phase of period, period of precession, phase of precession and phase angle
    Return:
        Synthetic LC, dict {'time': time, 'mst': mag, 'mr': mr, 'mz': mz, 'el': el, 'range': dist}
    """
    p, p_phase, p_pr, p_pr_phase, pr_angle = var_params  # theta
    # print("PHASE=", p_phase)

    temp_dir_name = conf_res["temp_dir_name"]  # "/home/vkudak/tmp"
    temp_dir_path = pathlib.Path(temp_dir_name)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    tmp_script_path = os.path.join(temp_dir_name, "temp_blender_script_topex.py")
    # video_file = os.path.join(conf_res["temp_dir_name"], "rendered_file.mp4")
    rnd_gen = "_" + gen_random_str()

    video_file = make_blender_script(tmp_script_path=tmp_script_path + rnd_gen,
                                          conf_res=conf_res,
                                          sat_spin=p,
                                          p_phase=p_phase,
                                          p_pr=p_pr,
                                          p_pr_phase=p_pr_phase,
                                          pr_angle=pr_angle
                                          )

    if video_file is False:
        sys.exit()

    # print("video_file=", video_file)
    # print("isfile", os.path.isfile(video_file))
    # video_file = os.path.normpath(video_file)
    # print("normed video_file=", video_file)
    # # print("video_file=", video_file)
    # print("isfile", os.path.isfile(video_file))


    # generate new video file
    res_code = blender_render(blender_path=conf_res["blender_path"], tmp_script_path=tmp_script_path + rnd_gen, log_dir_path=temp_dir_path)
    # print('res_code', res_code)
    if res_code != 0:
        sys.exit()

    # print("video_file=", video_file)
    # print("isfile", os.path.isfile(video_file))

    # if os.path.isfile(video_file):
    #     flux_res = process_video(video_file)
    #     os.remove(video_file)
    # else:
    #     sys.exit()

    flux_res = process_video(video_file)
    os.remove(video_file)
    os.remove(tmp_script_path + rnd_gen)

    # process flux and get LC
    synth_lc = make_lc(N=flux_res['count'], flux=flux_res['flux'],
                        s_date=conf_res['lc_start_date'], s_time=conf_res['lc_start_time'],
                        norad=conf_res['sat_norad'], fps=conf_res["fps"],
                        st_user=conf_res['st_user'], st_pass=conf_res['st_pass']
                        )

    return synth_lc


def lnlike(var_params, lc_time, lc_mag, lc_mag_err, conf_res):
    synth_lc = model(var_params, conf_res)

    m_diff = model_diff(synth_lc['time'], synth_lc['mst'], lc_time, lc_mag, norm_mag=False)

    # return -0.5 * np.sum(((y - y_model) / yerr) ** 2)

    # ll = conf_res['lc_duration'] * conf_res['fps']
    # return -0.5 * np.sum((m_diff / lc_mag_err[:ll]) ** 2)
    return -0.5 * np.sum((m_diff / 1.0) ** 2)


def lnprior(var_params):
    # Find in OCFit !!!!
    p, p_phase, p_pr, p_pr_phase, pr_angle = var_params
    if (5.0 < p < 15.0) and (0.0 <= p_phase < 359) and (0.0 <= p_pr < 500.0) and (0. <= p_pr_phase < 359) and (0. <= pr_angle < 60):
        return 0.0
    return -np.inf


def lnprob(var_params, lc_time, lc_mag, lc_mag_err, conf_res):
    lp = lnprior(var_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(var_params, lc_time, lc_mag, lc_mag_err, conf_res)

# data = (age, T,Terr)   #time, mag, merr
# nwalkers = 128
# niter = 500
# initial = np.array([1.0, 1.0, 1.0, 26000., 41000.,100000.,-4.5])
# ndim = len(initial)
# p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in xrange(nwalkers)]   steps !!!!


def run_mcmc(p0, nwalkers, niter, ndim, lnprob, obs_lc_data):
    # params= (conf_res, var_params, lc_time, lc_mag, lc_mag_err)

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    save_file = conf_res['save_mcmc_file']
    backend = None
    if save_file is not None:
        if os.path.isfile(save_file):
            os.remove(save_file)
        filename = save_file
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=obs_lc_data, backend=backend)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 10, progress=True)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state


def run_mcmc_pool(p0, nwalkers, niter, ndim, lnprob, obs_lc_data, ncpus=cpu_count()):
    # params= (conf_res, var_params, lc_time, lc_mag, lc_mag_err)

    with Pool(processes=ncpus) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=obs_lc_data, pool=pool)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LC simulation with MCMC method and Blender software')
    parser.add_argument('-c', '--config', help='Specify config file', required=False)
    parser.add_argument('-l', '--observed_lc', help="Path to observed LC", required=True)
    args = vars(parser.parse_args())

    if args["config"]:
        config_name = args["config"]
    else:
        print("Search for configuration in default filename - config.ini")
        config_name = "config.ini"

    if args["observed_lc"]:
        obs_lc_path = args["observed_lc"]

    conf_res = read_config(conf_file=config_name)

    # print(conf_res['p_spin_lim'][0])
    # sys.exit()

    lc_time, lc_mag, lc_mag_err = read_original_lc(obs_lc_path)

    var_params = (conf_res['p_spin'], conf_res['p_phase'], conf_res['pr_period'], conf_res['pr_phase'], conf_res['pr_angle'])

    params = (conf_res, var_params, lc_time, lc_mag, lc_mag_err)

    # From example..... https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
    nwalkers = conf_res['nwalkers']  # 20  # 128
    niter = conf_res['niter']  # 100  # 500
    # initial = np.array([5.0, 1.0, 1.0, 26000., 41000.,100000.,-4.5])
    initial = np.array(var_params)
    ndim = len(initial)

    # p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]   #steps !!!!

    # p0 = [randrange_float(p_min, p_max, p_step),
    #         randrange_float(p_phase_min, p_phase_max, p_phase_step),
    #         ......
    #         ] * nwalker times

    p0 = [ 
                        np.array([
                        randrange_float(conf_res['p_spin_lim'][0], conf_res['p_spin_lim'][1], conf_res['p_spin_lim'][2]),
                        randrange_float(conf_res['p_phase_lim'][0], conf_res['p_phase_lim'][1], conf_res['p_phase_lim'][2]),
                        randrange_float(conf_res['pr_period_lim'][0], conf_res['pr_period_lim'][1], conf_res['pr_period_lim'][2]),
                        randrange_float(conf_res['pr_phase_lim'][0], conf_res['pr_phase_lim'][1], conf_res['pr_phase_lim'][2]),
                        randrange_float(conf_res['pr_angle_lim'][0], conf_res['pr_angle_lim'][1], conf_res['pr_angle_lim'][2])
                        ])
        for i in range(nwalkers)
    ]

    # print(np.shape(p0))
    # print(np.shape(p2))
    # sys.exit()

    start_time = time.time()

    # sampler, pos, prob, state = run_mcmc_pool(p0, nwalkers, niter, ndim, lnprob, obs_lc_data=(lc_time, lc_mag, lc_mag_err, conf_res), ncpus=2)
    sampler, pos, prob, state = run_mcmc(p0, nwalkers, niter, ndim, lnprob, obs_lc_data=(lc_time, lc_mag, lc_mag_err, conf_res))
    samples = sampler.flatchain
    print("Fitted parameters:")
    print(samples[np.argmax(sampler.flatlnprobability)])

    print(f"--- {(time.time() - start_time) / 60.0} minutes ---")

    # Plot best result
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    best_synth_lc = model(theta_max, conf_res)

    m_diff = model_diff(best_synth_lc['time'], best_synth_lc['mst'], lc_time, lc_mag, norm_mag=False, save_plot=True, plot_title=theta_max)

    # plt.plot(age,T,label='Change in T')
    # plt.plot(age,best_fit_model,label='Highest Likelihood Model')
    # plt.show()
    # print 'Theta max: ',theta_max

    # Posterior Spread or Cornerplot
    labels = ['P', 'p_phase', 'P_pr', 'pr_phase', 'pr_angle']
    fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    fig.savefig(os.path.join(conf_res['temp_dir_path'], "cornr_plot.svg"))
