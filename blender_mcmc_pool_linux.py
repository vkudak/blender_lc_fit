import numpy as np
import argparse
import pathlib
import time
from multiprocessing import Pool, cpu_count
import platform
import corner

import emcee
from dotenv import load_dotenv
from dime_sampler import DIMEMove

from blender_support import *


def model(var_params, conf_res, delete_tmp=True):
    """
    Generate synthetic LC with Blender and python script defined in config file
    Args:
        conf_res: dict of config parameters
        var_params: Spin period, phase of period, period of precession, phase of precession and phase angle
    Return:
        Synthetic LC, dict {'time': time, 'mst': mag, 'mr': mr, 'mz': mz, 'el': el, 'range': dist}
    """
    temp_dir_name = conf_res["temp_dir_name"]  # "/home/vkudak/tmp"
    temp_dir_path = pathlib.Path(temp_dir_name)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    tmp_script_path = os.path.join(temp_dir_name, "temp_blender_script.py")
    # video_file = os.path.join(conf_res["temp_dir_name"], "rendered_file.mp4")
    rnd_gen = "_" + gen_random_str()

    # var_params = (var['value'] for var in conf_res['var_params_list'])
    # print("problem....")
    video_file = make_blender_script(tmp_script_path=tmp_script_path + rnd_gen,
                                     conf_res=conf_res, var_list=var_params  # conf_res['var_params_list']
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
    res_code = blender_render(blender_path=conf_res["blender_path"],
                              tmp_script_path=tmp_script_path + rnd_gen,
                              log_dir_path=temp_dir_path)
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

    flux_res = process_video(video_file, w=0)
    if delete_tmp:
        os.remove(video_file)
        os.remove(tmp_script_path + rnd_gen)

    # process flux and get LC
    synth_lc = make_lc(N=flux_res['count'], flux=flux_res['flux'],
                       s_date=conf_res['lc_start_date'], s_time=conf_res['lc_start_time'],
                       norad=conf_res['sat_norad'], fps=conf_res["fps"],
                       # st_user=conf_res['st_user'], st_pass=conf_res['st_pass']
                       tle_line1=conf_res['tle_line1'], tle_line2=conf_res['tle_line2']
                       )

    return synth_lc


def lnlike(var_params, lc_time, lc_mag, lc_mag_err, conf_res):
    synth_lc = model(var_params, conf_res)
    # m_diff = model_diff(synth_lc['time'], synth_lc['mst'], lc_time, lc_mag, conf_res=conf_res, norm_mag=False)
    m_diff = model_diff(synth_lc['time'], synth_lc['mst'], lc_time, lc_mag, conf_res=conf_res,
                        norm_mag=True, norm_range=(0, 5))

    ##########################################################
    # Write var_param.txt file with all parameters and Residual
    fv_filename = os.path.join(conf_res['temp_dir_name'], "var_params.txt")
    with open(fv_filename, "a") as fv:
        mlist = np.append(np.array(var_params), -0.5 * np.sum((m_diff / 1.0) ** 2))
        np.savetxt(fv, mlist, fmt='%10.2f', delimiter=" ", newline=" ")
        fv.write("\n")
    ###############################################################

    # return -0.5 * np.sum(((y - y_model) / yerr) ** 2)

    # ll = conf_res['lc_duration'] * conf_res['fps']
    # return -0.5 * np.sum((m_diff / lc_mag_err[:ll]) ** 2)
    return -0.5 * np.sum((m_diff / 1.0) ** 2)


def lnprior(var_params):
    # var_params =  [  40.39547474  229.66830222 1834.72259637  149.58446112   24.92478567]
    for i, param in enumerate(var_params):
        # print(i, ",", param)
        g_par = g_conf_res['var_params_list'][i]
        if g_par['min_val'] > param or param > g_par['max_val']:
            return -np.inf
    return 0.0


def lnprob(var_params):
    # print('lnprob params:')
    # print(var_params)
    # sys.exit()

    lc_time, lc_mag, lc_mag_err = g_data

    lp = lnprior(var_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(var_params, lc_time, lc_mag, lc_mag_err, g_conf_res)


def init_pool(data, conf_res):
    if platform.system() == "Windows":
        global g_data
        g_data = data
        global g_conf_res
        g_conf_res = conf_res
    if platform.system() == "Linux":
        g_data = data
        g_conf_res = conf_res


def run_mcmc_pool(p0, nwalkers, niter, ndim, lnprob, ncpus=cpu_count()):
    # params= (conf_res, var_params, lc_time, lc_mag, lc_mag_err)
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    backend = None
    save_file = conf_res['save_mcmc_file']
    if save_file is not None:
        if os.path.isfile(save_file):
            if conf_res['rewrite_h5'] is True:
                os.remove(save_file)
        backend = emcee.backends.HDFBackend(save_file)
        backend.reset(nwalkers, ndim)

    with (Pool(processes=ncpus, initializer=init_pool, initargs=(obs_lc_data, conf_res)) as pool):

        # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend, moves=DIMEMove())
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        pool=pool,
                                        backend=backend,
                                        moves=[
                                            (emcee.moves.DEMove(), 0.8),
                                            (DIMEMove(), 0.2)
                                        ]
                                        )

        print("Running burn-in...")
        # p0, _, _
        p0 = sampler.run_mcmc(p0, conf_res['niter_burn'], progress=True)  # 100
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state


if __name__ == "__main__":
    load_dotenv('.env')
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
    else:
        print("Enter observed LC [-l or --observed_lc] parameter")
        sys.exit()

    conf_res = read_config(conf_file=config_name)

    conf_res['st_user'] = os.getenv('ST_USER', default='None')
    conf_res['st_pass'] = os.getenv('ST_PASS', default='None')

    lc_time, lc_mag, lc_mag_err = read_original_lc(obs_lc_path)
    obs_lc_data = [lc_time, lc_mag, lc_mag_err]

    var_params = (var['value'] for var in conf_res['var_params_list'])

    # From example..... https://prappleizer.github.io/Tutorials/MCMC/MCMC_Tutorial.html
    nwalkers = conf_res['nwalkers']  # 20  # 128
    niter = conf_res['niter']  # 100  # 500
    # initial = np.array([5.0, 1.0, 1.0, 26000., 41000.,100000.,-4.5])

    initial = np.array(list(var_params))
    ndim = len(initial)

    # Initial parameters for each walker
    p0 = [
        np.array([randrange_float(var['min_val'], var['max_val'], var['step']) for var in conf_res['var_params_list']])
        for i in range(nwalkers)
    ]

    labels = [var['name'] for var in conf_res['var_params_list']]
    
    np.savetxt(os.path.join(conf_res['temp_dir_name'], "p0.txt"), p0, fmt='%10.2f', header="    ".join(labels))

    fv_filename = os.path.join(conf_res['temp_dir_name'], "var_params.txt")
    f = open(fv_filename, "w")
    f.write("   " + "    ".join(labels) + "    resid\n")
    f.close()

    start_time = time.time()

    sampler, pos, prob, state = run_mcmc_pool(p0, nwalkers, niter, ndim, lnprob, ncpus=conf_res['ncpu'])
    samples = sampler.flatchain
    print("Fitted parameters:")
    print(samples[np.argmax(sampler.flatlnprobability)])

    t_hour = (time.time() - start_time) / 3600.0

    # print(f"--- {(time.time() - start_time) / 60.0} minutes ---")
    print(f"---  %2dh %2dm  ---" % (t_hour, (t_hour % 1 * 60)))

    # Plot best result
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    best_synth_lc = model(theta_max, conf_res, delete_tmp=False)

    m_diff = model_diff(best_synth_lc['time'], best_synth_lc['mst'], lc_time, lc_mag, norm_mag=True, save_plot=True,
                        plot_title=theta_max, conf_res=conf_res, norm_range=(0, 1))

    # Posterior Spread or Cornerplot
    # labels = ['P', 'p_phase', 'P_pr', 'pr_phase', 'pr_angle']
    fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    fig.tight_layout()
    plt.savefig(os.path.join(conf_res['temp_dir_name'], "corner_plot.svg"))
