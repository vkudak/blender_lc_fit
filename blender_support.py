import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
from datetime import datetime, timedelta
import subprocess
import configparser

from jinja2 import Template
import cv2
# from tqdm import tqdm
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from skyfield.api import load, wgs84, utc

from sklearn.preprocessing import minmax_scale
from scipy.interpolate import BSpline, make_interp_spline
import random
import string
import json


def read_config(conf_file):
    """
    Args:
        conf_file: name of config file
    Return:
        dict of parameters
    """
    config = configparser.ConfigParser(inline_comment_prefixes="#")

    if os.path.isfile(conf_file):
        try:
            config.read(conf_file)

            temp_dir_name = config.get("options", 'temp_dir_name', fallback="tmp")
            template_path = config.get('options', 'template_path')
            blender_path = config.get('options', 'blender_path')

            ncpu = config.getint('mcmc_params', 'ncpu', fallback=1)
            niter_burn = config.getint('mcmc_params', 'niter_burn')
            niter = config.getint('mcmc_params', 'niter')
            nwalkers = config.getint('mcmc_params', 'nwalkers')
            save_mcmc_file = config.get('mcmc_params', 'save_mcmc_file', fallback=None)

            sat_name = config.get('satellite', 'name')
            sat_cospar = config.get('satellite', 'cospar')
            sat_norad = config.getint('satellite', 'norad')
            sat_model_path = config.get('satellite', 'sat_model_path')

            lc_start_time = config.get('blender_lc', 'start_time')
            lc_start_date = config.get('blender_lc', 'start_date')
            lc_duration = config.getint('blender_lc', 'lc_duration')
            fps = config.getint('blender_lc', 'fps')
            fr = config.get('blender_lc', 'frame_size')
            fr_x, fr_y = fr.split('x')
            fr_x = int(fr_x)
            fr_y = int(fr_y)
            tle_line1 = config.get('blender_lc', 'tle_line1')
            tle_line2 = config.get('blender_lc', 'tle_line2')

            var_params = config.get('var_params', 'value')
            var_params = json.loads(var_params)

            return {'temp_dir_name': temp_dir_name,
                    'template_path': template_path,
                    'blender_path': blender_path,

                    'ncpu': ncpu,
                    'nwalkers': nwalkers,
                    'niter': niter,
                    'niter_burn': niter_burn,
                    'save_mcmc_file': save_mcmc_file,

                    'sat_name': sat_name,
                    'sat_cospar': sat_cospar,
                    'sat_norad': sat_norad,
                    'sat_model_path': sat_model_path,

                    'lc_start_time': lc_start_time.strip(),
                    'lc_start_date': lc_start_date.strip(),
                    'lc_duration': lc_duration,
                    'fps': fps,
                    'frame_res': [fr_x, fr_y],
                    'tle_line1': tle_line1,
                    'tle_line2': tle_line2,
                    'var_params_list': var_params

                    }

        except Exception as E:
            print("Error in INI file\n", E)
            sys.exit()
    else:
        print("Error. Cant find config_sat.ini")
        sys.exit()


def get_tle(username, password, epoch=None, norad=None,):
    """
    Get TLE from Space-track
    epoch(datetime): python datetime
    norad(int): NORAD ID

    Return:
        None if fail
        or TLE filename if success
    """

    # st = SpaceTrackClient('labLKD', 'lablkdSpace2013')
    st = SpaceTrackClient(username, password)

    if epoch and norad:
        drange = op.inclusive_range(epoch, epoch + timedelta(days=1))
        # print(drange)
        lines = st.tle_publish(iter_lines=True,
                               publish_epoch=drange,
                               # orderby='TLE_LINE1',
                               format='tle',
                               norad_cat_id=[norad])
        lines = st.tle_publish(iter_lines=True, publish_epoch=drange, format='tle', norad_cat_id=[norad])

        with open(f'tle_{epoch.strftime("%Y_%m_%d")}_{norad}.txt', 'w') as fp:
            for line in lines:
                fp.write(line + "\n")

        return os.path.abspath(f'tle_{epoch.strftime("%Y_%m_%d")}_{norad}.txt')
    else:
        return None


def process_video(video_file_path, w=30):
    """
    Args:
        video_file_path: blender video file
        w: boders on frame to spip (in pixels)
    Return:
        dictionary {'count':count, 'flux':flux}
    """
    if not os.path.isfile(video_file_path):
        print(f"cant find video file {video_file_path}. \nTerminate program.")
        sys.exit()

    cap = cv2.VideoCapture(video_file_path)
    success, image = cap.read()
    count = 0

    # fps = cap.get(cv2.CAP_PROP_FPS)
    # N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(3))  # float `width`
    height = int(cap.get(4))  # float `height`
    # print(width, height)
    # print(f"FPS = {fps}, N_frames = {N_frames}, WxH={width}x{height}")

    # print(width, height)
    res = {'count': [],
            'flux': []}

    # pbar = tqdm(total=N_frames)
    while success:
        count += 1
        gray_img = cv2.cvtColor(image[w:width - w, w:height - w], cv2.COLOR_BGR2GRAY)
        res["count"].append(count)
        res["flux"].append(gray_img.sum())
        # pbar.update(1)
        success, image = cap.read()
    # pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    return res


# def make_blender_script(template_path, tmp_script_path, sat_model_path, output_dir, resolution, fps, sat_spin):
def make_blender_script(tmp_script_path, conf_res, var_list):
    """
    Prepare Blender script from template and passed parameters
    Args:
        tmp_script_path: temporary created script path
        conf_res: all data from config file
        var_list: list of variable parameters [{"name:spin", "value":5, "min_val:3", 'max_val':6}, {}, {}]
    Return:
        True or False. Script saved in entered path dir.
    """
    try:
        with open(conf_res['template_path'], mode="r") as file:
            script_template = Template(file.read())

        # print("point 1")
        with open(tmp_script_path, mode="w") as fp:
            video_file_full_path = os.path.join(conf_res['temp_dir_name'], "rendered_file_" + gen_random_str() + ".mp4")

            rendered_script = script_template.render(
                conf_res=conf_res,
                var_list=var_list,
                rendered_videofile_name=video_file_full_path
            )

            fp.write(rendered_script)
        return video_file_full_path

    except Exception as error:
        print('Error:', error)
        # sys.exit()
        return False


def blender_render(blender_path, tmp_script_path, log_dir_path):
    file = open(os.path.join(log_dir_path, 'blender_output.log'), "w")

    res = subprocess.run([blender_path, "-b", "-P", tmp_script_path, 
                        "--log-file", os.path.join(log_dir_path, 'blender_log.log')], 
                        stdout=file
                        )
    # --log-level 0 for fewer details
    return res.returncode


def make_lc(N, flux, s_date, s_time, norad, fps, st_user, st_pass, location=[+48.63, 22.33, 180], A=16.4, plot=False):
    """
    Create a LC from Counts and Flux data
    Args:
        N: counts
        Flux: Flux
        s_date: start date (str) 2020-11-31
        s_time: start time (str) 20:15:31
        norad: NORAD ID of satellite
        fps: frame per second
        location: [latitude, longitude, height], default is Uzhhorod
        A: zero-point for mag standartization


        plot: False or True
    Returns:
        res: dictionary {'time':datetime, 'mag':mag ....}
    """
    start_date = datetime.strptime(s_date + " " + s_time, '%Y-%m-%d %H:%M:%S')
    date_time = [start_date + timedelta(seconds=x/fps) for x in N]

    if not os.path.isfile(f'tle_{start_date.date().strftime("%Y_%m_%d")}_{norad}.txt'):
        tle_filename = get_tle(st_user, st_pass, epoch=start_date.date(), norad=norad)
    else:
        tle_filename = os.path.abspath(f'tle_{start_date.date().strftime("%Y_%m_%d")}_{norad}.txt')

    if tle_filename:
        satellites = load.tle_file(url=tle_filename)
        # for sat in satellites:
        #     print(sat.model.jdsatepoch)
        # sys.exit()
        sat = satellites[-1]
        # print("SATELLITE", sat)
    else:
        print("Cant load TLE. Cannot continue")
        sys.exit()

    mag = [-2.5 * math.log10(f) for f in flux]
    mag = np.array(mag) + A

    uzh = wgs84.latlon(location[0], location[1], location[2])  # wgs84.latlon(+48.63, 22.33, 180)
    ts = load.timescale()

    azm = []
    el = []
    dist = []
    mr = []
    mz = []
    time = []

    for i in range(0, len(date_time)):
        t = ts.from_datetime(date_time[i].replace(tzinfo=utc))

        difference = sat - uzh
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        # print(t.utc, alt, alt.degrees)

        azm.append(az.degrees)
        el.append(alt.degrees)
        dist.append(distance.km)
        mr.append(-5 * math.log10(distance.km / 1000.))

        mzz = 1. / (math.cos((math.pi / 2.) - math.radians(el[i])))  # 1/cos(90-el)  cos(0)=1 ????  mz in (el=90) = 1
        mz.append(mzz)
        # mr = -5 * math.log10(distance.m / 1000.0)

        # mst = mag[i] + mz[i] + mr[i]
        # dd = date_time[i].strftime("%Y-%m-%d %H:%M:%S.%f")
        # time.append(dd)
        time.append(date_time[i])

    mag = np.array(mag) + np.array(mz) + np.array(mr)

    # print(date_time)

    if plot:
        plt.rcParams['figure.figsize'] = [12, 6]

        plt.ylabel('m_st*')
        plt.xlabel('UT')
        ax = plt.gca()
        ax.xaxis.grid()
        ax.yaxis.grid()

        dm = max(mag) - min(mag)
        dm = dm * 0.1
        plt.axis([min(date_time), max(date_time), max(mag) + dm, min(mag) - dm])
        plt.plot_date(date_time, mag, "xr-", linewidth=0.5, fillstyle="none", markersize=3)

        # plt.gcf().autofmt_xdate()

        date_format = mpl_dates.DateFormatter('%H:%M:%S')
        plt.gca().xaxis.set_major_formatter(date_format)

        plt.tight_layout()
        plt.savefig(str(norad) + s_date + s_time + ".png")
        plt.show()

    return {'time': time, 'mst': mag, 'mr': mr, 'mz': mz, 'el': el, 'range': dist}


def read_original_lc(filename):
    """
    for R filter, phR format
    """
    with open(filename) as f:
        flux, m, m_err = np.genfromtxt(f, skip_header=True, usecols=(6, 8, 9), unpack=True, comments='#')
        
        f.seek(0)
        lcd, lct = np.genfromtxt(f, skip_header=True, unpack=True, usecols=(0, 1,),
                                 dtype=None, encoding="utf-8", comments='#')
        
        lctime = list(zip(lcd, lct))
        lctime = [x[0] + " " + x[1] for x in lctime]
        lctime = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f") for x in lctime]

    return lctime, m, m_err


def dt2timestamp(datetime_list):
    # print(datetime_list[0], type(datetime_list[0]))
    return [x.timestamp() for x in datetime_list]


def timestamp2dt(stamp_list):
    return [datetime.fromtimestamp(x) for x in stamp_list]


def model_diff(synth_time, synth_mag, lc_time, lc_mag, conf_res, norm_mag=True, save_plot=False, plot_title=None):
    """
    Calculate difference between original LC and synthetic
    Interpolate with spline original LC and produce new one where time is same as in synthetic LC
    Args:
        synth_time: datetime of synth LC
        synth_mag: magnitudes of synth LC
        lc_time: datetime of observed LC
        lc_mag: magnitudes of observed LC
        conf_res: config data
        norm_mag: norm magnitudes if True
        save_plot: False, Save plot if True
        plot_title: None, set plot title if needed (model parameters can be printed)
    Return:
        Array of mags (lc_mag - synth_mag)
    """

    if norm_mag:
        lc_mag_norm = minmax_scale(-1 * lc_mag, feature_range=(0, 1))
        synth_mag_norm = minmax_scale(-1 * synth_mag, feature_range=(0, 1))
        lc_data = {'time': lc_time, "mag": lc_mag, "timestamp": dt2timestamp(lc_time), "norm_mag": lc_mag_norm}
        synth_data = {'time': synth_time, "mag": synth_mag, "timestamp": dt2timestamp(synth_time), "norm_mag": synth_mag_norm}
        spl = make_interp_spline(lc_data['timestamp'], lc_data['norm_mag'])  # , bc_type="natural")
        lc_mag_new = spl(synth_data['timestamp']).T
    else:
        lc_data = {'time': lc_time, "mag": lc_mag, "timestamp": dt2timestamp(lc_time)}
        synth_data = {'time': synth_time, "mag": synth_mag, "timestamp": dt2timestamp(synth_time)}
        spl = make_interp_spline(lc_data['timestamp'], lc_data['mag'])  # , bc_type="natural")
        lc_mag_new = spl(synth_data['timestamp']).T

    if save_plot:
        # norm data even if norm_mag=False
        lc_mag_norm = minmax_scale(-1 * lc_mag, feature_range=(0, 1))
        synth_mag_norm = minmax_scale(-1 * synth_mag, feature_range=(0, 1))
        lc_data = {'time': lc_time, "mag": lc_mag, "timestamp": dt2timestamp(lc_time), "norm_mag": lc_mag_norm}
        synth_data = {'time': synth_time, "mag": synth_mag, "timestamp": dt2timestamp(synth_time), "norm_mag": synth_mag_norm}
        spl = make_interp_spline(lc_data['timestamp'], lc_data['norm_mag'])  # , bc_type="natural")
        lc_mag_new = spl(synth_data['timestamp']).T

        plt.rcParams['figure.figsize'] = [12, 8]

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # ax1.plot(lc_data['time'], lc_data['norm_mag'], '-o', markersize=3, label='original LC')
        ax1.plot(synth_data['time'], lc_mag_new, '-o', markersize=3, label='original LC')
        ax1.plot(synth_data['time'], synth_data['norm_mag'], '-o', markersize=3, label='blender LC')
        ax1.legend()
        ax1.tick_params('x', labelbottom=True)

        ax2.plot(synth_data['time'], lc_mag_new - synth_data['norm_mag'], markersize=3, label='residuals', color='black')
        # zero line
        min_t = min(synth_data['time'])
        max_t = max(synth_data['time'])
        x = [min_t, max_t]
        y = [0, 0]
        ax2.plot(x, y, linestyle='dotted', color='red')
        ax2.legend()

        mdif = lc_mag_new - synth_mag
        name = -0.5 * np.sum((mdif / 1.0) ** 2)  # name will be -0.5*(y-y_model)^2
        name = str(name)

        if plot_title is not None:
            fig.suptitle(plot_title, fontsize=14)
        else:
            fig.suptitle('Observed and synthetic LC', fontsize=14)
        fig.tight_layout()
        plt.savefig(os.path.join(conf_res["temp_dir_name"], "resid_" + name.replace(".", "_") + ".png"))

    return lc_mag_new - synth_mag


def randrange_float(start, stop, step):
    """
    usage:
        randrange_float(2.1, 4.2, 0.3) # returns 2.4
    Args:
        start: min value
        stop: max value
        step: step size
    Return:
        float: random in given range of values
    """
    return random.randint(0, int((stop - start) / step)) * step + start


def gen_random_str(n=5):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))
