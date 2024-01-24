"""
Created on 2024-01-24
@author: Viktor Kudak

Script to plot the residuals of observed LC and generated by Blender synthetic LC
"""
import argparse
from blender_support import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot residuals for observed LC and Blender video file')
    parser.add_argument('-c', '--config', help='Specify config file', required=False)
    parser.add_argument('-l', '--observed_lc', help="Path to observed LC", required=True)
    parser.add_argument('-v', '--synth_video_file', help="Path to Blender video", required=True)
    args = vars(parser.parse_args())

    if args["config"]:
        config_name = args["config"]
    else:
        print("Search for configuration in default filename - config.ini")
        config_name = "config.ini"

    obs_lc_path = args["observed_lc"]
    video_file_path = args["synth_video_file"]

    conf_res = read_config(conf_file=config_name)
    conf_res['st_user'] = os.getenv('ST_USER', default='None')
    conf_res['st_pass'] = os.getenv('ST_PASS', default='None')

    lc_time, lc_mag, lc_mag_err = read_original_lc(obs_lc_path)
    obs_lc_data = [lc_time, lc_mag, lc_mag_err]

    res = process_video(video_file_path, w=30)
    synth_lc = make_lc(N=res['count'], flux=res['flux'],
                       s_date=conf_res['lc_start_date'], s_time=conf_res['lc_start_time'],
                       norad=conf_res['sat_norad'], fps=conf_res["fps"],
                       st_user=conf_res['st_user'], st_pass=conf_res['st_pass']
                       )
    # Plot in tmp dir from config file
    model_diff(synth_lc['time'], synth_lc['mst'], lc_time, lc_mag,
               norm_mag=False, save_plot=True, plot_title='Residuals', conf_res=conf_res)
