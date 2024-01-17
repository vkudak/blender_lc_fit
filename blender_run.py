"""
Run Blender rendering in beckgraund to produce synth LC
then search spin parameters with MCMC
"""
import sys
import os
import pathlib

import argparse
from blender_support import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LC simulation with MCMC method and Blender software')
    parser.add_argument('-c', '--config', help='Specify config file', required=False)
    args = vars(parser.parse_args())

    if args["config"]:
        config_name = args["config"]
    else:
        print("Search for configuration in default filename - config.ini")
        config_name = "config.ini"
    conf_res = read_config(conf_file=config_name)

    temp_dir_name = conf_res["temp_dir_name"]  # "/home/vkudak/tmp"
    temp_dir_path = pathlib.Path(temp_dir_name)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    template_path = conf_res["template_path"]  # "/home/vkudak/PycharmProjects/blender_fit/template_blender_script_topex.py"
    tmp_script_path = os.path.join(temp_dir_name, "temp_blender_script_topex.py")
    blender_path = conf_res["blender_path"]  # "/home/vkudak/software/blender-3.6.5-linux-x64/blender"

    # blender_script(template_path=template_path, temp_script_path=temp_script_path,
    #               sat_model_path="/home/vkudak/PycharmProjects/blender_fit/Topex_size_princ_bsdf.blend",
    #               output_dir=temp_dir_name,
    #               resolution=conf_res['frame_res'], #[640, 480],
    #               fps=5,
    #               sat_spin=9.85)
    # # sys.exit()
    # blender_render(blender_path=blender_path, tmp_script_path=tmp_script_path, log_dir_path=temp_dir_path)

    video_file = os.path.join(temp_dir_name, "rendered_file.mp4")

    # make_lc(video_file)
    n, flux = process_video(video_file)

    # process flux and get LC
    res_lc = make_lc(N=n, flux=flux, s_date='', s_time='', norad='', fps=conf_res["fps"],
                     st_user=conf_res['st_user'], st_pass=conf_res['st_pass']
                     )
