import numpy as np
import argparse
import pathlib
import time
from multiprocessing import Pool, cpu_count
import platform
import os
import sys
from datetime import timedelta

# Вимикаємо локи HDF5
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import emcee
import pygad  # <-- ДОДАНО ПАКЕТ
from dotenv import load_dotenv
from dime_sampler import DIMEMove
import matplotlib.pyplot as plt
import matplotlib.path
import corner
from packaging import version

from blender_support import *

# [Усі ваші допоміжні функції: fixed_path_deepcopy, model, lnlike, lnprior, lnprob, init_pool, run_mcmc_pool залишаються БЕЗ ЗМІН]

"""
На що варто звернути увагу перед першим запуском:
Кількість ядер (ncpu): Оскільки спочатку pygad.GA паралелить процеси через свій менеджер, а потім Pool в MCMC через свій, 
переконайтеся, що об'єм оперативної пам'яті витримає активну роботу Blender на максимальній кількості потоків.

Параметри поколінь GA: Для початку я поставив помірні налаштування (num_generations=30, sol_per_pop=16). Це дозволить 
алгоритму швидко (за 30 кроків еволюції) звузити область пошуку до потрібного мінімуму. Якщо простір параметрів 
занадто дикий — збільшіть num_generations до 50–70.

Хмара розкиду блукачів: Зверніть увагу на цей рядок:
spread = (var['max_val'] - var['min_val']) * 0.02
Це означає, що MCMC почне свій шлях із компактної хмари точок (розкид у всього 2% від ширини вашого конфігу) навколо 
знайденого генетичним алгоритмом рішення. Це збереже сотні ітерацій, які раніше йшли на фазу "burn-in" 
(вигорання марковського ланцюга).
"""


def model(var_params, conf_res, delete_tmp=True, sub_name=""):
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
    for i, param in enumerate(var_params):
        g_par = g_conf_res['var_params_list'][i]
        if g_par['min_val'] > param or param > g_par['max_val']:
            return -np.inf
    return 0.0


def lnprob(var_params):
    lc_time, lc_mag, lc_mag_err = g_data
    lp = lnprior(var_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(var_params, lc_time, lc_mag, lc_mag_err, g_conf_res)


def init_pool(data, conf_res):
    global g_data, g_conf_res
    g_data = data
    g_conf_res = conf_res


# --- НОВА ФУНКЦІЯ ОЦІНКИ ДЛЯ PYGAD ---
def pygad_fitness_func(ga_instance, solution, solution_idx):
    """
    Фітнес-функція еволюційного алгоритму.
    PyGAD максимізує значення, тому повертаємо інверсію від суми квадратів залишків.
    """
    try:
        # Використовуємо глобальні змінні даних, які ми зчитали в __main__
        lc_time, lc_mag, _ = g_data
        
        # 1. Генеруємо синтетичну криву за допомогою вашої моделі
        synth_lc = model(solution, g_conf_res, delete_tmp=True)
        
        # 2. Рахуємо нев'язку (різницю), як у вашій lnlike
        m_diff = model_diff(synth_lc['time'], synth_lc['mst'], lc_time, lc_mag, 
                            conf_res=g_conf_res, norm_mag=True, norm_range=(0, 5))
        
        # 3. Рахуємо суму квадратів помилок
        chi2 = np.sum((m_diff / 1.0) ** 2)
        
        # Максимізуємо значення. Чим менше chi2, тим вищий fitness.
        fitness = 1.0 / (chi2 + 1e-6)
        return fitness
    except Exception as e:
        # Якщо Blender видав помилку або файл пошкодився — повертаємо нуль, еволюція відкине цю особину
        return 0.0


def run_mcmc_pool(p0, nwalkers, niter, ndim, lnprob, ncpus=cpu_count()):
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
    parser = argparse.ArgumentParser(description='LC simulation with MCMC and GA optimization')
    parser.add_argument('-c', '--config', help='Specify config file', required=False)
    parser.add_argument('-l', '--observed_lc', help="Path to observed LC", required=True)
    args = vars(parser.parse_args())

    os.environ.pop("XDG_RUNTIME_DIR", None)

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
    
    # Ініціалізуємо глобальні змінні для роботи функцій всередині пулів процесу
    init_pool(obs_lc_data, conf_res)

    labels = [var['name'] for var in conf_res['var_params_list']]
    ndim = len(conf_res['var_params_list'])

    # =========================================================================
    # ЕТАП 1: ГЕНЕТИЧНИЙ АЛГОРИТМ (PYGAD) ДЛЯ ПОШУКУ ГЛОБАЛЬНОГО ОПТИМУМУ
    # =========================================================================
    print("\n--- ЗАПУСК ГЕНЕТИЧНОГО АЛГОРИТМУ (PyGAD) ДЛЯ ПОШУКУ ОПТИМАЛЬНОГО СТАРТУ ---")
    
    # Автоматично формуємо межі параметрів (gene_space) на основі вашого config.ini
    gene_space = []
    for var in conf_res['var_params_list']:
        gene_space.append({'low': var['min_val'], 'high': var['max_val']})

    ga_start_time = time.time()
    
    ga_instance = pygad.GA(
        num_generations=30,          # Кількість еволюційних епох (поколінь)
        num_parents_mating=6,        # Кількість батьків для схрещування
        fitness_func=pygad_fitness_func,
        sol_per_pop=16,              # Розмір популяції (скільки Blender-сценаріїв рахуємо за раз)
        num_genes=ndim,
        gene_space=gene_space,
        parent_selection_type="sss", # Steady state selection
        keep_parents=2,              # Елітизм (зберігаємо найкращі рішення)
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20,   # Шанс мутації параметрів
        # Використовуємо мультипроцесинг самого PyGAD для паралельних викликів Blender
        parallel_processing=["process", conf_res['ncpu']]
    )

    ga_instance.run()
    
    # Отримуємо найкращий набір параметрів з точки зору еволюції
    best_ga_solution, best_ga_fitness, _ = ga_instance.best_solution()
    ga_duration = timedelta(seconds=int(time.time() - ga_start_time))
    print(f"--- GA завершено за: {ga_duration} ---")
    print(f"Найкращі знайдені GA параметри: {best_ga_solution}")

    # =========================================================================
    # ЕТАП 2: ПІДГОТОВКА СТАРТОВИХ ТОЧОК (p0) ДЛЯ MCMC НА ОСНОВІ РЕЗУЛЬТАТУ GA
    # =========================================================================
    nwalkers = conf_res['nwalkers']
    niter = conf_res['niter']

    print(f"\nГенерація {nwalkers} блукачів навколо оптимальної точки GA...")
    
    # Замість повністю випадкового розкиду по всьому простору, ми створюємо локальну хмару (плями)
    # навколо найкращого рішення еволюційного алгоритму:
    p0 = []
    for i in range(nwalkers):
        walker_pos = []
        for idx, var in enumerate(conf_res['var_params_list']):
            # Робимо невеликий випадковий розкид (в межах 2-5% від кроку або діапазону параметрів) навколо центру
            spread = (var['max_val'] - var['min_val']) * 0.02
            random_offset = np.random.uniform(-spread, spread)
            val = best_ga_solution[idx] + random_offset
            # Обмежуємо, щоб значення випадково не вискочили за ліміти конфігу
            val = np.clip(val, var['min_val'], var['max_val'])
            walker_pos.append(val)
        p0.append(np.array(walker_pos))

    np.savetxt(os.path.join(conf_res['temp_dir_name'], "p0.txt"), p0, fmt='%10.2f', header="    ".join(labels))

    fv_filename = os.path.join(conf_res['temp_dir_name'], "var_params.txt")
    with open(fv_filename, "w") as f:
        f.write("   " + "    ".join(labels) + "    resid\n")

    # =========================================================================
    # ЕТАП 3: КЛАСИЧНИЙ ЗАПУСК MCMC ДЛЯ КІНЦЕВОЇ СТАТИСТИКИ
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
        print("\n--- ОБЧИСЛЕНІ ПАРАМЕТРИ ---")
        print(f"Maximum Likelihood (Log_prob max): {theta_max_prob}")
        print(f"Median (Cornerplot center): {theta_median}")

        f_out.write("--- MCMC Results ---\n")
        f_out.write(f"Maximum Likelihood Parameters (Max Log_prob): {theta_max_prob}\n")
        f_out.write(f"Median Parameters (Cornerplot): {theta_median}\n")

        duration = timedelta(seconds=int(time.time() - start_mcmc_time))
        f_out.write(f"--- MCMC Exec Time: {duration} (Days, HH:MM:SS) ---\n")
        f_out.write(f"--- Full Engine Time (GA + MCMC): {timedelta(seconds=int(time.time() - ga_start_time))} ---\n")

    # Генерація графіків залишків для Максимуму та Медіани
    print("\nGenerating LC and residuals for: MAXIMUM LIKELIHOOD...")
    best_synth_lc_max = model(theta_max_prob, conf_res, delete_tmp=False, sub_name="max")
    m_diff_max = model_diff(best_synth_lc_max['time'], best_synth_lc_max['mst'], lc_time, lc_mag,
                            norm_mag=True, save_plot=True, plot_title=f"Max_LogProb_{theta_max_prob}",
                            conf_res=conf_res, sub_name="max", norm_range=(0, 1))

    print("\nGenerating LC and residuals for: MEDIAN...")
    best_synth_lc_median = model(theta_median, conf_res, delete_tmp=False, sub_name="median")
    m_diff_median = model_diff(best_synth_lc_median['time'], best_synth_lc_median['mst'], lc_time, lc_mag,
                               norm_mag=True, save_plot=True, plot_title=f"Median_{theta_median}",
                               conf_res=conf_res, sub_name="median", norm_range=(0, 1))

    # Побудова Corner Plot
    fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    fig.tight_layout()
    plt.savefig(os.path.join(conf_res['temp_dir_name'], "corner_plot.svg"))
    print("\nУсі етапи успішно завершено! Результати збережено.")