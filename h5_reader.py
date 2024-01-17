import numpy as np
import matplotlib.pyplot as plt
import emcee
import sys
import corner
# plt.rcParams["font.size"] = 16
# plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.sans-serif"] = ["Computer Modern Sans"]
plt.rcParams["text.usetex"] = True
plt.rcParams["backend"] = "ps" #'TkAgg'

filename = sys.argv[1]

reader = emcee.backends.HDFBackend(filename)

# tau = reader.get_autocorr_time()
# print(tau)
# burnin = int(2 * np.max(tau))
# thin = int(0.5 * np.min(tau))
samples = reader.get_chain(flat=True)#discard=burnin, flat=True, thin=thin)

labels = [r'P', r'$P_{phase}$', r'$P_{pr}$', r'$Pr_{phase}$', r'$Pr_{angle}$']
fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
# fig.savefig(os.path.join(conf_res['temp_dir_name'], "cornr_plot.svg"))
plt.show()
