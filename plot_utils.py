from collections import defaultdict
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import matplotlib.pyplot as plt
from matplotlib import ticker, get_backend, rc
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['agg.path.chunksize'] = 10000

grey, gold, lightblue, green = '#808080', '#cab18c', '#0096d6', '#008367'
pink, yellow, orange, purple = '#ef7b9d', '#fbd349', '#ffa500', '#a35cff'
darkblue, brown, red = '#004065', '#731d1d', '#E31937'

_int_backends = ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg',
                 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo',
                 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo']

_backend = get_backend()

if _backend in _int_backends:
    fontsize = 12
    fig_scale = 1
else:
    fontsize = 10
    fig_scale = 1

quiver_params = {'angles': 'xy',
                 'scale_units': 'xy',
                 'scale': 1,
                 'width': 0.012}

grid_params = {'linewidth': 0.3,
               'alpha': 0.2}


def set_rc(func):
    def wrapper(*args, **kwargs):
        rc('font', family='serif', size=fontsize)
        rc('axes', axisbelow=True, titlesize=12)
        rc('lines', linewidth=1)
        func(*args, **kwargs)
    return wrapper


@set_rc
def plot_ecm(track_energy,
             track_magnetism,
             track_spin_correlation,
             n,
             energy,
             initial_temperature):

    """
    Plot energy correlation magnetism.
    """

    host = host_subplot(111, axes_class=axisartist.Axes)
    plt.subplots_adjust(right=0.75, left=0.15)

    par1 = host.twinx()
    par2 = host.twinx()

    par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(60, 0))

    par1.axis["right"].toggle(all=True)
    par2.axis["right"].toggle(all=True)

    p1, = host.plot(track_energy, color=red, linewidth=1.5, zorder=3, label="Energy")

    p2, = par1.plot(track_magnetism, color=(65/255, 105/255, 225/255),
                    linewidth=1.5, zorder=1, label="Magnetism")

    p3, = par2.plot(track_spin_correlation, color=orange, linewidth=1.5,
                    zorder=2, label="Q correlation")

    host.spines['left'].set_linewidth(0.5)
    host.spines['bottom'].set_linewidth(0.5)
    host.spines['right'].set_color('none')
    host.spines['top'].set_color('none')

    # host.set_xlim(0, 2)
    # host.set_ylim(0, 2)
    # par1.set_ylim(0, 4)
    # par2.set_ylim(1, 65)

    host.set_xlabel("MC time steps")
    host.set_ylabel("Energy")
    par1.set_ylabel("Magnetism")
    par2.set_ylabel("Q correlation")

    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())

    plt.title(" Spin_size = {} , Temperature = {}".format(n, initial_temperature))
    plt.savefig("Correlation Function: {}.jpeg".format(energy), dpi=1000)
    plt.close()

    return None


def plot_upper_bound(interaction_dict: defaultdict,
                     spins: np.ndarray):

    spin_matrix_size = len(spins)

    visualization_x_positives = []
    visualization_y_positives = []

    visualization_x_negatives = []
    visualization_y_negatives = []

    for element in range(1, spin_matrix_size):

        if spins[element] == 1:
            visualization_x_positives.append(interaction_dict[element][0] - interaction_dict[element][1])
            visualization_y_positives.append(interaction_dict[element][3] - interaction_dict[element][2])

        if spins[element] == -1:
            visualization_x_negatives.append(interaction_dict[element][0] - interaction_dict[element][1])
            visualization_y_negatives.append(interaction_dict[element][3] - interaction_dict[element][2])

    plt.scatter(visualization_x_negatives, visualization_y_negatives,
                marker='o', edgecolor='k', color=(65/255, 105/255, 225/255), linewidth=0.2, s=5, label="-1")
    plt.scatter(visualization_x_positives, visualization_y_positives,
                marker='o', edgecolor='k', color=(86/255, 101/255, 105/255),
                linewidth=0.2, s=5, label="1")
    plt.legend()
    # plt.savefig("Bounds0.jpeg", dpi=1000)
    plt.close()

    return None


def plot_probability(interaction_dict: defaultdict,
                     spins: np.ndarray):

        spin_matrix_size = len(spins)

        visualization_sum_positives = []
        visualization_sum_negatives = []

        for element in range(1, spin_matrix_size):

            if spins[element-1] == 1:
                visualization_sum_positives.append(np.sum(interaction_dict[element]))

            if spins[element-1] == -1:
                visualization_sum_negatives.append(np.sum(interaction_dict[element]))

        print(np.mean(visualization_sum_negatives))
        print(np.mean(visualization_sum_positives))

        return None


def plot_mobility(spin_mobility: np.ndarray,
                  n: int,
                  final_energy: float) -> None:

    cmaps = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
             'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar']

    plt.imshow(np.reshape(spin_mobility, (n, n)), cmap="gnuplot")
    plt.savefig("Mob: spin_size{}_ising_energy{}.jpeg".format(n, final_energy), dpi=1000)
    plt.close()

    sns.kdeplot(data=np.reshape(spin_mobility, -1), bw_adjust=.5)
    plt.savefig("Mobility{}_{}.jpeg".format(n, final_energy), dpi=1000)
    plt.close()
    return None

    # fig, ax1, ax2, ax3 = plt.subplots()
    #
    # ax1.set_xlabel("Time step")
    # ax1.set_ylabel("Correlation Q", color='green')
    # ax1.plot(track_spin_correlation, color='green')
    #
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Energy", color='red')
    # ax2.plot(track_energy, color='red')
    #
    # ax3 = ax1.twinx()
    # ax3.set_ylabel("Magnetism", color='blue')
    # ax3.plot(track_magnetism, color='blue')
    #
    # plt.title("Correlation Function:"
    #           " spin_size = {} , temperature = {}".format(n, initial_temperature))
    # plt.savefig("Correlation Function:"
    #             " spin_size{}_ising_energy{}_temperature{}.jpeg".format(n, energy, initial_temperature), dpi=1000)
    # plt.close()


# temperature = np.arange(1, 100, 1)
# energy = np.arange(1, 100, 1)
#
# prob = list(map(lambda t: np.exp(-t/4), energy))
# prob2 = list(map(lambda t: np.exp(-t/5), energy))
# prob3 = list(map(lambda t: np.exp(-t/6), energy))
# prob4 = list(map(lambda t: np.exp(-t/7), energy))
#
# print(temperature)
# plt.plot(energy, prob, label='4')
# plt.plot(energy, prob2, label='5')
# plt.plot(energy, prob3, label='6')
# plt.plot(energy, prob4, label='7')
# plt.legend()
#
# plt.show()