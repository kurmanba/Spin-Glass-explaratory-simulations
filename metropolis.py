from data_reduction_indexing import *
from tqdm import tqdm
from plot_utils import *
from metropolis_env import Metropolis
from matplotlib import animation
from skvideo.io import FFmpegWriter
from matplotlib import cm

lattice_size = 50                                               # Only even numbers to not break checkerboard formation
# temperature = 20                                              # beta = 1/kT  here k = 1
metropolis_steps = 1_000_000_00                                 # of metropolis iterations


def initial_energy(structured_data: tuple,
                   external_field: float) -> float:

    """
    Function computes energy at the first monte-carlo run.
    To look at data structure generated see: data_reduction_indexing.

    Parameters:
    __________
    structured_data: indexes, neighbors, spins, etc. see data_reduction_indexing
    metropolis_selection: proposal to change spin

    Return:
    ______
    hamiltonian: hamiltonian calculated by data reduction technique and
    checker board principle.
    """

    m_north, m_south, m_east, m_west = structured_data[0], structured_data[1], structured_data[2], structured_data[3]
    j_north, j_south, j_east, j_west = structured_data[4], structured_data[5], structured_data[6], structured_data[7]

    sigma = structured_data[10]

    hamiltonian_north = j_north * ((m_north @ sigma) * sigma)
    hamiltonian_south = j_south * ((m_south @ sigma) * sigma)
    hamiltonian_east = j_east * ((m_east @ sigma) * sigma)
    hamiltonian_west = j_west * ((m_west @ sigma) * sigma)

    hamiltonian = -(hamiltonian_north + hamiltonian_south + hamiltonian_east + hamiltonian_west)

    if external_field != 0:
        hamiltonian -= np.sum(sigma) * external_field

    return hamiltonian


def rejection_condition(de: float,
                        beta: float) -> bool:
    """
    Acceptance of metropolis step is controlled with temperature
    and evaluation of objective function improvement.

    Parameters:
    __________
    de: change in Hamiltonian (Energy)
    beta: kT Boltzmann factor

    Return:
    ______
    bool: boolean condition of acceptance.
    """

    return de <= 0 or np.exp(-beta * de) > np.random.random()


def energy_change(structured_data: tuple,
                  metropolis_selection: int,
                  external_field: float) -> float:
    """
    Energy change from proposed step is calculated using the predefined data
    structure. see data_reduction_indexing

    Parameters:
    __________
    structured_data: indexes, neighbors, spins, etc. see data_reduction_indexing
    metropolis_selection: proposal to change spin

    Return:
    ______
    dh: change in hamiltonian
    """

    relation_dict, full_interaction_dict, spin_vector = structured_data[8], structured_data[9], structured_data[10]

    j = [(full_interaction_dict[metropolis_selection])[0][0],
         (full_interaction_dict[metropolis_selection])[1][0],
         (full_interaction_dict[metropolis_selection])[2][0],
         (full_interaction_dict[metropolis_selection])[3][0]]

    indexes = relation_dict[metropolis_selection]

    sigma_xy = np.array([spin_vector[int(indexes[0]) - 1], spin_vector[int(indexes[1]) - 1],
                         spin_vector[int(indexes[2]) - 1], spin_vector[int(indexes[3]) - 1]])

    dh = 2 * spin_vector[metropolis_selection - 1] * np.dot(j, sigma_xy)
    dh += 2 * spin_vector[metropolis_selection - 1] * external_field

    return dh


def run_metropolis(time_steps: int,
                   indexation: tuple,
                   external_mag: float,
                   initial_temperature,
                   n=lattice_size,
                   include_plot=True):
    """
    Metropolis or single bath monte carlo where transition
    probability is controlled with temperature is used for
    solving the molecular dynamics problem.

    Parameters:
    __________
    time_steps: MC steps
    indexation: indexes, neighbors, spins, etc. see data_reduction_indexing
    metropolis_selection: proposal to change spin
    initial_temperature: temperature
    n: lattice size

    Return:
    ______
    spin: spin for estimated final lowest energy for given
    temperature.
    energy: estimated final energy.
    """

    beta = 1 / initial_temperature
    accepted = 0
    instructions = indexation

    energy = initial_energy(instructions, external_mag)
    magnetism = np.sum(instructions[10])
    initial_spin = instructions[10].copy()
    spin_correlation = 1

    count_flips = np.zeros((n**2))
    total_selection = np.zeros((n**2))

    track_energy = np.zeros(time_steps)
    track_magnetism = np.zeros(time_steps)                                                            # ensemble average
    track_spin_correlation = np.zeros(time_steps)                                                     # ensemble average

    track_energy[0] = energy
    track_magnetism[0] = magnetism
    track_spin_correlation[0] = spin_correlation

    flip_counter = 0

    if include_plot:
        plt.imshow(np.reshape(np.reshape(instructions[10], -1), (n, n)), cmap="gnuplot")
        plt.savefig("Initial: spin_size{}_ising_energy{}.jpeg".format(n, energy), dpi=1000)
        plt.close()

    print("Size = {} lattice was indexed and relations are stored. Initial E = {} eV".format(n, energy))
    print("Starting Metropolis")

    for i in tqdm(range(1, time_steps)):                                                         # Metropolis loop

        metropolis_selection = np.random.randint(1, n**2 + 1)
        de = energy_change(instructions, metropolis_selection, external_mag)
        total_selection[metropolis_selection - 1] += 1

        if rejection_condition(de, beta):
            accepted += 1                                                                        # increment acceptance

            count_flips[metropolis_selection - 1] += 1

            spin_correlation += -2 * instructions[10][metropolis_selection - 1] * \
                initial_spin[metropolis_selection-1]/n**2
            energy += de                                                                         # increment energy
            magnetism += - 2 * instructions[10][metropolis_selection - 1]                        # increment magnetism
            instructions[10][metropolis_selection - 1] *= -1                                     # flip spin

        track_spin_correlation[i] = spin_correlation
        track_energy[i] = energy
        track_magnetism[i] = magnetism

        # if i % 1_000_000 == 0:
        #     sns.kdeplot(data=count_flips, bw_adjust=.5)
        #     plt.savefig("stats_{}.jpeg".format(flip_counter), dpi=200)
        #     plt.close()
        #     flip_counter += 1
        #     plt.imshow(np.reshape(np.reshape(count_flips, -1), (n, n)) - np.sum(count_flips)/n**2, interpolation="sinc", cmap="jet")
        #     # plt.title("Spin flip per 1 mln MC steps", fontsize=19)
        #     plt.xticks([])
        #     plt.yticks([])
        #     # plt.colorbar()
        #     plt.savefig("flip_{}.jpeg".format(flip_counter), dpi=800)
        #     plt.close()
        #     count_flips = np.zeros((n ** 2))

    # clear_output(wait=False)

    if include_plot:
        plot_upper_bound(instructions[9], instructions[10])
        plot_probability(instructions[9], instructions[10])
        plot_ecm(track_energy / n**2, track_magnetism / n**2, track_spin_correlation,
                 n, energy, initial_temperature)
        plot_mobility(count_flips, lattice_size, energy)

    print(accepted)
    print("Final energy E = {} eV".format(energy))

    return instructions[10], energy


pre_stored_data = relation_matrix(lattice_size)
energy_vs_temperature = []
external_magnetic_field = 0
t_min, t_max = 1, 2
temperature_range = [.5]  # np.linspace(0.1, 1, 10)

for temperature in temperature_range:

    spins, final_energy = run_metropolis(metropolis_steps,
                                         pre_stored_data,
                                         external_magnetic_field,
                                         temperature,
                                         lattice_size,
                                         include_plot=True)
    energy_vs_temperature.append(final_energy)


plt.scatter(temperature_range, energy_vs_temperature)
plt.xlabel("Temperature")
plt.ylabel("Energy", color='blue')
plt.title("Energy vs Temperature for {} by {} lattice".format(lattice_size, lattice_size))
plt.savefig("Temperature_vs_energy3.jpeg", dpi=1000)

cmaps = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
         'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar']

plt.imshow(np.reshape(np.reshape(spins, -1), (lattice_size, lattice_size)), cmap="gnuplot")
plt.savefig("Final_spin: spin_size{}_ising_energy{}.jpeg".format(lattice_size, final_energy), dpi=1000)
