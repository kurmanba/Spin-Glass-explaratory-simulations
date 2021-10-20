from matplotlib import pyplot as plt
from data_reduction_indexing import *
from tqdm import tqdm
from plot_utils import plot_ecm, plot_upper_bound, plot_probability

lattice_size = 1000                                              # Only even numbers to not break checkerboard formation
# temperature = 20                                               # beta = 1/kT  here k = 1
metropolis_steps = 1_000_0000                                     # of metropolis iterations


def initial_energy(structured_data: tuple) -> float:

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
                  metropolis_selection: int) -> float:

    """
    Energy change from proposed step is calculated using the predefined data
    structure. see data_reduction_indexing

    Parameters:
    __________
    structured_data: indexes, neighbors, spins, etc. see data_reduction_indexing
    metropolis_selection: proposal to change spin

    Return:
    ______
    de: change in hamiltonian

    """

    relation_dict, full_interaction_dict, spin_vector = structured_data[8], structured_data[9], structured_data[10]

    j = [np.vstack(full_interaction_dict[metropolis_selection])[0][0],
         np.vstack(full_interaction_dict[metropolis_selection])[1][0],
         np.vstack(full_interaction_dict[metropolis_selection])[2][0],
         np.vstack(full_interaction_dict[metropolis_selection])[3][0]]

    indexes = relation_dict[metropolis_selection]

    sigma_xy = np.array([spin_vector[int(indexes[0]) - 1], spin_vector[int(indexes[1]) - 1],
                         spin_vector[int(indexes[2]) - 1], spin_vector[int(indexes[3]) - 1]])

    return 2 * spin_vector[metropolis_selection - 1] * np.dot(j, sigma_xy)


def run_simulated_annealing(time_steps: int,
                            indexation: tuple,
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

    annealing_temperature = initial_temperature
    beta = 1 / annealing_temperature
    accepted = 0
    instructions = indexation

    energy = initial_energy(instructions)
    magnetism = sum(instructions[10])
    initial_spin = instructions[10].copy()
    spin_correlation = 1

    track_energy = np.zeros(time_steps)
    track_magnetism = np.zeros(time_steps)                                                            # ensemble average
    track_spin_correlation = np.zeros(time_steps)                                                     # ensemble average

    track_energy[0] = energy
    track_magnetism[0] = magnetism / n**2
    track_spin_correlation[0] = spin_correlation

    # plt.imshow(np.reshape(np.reshape(instructions[10], -1), (n, n)), cmap="gnuplot")
    # plt.savefig("Initial: spin_size{}_ising_energy{}.jpeg".format(n, energy), dpi=1000)
    # plt.close()

    print("Size = {} lattice was indexed and relations are stored. Initial E = {} eV".format(n, energy))
    print("Starting Metropolis")

    for i in tqdm(range(1, time_steps)):                                                         # Metropolis loop

        metropolis_selection = np.random.randint(1, n**2)
        de = (energy_change(instructions, metropolis_selection))

        if i % 100 == 0:
            beta *= 1/0.999

        if rejection_condition(de, beta):
            accepted += 1                                                                        # increment acceptance
            spin_correlation += -2*instructions[10][metropolis_selection-1]*initial_spin[metropolis_selection-1]/n**2
            energy += de                                                                         # increment energy
            magnetism += - 2 * instructions[10][metropolis_selection-1]                            # increment magnetism
            instructions[10][metropolis_selection-1] *= -1                                         # flip spin

        track_spin_correlation[i] = spin_correlation
        track_energy[i] = energy
        track_magnetism[i] = magnetism

    if include_plot:
        plot_upper_bound(instructions[9], instructions[10])
        plot_probability(instructions[9], instructions[10])
        plot_ecm(track_energy/n**2, track_magnetism, track_spin_correlation,
                 n, energy, initial_temperature)

    print(annealing_temperature)
    print(accepted)
    print("Final energy E = {} eV".format(energy))

    return instructions[10], energy


pre_stored_data = relation_matrix(lattice_size)
energy_vs_temperature = []
t_min, t_max = 40, 41


for temperature in range(t_min, t_max):
    spins, final_energy = run_simulated_annealing(metropolis_steps, pre_stored_data, temperature, lattice_size)
    energy_vs_temperature.append(final_energy)


plt.scatter(np.arange(t_min, t_max, 1), energy_vs_temperature)
plt.xlabel("Temperature")
plt.ylabel("Energy", color='blue')
plt.title("Energy vs Temperature for {} by {} lattice".format(lattice_size, lattice_size))
plt.savefig("Temperature_vs_energy1.jpeg", dpi=1000)

cmaps = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
         'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar']

# plt.imshow(np.reshape(np.reshape(spins, -1), (lattice_size, lattice_size)), cmap="gnuplot")
# plt.savefig("Final_spin: spin_size{}_ising_energy{}.jpeg".format(lattice_size, final_energy), dpi=1000)
