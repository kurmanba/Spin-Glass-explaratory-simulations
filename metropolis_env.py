from data_reduction_indexing import relation_matrix
from tqdm import tqdm
import sqlite3
from itertools import product
from plot_utils import *
# import scipy.io
import os
# import cProfile
# import pstats


class Metropolis:
    def __init__(self,
                 lattice_size: int,
                 metropolis_steps: int,
                 temperature: float,
                 seed: int,
                 external_magnetic_field: float,
                 instructions: tuple,
                 database_address: str,
                 storage_address: str):

        self.metropolis_steps = metropolis_steps
        self.instructions = instructions
        self.emf = external_magnetic_field
        self.temperature = temperature
        self.lattice_size = lattice_size
        self.seed = seed

        self.beta = 1 / self.temperature
        self.accepted = 0
        self.energy = self.initial_energy()

        # Book keeping
        self.magnetism = np.sum(self.instructions[10])
        self.initial_spin = self.instructions[10].copy()
        self.spin_correlation = 1

        self.count_flips = np.zeros((self.lattice_size ** 2))                  # spin mobility tracking (acceptance)
        self.total_selection = np.zeros((self.lattice_size ** 2))              # spin mobility tracking (tot. selection)

        self.track_energy = np.zeros(self.metropolis_steps)
        self.track_magnetism = np.zeros(self.metropolis_steps)                 # ensemble average
        self.track_spin_correlation = np.zeros(self.metropolis_steps)          # ensemble average

        self.track_energy[0] = self.energy
        self.track_magnetism[0] = self.magnetism
        self.track_spin_correlation[0] = 1
        self.metropolis_counter = 0

        # Push to database and save .npy files
        self.database_address = database_address
        self.storage_address = storage_address

    def initial_energy(self) -> float:
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

        m_north, m_south, m_east, m_west = self.instructions[0], self.instructions[1],\
            self.instructions[2], self.instructions[3]

        j_north, j_south, j_east, j_west = self.instructions[4], self.instructions[5],\
            self.instructions[6], self.instructions[7]

        sigma = self.instructions[10]

        hamiltonian_north = j_north * ((m_north @ sigma) * sigma)
        hamiltonian_south = j_south * ((m_south @ sigma) * sigma)
        hamiltonian_east = j_east * ((m_east @ sigma) * sigma)
        hamiltonian_west = j_west * ((m_west @ sigma) * sigma)

        hamiltonian = -(hamiltonian_north + hamiltonian_south + hamiltonian_east + hamiltonian_west)
        hamiltonian -= np.sum(sigma) * self.emf

        return hamiltonian

    @staticmethod
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
        bool: condition of acceptance.
        """
        return de <= 0 or np.exp(-beta * de) > np.random.random()

    def energy_change(self,
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
        dh: change in hamiltonian
        """

        j = [(self.instructions[9][metropolis_selection])[0][0],
             (self.instructions[9][metropolis_selection])[1][0],
             (self.instructions[9][metropolis_selection])[2][0],
             (self.instructions[9][metropolis_selection])[3][0]]

        indexes = self.instructions[8][metropolis_selection]

        sigma_xy = np.array([self.instructions[10][int(indexes[0]) - 1], self.instructions[10][int(indexes[1]) - 1],
                             self.instructions[10][int(indexes[2]) - 1], self.instructions[10][int(indexes[3]) - 1]])

        dh = 2 * self.instructions[10][metropolis_selection - 1] * np.dot(j, sigma_xy)
        dh += 2 * self.instructions[10][metropolis_selection - 1] * self.emf

        return dh

    def step_metropolis(self):
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

        metropolis_selection = np.random.randint(1, self.lattice_size ** 2 + 1)
        de = self.energy_change(metropolis_selection)
        self.total_selection[metropolis_selection - 1] += 1                                        # track selection

        if self.rejection_condition(de, self.beta):

            self.count_flips[metropolis_selection - 1] += 1                                        # track mobility
            self.spin_correlation += -2 * self.instructions[10][metropolis_selection - 1] * \
                self.initial_spin[metropolis_selection - 1] / self.lattice_size ** 2               # track correlation

            self.energy += de                                                                      # increment energy
            self.magnetism += - 2 * self.instructions[10][metropolis_selection - 1]                # increment magnetism
            self.instructions[10][metropolis_selection - 1] *= -1                                  # flip spin

        self.track_spin_correlation[self.metropolis_counter] = self.spin_correlation               # stats correlation
        self.track_energy[self.metropolis_counter] = self.energy                                   # stats energy
        self.track_magnetism[self.metropolis_counter] = self.magnetism                             # stats magnetism
        self.accepted += 1                                                                         # acceptance
        self.metropolis_counter += 1

        return None

    def extract_plots(self) -> None:
        """
        Return:
        ______
        plots:
        """

        plot_upper_bound(self.instructions[9], self.instructions[10])
        plot_probability(self.instructions[9], self.instructions[10])

        plot_ecm(self.track_energy / self.lattice_size ** 2, self.track_magnetism / self.lattice_size ** 2,
                 self.track_spin_correlation, self.lattice_size, self.energy, self.temperature)

        plot_mobility(self.count_flips, self.lattice_size, self.energy)

        return None

    def push_results(self) -> None:
        """
        Push results to database.
        """
        push_dict = defaultdict()

        push_dict["lattice_size"] = self.lattice_size
        push_dict["iterations"] = self.metropolis_steps
        push_dict["temperature"] = self.temperature
        push_dict["emf"] = self.emf
        push_dict["seed"] = self.seed
        push_dict["results_final_energy"] = str(self.energy)

        mag_file = "{}_{}_{}_{}_{}_magnetism.npy".format(self.lattice_size,
                                                         self.metropolis_steps,
                                                         self.temperature,
                                                         self.emf,
                                                         self.seed)
        np.save(os.path.join(self.storage_address, mag_file), self.track_magnetism)
        push_dict["result_magnetism_array"] = mag_file

        energy_file = "{}_{}_{}_{}_{}_energy.npy".format(self.lattice_size,
                                                         self.metropolis_steps,
                                                         self.temperature,
                                                         self.emf,
                                                         self.seed)
        np.save(os.path.join(self.storage_address, energy_file), self.track_energy)
        push_dict["result_energy_array"] = energy_file

        correlation_file = "{}_{}_{}_{}_{}_correlation.npy".format(self.lattice_size,
                                                                   self.metropolis_steps,
                                                                   self.temperature,
                                                                   self.emf,
                                                                   self.seed)
        np.save(os.path.join(self.storage_address, correlation_file), self.track_spin_correlation)
        push_dict["result_correlation_array"] = correlation_file

        spin_file = "{}_{}_{}_{}_{}_spin.npy".format(self.lattice_size,
                                                     self.metropolis_steps,
                                                     self.temperature,
                                                     self.emf,
                                                     self.seed)
        np.save(os.path.join(self.storage_address, spin_file), self.instructions[10])
        push_dict["result_final_spin_array"] = spin_file

        mobility_file = "{}_{}_{}_{}_{}_mobility.npy".format(self.lattice_size,
                                                             self.metropolis_steps,
                                                             self.temperature,
                                                             self.emf,
                                                             self.seed)
        np.save(os.path.join(self.storage_address, mobility_file), self.count_flips)
        push_dict["results_mobility_array"] = mobility_file

        db_conn = sqlite3.connect(self.database_address)
        db = db_conn.cursor()
        columns = ', '.join(push_dict.keys())
        placeholders = ':' + ', :'.join(push_dict.keys())
        query = 'INSERT INTO simulations (%s) VALUES (%s)' % (columns, placeholders)
        db.execute(query, push_dict)
        db_conn.commit()
        db_conn.close()

        return None


if __name__ == '__main__':

    lattice_sizes_sweep = [300]
    seeds_sweep = [int(i) for i in np.linspace(117, 119, 3, dtype=int)]
    metropolis_steps_sweep = [int(100*i**2) for i in lattice_sizes_sweep]
    external_magnetic_fields = [float(i) for i in np.linspace(-5, 5, 11)]
    temperatures = [float(i) for i in np.linspace(0.1, 5, 15)]
    data_storage = "/Users/alisher/PycharmProjects/SpinGlass/data_storage"

    # Iterate over given data
    iterator = product(seeds_sweep,
                       lattice_sizes_sweep,
                       metropolis_steps_sweep,
                       external_magnetic_fields,
                       temperatures)

    k_1_old, k_0_old = 0, -1
    total_iterations = int(len(list(iterator)))

    iterator = product(seeds_sweep,
                       lattice_sizes_sweep,
                       metropolis_steps_sweep,
                       external_magnetic_fields,
                       temperatures)

    with tqdm(total=total_iterations) as progress_bar:
        for k in iterator:
            if k[1] != k_1_old or k[0] != k_0_old:
                generate_instructions = relation_matrix(k[1], k[0])

            metro = Metropolis(lattice_size=k[1],
                               metropolis_steps=k[2],
                               temperature=k[4],
                               seed=k[0],
                               external_magnetic_field=k[3],
                               instructions=generate_instructions,
                               database_address="metropolis.db",
                               storage_address=data_storage)

            k_1_old, k_0_old = k[1], k[0]

            np.random.seed(int(k[0]))
            for i in range(k[2]):
                metro.step_metropolis()
            metro.push_results()
            progress_bar.update(1)

    # PROFILING performance
    # for temperature in temperature_range:
    #     with cProfile.Profile() as pr:
    #         metro = Metropolis(metropolis_steps,
    #                            temperature,
    #                            external_magnetic_field,
    #                            lattice_size,
    #                            pre_stored_data)
    #
    #         for i in tqdm(range(metropolis_steps)):
    #             metro.step_metropolis()
    #         metro.extract_plots()
    #
    #     stats = pstats.Stats(pr)
    #     stats.sort_stats(pstats.SortKey.TIME)
    #     # stats.print_stats()
    #     stats.dump_stats(filename='check_metro.prof')
