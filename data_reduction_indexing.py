import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, dok_matrix

"""
    This section of code reduces 2D Ising model 
    to a comfortable for computation form (matrices). 
    
    Hamiltonian (energy) of a system is given as:
    
    E = - sum_<x,y> (J * sigma_<x> * sigma_<y>)  (no external field)
    
    Proposed data reduction (decomposition):
    
    E = - (J_north * ([Mask_north * sigma] ° sigma) + ... )  (°Hadamard)
    
    Here: J_north: is N**2 matrix of interactions

          Mask_North: is N**2xN**2 matrix so that when multiplied by
                      sigma_<x> * sigma_<y> is obtained
                      
          J_south, J_west, J_east, M_south... etc are similar matrices.
          
          sigma: N**2x1 matrix containing spins 
                      
    CHECKER BOARD structure is used with some minor
    modifications. Memory requirements might jump a bit
    as some masking matrices needs to be stored, yet 
    sparse nature of matrices  doesn't contribute much to memory demand.
 """


def relation_matrix(n: int,
                    seed: int) -> tuple:
    """
    Function was written to structure the neighborhood.
    Fore each cell it returns neighbors in a dict format.

    Parameters:
    __________
    n: size of the lattice n x n

    Return:
    ______
    mask_north: Mask to get expression sigma_x * sigma_y
    mask_south: Mask to get expression sigma_x * sigma_y
    mask_east: Mask to get expression sigma_x * sigma_y
    mask_west: Mask to get expression sigma_x * sigma_y

    interaction_north: Interaction matrix to get expression J * sigma_x * sigma_y
    interaction_south: Interaction matrix to get expression J * sigma_x * sigma_y
    interaction_east: Interaction matrix to get expression J * sigma_x * sigma_y
    interaction_west: Interaction matrix to get expression J * sigma_x * sigma_y

    relation_dict: Neighborhood relations. (Indexation)
    full_interaction_dict: Interaction among spins.
    spins_vector: +1 , -1 spins
    """

    np.random.seed(int(seed))

    relation_dict = defaultdict()                                           # Relation of each neighbor to each other
    interaction_dict = defaultdict()                                        # Interactions saved with checker board rule
    full_interaction_dict = defaultdict()                                   # Full interaction dic

    spins = np.random.choice([-1, 1], (n, n))
    spins_vector = np.reshape(spins, -1)

    original = np.reshape(np.linspace(1, n ** 2, n ** 2), (n, n))

    periodic = np.hstack((np.vstack((original, original, original)),
                          np.vstack((original, original, original)),
                          np.vstack((original, original, original))))

    checker_board_selection = np.zeros((n, n), dtype=int)
    checker_board_selection[1::2, ::2] = 1
    checker_board_selection[::2, 1::2] = 1

    for x in range(n, 2 * n):
        for y in range(n, 2 * n):

            relation_dict[periodic[x][y]] = [periodic[x - 1][y],                               # North
                                             periodic[x + 1][y],                               # South
                                             periodic[x][y + 1],                               # East
                                             periodic[x][y - 1]]                               # West

            if checker_board_selection[x-n][y-n] == 1:
                mean, var = 0, 1                                                              # integers mean & variance
                interaction_dict[periodic[x][y]] = [np.random.normal(np.random.randint(mean, var), size=1),  # North
                                                    np.random.normal(np.random.randint(mean, var), size=1),  # South
                                                    np.random.normal(np.random.randint(mean, var), size=1),  # East
                                                    np.random.normal(np.random.randint(mean, var), size=1)]  # West

    for x in range(n, 2 * n):                                                         # full interactions for simulation
        for y in range(n, 2 * n):
            if periodic[x][y] not in interaction_dict:

                instructions = relation_dict[periodic[x][y]]
                full_interaction_dict[periodic[x][y]] = [interaction_dict[instructions[0]][1],            # North
                                                         interaction_dict[instructions[1]][0],            # South
                                                         interaction_dict[instructions[2]][3],            # East
                                                         interaction_dict[instructions[3]][2]]            # West
            else:
                full_interaction_dict[periodic[x][y]] = interaction_dict[periodic[x][y]]

    # declare masking matrices
    mask_north, mask_south = dok_matrix((n**2, n**2), dtype=np.float32), dok_matrix((n**2, n**2), dtype=np.float32)
    mask_east, mask_west = dok_matrix((n**2, n**2), dtype=np.float32), dok_matrix((n**2, n**2), dtype=np.float32)

    # declare interactions masks
    interaction_north, interaction_south = np.zeros((n**2)), np.zeros((n**2))
    interaction_east, interaction_west = np.zeros((n**2)), np.zeros((n**2))

    for i in range(0, n**2):                                                                  # construct mask

        if (i+1) in interaction_dict:

            selected_cell_relations = relation_dict[i + 1]

            mask_north[i, int(selected_cell_relations[0]) - 1] = 1
            mask_south[i, int(selected_cell_relations[1]) - 1] = 1
            mask_east[i, int(selected_cell_relations[2]) - 1] = 1
            mask_west[i, int(selected_cell_relations[3]) - 1] = 1

            selected_cell_interaction = interaction_dict[i + 1]

            interaction_north[i] = selected_cell_interaction[0]
            interaction_south[i] = selected_cell_interaction[1]
            interaction_east[i] = selected_cell_interaction[2]
            interaction_west[i] = selected_cell_interaction[3]

    interaction_north = csr_matrix(interaction_north)                                         # convert to sparse matrix
    interaction_south = csr_matrix(interaction_south)
    interaction_east = csr_matrix(interaction_east)
    interaction_west = csr_matrix(interaction_west)

    return (mask_north, mask_south, mask_east, mask_west,                                     # Masks
            interaction_north, interaction_south, interaction_east, interaction_west,         # Interaction matrices
            relation_dict, full_interaction_dict, spins_vector)                               # Simulation helper
