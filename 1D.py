from itertools import permutations
import numpy as np
import datetime

# SYMMETRY in RING!!! Search for desired states!!!
# Following piece of code is continuation of the work performed in
# simulations of the series parallel system. Exersice in LINEAR Programming.

print('1. Started computation at ' + str(datetime.datetime.now()))

# INPUT:
# SET SIZE: [x_1,x_2,........x_n]
# OUTPUT:
# Maximum unifrom distance between grid elements in circular ring so that:
# For all elements pairwise distance is more than some value which needs to
# be figured out.

n = 9
cheb_limit = int(n / 2)
grid = np.arange(1, n + 1, 1)
a = list(permutations(grid))

print('2. Permutations are stored ' + str(datetime.datetime.now()))

neighbor_size = np.zeros((1, cheb_limit))
helper_struct = np.transpose(np.hstack((grid, grid, grid)))
for i in range(1, cheb_limit):  # preapre indexes for indexation dictionary
    count1 = 2
    count2 = 2
    neighbor_size[0][i - 1] = count1
neighbor_size[0][-1] = count1


def get_neighbors_chebychev1D(helper_struct, i, m):  # Neighbor extraction in circular ring
    helper_array = np.zeros(2)
    helper_array[0] = helper_struct[i + m + (n - 1)]
    helper_array[1] = helper_struct[i - m + (n - 1)]
    if m != cheb_limit:
        return (helper_array)
    elif m == cheb_limit:
        return np.unique(helper_array)


neighbor_struct = {i: np.zeros((n, int(neighbor_size[0][i - 1]))) for i in range(1, cheb_limit + 1)}

for m in range(1, cheb_limit + 1):
    z = 0
    neighbor_temp = np.zeros((n, 2))
    for x in range(1, n + 1):
        neighbor_temp[z] = get_neighbors_chebychev1D(helper_struct, x, m)
        neighbor_struct[m] = neighbor_temp
        z += 1

    # print(get_neighbors_chebychev1D(helper_struct, 2, 4))


# Iterating through the permutations to find the state which satisfies the
# uniformity condition best

def periodic_cheb_distance(state, i, j, grid, neighbor_struct):
    x_1 = grid[i - 1]
    x_2 = grid[j - 1]
    # print(x_1,x_2)
    for dist in range(1, cheb_limit + 1):
        if (neighbor_struct[dist][x_1 - 1][0] == x_2) or (neighbor_struct[dist][x_1 - 1][1] == x_2):
            return dist


def linear_programming_optimizer(neighbor_struct, state, grid):

    saved_diR = np.zeros(n)
    saved_diL = np.zeros(n)

    for i, x_i in enumerate(state):
        for j, x_j in enumerate(state):
            if state[j] == (neighbor_struct[1][x_i - 1][0]) and j != i:
                saved_diR[i] = periodic_cheb_distance(state, i, j, grid, neighbor_struct)
            elif state[j] == (neighbor_struct[1][x_i - 1][1]) and j != i:
                saved_diL[i] = periodic_cheb_distance(state, i, j, grid, neighbor_struct)

    diR = min(set(saved_diR))
    diL = min(set(saved_diL))

    G_min = [diR, diL]
    mean = [saved_diR, saved_diL]
    std = np.std(mean)
    return min(G_min), std, saved_diR, saved_diL


# def visualize_energy_function(): # Visualization of neighbor distribution
# return
# MAIN LOOP
request = linear_programming_optimizer(neighbor_struct, a[0], grid)
min_dist_old = request[0]
mean_dist_old = 5
answer_state = {}
neighbors = {}
total_dist = {}

standard_deviation = np.zeros(len(a))
change = 0
print('3. Starting search ' + str(datetime.datetime.now()))
for counter, state in enumerate(a):
    request = linear_programming_optimizer(neighbor_struct, state, grid)
    min_dist = request[0]
    mean_dist = np.std(request[3]) + np.std(request[2])
    standard_deviation[counter] = request[1]
    neighbors[counter] = request[3] + request[2]
    total_dist[counter] = sum(neighbors[counter])
    # if min_dist >= min_dist_old and mean_dist <= mean_dist_old:
    if mean_dist == 0:
        answer_state[change] = state
        break
        min_dist_old = min_dist
        mean_dist_old = mean_dist
        change += 1
        print(min_dist, mean_dist, counter, change, '(', np.std(request[2]), ',', np.std(request[1]), ')')
answer_min_dist = min_dist_old
answer_mean_dist = mean_dist_old
print('N. All done Look at the graph and ENJOY ' + str(datetime.datetime.now()))
