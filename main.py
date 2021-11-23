"""code was written by Alisher Kurmanbay"""
"""My first code on python after making switch from MATLAB"""
"""Code is remake of cliff walking exercise in resizable grid
in toroid space. """

import numpy as np
import matplotlib.pyplot as plt
import scipy


# from sklearn import svm, datasets
# ______________________________________________________________________________
# 1.Data preparation (Data Structure)

""" In this section user defined parameters are inputted.
    From the inputted data necessary calculations are performed 
    to determine how the data will be managed."""

n = 31  # define size of the matrix
start_grid = 56  # starting grid
finish_grid = 221  # finish grid
# Goal_cells = [221,220,222,199,200,201,243,241,242]
# Goal_cells = [1,2,30,31,931,932,960,961,300,400]
Goal_cells = [449, 450, 451, 480, 481, 482, 511, 512, 513, 1, 31, 931, 961, 16, 466, 496, 946]
# Goal_cells = [481,1,31,931,961,16,466,496,946]
Transition = [1, 3, 6, 9, 10, 11, 14]
# Transition = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# Transition = [2,4,6,8,10,12,14]
# Transition = [1,2,3,13,14,15]
# Transition = [3,5,11,15]                     # allowed travel distance\s (chebychev)
T_prob = [0.05, 0.15, 0.05, 0.25, 0.5, 0.15, 0.05]
# T_prob = [0.1,0,2,0.3,0.4,0.3,0.2,0.1]
# T_prob = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75]
# Transition = [1,3,5,7,9,11,13,15]
# Goal_cells = [25,400,221,140,190,67,73]
# Goal_cells = np.random.randint(n**2, size=4)
cheb_limit = int(n / 2)  # determine maximum possible distance (chebychev)
count1 = 0
count2 = 0
neighbor_size = np.zeros((1, cheb_limit))
policy = np.zeros((1, cheb_limit))

for i in range(1, cheb_limit):  # prepare indexes for indexation dictionary
    count1 = 8 * i
    count2 = count2 + count1
    neighbor_size[0][i - 1] = count1
neighbor_size[0][-1] = n ** 2 - count2 - 1

a = np.linspace(1, n ** 2, n ** 2)
b = np.reshape(a, (n, n))  # original 2D array
t1 = np.vstack((b, b, b))
t2 = np.hstack((t1, t1, t1))

print('Check point 1: Matrices are initialized \n')  # Check point

# ______________________________________________________________________________

# 1.Define gridworld neighborhood and possible state transitions.


def get_neighbors_chebychev(grid, i, j, m):

    counter1 = 0                                                        # initialize internal counter
    helper_array = np.zeros((1, m * 8))
    n_s = m * 2 + 1                                                     # north south helper counter
    e_w = m * 2 - 1                                                     # east west helper counter

    for x in range(0, n_s):                                             # [NW,N...]
        helper_array[0][x] = grid[i - m][j - m + x]
    counter1 += n_s
    for x in range(1, e_w + 1):
        helper_array[0][counter1 - 1 + x] = grid[i - m + x][j + m]
    counter1 += e_w
    for x in range(0, n_s):                                             # [S,SE..]
        helper_array[0][counter1 + x] = grid[i + m][j + m - x]
    counter1 += n_s
    for x in range(1, e_w + 1):                                         # [W,SW,NW..]
        helper_array[0][counter1 - 1 + x] = grid[i + m - x][j - m]
    if m != cheb_limit:
        return helper_array
    elif m == cheb_limit:
        return np.unique(helper_array)


neighbor_struct = {i: np.zeros((n ** 2, int(neighbor_size[0][i - 1]))) for i in range(1, cheb_limit + 1)}

for m in range(1, cheb_limit + 1):
    z = 0
    neighbor_temp = np.zeros((n ** 2, int(neighbor_size[0][m - 1])))
    for x in range(n, 2 * n):
        for y in range(n, 2 * n):
            neighbor_temp[z] = get_neighbors_chebychev(t2, x, y, m)
            neighbor_struct[m] = neighbor_temp
            z += 1
# ______________________________________________________________________________

print('Check point 2: Indexation was succesfull \n')  # Check point


class MDP:  # Markov Decision Process

    """ Current state is inputted to the class and the output is matrix of
    possible actions and rewards under policy pi. The following class also
    updates the value matrix. (Policy iteration)"""

    def __init__(self, state, neighbor_struct, V, teta, b, Goal_cells, Transition, T_prob):

        self.state = state  # Current State S
        self.neighbor_struct = neighbor_struct  # GRIDWORLD ENVIRONMENT
        self.teta = teta
        self.Value = V  # Value Matrix
        self.index_matrix = b  # Indexed Gridworld
        self.actions = []
        self.policy = []
        self.transition_prob = []
        self.reward = []
        self.Goal = Goal_cells
        self.Transition = Transition
        self.T_prob = T_prob

    def get_transition_prob(self):  # Define transition probabilities to actions
        temporary = []
        for i in enumerate(self.Transition):
            pdf = self.T_prob[i[0]]
            if i[0] == 0:
                temporary = neighbor_struct[i[1]][self.state - 1]
                prob_temp = (np.ones(np.size(temporary))) * pdf
            else:
                temporary2 = neighbor_struct[i[1]][self.state - 1]
                prob_temp2 = (np.ones(np.size(temporary2))) * pdf
                temporary = np.concatenate((temporary2, temporary), axis=0)
                prob_temp = np.concatenate((prob_temp2, prob_temp), axis=0)
        self.actions = temporary
        self.transition_prob = np.transpose(prob_temp)

    def get_policy(self):
        if self.state not in self.Goal:
            self.policy = 1 / (np.size(self.actions))
        else:
            self.policy = 0

    def get_rewards(self):  # calculate rewards
        self.reward = np.zeros((1, np.size(self.actions)))
        for i in enumerate(self.actions):
            if i[1] not in self.Goal:
                self.reward[0][int(i[0])] = - 1
            else:
                self.reward[0][int(i[0])] = 100

    def run(self):
        self.get_transition_prob()
        self.get_policy()
        self.get_rewards()
        cell_index = np.argwhere(self.index_matrix == self.state)
        V_temp = 0
        for i in enumerate(self.actions):
            act_index = np.argwhere(self.index_matrix == i[1])
            V_temp += self.policy * self.transition_prob[i[0]] * (
                    self.reward[0][i[0]] + self.teta * self.Value[act_index[0][0]][act_index[0][1]])
        self.Value[cell_index[0][0]][cell_index[0][1]] = V_temp

    def reset(self):
        self.state = 0


# ______________________________________________________________________________
Value = np.reshape(np.zeros((1, n ** 2)), (n, n))
iterations = 15
counter = 0
teta = 0.4
# ______________________________________________________________________________

# Main Loop
while counter < iterations:
    for cell in a:
        cell_index = np.argwhere(b == cell)
        state = int(cell)
        value_iteration = MDP(state, neighbor_struct, Value, teta, b, Goal_cells, Transition, T_prob)
        value_iteration.run()
    counter += 1
    print('Iteration', counter, ' ------> ', round(counter / iterations * 100, 1), ' % completed')

plt.imshow(Value, cmap='hot', interpolation='none')
plt.xticks([])
plt.yticks([])
plt.savefig('bellman1.jpg', bbox_inches='tight', dpi=1000)
print('Check point 3: Plots are now rendering')  # Check point
plt.show()
