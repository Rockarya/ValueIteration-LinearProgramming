import cvxpy as cp
import numpy as np
import random
import json
import math

step_cost = 20
x = 64
gamma = 0.999
delta = 1e-3
mm_health = 100
final_reward = 50  # when IJ kills the MM(health of MM becomes 0)
bad_reward = 40  # when MM attacks IJ succesfully it loses -40 reward

time_instance = 200  # max time to converge
positions = 5  # 5 different positions are possible('W'(0), 'N'(1), 'E'(2), 'S'(3), 'C'(4))..in this order
material = 3  # 0,1,2 material
arrows = 4  # 0,1,2,3 arrows can be possible
state = 2  # 2 states are possible 0:dormant 1:ready
health = 101  # health value from 0-100
action = 9  # 9 actions in order:- UP(0),DOWN(1),LEFT(2),RIGHT(3),STAY(4),SHOOT(5),HIT(6),CRAFT(7),GATHER(8), NONE(9) (SHOOT the arrow and hit with the blade)

choose_action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'SHOOT', 'HIT', 'CRAFT', 'GATHER', 'NONE']
choose_position = ['W', 'N', 'E', 'S', 'C']
choose_state = ['D', 'R']

# defining alpha matrix(total 600 possible states)
alpha = np.zeros(shape=(600,1))
# start state is last possible state (C,2,3,R,100)
alpha[599][0] = 1.0

# defining reward array
r = np.zeros(shape=(1,1936))


# making A array
A = np.zeros(shape=(600, 1936))
column_action = np.zeros(shape=1936)
no_of_columns = -1


def give_row(pos, mat, arr, sta, hea):
    row_num = 120 * pos + 5 * 2 * 4 * mat + 5 * 2 * arr + 5 * sta + int(hea / 25)
    return row_num


def add_column():
    global no_of_columns
    no_of_columns += 1


def set_a(pos, mat, arr, sta, hea):
    global no_of_columns
    global r
    
    
    if hea == 0:
        add_column()
        column_action[no_of_columns] = 9
        A[give_row(pos, mat, arr, sta, hea), no_of_columns] = 1.0
        r[0][no_of_columns] = 0.0
        return

    # best_action = ''
    value = 0.0
    ma = -100000000.0  # setting ma value to big -ve
    for act in range(action):
        value = -100000000.0  # setting ma value to big -ve

        # west square (IJ will not be affected by MM's attack here)
        if pos == 0:
            # dormant state
            if sta == 0:
                # movement
                if act == 3:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(4, mat, arr, 0, hea), no_of_columns] = -1.0*0.8
                    A[give_row(4, mat, arr, 1, hea), no_of_columns] = -1.0*0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0*0.2
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -1.0*0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                    # arrow shoot
                elif act == 5 and arr > 0 and hea > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(pos, mat, arr-1, 0, hea), no_of_columns] = -1.0 * 0.8 * 0.75
                    A[give_row(pos, mat, arr-1, 1, hea), no_of_columns] = -1.0 * 0.2 * 0.75
                    A[give_row(pos, mat, arr-1, 0, hea - 25), no_of_columns] = -1.0 * 0.8 * 0.25
                    A[give_row(pos, mat, arr-1, 1, hea - 25), no_of_columns] = -1.0 * 0.2 * 0.25

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                    # adding final reward if the MM is killed

            # ready state (sta=1)
            else:
                # movement
                if act == 3:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(4, mat, arr, 1, hea), no_of_columns] = -1.0 * 0.5
                    A[give_row(4, mat, arr, 0, hea), no_of_columns] = -1.0 * 0.5

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0 * 0.5
                    A[give_row(0, mat, arr, 0, hea), no_of_columns] = -1.0 * 0.5

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                    # arrow shoot
                elif act == 5 and arr > 0 and hea > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(0, mat, arr, 1, hea - 25), no_of_columns] = -1.0 * 0.5 * 0.25
                    A[give_row(0, mat, arr, 1, hea), no_of_columns] = -1.0 * 0.5 * 0.75
                    A[give_row(0, mat, arr, 0, hea), no_of_columns] = -1.0 * 0.5 * 0.75
                    A[give_row(0, mat, arr, 0, hea - 25), no_of_columns] = -1.0 * 0.5 * 0.25

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

        # north square (IJ will not be affected by MM's attack here)
        elif pos == 1:
            # dormant state
            if sta == 0:
                # movement
                if act == 1:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.8
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.2
                    A[give_row(4, mat, arr, 0, hea), no_of_columns] = -0.85 * 0.8
                    A[give_row(4, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 0.15 + 0.85 * 0.2
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.8
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.2
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                # material crafting
                elif act == 7 and mat > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(pos, mat - 1, min(arr + 1, 3), 0, hea), no_of_columns] = -0.5 * 0.8
                    A[give_row(pos, mat - 1, min(arr + 1, 3), 1, hea), no_of_columns] = -0.5 * 0.2
                    A[give_row(pos, mat - 1, min(arr + 2, 3), 0, hea), no_of_columns] = -0.35 * 0.8
                    A[give_row(pos, mat - 1, min(arr + 2, 3), 1, hea), no_of_columns] = -0.35 * 0.2
                    A[give_row(pos, mat - 1, min(arr + 3, 3), 0, hea), no_of_columns] = -0.15 * 0.8
                    A[give_row(pos, mat - 1, min(arr + 3, 3), 1, hea), no_of_columns] = -0.15 * 0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

            # ready state (sta=1)
            else:
                # movement
                if act == 1:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(4, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.5
                    A[give_row(4, mat, arr, 0, hea), no_of_columns] = -0.85 * 0.5
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.5
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.5

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 0.15 + 0.85 * 0.5
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.5
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.5
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = -0.85 * 0.5

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                    # material crafting
                elif act == 7 and mat > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(pos, mat - 1, min(arr + 1, 3), 1, hea), no_of_columns] = -0.5 * 0.5
                    A[give_row(pos, mat - 1, min(arr + 1, 3), 0, hea), no_of_columns] = -0.5 * 0.5
                    A[give_row(pos, mat - 1, min(arr + 2, 3), 1, hea), no_of_columns] = -0.35 * 0.5
                    A[give_row(pos, mat - 1, min(arr + 2, 3), 0, hea), no_of_columns] = -0.35 * 0.5
                    A[give_row(pos, mat - 1, min(arr + 3, 3), 1, hea), no_of_columns] = -0.15 * 0.5
                    A[give_row(pos, mat - 1, min(arr + 3, 3), 0, hea), no_of_columns] = -0.15 * 0.5

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

        # east square (IJ will be affected by MM's attack now)
        elif pos == 2:
            # dormant state
            if sta == 0:
                # movement
                if act == 2:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(4, mat, arr, 0, hea), no_of_columns] = -1.0 * 0.8
                    A[give_row(4, mat, arr, 1, hea), no_of_columns] = -1.0 * 0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 0.2
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                # arrow shoot
                elif act == 5 and hea > 0 and arr > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(pos, mat, arr - 1, 1, hea - 25), no_of_columns] = -0.9 * 0.2
                    A[give_row(pos, mat, arr - 1, 0, hea - 25), no_of_columns] = -0.9 * 0.8
                    A[give_row(pos, mat, arr - 1, 1, hea), no_of_columns] = -0.1 * 0.2
                    A[give_row(pos, mat, arr - 1, 0, hea), no_of_columns] = -0.1 * 0.8

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                # blade hit
                elif act == 6 and hea > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 0.2*0.8 + 0.2
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -0.2 * 0.8
                    A[give_row(pos, mat, arr, 0, max(0, hea - 50)), no_of_columns] = -0.8 * 0.2
                    A[give_row(pos, mat, arr, 1, max(0, hea - 50)), no_of_columns] = -0.2 * 0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

            # ready state (sta=1)
            else:
                # movement
                if act == 2:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(4, mat, arr, 1, hea), no_of_columns] = -1.0 * 0.5
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -1.0 * 0.5

                    reward = -1 * step_cost -1.0 * bad_reward
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 0.5
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -1.0 * 0.5

                    reward = -1 * step_cost -1.0 * bad_reward
                    r[0][no_of_columns] = reward

                # arrow shoot
                elif act == 5 and hea > 0 and arr > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(pos, mat, arr - 1, 1, hea), no_of_columns] = -0.1 * 0.5
                    A[give_row(pos, mat, 0, 0, min(hea + 25, 100)), no_of_columns] = -0.5
                    A[give_row(pos, mat, arr - 1, 1, hea - 25), no_of_columns] = -0.9 * 0.5

                    reward = -1 * step_cost -1.0 * bad_reward
                    r[0][no_of_columns] = reward

                # blade hit
                elif act == 6 and hea > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 0.5 + 0.5*0.2
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -0.5
                    A[give_row(pos, mat, arr, 1, max(0, hea - 50)), no_of_columns] = -0.5 * 0.20

                    reward = -1 * step_cost + -1.0 * bad_reward
                    r[0][no_of_columns] = reward

        # south square (IJ will not be affected by MM's attack here)
        elif pos == 3:
            # dormant state
            if sta == 0:
                # movement
                if act == 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(4, mat, arr, 0, hea), no_of_columns] = -0.85 * 0.8
                    A[give_row(4, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.2
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.8
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 0.15 + 0.85 * 0.2
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.2
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.2
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.8

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                # material gathering
                elif act == 8 and mat < 2:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 0.75 + 0.25 * 0.2
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -0.25 * 0.2
                    A[give_row(pos, mat + 1, arr, 1, hea), no_of_columns] = -0.75 * 0.2
                    A[give_row(pos, mat + 1, arr, 0, hea), no_of_columns] = -0.75 * 0.8

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward
                    
                elif act == 8 and mat == 2:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 0.2
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward


            # ready state (sta=1)
            else:
                # movement
                if act == 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(4, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.5
                    A[give_row(4, mat, arr, 0, hea), no_of_columns] = -0.85 * 0.5
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.5
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.5

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 0.15 + 0.85 * 0.5
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = -0.85 * 0.5
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.5
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.5

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                # material gathering
                elif act == 8 and mat < 2:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 0.25 * 0.5 + 0.75
                    A[give_row(pos, mat + 1, arr, 1, hea), no_of_columns] = -0.75 * 0.5
                    A[give_row(pos, mat + 1, arr, 0, hea), no_of_columns] = -0.75 * 0.5
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = -0.25 * 0.5

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward
                    
                    
                elif act == 8 and mat == 2:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 0.2
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward


        # Center square (IJ will not be affected by MM's attack here)
        elif pos == 4:
            # dormant state
            if sta == 0:
                # movement
                if act == 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(1, mat, arr, 0, hea), no_of_columns] = -0.85 * 0.8
                    A[give_row(1, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.2
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.2
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.8

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 1:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(3, mat, arr, 0, hea), no_of_columns] = -0.85 * 0.8
                    A[give_row(3, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.2
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.2
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.8

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 3:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -1.0 * 0.8
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -1.0 * 0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 2:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(0, mat, arr, 0, hea), no_of_columns] = -0.85 * 0.8
                    A[give_row(0, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.2
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.2
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.8

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 0.85 * 0.2 + 0.15
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.2
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.2
                    A[give_row(2, mat, arr, 0, hea), no_of_columns] = -0.15 * 0.8

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                # arrow shoot
                elif act == 5 and hea > 0 and arr > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 1.0
                    A[give_row(pos, mat, arr - 1, 0, hea - 25), no_of_columns] = -0.5 * 0.8
                    A[give_row(pos, mat, arr - 1, 1, hea - 25), no_of_columns] = -0.5 * 0.2
                    A[give_row(pos, mat, arr - 1, 1, hea), no_of_columns] = -0.5 * 0.2
                    A[give_row(pos, mat, arr - 1, 0, hea), no_of_columns] = -0.5 * 0.8

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

                # blade hit
                elif act == 6 and hea > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 0, hea), no_of_columns] = 0.1 + 0.9 * 0.2
                    A[give_row(pos, mat, arr, 1, max(0, hea - 50)), no_of_columns] = -0.1 * 0.2
                    A[give_row(pos, mat, arr, 0, max(0, hea - 50)), no_of_columns] = -0.1 * 0.8
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = -0.9 * 0.2

                    reward = -1 * step_cost
                    r[0][no_of_columns] = reward

            # ready state (sta=1)
            else:
                # movement
                if act == 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(1, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.5
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -0.5
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.5

                    reward = -1 * step_cost + -1.0 * bad_reward
                    r[0][no_of_columns] = reward

                elif act == 1:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(3, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.5
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -0.5
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.5

                    reward = -1 * step_cost + -1.0 * bad_reward
                    r[0][no_of_columns] = reward

                elif act == 3:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -1.0 * 0.5
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -0.5

                    reward = -1 * step_cost + -1.0 * bad_reward
                    r[0][no_of_columns] = reward

                elif act == 2:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(0, mat, arr, 1, hea), no_of_columns] = -0.85 * 0.5
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -0.5
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.5

                    reward = -1 * step_cost + -1.0 * bad_reward
                    r[0][no_of_columns] = reward

                elif act == 4:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 0.5 + 0.5 * 0.15
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -0.5
                    A[give_row(2, mat, arr, 1, hea), no_of_columns] = -0.15 * 0.5

                    reward = -1 * step_cost + -1.0 * bad_reward
                    r[0][no_of_columns] = reward

                # arrow shoot
                elif act == 5 and hea > 0 and arr > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 1.0
                    A[give_row(pos, mat, arr - 1, 1, hea), no_of_columns] = -0.5 * 0.5
                    A[give_row(pos, mat, arr - 1, 1, hea - 25), no_of_columns] = -0.5 * 0.5
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -0.5

                    reward = -1 * step_cost + -1.0 * bad_reward
                    r[0][no_of_columns] = reward

                # blade hit
                elif act == 6 and hea > 0:
                    add_column()
                    column_action[no_of_columns] = act
                    A[give_row(pos, mat, arr, 1, hea), no_of_columns] = 0.5 + 0.5 * 0.1
                    A[give_row(pos, mat, arr, 1, min(0, hea - 50)), no_of_columns] = -0.1 * 0.5
                    A[give_row(pos, mat, 0, 0, min(100, hea + 25)), no_of_columns] = -0.5

                    reward = -1 * step_cost + -1.0 * bad_reward
                    r[0][no_of_columns] = reward


def main():
    global health, positions, material, arrows, state
    for a in range(positions):
        for b in range(material):
            for c in range(arrows):
                for d in range(state):
                    for e in range(5):
                        set_a(a, b, c, d, e * 25)


main()
print(no_of_columns)

# creating x matrix
x = cp.Variable(shape=(1936, 1), name="x")

# for i in range(600):
#     constraints+= [ 
#     cp.matmul(A[i], x) == alpha[i][0], x>=0]

constraints = [cp.matmul(A, x) == alpha, x >= 0]
objective = cp.Maximize(cp.matmul(r, x))
problem = cp.Problem(objective, constraints)

solution = problem.solve()
print(solution)

print(x.value)



