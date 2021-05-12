import numpy as np
import cvxpy as cp

# alpha = np.zeros(shape=(600,1))
# # start state is last possible state (C,2,3,R,100)
# alpha[599][0] = 1.0

# A = np.zeros(shape=(600, 1936))

# r = np.zeros(shape=(1,1936))



# # creating x matrix
# x = cp.Variable(shape=(1936, 1), name="x")


# constraints = [cp.matmul(A, x) <= alpha, x >= 0]
# objective = cp.Maximize(cp.matmul(r, x))
# problem = cp.Problem(objective, constraints)

# solution = problem.solve()
# # print(solution)

# # print(x.value)
# print(r)
hea = 100
print(int(hea/25)+1)