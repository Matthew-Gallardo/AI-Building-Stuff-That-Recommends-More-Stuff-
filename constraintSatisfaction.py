from constraint import *

problem = Problem()
problem.addVariable('a', range(10))
problem.addVariable('b', range(10))
problem.addConstraint(lambda a, b: a * 2 == b)

solutions = problem.getSolutions()
print(solutions)

def magic_square(matrix_ms):
    iSize = len(matrix_ms[0])
    sum_list = []

    for col in range(iSize):
        sum_list.append(sum(row[col] for row in matrix_ms))
        sum_list.extend([sum(lines) for lines in matrix_ms])

    dlResult = 0  # Initialize dlResult
    for i in range(0, iSize):
        dlResult += matrix_ms[i][i]
        sum_list.append(dlResult)

    drResult = 0  # Initialize drResult
    for i in range(iSize-1, -1, -1):
        drResult += matrix_ms[i][i]
        sum_list.append(drResult)

    if len(set(sum_list)) > 1:
        return False

    return True

print(magic_square([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

denominations = [1, 2, 5, 10, 20, 50, 100]

def returnChange(change, denominations):
    toGiveBack = [0] * len(denominations)

    for pos, coin in enumerate(reversed(denominations)):
        while coin <= change:
            change = change - coin
            toGiveBack[pos] += 1

    return toGiveBack

print(returnChange(30, denominations))
