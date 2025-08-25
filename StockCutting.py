import pulp
from pulp import LpVariable


def initialize_patterns(lengthofEachRoll, boards) -> list[list[int]]:
    # Starting feasible solution where only cutting one board width at the time from the roll

    m = len(boards)
    patterns = []
    for i in range(m):
        pattern = [0] * m
        pattern[i] = lengthofEachRoll // boards[i] # take the integer value, basically the number of boards of length i that can be cut from W (roll size)
        patterns.append(pattern)
    return patterns


def solve_master(patterns, demandOfEachBoardPattern, typesOfBoardLength) -> (list[float],pulp.LpProblem):
    # Solve the restricted master problem RMP
    # For a Minimization problem, a -ve (C[j] - Z[j]) enters the basis
    # x_j variable represents a pattern cut out of a roll. So, a pattern means use of a roll
    rmp = pulp.LpProblem("CuttingStockMaster", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, cat="Continuous") for j in range(len(patterns))] # note that the var here is continuous for RMP. We need an LP to get the dual values.

    # Objective: minimize rolls
    rmp += pulp.lpSum(x) # objective is to minimize sum[x_j] i.e the number of rolls used

    # Demand constraints with existing patterns
    for i in range(typesOfBoardLength):
        rmp += pulp.lpSum(patterns[j][i] * x[j] for j in range(len(patterns))) >= demandOfEachBoardPattern[i]

    rmp.solve(pulp.HiGHS(msg=False))

    varValues = [(j.name,j.value()) for j in x]
    dual = [c.pi for c in rmp.constraints.values()]
    return dual, rmp


def solve_subproblem(dualFromRMP, boards, lengthofEachRoll) -> (float, list[int]):
    # Solve the column generation subproblem CGSP

    m = len(boards)
    sub = pulp.LpProblem("Subproblem", pulp.LpMaximize)

    # Here y_j represents the number of unit of each board type to be taken in this pattern
    # Hypothetical y_0 = 2, y_1 = 1, y_2 = 0 represents that in the new pattern [2,1,0] we have chosen to cut 2 unit of 4 board type and 1 unit of 6 board type
    y = [pulp.LpVariable(f"y_{i}", lowBound=0, cat="Integer") for i in range(m)] # see the type of var is integer

    # Objective: maximize dual value
    # We want a pattern such that it gives a -ve (C[j] - Z[j]) or
    # a positive reduced cost (Z[j] - C[j]) . A positive reduced cost or a negative (C[j] - Z[j]) enters the basis
    # c[j] for any new pattern is 1 as the goal is to minimize the number of rolls (patterns) to be used
    # For a pattern j of type [a1, b1, c1] and c[j] =1, (C[j] - Z[j]) = 1 - (dual of constr1 * a1 + dual of constr2 * b1 + dual of constr3 * c1)
    # to obtain a -ve (C[j] - Z[j]), we can maximize the Z[j] = (dual of constr1 * a1 + dual of constr2 * b1 + dual of constr3 * c1)
    # the constraint will be 4*a1 + 6*b1 + 7*c1 <= EachRollLength
    sub += pulp.lpSum(dualFromRMP[i] * y[i] for i in range(m))

    # Width constraint
    sub += pulp.lpSum(boards[i] * y[i] for i in range(m)) <= lengthofEachRoll

    sub.solve(pulp.HiGHS(msg=False))

    NegativeOfReducedCost_CjMinusZj = 1 - pulp.value(sub.objective)
    varValues = [(j.name, j.value()) for j in y]
    pattern = [int(y[i].varValue) for i in range(m)]
    return NegativeOfReducedCost_CjMinusZj, pattern


def solve_final_integer_master(allPatterns, demandOfEachBoardPattern, typesOfBoardLength)-> (list[LpVariable], pulp.LpProblem):
    # get the optimal solution for the RMP using the column generated in CGSP

    master = pulp.LpProblem("FinalMaster", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, cat="Integer") for j in range(len(allPatterns))] # note: we have int var here unlike in RMP

    master += pulp.lpSum(x)

    for i in range(typesOfBoardLength):
        master += pulp.lpSum(allPatterns[j][i] * x[j] for j in range(len(allPatterns))) >= demandOfEachBoardPattern[i]

    master.solve(pulp.HiGHS(msg=False))
    varValues = [(j.name, j.value()) for j in x]
    return x, master


def get_solution(varOfFinalProblem, allPatternsConsideredInFinalProblem):
    # display solution

    print("\nCutting Patterns Used:")
    for i, var in enumerate(varOfFinalProblem):
        if var.varValue > 0:
            print(f"Pattern {allPatternsConsideredInFinalProblem[i]} is used {int(var.varValue)} times")
    print(f"\nTotal rolls used: {int(pulp.value(pulp.lpSum(var for var in varOfFinalProblem)))}")


if __name__ == "__main__":

    W = 15 #length/Width of Each roll from which the boards will be cut
    boards = [4, 6, 7, 5] #Lengths of boards to be cut from the rolls
    requirements = [800, 500, 100, 200] #Requirements for each length like 80 boards of 4ft length
    m = len(boards)

    patterns = initialize_patterns(lengthofEachRoll= W, boards= boards) # this will give the first set of basic variables

    while True:  # keep iterating (exiting and entering columns until all reduced costs are negative)
        dual, RMP = solve_master(patterns= patterns, demandOfEachBoardPattern = requirements, typesOfBoardLength= m)
        NegativeOfReducedCost_CjMinusZj, new_pattern = solve_subproblem(dualFromRMP= dual, boards= boards, lengthofEachRoll= W)

        if NegativeOfReducedCost_CjMinusZj >= -1e-3: # if the value is less than equal to -0.001 i.e. almost 0 or +ve
            break  # optimality

        patterns.append(new_pattern)

    varOfFinalProblem, finalProblem = solve_final_integer_master(allPatterns= patterns,
                                                                 demandOfEachBoardPattern= requirements,
                                                                 typesOfBoardLength= m) # after adding all the patterns which has +ve reduced cost or -ve (C[j] - Z[j]), solve the problem as an integer problem
    # prints the solution
    get_solution(varOfFinalProblem, patterns)
