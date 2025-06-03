import random


def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])

def approach_two(first_distrib, samples):
    elim_counts = [0] * len(first_distrib)
    for i in range(samples):
        remaining = set()
        while len(remaining) != 1:
            darts = set(random.choices(range(len(first_distrib)), first_distrib, k=len(first_distrib)-1))
            remaining = set(range(len(first_distrib))) - darts
        for candidate in remaining:
            elim_counts[candidate] += 1

    return [count / samples for count in elim_counts]


def approach_two_exact(counts, eliminated=[]):
    if counts.count(0) > 1:
        return [1/counts.count(0) if x == 0 else 0 for x in counts]

    factors = []
    for i in range(len(counts)):
        if counts[i] == -1:
            factors.append(0)
            continue
        factor = 1
        for j in range(len(counts)):
            if counts[j] == -1:
                continue
            if i != j:
                factor *= counts[j]
        factors.append(factor)
    norm_sum = sum(factors)

    return [f/norm_sum for f in factors]


def util_to_first_distrib(utils, eliminated=[]):
    utils = normalize_utils(utils)
    first_distrib = [0] * len(utils[0])
    for profile in utils:
        while argmax(profile) in eliminated:
            profile[argmax(profile)] = -1
        first_distrib[argmax(profile)] += 1
    first_distrib = [f/sum(first_distrib) for f in first_distrib]
    first_distrib = [-1 if i in eliminated else first_distrib[i] for i in range(len(first_distrib))] 
    return first_distrib

def util_to_harmonic(utils, eliminated=[]):
    utils = normalize_utils(utils)
    harmonic = [0] * len(utils[0])
    for profile in utils:
        for i in eliminated:
            profile[i] = -1
        order = sorted(range(len(profile)), key=lambda i: profile[i], reverse=True)
        for i, alt in enumerate(order):
            if alt in eliminated:
                continue
            harmonic[alt] += 1 / (i + 1)

    harmonic = [-1 if i in eliminated else harmonic[i] for i in range(len(harmonic))] 
    return harmonic

def util_to_borda(utils, eliminated=[]):
    utils = normalize_utils(utils)
    borda = [0] * len(utils[0])
    m = len(utils[0]) - len(eliminated)
    for profile in utils:
        for i in eliminated:
            profile[i] = -1
        order = sorted(range(len(profile)), key=lambda i: profile[i], reverse=True)
        for i, alt in enumerate(order):
            if alt in eliminated:
                continue
            borda[alt] += m - i - 1  # Borda count is m - position in the order

    borda = [-1 if i in eliminated else borda[i] for i in range(len(borda))] 
    return borda


def run_irv(utils, count_rule=util_to_first_distrib, elim_rule=approach_two_exact, eliminated=[]):
    if len(eliminated) == len(utils[0]) - 1:
        win_distrib = [0 if i in eliminated else 1 for i in range(len(utils[0]))]
        return win_distrib

    win_distrib = [0] * len(utils[0])
    counts = count_rule(utils, eliminated)

    # print(counts)

    elim_probs = elim_rule(counts)

    # print(f"with {eliminated} eliminated, the first_distrib is {first_distrib} and this round's elimination probabilities are {elim_probs}")

    for elim in range(len(elim_probs)):
        if elim in eliminated:
            continue

        next_round = run_irv(utils, count_rule, elim_rule, eliminated + [elim])
        # print(f"with {eliminated + [elim]} eliminated, next round's win probabilities are {next_round}")
        
        for alt in range(len(win_distrib)):
            if alt != elim:
                win_distrib[alt] += elim_probs[elim] * next_round[alt]

    return win_distrib

def social_wellfare(distrib, utils) -> float:
    sw = 0
    for i in range(len(distrib)):
        for j in range(len(utils)):
            sw += distrib[i] * utils[j][i]
    return sw


def normalize_utils(utils):
    return [[u/sum(utils[i]) for u in utils[i]] for i in range(len(utils))]


def main():
    # utils = []
    # m = 4
    # for i in range(m):
    #     for j in range(i+1):
    #         utils.append([0 if i != j else 1 for j in range(m)] + [1])

    utils = (
        [[1, 0, 0, 0, 1]] * 100
        + [[0, 1, 0, 0, 1]] * 133
        + [[0, 0, 1, 0, 1]] * 167
        + [[0, 0, 0, 1, 1]] * 200
        )


    
    # print(utils)

    utils = normalize_utils(utils)

    # for i in range(m, 0, -1):
    #     print(list(range(m, i, -1)))
    #     print(util_to_first_distrib(utils, list(range(m, i, -1))))
    #     print(approach_two_exact(util_to_first_distrib(utils, list(range(m, i, -1)))))

    win_distrib = run_irv(utils)
    sw = social_wellfare(win_distrib, utils)
    print(f"win distrib is {win_distrib}")
    print(f"social welfare is {sw}")

    alternatives_util = [0] * len(utils[0])
    for profile in utils:
        for i in range(len(profile)):
            alternatives_util[i] += profile[i]

    max_sw = max(alternatives_util)
    print(f"the maximum social welfare is alternative {argmax(alternatives_util)} with utility {max_sw}")

    print(f"distortion = {max_sw/sw}")

def ballot_to_utils(ballots):
    m = len(ballots[0])
    utils = [[0] * m for _ in range(len(ballots))]
    
    for i, ballot in enumerate(ballots):
        for j, candidate in enumerate(ballot):
            utils[i][candidate - 1] = m - j  # Assign utility based on position in the ballot

    return utils

def main2():

    election = (
        [[1,2,3,4]] +
        [[1,2,3,4]] +
        [[1,3,2,4]] +
        [[1,3,2,4]] +
        [[3,2,1,4]] +
        [[4,3,2,1]] +
        [[4,2,1,3]] +
        [[2,1,3,4]]
    )


    election2 = (
        [[1,2,3]] +
        [[1,2,3]] +
        [[1,3,2]] +
        [[1,3,2]] +
        [[3,2,1]] +
        [[3,2,1]] +
        [[2,1,3]] +
        [[2,1,3]]
    )


    utils = ballot_to_utils(election)
    win_distrib = run_irv(utils, count_rule=util_to_first_distrib)

    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_first_distrib)

    print(f"win distrib is {win_distrib2}")

    one_diff = win_distrib2[0] - win_distrib[0]
    # four_diff = win_distrib2[3] - win_distrib[3]
    print(f"the difference in the first candidate's win probability is {one_diff}")
    # print(f"the difference in the fourth candidate's win probability is {four_diff}")

    # winner = argmax(win_distrib)
    # winner2 = argmax(win_distrib2)

    # print(f"the winner of the first election is {winner + 1} and the second is {winner2 + 1}")

def main3():
    # c = 1
    # for i in range(1, 5):
    #     a = [1/2] + [1/ (2 * i)] * i
    #     b = approach_two_exact(a)
    #     c *= 1 - b[0]

    # print(c)

    election = [[1,2,3], [1,2,3], [1,2,3], [1,2,3], [3,2,1], [3,2,1], [2,1,3], [2,1,3]]

    election2 = [[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4], [3,2,1,4], [4,3,2,1], [4,2,1,3], [2,1,3,4]]

    utils = ballot_to_utils(election)
    win_distrib = run_irv(utils, count_rule=util_to_first_distrib)

    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_first_distrib)

    print(f"win distrib is {win_distrib2}")

def prcv_counterexample_main():
    base = (
        [[1, 2, 3, 4]] * 10000 +
        [[2, 3, 4, 1]] * 1 +
        [[2, 4, 3, 1]] * 1 +
        [[3, 1, 2, 4]] * 50 +
        [[4, 1, 2, 3]] * 50
    )

    election = base + [[2, 1, 3, 4]]


    election2 = base + [[1, 2, 3, 4]]


    utils = ballot_to_utils(election)
    win_distrib = run_irv(utils, count_rule=util_to_first_distrib)

    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_first_distrib)

    print(f"win distrib is {win_distrib2}")

    one_diff = win_distrib2[0] - win_distrib[0]
    print(f"the difference in the first candidate's win probability is {one_diff}")

def main5():
    base = (
        [[3, 1, 4, 5, 6, 7, 2]] + 
        [[3, 1, 7, 4, 5, 6, 2]] + 
        [[3, 1, 5, 6, 7, 4, 2]] +
        [[3, 1, 6, 7, 4, 5, 2]] +
        [[4, 1, 3, 5, 6, 7, 2]] + 
        [[4, 1, 7, 3, 5, 6, 2]] + 
        [[4, 1, 5, 6, 7, 3, 2]] +
        [[4, 1, 6, 7, 3, 5, 2]] +
        [[5, 1, 3, 4, 6, 7, 2]] + 
        [[5, 1, 7, 3, 4, 6, 2]] + 
        [[5, 1, 4, 6, 7, 3, 2]] +
        [[5, 1, 6, 7, 3, 4, 2]] +
        [[6, 1, 3, 5, 7, 4, 2]] + 
        [[6, 1, 5, 7, 4, 3, 2]] +
        [[6, 1, 7, 4, 3, 5, 2]] +
        [[6, 1, 4, 3, 5, 7, 2]] +
        [[7, 1, 5, 6, 4, 3, 2]] +
        [[7, 1, 6, 4, 3, 5, 2]] +
        [[7, 1, 4, 3, 5, 6, 2]] +
        [[7, 1, 3, 5, 6, 4, 2]] +         
        [[1, 2, 3, 4, 5, 6, 7]] * 1000 + 
        [[1, 2, 4, 5, 6, 7, 3]] * 1000 +
        [[1, 2, 5, 6, 7, 3, 4]] * 1000 + 
        [[1, 2, 6, 7, 3, 4, 5]] * 1000 + 
        [[1, 2, 7, 3, 4, 5, 6]] * 1000
    )

    election = base + [[2, 1, 3, 4]]


    election2 = base + [[1, 2, 3, 4]]


    utils = ballot_to_utils(election)
    win_distrib = run_irv(utils, count_rule=util_to_harmonic)

    round1_elim_probs = approach_two_exact(util_to_harmonic(utils))

    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_harmonic)

    round1_elim_probs2 = approach_two_exact(util_to_harmonic(utils2))

    print(f"win distrib is {win_distrib2}")

    one_diff = win_distrib2[0] - win_distrib[0]
    print(f"the difference in the first candidate's win probability is {one_diff}")

    change_in_elim_probs = [a - b for a, b in zip(round1_elim_probs2, round1_elim_probs)]

    print(f"change in round 1 elimination probabilities is {change_in_elim_probs}")

    for i in [1,2,3,4,5,6]:
        win_prob = run_irv(utils2, count_rule=util_to_harmonic, eliminated=[i])[0]
        print(win_prob, win_prob * change_in_elim_probs[i])

    print(sum([run_irv(utils2, count_rule=util_to_harmonic, eliminated=[i])[0] * change_in_elim_probs[i] for i in [1,2,3,4,5,6]]))

def main6():

    election = [[1, 2, 3, 4, 5]]

    utils = ballot_to_utils(election)

    print(util_to_borda(utils))

    win_distrib = run_irv(utils, count_rule=util_to_borda)

    print(f"win distrib is {win_distrib}")

def main7():

    election = [[1, 2, 3, 4]]

    utils = ballot_to_utils(election)

    print(util_to_harmonic(utils))

    win_distrib = run_irv(utils, count_rule=util_to_harmonic)

    print(f"win distrib is {win_distrib}")

def borda_insertion_counterexample():
    election = (
        [[1,2,3]] * 2 +
        [[1,3,2]] * 2 +
        [[2,3,1]] * 2 +
        [[3,2,1]] * 2
    )

    election2 = (
        [[1,4,2,3]] * 2 +
        [[1,4,3,2]] * 2 +
        [[2,3,1,4]] +
        [[3,2,1,4]] +
        [[2,3,1,4]] +
        [[3,2,1,4]]
    )


    for rule in [util_to_first_distrib, util_to_harmonic, util_to_borda]:
        print()
        print(f"using {rule.__name__} as the counting rule")
        utils = ballot_to_utils(election)
        win_distrib = run_irv(utils, count_rule=rule)

        print(f"win distrib is {win_distrib}")

        utils2 = ballot_to_utils(election2)
        win_distrib2 = run_irv(utils2, count_rule=rule)

        print(f"win distrib is {win_distrib2}")

        one_diff = win_distrib2[0] - win_distrib[0]
        print(f"the difference in the first candidate's win probability is {one_diff}")

def main8():
    election = (
        [[1,2,3]] * 2 +
        [[1,3,2]] * 2 +
        [[2,3,1]] * 2 +
        [[3,2,1]] * 2
    )

    election2 = (
        [[1,4,2,3]] * 2 +
        [[1,4,3,2]] * 2 +
        [[2,3,1,4]] +
        [[3,2,1,4]] +
        [[2,3,1,4]] +
        [[3,2,1,4]]
    )

    print(election)
    print(election2)

    for rule in [util_to_first_distrib, util_to_harmonic, util_to_borda]:
        print()
        print(f"using {rule.__name__} as the counting rule")
        utils = ballot_to_utils(election)
        win_distrib = run_irv(utils, count_rule=rule)

        print(f"win distrib is {win_distrib}")

        utils2 = ballot_to_utils(election2)
        win_distrib2 = run_irv(utils2, count_rule=rule)

        print(f"win distrib is {win_distrib2}")

        one_diff = win_distrib2[0] - win_distrib[0]
        print(f"the difference in the first candidate's win probability is {one_diff}")

if __name__ == "__main__":
    main8()


