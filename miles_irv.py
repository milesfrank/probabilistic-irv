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


def util_to_first_distrib(utils, eliminated=tuple()):
    utils = normalize_utils(utils)
    first_distrib = [0] * len(utils[0])
    for profile in utils:
        while argmax(profile) in eliminated:
            profile[argmax(profile)] = -1
        first_distrib[argmax(profile)] += 1
    first_distrib = [f/sum(first_distrib) for f in first_distrib]
    first_distrib = [-1 if i in eliminated else first_distrib[i] for i in range(len(first_distrib))] 
    return first_distrib

def util_to_harmonic(utils, eliminated=tuple()):
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

def util_to_borda(utils, eliminated=tuple()):
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

def util_to_f(utils, f, eliminated=tuple()):
    utils = normalize_utils(utils)
    count = [0] * len(utils[0])
    m = len(utils[0]) - len(eliminated)
    for profile in utils:
        for i in eliminated:
            profile[i] = -1
        order = sorted(range(len(profile)), key=lambda i: profile[i], reverse=True)
        for i, alt in enumerate(order):
            if alt in eliminated:
                continue
            count[alt] += f(m, i)  # m is the number of alternatives, i is the position in the candidate in the ballot

    count = [-1 if i in eliminated else count[i] for i in range(len(count))] 
    return count


from functools import cache
@cache
def run_irv(utils, count_rule=util_to_first_distrib, elim_rule=approach_two_exact, eliminated=tuple()):
    if len(eliminated) == len(utils[0]) - 1:
        win_distrib = [0 if i in eliminated else 1 for i in range(len(utils[0]))]
        return win_distrib

    win_distrib = [0] * len(utils[0])
    counts = count_rule(utils, eliminated)

    elim_probs = elim_rule(counts)

    for elim in range(len(elim_probs)):
        if elim in eliminated:
            continue

        next_round = run_irv(utils, count_rule, elim_rule, tuple((eliminated + [elim]).sort()))
        
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
    # also makes it into a list of lists rather than tuples
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

    election = [[1,2,3], [3,2,1], [2,1,3], [2,1,3]]

    election2 = [[1,2,3,4], [3,2,1,4], [4,2,1,3], [2,1,3,4]]

    utils = ballot_to_utils(election)
    win_distrib = run_irv(utils, count_rule=util_to_first_distrib)

    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_first_distrib)

    print(f"win distrib is {win_distrib2}")

def prcv_counterexample_main():
    base = (
        [[1, 2, 3, 4]] * 500 +
        [[2, 3, 4, 1]] * 2 +
        [[2, 4, 3, 1]] * 2 +
        [[3, 1, 2, 4]] * 50 +
        [[4, 1, 2, 3]] * 50
    )

    election = base + [[2, 1, 3, 4]]


    election2 = base + [[1, 2, 3, 4]]


    utils = ballot_to_utils(election)
    win_distrib = run_irv(utils, count_rule=util_to_first_distrib)

    # print(approach_two_exact(util_to_first_distrib(utils)))

    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_first_distrib)

    # print(approach_two_exact(util_to_first_distrib(utils2)))

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
        [[1,2,3]] +
        [[1,3,2]] +
        [[2,3,1]] +
        [[3,2,1]] 
    )

    election2 = (
        [[1,4,2,3]] +
        [[1,4,3,2]] +
        [[2,3,1,4]] +
        [[3,2,1,4]]
    )


    utils = ballot_to_utils(election)
    win_distrib = run_irv(utils, count_rule=util_to_borda)

    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_borda)

    print(f"win distrib is {win_distrib2}")

    one_diff = win_distrib2[0] - win_distrib[0]
    print(f"the difference in the first candidate's win probability is {one_diff}")

def borda_insertion_counterexample2():
    election = (
        [[1,2]] +
        [[2,1]]
    )

    election2 = (
        [[1,3,2]] +
        [[2,1,3]]
    )


    utils = ballot_to_utils(election)
    win_distrib = run_irv(utils, count_rule=util_to_borda)

    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_borda)

    print(f"win distrib is {win_distrib2}")

    one_diff = win_distrib2[0] - win_distrib[0]
    print(f"the difference in the first candidate's win probability is {one_diff}")

def borda_insertion_counterexample3(i=3):
    election = (
        [list(range(3,i)) + [1,2]] +
        [list(range(3,i)) + [2,1]]
    )

    election2 = (
        [list(range(3,i)) +[1,i,2]] +
        [list(range(3,i)) +[2,1,i]]
    )


    utils = ballot_to_utils(election)
    win_distrib = run_irv(utils, count_rule=util_to_borda)

    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_borda)

    print(f"win distrib is {win_distrib2}")

    one_diff = win_distrib2[0] - win_distrib[0]
    print(f"the difference in the first candidate's win probability is {one_diff}")

def borda_as_f(m, i):
    return (m - i - 1) / (m - 1) # Borda count normalized to [0, 1]

def harmonic_0_as_f(m, i):
    if i == m - 1:
        return 0  # Last position gets 0
    return 1 / (i + 1)

def harmonic_as_f(m, i):
    return 1 / (i + 1)

def kapproval_as_f(m, i):
    if i == 0:
        return 1
    if i == 1:
        return 1
    return 0

def kveto_as_f(m, i):
    if i >= m - 2:
        return 0
    return 1

def fptp_as_f(m, i):
    if i == 0:
        return 1
    return 0

def main8():
    def rule(utils, eliminated=[]):
        return util_to_f(utils, f=borda_as_f, eliminated=eliminated)
    
    previous = 0

    for i in range(4, 10):
        election = (
            [[1] + list(range(4, i)) + [2, 3]] +
            [[1] + list(range(4, i)) + [3, 2]] 
            # [list(range(1, i))]
        )

        print(election)

        utils = ballot_to_utils(election)

        print(rule(utils))
        print(approach_two_exact(rule(utils)))


        win_distrib = run_irv(utils, count_rule=rule)

        print(f"win distrib is {win_distrib}")
        one_diff = win_distrib[0] - previous
        print(f"the difference in the first candidate's win probability is {one_diff}")
        print()
        previous = win_distrib[0]

def main9():
    def rule(utils, eliminated=[]):
        return util_to_f(utils, f=kveto_as_f, eliminated=eliminated)
    
    previous = 0
    previous2 = 0

    for i in range(3, 10):
        election = (
            [list(range(1, i)),
             list(range(i-1, 0, -1)), 
            ]
        )

        print(election)

        utils = ballot_to_utils(election)

        print(rule(utils))
        print(approach_two_exact(rule(utils)))


        win_distrib = run_irv(utils, count_rule=rule)

        print(f"win distrib is {win_distrib}")
        last_diff = win_distrib[-2] - previous
        one_diff = win_distrib[0] - previous2
        print(f"the difference in the first candidate's win probability is {one_diff}")
        print(f"the difference in candidate {i-2}'s win probability is {last_diff}")
        print()
        previous = win_distrib[-1]
        previous2 = win_distrib[0]

def harmonic_testing():
    k = 100
    base = (
        [[1] + list(range(5, 101)) + [3,4,2]] * int(1.5 * k) + 
        [[1] + list(range(5, 101)) + [4,3,2]] * int(1.5 * k) + 
        [[1,3] + list(range(5, 101)) + [4,2]] * k + 
        [[1,4] + list(range(5, 101)) + [3,2]] * k
        
    )

    election = base + [[2,1] + list(range(5, 101)) + [4,3]] 
    election2 = base + [[1,2] + list(range(5, 101)) + [4,3]] 

    utils = ballot_to_utils(election)

    win_distrib = run_irv(utils, count_rule=util_to_harmonic)
    print(f"win distrib is {win_distrib}")

    utils2 = ballot_to_utils(election2)
    win_distrib2 = run_irv(utils2, count_rule=util_to_harmonic)
    print(f"win distrib is {win_distrib2}")

    one_diff = win_distrib2[0] - win_distrib[0]
    print(f"the difference in the first candidate's win probability is {one_diff}")

    # scores = util_to_harmonic(utils)

    # print([x / sum(scores) for x in scores][:4])
    # print([x / sum(scores[:4]) for x in scores[:4]])
    # print(approach_two_exact(util_to_harmonic(utils))[:4])

if __name__ == "__main__":
    harmonic_testing()


# how non monotonic is it (bound for this)
# come up with a counterexample given scoring vector and m
# is borda still non insertion monotonic for higher m
# threshhold for being insertion monotonic (what are we even thresholding (maybe biggest difference between two scoring positions, or first/last, or probably something else))