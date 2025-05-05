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


def approach_two_exact(first_distrib, eliminated=[]):
    if first_distrib.count(0) > 1:
        return [1/first_distrib.count(0) if x == 0 else 0 for x in first_distrib]

    factors = []
    for i in range(len(first_distrib)):
        if first_distrib[i] == -1:
            factors.append(0)
            continue
        factor = 1
        for j in range(len(first_distrib)):
            if first_distrib[j] == -1:
                continue
            if i != j:
                factor *= first_distrib[j]
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


def run_irv(utils, elim_rule=approach_two_exact, eliminated=[]):
    if len(eliminated) == len(utils[0]) - 1:
        win_distrib = [0 if i in eliminated else 1 for i in range(len(utils[0]))]
        return win_distrib

    win_distrib = [0] * len(utils[0])
    first_distrib = util_to_first_distrib(utils, eliminated)

    elim_probs = elim_rule(first_distrib)

    # print(f"with {eliminated} eliminated, the first_distrib is {first_distrib} and this round's elimination probabilities are {elim_probs}")

    for elim in range(len(elim_probs)):
        if elim in eliminated:
            continue

        next_round = run_irv(utils, elim_rule, eliminated + [elim])
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



if __name__ == "__main__":
    main()


