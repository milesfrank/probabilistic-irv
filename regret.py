from itertools import permutations

ballots = list(permutations([0, 1, 2, 3]))

# print(ballots)

# print(len(ballots))

# ballots = [
#     [0,1,2,3],
#     [1,2,3,0],
#     [2,3,0,1],
#     [3,0,1,2],
# ]

# ballots = [
#     [0, 1, 2, 3],
#     [1, 2, 0, 3],
#     [2, 0, 1, 3],
# ]

# possible_utilities = [
#     [0, 0, 0],
#     [1, 0, 0],
#     [0, 0, -1],
# ]

possible_utilities = [
    [0,0,0,0],
    [1,0,0,0],
    # [0.5,0.5,0,0],
    [0,0,0,-1],
    # [0,0,-0.5,-0.5]
]

attempts = {}

n = len(ballots)
m = len(ballots[0])

max_regret = 0
profiles = []

def search(ballot_num, utility_profiles):
    global max_regret, profiles
    if ballot_num == n:
        utilities = [sum(utility_profiles[i][j] for i in range(n)) for j in range(m)]
        if max(utilities) != utilities[0]:
            return
        regret = (utilities[0] - (sum(utilities))/m )/n
        if regret < max_regret:
            return
        elif regret > max_regret:
            max_regret = regret
            profiles.clear()
        profiles.append(utility_profiles)
        return
    for utility in possible_utilities:
        profile = [0] * m
        for i, alt in enumerate(ballots[ballot_num]):
            profile[alt] = utility[i]

        if profile[0] < 0:
            continue
        search(ballot_num + 1, utility_profiles + [tuple(profile)])

search(0, [])

regrets = {}

# print(attempts)

print("Max regret:", max_regret)
print("Profiles with max regret:")
for profile in profiles:
    print(profile)
    utilities = [sum(profile[i][j] for i in range(n)) for j in range(m)]

# for profile in attempts:
#     utilities = attempts[profile]
#     regret = (max(utilities) - (sum(utilities))/m )/n
#     regrets[profile] = regret

# worst_regret = max(regrets.values())
# worst_cases = [profile for profile, regret in regrets.items() if regret == worst_regret]
# print("Worst regret:", worst_regret)
# print("Worst cases:")
# for case in worst_cases:
#     print(case)
#     print("Utilities:", attempts[case])
