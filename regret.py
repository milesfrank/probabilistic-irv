from itertools import permutations

ballots = list(permutations([0, 1, 2, 3]))

possible_utilities = [
    # [0,0,0,0],
    # [1,0,0,0],
    [0.5,0.5,0,0],
    [0,0,0,-1],
    [0,0,-0.5,-0.5]
]

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
    
    if ballots[ballot_num][0] == 0:
        search(ballot_num + 1, utility_profiles + [(1, 0, 0, 0)])
        return
    elif ballots[ballot_num][-1] == 0:
        search(ballot_num + 1, utility_profiles + [(0, 0, 0, 0)])
        return

    for utility in possible_utilities:
        profile = [0] * m
        
        for i, alt in enumerate(ballots[ballot_num]):
            profile[alt] = utility[i]

        search(ballot_num + 1, utility_profiles + [tuple(profile)])

search(0, [])

regrets = {}

# print(attempts)

print("Max regret:", max_regret)
print("Profiles with max regret:")
for profile in profiles[:5]:
    print(profile)
    utilities = [sum(profile[i][j] for i in range(n)) for j in range(m)]
    print("Utilities:", utilities)