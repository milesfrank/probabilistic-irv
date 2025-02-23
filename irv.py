import random
import copy



def test_sample(distrib, samples):
    counts = [0] * len(distrib)
    choik = range(len(distrib))
    for i in range(samples):
        counts[random.choices(choik, weights=distrib, k=1)[0]] += 1
    return [c/samples for c in counts]



def approach_one(distrib, samples):
    counts = [0] * len(distrib)
    choik = list(range(len(distrib)))
    for i in range(samples):
        choik = list(range(len(distrib)))
        distrib2 = [0]*len(distrib)
        for j in range(len(distrib)):
            distrib2[j] = distrib[j]
        while len(choik) > 1:
            c1 = random.choices(choik, weights=distrib2, k=1)[0]
            ind = choik.index(c1)
            choik.pop(ind)
            distrib2.pop(ind)
        counts[choik[0]] += 1
    return [c/samples for c in counts]



def approach_two(distrib, samples):
    counts = [0] * len(distrib)
    choik = list(range(len(distrib)))
    for i in range(samples):
        encountered = []
        while len(encountered) != len(distrib) - 1:
            c1 = random.choices(choik, weights=distrib, k=1)[0]
            if c1 in encountered:
                encountered = []
                continue
            else:
                encountered.append(c1)
        
        missing = 0
        while(missing in encountered):
            missing += 1

        counts[missing] += 1
    return [c/samples for c in counts]




def approach_two_exact(distrib):
    ret = [0] * len(distrib)
    for i in range(len(ret)):
        product = 1
        for j in range(len(ret)):
            if j != i:
                product *= distrib[j]
        ret[i] = product
    
    return [r/sum(ret) for r in ret]


def fptp(ballot, cand_id):
    scores = [0]*len(cand_id.keys())
    scores[cand_id[ballot[0]]] += 1
    return scores


def harmonic(ballot, cand_id):
    scores = [0]*len(cand_id.keys())
    for i in range(len(ballot)):
        scores[cand_id[ballot[i]]] += 1/(1+i)
    return scores


def least_then_lexicographic(counts, cand_id, cand_list):
    mi = min(counts)

    bottom = []

    for i in range(len(counts)):
        if counts[i] == mi:
            bottom.append(cand_list[i])
    
    loser = max(bottom)
    return [loser]




def least_then_random(counts, cand_id, cand_list):
    mi = min(counts)

    bottom = []

    for i in range(len(counts)):
        if counts[i] == mi:
            bottom.append(cand_list[i])
    
    
    return [bottom[int(random.random()*len(bottom))]]


def inverse_probabilistic(counts, cand_id, cand_list):
    counts = copy.deepcopy(counts)
    counts = [c/sum(counts) for c in counts]
    distrib = approach_two_exact(counts)
    return random.choices(cand_list, weights=distrib, k=1)



def irv(ballots, count_rule=fptp, loser_rule=least_then_lexicographic):
    ballots = copy.deepcopy(ballots)
    if len(ballots) == 0: #invalid
        return
    

     # find all cands still in
    cands_in = {}
    for ballot in ballots:
        for i in ballot:
            cands_in[i] = True
        
    cand_list = []
    for v in cands_in.keys():
        cand_list.append(v)
        
    #print(str(cand_list))

    cand_id = {}
    output = []
    for i in range(len(cand_list)):
        cand_id[cand_list[i]] = i
    
    while len(cand_list) > 0:
        #for ballot in ballots:
            #print(str(ballot))
        #print()
        # count scores according to the rule
        counts = [0] * len(cand_list)
        for ballot in ballots:
            score = count_rule(ballot, cand_id)
            for i in range(len(score)):
                counts[i] += score[i]
            
        # determine loser according to the rule

        losers = loser_rule(counts, cand_id, cand_list)

        # update cand_list and cand_id

        for loser in losers:
            cand_list.remove(loser)
        
        cand_id = {}

        for i in range(len(cand_list)):
            cand_id[cand_list[i]] = i
        
        # update all of the ballots

        i = 0
        while i < len(ballots):
            for loser in losers:
                if loser in ballots[i]:
                    ballots[i].remove(loser)
            if len(ballots[i]) == 0:
                ballots.pop(i)
            else:
                i += 1
        
        random.shuffle(losers)

        for loser in losers:
            output.append(loser)
        
    output.reverse()

    return output

def filter_indices(lst, ignore_indices):
    return [val for idx, val in enumerate(lst) if idx not in ignore_indices]


def approval_irv(ballots, loser_rule=least_then_lexicographic):
    output = []
    ballots = copy.deepcopy(ballots)

    if len(ballots) == 0: #invalid
        return


    cand_list = list(range(1, len(ballots[0]) + 1))
        
    #print(str(cand_list))

    cand_id = {}
    output = []
    for i in range(len(cand_list)):
        cand_id[cand_list[i]] = i

    for i in range(len(ballots)):
        if sum(ballots[i]) > 0:
            ballots[i] = [s/sum(ballots[i]) for s in ballots[i]]

    while(len(cand_list) > 0):
        counts = [0] * len(cand_list)
        for ballot in ballots:
            for i in range(len(ballot)):
                counts[i] += ballot[i]
            
        losers = loser_rule(counts, cand_id, cand_list)
        loser_inds = [cand_id[loser] for loser in losers]
        for i in range(len(ballots)):
            ballots[i] = filter_indices(ballots[i], loser_inds)
            if sum(ballots[i]) > 0:
                ballots[i] = [s/sum(ballots[i]) for s in ballots[i]]
        for loser in losers:
            cand_list.remove(loser)

        cand_id = {}

        for i in range(len(cand_list)):
            cand_id[cand_list[i]] = i        

        for loser in losers:
            output.append(loser)


    output.reverse()
    return output
    pass


def ahead_in_ballot(c1, c2, ballot):
    for c in ballot:
        if c == c1:
            return 1
        elif c == c2:
            return -1
    return 0

def ahead_in_ballots(c1, c2, ballots):
    c1_count = 0
    c2_count = 0
    for ballot in ballots:
        v = ahead_in_ballot(c1, c2, ballot)
        if v == 1:
            c1_count += 1
        elif v == -1:
            c2_count += 1
    #print(str(c1) + " : " + str(c1_count))
    #print(str(c2) + " : " + str(c2_count) + "\n\n")
    if c1_count > c2_count:
        return 1
    elif c2_count > c1_count:
        return -1
    return 0


def find_condorcet_winner(ballots):
    if len(ballots) == 0: #invalid
        return
    
     # find all cands still in
    cands_in = {}
    for ballot in ballots:
        for i in ballot:
            cands_in[i] = True
        
    cand_list = []
    for v in cands_in.keys():
        cand_list.append(v)

    for c1 in cand_list:
        condorcet = True
        for c2 in cand_list:
            if c1 == c2:
                continue
            if ahead_in_ballots(c1, c2, ballots) != 1:
                condorcet = False
                break
        if condorcet:
            return c1
    return None
            
    


    #print(str(cand_list))




def duplicate(ballots, n):
    new_ballots = []

    for i in range(n):
        for ballot in ballots:
            new_ballots.append(copy.deepcopy(ballot))

    return new_ballots 


def recursive_construction(level):
    pass

    if level < 1:
        return

    if level == 1:
        return [[2]]

    else:
        ballots = recursive_construction(level - 1)
        ballots = duplicate(ballots, level)
        for i in range(len(ballots)//level * (level-1), len(ballots)):
            nballot = []
            nballot.append(level + 1)
            for c in ballots[i]:
                nballot.append(c)
            ballots[i] = nballot
    

    #print(str(ballots))
    return ballots



def fill_ballot(ballot, max_cand):
    for i in range(1, max_cand + 1):
        if i not in ballot:
            if i != 1:
                ballot.insert(1, i)
            else:
                ballot.append(1)


def constr(level, a_mult):
    ballots = recursive_construction(level)
    for i in range(int(len(ballots) * a_mult)):
        ballots.append([1])
    
    for ballot in ballots:
        fill_ballot(ballot, level + 1)
    return ballots


def generate_all_perms(ranking):

    if len(ranking) == 0:
        return []

    if len(ranking) == 1:
        return [ranking]

    if len(ranking) == 2:
        return [[ranking[0], ranking[1]], [ranking[1], ranking[0]]]
    
    rankings = []

    for i in range(len(ranking)):
        subranking = []
        for j in range(len(ranking)):
            if i != j:
                subranking.append(ranking[j])
        subrankings = generate_all_perms(subranking)
        for sub in subrankings:
            rankings.append([ranking[i]] + sub)
    return rankings

def generate_random_permutations(k, n):
    """
    Generate n random permutations of the list [1, 2, ..., k].

    :param n: Number of permutations to generate
    :param k: The range of numbers to permute
    :return: List of n random permutations
    """
    base_list = list(range(1, k + 1))
    permutations = [random.sample(base_list, k) for _ in range(n)]
    return permutations


def main():

    app_election = [
        [0.6, 0.4, 0],
        [0.6, 0.4, 0],
        [0.7, 0.3, 0],
        [0, 0.1, 0.9]
    ]

    print(str(approval_irv(app_election)))
    #return

    election = [
        [1, 3, 2],
        [1, 3, 2],
        [2, 3, 1],
        [2, 3, 1],
        [3, 2, 1],
        [3, 1, 2],

    ]

    print(irv(election))

    for i in range(10):
        print(irv(election, loser_rule=least_then_random))
    

    election2 = [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],

        [2, 1, 3],
        [2, 1, 3],
        [2, 1, 3],
        [2, 1, 3],
        [2, 1, 3],

        [2, 3, 1],
        [2, 3, 1],
        [2, 3, 1],

        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
    ]



    


    winners = [0] * 3
    for i in range(1000000):
        winners[irv(election2, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))


def main2():
    election = [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        

        [2, 3, 1],
        [2, 3, 1],
        [2, 3, 1],
        [2, 3, 1],
        [2, 3, 1],
        [2, 3, 1],
        [2, 3, 1],
        [2, 3, 1],
        
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],

        [3, 1, 2],
        [3, 1, 2],
        [3, 1, 2],
        [3, 1, 2],
        [3, 1, 2],
        [3, 1, 2],
        [3, 1, 2],
        [3, 1, 2],
        [3, 1, 2],
    ]

    winners = [0] * 3
    for i in range(500000):
        winners[irv(election, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))


def main3():
    n = 10
    j = 4
    k = 5
    election = []

    for i in range(n - j - 1):
        election.append([1, 2, 3])
    for i in range(n):
        election.append([2, 1, 3])
    for i in range(j):
        election.append([1, 3, 2])
    for i in range(k):
        election.append([3, 1, 2])
    for i in range(k):
        election.append([3, 2, 1])
    
    print(str(find_condorcet_winner(election)))

    winners = [0] * 3
    for i in range(100000):
        winners[irv(election, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))

    # demonstrates condorcet winner is not necessarily most likely win





def main4():
    election = constr(5, 5)
    for ballot in election:
        print(str(ballot))

    print("\n")
    #for ballot in election:
    #    if ballot[0] == 3:
    #        print(str(ballot))
    
    parity = False
    for i in range(len(election)):
        if election[i] == [3, 6, 5, 4, 2, 1]:
            if parity:
                election[i] = [2, 3, 6, 5, 4, 1]
                parity = False
            else:
                parity = True

    for i in range(len(election)):
        if election[i] == [3, 6, 5, 4, 2, 1]:
            print(str(election[i]))

    winners = [0] * 6
    for i in range(20000):
        winners[irv(election, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))


    pass


def main5():
    election = [
        [1,2,3],
        [1,2,3],
        [1,3,2],
        [2,1,3],
        [2,3,1],
        [3,1,2],
        [3,2,1],
        [3,2,1]
    ]
    election = generate_all_perms([1,2,3,4])
    election.append([1,2,3,4])
    election.append([1,2,4,3])
    election.append([4,3,2,1])

    election = [
        [3,2,1],
        [3,2,1],
        [3,2,1],
        [3,2,1],
        [3,2,1],
        [3,2,1],
        [3,2,1],
        [3,2,1],
        [3,2,1],
        [3,2,1],
        [2,1,3]
    ]

    winners = [0] * len(election[0]) # only works if same length
    for i in range(1000000):
        winners[irv(election, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))


    election.append([1,2,3])

    winners = [0] * len(election[0]) # only works if same length
    for i in range(1000000):
        winners[irv(election, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))



def main6():
    """
    compare: with vs without 3, see if adding a cand can make 1 do better
    """


    election = [
        [1,2,3],
        [1,2,3],
        [1,3,2],
        [1,3,2],
        [3,2,1],
        [3,2,1],
        [2,1,3],
        [2,1,3],
    ]

    winners = [0] * len(election[0]) # only works if same length
    for i in range(1000000):
        winners[irv(election, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))


def main7():
    """
    compare: with vs without 3, see if adding a cand can make 1 do better
    """


    election = generate_random_permutations(4, 10)
    print(election)
    election2 = []
    election3 = []
    for b in election:
        election2.append(b)
        election3.append(b)

    election2.append([1,2,3,4])
    election3.append([2,1,3,4])

    winners = [0] * len(election2[0]) # only works if same length
    for i in range(10000000):
        winners[irv(election2, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))


    winners = [0] * len(election3[0]) # only works if same length
    for i in range(10000000):
        winners[irv(election3, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))



def main8():
    """
    compare: even vs uneven spread, which is worse for 1
    """


    election = [
        [1,2,3,4],
        [1,2,3,4],
        [2,3,4,1],
        [2,4,3,1],
        [3,2,4,1],
        [3,4,2,1],
        [4,2,3,1],
        [4,3,2,1],
    ]

    election2 = [
        [1,2,3,4],
        [1,2,3,4],
        [2,3,4,1],
        [2,3,4,1],
        [3,2,4,1],
        [3,2,4,1],
        [4,3,2,1],
        [4,3,2,1],
    ]

    winners = [0] * len(election[0]) # only works if same length
    for i in range(1000000):
        winners[irv(election, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))

    winners = [0] * len(election[0]) # only works if same length
    for i in range(1000000):
        winners[irv(election, loser_rule=inverse_probabilistic)[0] - 1] += 1
    
    winners = [w / sum(winners) for w in winners]

    print(str(winners))




if __name__ == "__main__":
    main8()
        
        




#irv([[1, 2, 4], [4, 2, 1], [4, 3], [3, 4, 6, 9]])













#print(str(approach_one([4, 3, 2, 1], 10000000)))
#print(str(approach_two_exact([4, 3, 2, 1])))
#print(str(approach_two_exact(approach_two_exact([4, 3, 2, 1]))))
#approach_one([3,2,1], 10000000)
#print(str(approach_one(approach_one([3,2,1], 30000000), 10000000)))