import numpy as np

def weighted_tournament(ballots):
    m = len(ballots[0])  # Number of alternatives (candidates)
    transition_matrix = np.zeros((m, m), dtype=float)  # Initialize the pairwise preference matrix
    for ballot in ballots:  # Fill in the matrix
        for i in range(m):
            for j in range(i+1, m):
                a, b = ballot[i], ballot[j]
                transition_matrix[a-1][b-1] += 1

    transition_matrix += np.identity(m) * 1e-10  # Add self-loops to avoid zero probabilities

    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)  # Normalize the matrix
    return transition_matrix

def converge_to_distrib(T):
    eig = np.linalg.eig(T.T)
    index = np.argmin(np.abs(eig[0] - 1))  # Get the index of the eigenvalue = 1
    eigenvec = eig[1][:, index]  # Get the eigenvector
    return eigenvec / np.sum(eigenvec)

def run_election(ballots, losers=[]):
    win_distrib = np.zeros(len(ballots[0]), dtype=float)  # Initialize the win distribution

    if len(losers) == len(ballots[0]) - 1:  # If all candidates are eliminated, return the distribution
        winner = set(ballots[0]) - set(losers)
        win_distrib[list(winner)[0] - 1] = 1.0
        return win_distrib

    sim_ballots = np.copy(ballots)  # Copy the ballots for simulation
    for i in range(len(sim_ballots)):
        for loser in losers:
            sim_ballots[i] = [loser] + sim_ballots[i][sim_ballots[i] != loser].tolist()  # Move eliminated candidates to the front so they are do not effect the graph

    G = weighted_tournament(sim_ballots)
    distrib = converge_to_distrib(G)
    for i in range(len(distrib)):
        win_distrib += distrib[i] * run_election(sim_ballots, losers + [i+1])  # Recursive call to run_election

    return win_distrib  # Return distribution of wins
    


def main():
    ballots = [
        [1, 2, 3, 4, 5],
        [1, 3, 2, 5, 4],
    ]
    G = weighted_tournament(ballots)
    print(G)

    print(converge_to_distrib(G))

    print(run_election(ballots))

if __name__ == "__main__":
    main()