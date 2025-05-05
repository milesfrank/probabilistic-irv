import numpy as np

def weighted_tournament(ballots):
    m = len(ballots[0])  # Number of alternatives (candidates)
    transition_matrix = np.zeros((m, m), dtype=float)  # Initialize the pairwise preference matrix
    for ballot in ballots:  # Fill in the matrix
        for i in range(m):
            for j in range(i+1, m):
                a, b = ballot[i], ballot[j]
                transition_matrix[a-1][b-1] += 1

    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)  # Normalize the matrix
    return transition_matrix

def converge_to_distrib(T):
    eig = np.linalg.eig(T.T)
    index = np.argmin(np.abs(eig[0] - 1))  # Get the index of the eigenvalue = 1
    eigenvec = eig[1][:, index]  # Get the eigenvector
    return eigenvec / np.sum(eigenvec)


def main():
    ballots = [
        [1, 2, 3, 4, 5],
        [1, 3, 2, 5, 4],
    ]
    G = weighted_tournament(ballots)
    print(G)

    print(converge_to_distrib(G))

if __name__ == "__main__":
    main()