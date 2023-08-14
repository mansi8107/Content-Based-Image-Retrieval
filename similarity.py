import numpy as np


def euclidean(fv1,
              fv2):  # takes 2 feature vectors and return the euclidean distance between them # both the vectors are
    # of same length
    distance = np.float64(0)

    for i in range(len(fv1)):
        distance += np.square(fv1[i] - fv2[i])

    return np.sqrt(distance)


def cosine(fv1, fv2):
    distance = np.dot(fv1, fv2) / (np.linalg.norm(fv1) * np.linalg.norm(fv2))

    return distance
