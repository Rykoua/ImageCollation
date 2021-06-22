import numpy as np
from tqdm import tqdm


def get_best_matches_axis(score_matrix, axis=0):
    j_max = np.argmax(score_matrix, axis=axis)
    matches = list()
    for i in range(score_matrix.shape[1-axis]):
        if axis == 1:
            matches.append((i, j_max[i]))
        else:
            matches.append((j_max[i], i))
    return matches


def get_recall_at_1(score_matrix, true_matches, axis=0):
    matches = get_best_matches_axis(score_matrix, axis=axis)
    nb_match_found = len([match for match in matches if match in true_matches])
    return nb_match_found/len(true_matches)


def get_recall_performance(score_matrix, true_matches):
    recall_0 = get_recall_at_1(score_matrix, true_matches, axis=0)
    recall_1 = get_recall_at_1(score_matrix, true_matches, axis=1)
    return (recall_0 + recall_1) / 2


def max_matrix(matrix, axis):
    if axis == 0:
        return np.max(matrix, axis=0).reshape(1, matrix.shape[1]).repeat(matrix.shape[0], axis=0)
    else:
        return np.max(matrix, axis=1).reshape(matrix.shape[0], 1).repeat(matrix.shape[1], axis=1)


def normalize_score_matrix(score_matrix):
    sc = score_matrix/max_matrix(score_matrix, axis=0)
    sl = score_matrix/max_matrix(score_matrix, axis=1)
    return sl+sc


def get_consistent_matches(score_matrix):
    m1 = np.zeros_like(score_matrix)
    m2 = np.zeros_like(score_matrix)
    m1[np.arange(score_matrix.shape[0]), np.argmax(score_matrix, axis=1)] = 1
    m2[np.argmax(score_matrix, axis=0), np.arange(score_matrix.shape[1])] = 1
    consistent_matches_coordinates = np.where(m1 * m2 == 1)
    return list(zip(consistent_matches_coordinates[0], consistent_matches_coordinates[1]))


def gaussian_func(d, std, alpha):
    var = std**2
    if var != 0:
        return 1 + alpha * np.exp(-d ** 2 / (2 * var))
    else:
        gaussian_value = np.ones_like(d)
        gaussian_value[np.where(d == 0)] = 2
        return gaussian_value


def create_gaussian_matrix(s1, s2, x, y, std, alpha):
    def function_to_apply(i, j):
        d = np.sqrt((x - i) ** 2 + (y - j) ** 2)
        return gaussian_func(d, std, alpha)
    return np.fromfunction(function_to_apply, (s1, s2))


def propagate_matches(score_matrix, std, alpha):
    consistent_matches = get_consistent_matches(score_matrix)
    s1, s2 = score_matrix.shape
    factor_matrix = np.ones((s1, s2))
    for match in tqdm(consistent_matches):
        x, y = match
        factor_matrix *= create_gaussian_matrix(s1, s2, x, y, std, alpha)
    return factor_matrix * score_matrix


def get_ordered_scores_matrix(score_matrix, axis=1):
    ordered_scores = np.flip(np.argsort(score_matrix, axis=axis), axis=axis)
    return ordered_scores


def get_nearest_neighbors_dict(score_matrix, number_nn=20):
    ordered_matrix = get_ordered_scores_matrix(score_matrix)[:number_nn]
    nearest_neighbours_dict = dict()
    for i in range(ordered_matrix.shape[0]):
        nearest_neighbours_dict[i] = list(ordered_matrix[i])
    return nearest_neighbours_dict
