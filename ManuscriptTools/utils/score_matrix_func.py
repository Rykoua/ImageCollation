import numpy as np


def get_best_matches_axis(score_matrix, axis=0):
    j_max = np.argmax(score_matrix, axis=1 - axis)
    matches = list()
    for i in range(score_matrix.shape[axis]):
        if axis == 0:
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
    best_couples_coordinates = np.where(m1 * m2 == 1)
    return list(zip(best_couples_coordinates[0], best_couples_coordinates[1]))


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


def propagate_matches(score_matrix, matches, std, alpha):
    new_score_matrix = score_matrix.copy()
    for match in matches:
        x, y = match
        new_score_matrix *= create_gaussian_matrix(score_matrix.shape[0], score_matrix.shape[1], x, y, std,
                                                   alpha)
    return new_score_matrix


def propagate_information(score_matrix, std, alpha):
    best_matches = get_consistent_matches(score_matrix)
    return propagate_matches(score_matrix, best_matches, std, alpha)


def get_ordered_scores_matrix(smatrix, axis=1):
    ordered_scores = np.flip(np.argsort(smatrix, axis=axis), axis=axis)
    return ordered_scores
