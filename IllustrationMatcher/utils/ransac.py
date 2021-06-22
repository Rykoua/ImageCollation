import random

import torch
import numpy as np


class Ransac:
    def __init__(self):
        pass

    @staticmethod
    def get_transformation_info(transformation_name):
        transformations_info = {"affine": {
            "transformation_function": Ransac.affine,
            "n": 3},
            "translation": {
                "transformation_function": Ransac.translation,
                "n": 1},
            "translation_scaling": {
                "transformation_function": Ransac.translation_scaling,
                "n": 2},
            "translation_scaling_2d": {
                "transformation_function": Ransac.translation_scaling_2d,
                "n": 2},
        }
        return (transformations_info[transformation_name]["n"],
                transformations_info[transformation_name]["transformation_function"])

    @staticmethod
    def solve_linear_regression(x, y):
        a = torch.matmul(x.permute(0, 2, 1), x).pinverse()
        b = torch.matmul(x.permute(0, 2, 1), y)
        w = torch.matmul(a, b)
        return w

    @staticmethod
    def solve_linear_regression_(x, y):
        # return torch.solve(y, x)[0]
        return torch.linalg.solve(y, x)[0]

    @staticmethod
    def affine(x, y):
        x = torch.cat((x, torch.ones((x.shape[0], x.shape[1], 1), device=x.device)), dim=2)
        return Ransac.solve_linear_regression(x, y)

    @staticmethod
    def translation(x, y):
        t = y - x
        return torch.cat((torch.eye(2).unsqueeze(0).repeat(t.shape[0], 1, 1), t.unsqueeze(1)), dim=1)

    @staticmethod
    def translation_scaling(x, y):
        x = torch.cat((x[:, :, 0].unsqueeze(2), x[:, :, 1].unsqueeze(2)), dim=-1)
        x = x.view(x.shape[0], 2 * x.shape[1], 1)
        v = torch.remainder(torch.arange(0, x.shape[1]), 2).repeat(x.shape[0], 1).unsqueeze(2).type(
            torch.float)
        x = torch.cat((x, 1 - v, v), dim=-1)
        y = y.view(y.shape[0], -1).unsqueeze(2)
        w = Ransac.solve_linear_regression(x, y)
        d = w[:, 0:1, :] * torch.eye(2).unsqueeze(0).repeat(x.shape[0], 1, 1)
        t = w[:, 1:, 0].unsqueeze(1)
        w = torch.cat((d, t), dim=1)

        return w

    @staticmethod
    def translation_scaling_2d(x, y):
        u = torch.zeros(x.shape[1], device=x.device).repeat(x.shape[0], 1).unsqueeze(2).repeat(1, 1, 2)
        x = torch.cat((x[:, :, 0].unsqueeze(2), u, x[:, :, 1].unsqueeze(2)), dim=-1)
        x = x.view(x.shape[0], 2 * x.shape[1], 2)
        v = torch.remainder(torch.arange(0, x.shape[1]), 2).repeat(x.shape[0], 1).unsqueeze(2).type(
            torch.float)
        x = torch.cat((x, 1 - v, v), dim=-1)
        y = y.view(y.shape[0], -1).unsqueeze(2)
        w = Ransac.solve_linear_regression(x, y)
        d = torch.diag_embed(w[:, :2, 0])
        t = w[:, 2:, 0].unsqueeze(1)
        w = torch.cat((d, t), dim=1)
        return w

    @staticmethod
    def compute_error(x, y, transformation_matrices):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = torch.cat((x, torch.ones((x.shape[0], x.shape[1], 1), device=x.device)), dim=2)
        y_pred = torch.matmul(x, transformation_matrices)
        return torch.sum((y - y_pred) ** 2, dim=2) ** 0.5

    @staticmethod
    def compute_score(match2, error, similarity, score_tolerance):
        score = similarity * torch.exp(-error ** 2 / (2 * score_tolerance ** 2))
        return Ransac.compute_final_score(match2.cuda(), score.cuda())

    @staticmethod
    def get_ransac_results(match1, match2, similarity, score_tolerance, samples, transformation_function):
        x = match1[samples]
        y = match2[samples]
        transformation_matrices = transformation_function(x, y)
        error = Ransac.compute_error(match1, match2, transformation_matrices)
        return transformation_matrices, Ransac.compute_score(match2, error, similarity,
                                                             score_tolerance)

    @staticmethod
    def generate_samples(nb_match, nb_iter, nb_sample, device='cpu'):
        if nb_sample == 3:
            samples = torch.randint(nb_match, (nb_iter, 3), device=device)
            conditions = torch.stack([
                samples[:, 0] == samples[:, 1],
                samples[:, 0] == samples[:, 2],
                samples[:, 1] == samples[:, 2]
            ], dim=1)
        elif nb_sample == 2:
            samples = torch.randint(nb_match, (nb_iter, 2), device=device)
            conditions = torch.stack([
                samples[:, 0] == samples[:, 1]
            ], dim=1)
        elif nb_sample == 1:
            samples = torch.randint(nb_match, (nb_iter, 1), device=device)
            conditions = torch.full((nb_iter, 1), True)
        else:
            samples = torch.randint(nb_match, (nb_iter, 2), device=device)
            conditions = torch.stack([
                samples[:, 0] == samples[:, 1]
            ], dim=1)
        duplicated_samples = torch.any(conditions, dim=1)
        unique_samples = samples[~duplicated_samples]
        return unique_samples

    @staticmethod
    # change nb iter 100 -> 1000
    def get_ransac_score(match1, match2, similarity, grid_size, feature_map_size, tolerance=1 / np.sqrt(50),
                         nb_iter=100,
                         transformation_name="affine", nb_max_iter=100):
        nb_match = len(match1)
        match1 = match1.cuda()
        match2 = match2.cuda()
        similarity = similarity.cuda()
        grid_size = grid_size.cuda()

        n, transformation_function = Ransac.get_transformation_info(transformation_name)
        unique_samples = Ransac.generate_samples(nb_match, nb_iter, n, device=match1.device)

        nb_loop = len(unique_samples) // nb_max_iter
        best_transformation, best_score = None, 0

        for i in range(nb_loop):

            transformation_matrices, scores = Ransac.get_ransac_results(match1, match2, similarity, tolerance,
                                                                        unique_samples.narrow(0,
                                                                                              i * nb_max_iter,
                                                                                              nb_max_iter),
                                                                        transformation_function)

            best = torch.argmax(scores)

            if scores[best] > best_score:
                best_transformation = transformation_matrices[best]
                best_score = scores[best]

        if len(unique_samples) - nb_loop * nb_max_iter > 0:
            transformation_matrices, scores = Ransac.get_ransac_results(match1, match2,
                                                                        similarity, tolerance,
                                                                        unique_samples.narrow(0,
                                                                                              nb_loop * nb_max_iter,
                                                                                              len(
                                                                                                  unique_samples) - nb_loop * nb_max_iter),
                                                                        transformation_function)

            best = torch.argmax(scores)
            if scores[best] > best_score:
                best_transformation = transformation_matrices[best]
                best_score = scores[best]

        error = Ransac.compute_error(match1, match2,
                                     best_transformation).squeeze(0)

        total_score = similarity * torch.exp(-error ** 2 / tolerance ** 2)

        # return Ransac.filter_best_matches(match1, match2, grid_size, feature_map_size, total_score,
        #                                   similarity)
        # return Ransac.compute_score(match2, error, similarity, tolerance)
        return Ransac.compute_final_score(match2, total_score, feature_map_size)
        # return best_score.item()/feature_map_size

    @staticmethod
    def get_idx_to_keep(matches, scores):
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        unique_matches, idx = torch.unique(matches, return_inverse=True, dim=0)
        M = -2 * torch.ones(scores.shape[0], unique_matches.shape[0], matches.shape[0],
                            device=matches.device).type(torch.float)
        M[:, idx, torch.arange(0, matches.shape[0])] = scores.type(torch.float)
        idx_to_keep = torch.argmax(M, dim=-1)
        return idx_to_keep

    @staticmethod
    def compute_final_score(match2, total_score, feature_map_size=None):
        idx_to_keep = Ransac.get_idx_to_keep(match2, total_score)
        if total_score.dim() == 1:
            total_score = total_score.unsqueeze(0)
        final_score = torch.sum(torch.gather(total_score, 1, idx_to_keep), dim=-1)

        if final_score.numel() == 1:
            final_score = final_score.item()

        if feature_map_size is not None:
            final_score /= feature_map_size

        return final_score
