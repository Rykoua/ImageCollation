import math
from pathlib import Path
import random

from PIL import Image
from os.path import join, splitext, isfile, exists
from os import listdir

from tqdm import tqdm

from .feature_matching import FeatureMatching
from .fast_feature_matching import FastFeatureMatching
from .models import get_conv4_model
import torch

from .ransac import Ransac
import numpy as np

from .score_matrix_func import normalize_score_matrix, propagate_matches, get_recall_performance
from .get_ground_truth import load_json_as_list

import re


class IllustrationMatching:
    def __init__(self, folder1, folder2, fast=True):
        self.folder1 = folder1
        self.folder2 = folder2
        if fast:
            self.feature_matching = FastFeatureMatching(get_conv4_model())
        else:
            self.feature_matching = FeatureMatching(get_conv4_model())
        self.ransac = Ransac()

    @staticmethod
    def get_file_extension(file):
        return splitext(file)[1][1:].lower()

    @staticmethod
    def natural_sort(my_list):
        def convert(text):
            if text.isdigit():
                return int(text)
            else:
                return text.lower()

        def alphanum_key(key):
            return [convert(c) for c in re.split('([0-9]+)', key)]

        return sorted(my_list, key=alphanum_key)

    @staticmethod
    def list_folder_images(folder, natural_sort=True):
        image_types = ['jpg', 'tif', 'png', 'bmp']
        paths = [join(folder, f) for f in listdir(folder) if
                 (isfile(join(folder, f)) and
                  (splitext(f)[1][1:].lower() in image_types))]
        if natural_sort:
            paths = IllustrationMatching.natural_sort(paths)
        return paths

    @staticmethod
    def get_images_list(folder):
        images_path = IllustrationMatching.list_folder_images(folder)
        return [Image.open(image_path).convert('RGB') for image_path in images_path]

    def compute_score(self, feats, feat_ref):
        match1, match2, similarity, grid_size, feature_map_size = self.feature_matching.compute_feature_matching(
            feats, feat_ref)
        score = self.ransac.get_ransac_score(match1, match2, similarity, grid_size,
                                             feature_map_size,
                                             tolerance=1/np.sqrt(50),
                                             nb_iter=100,
                                             transformation_name="affine", nb_max_iter=100)
        return score

    @staticmethod
    def load_if_exists(npy_file_path):
        if exists(npy_file_path):
            return np.load(npy_file_path)
        else:
            return None

    @staticmethod
    def get_saved_results(save_dir):
        if save_dir is None:
            return None, None, None

        score_matrix_1 = IllustrationMatching.load_if_exists(join(save_dir, "score_matrix_1.npy"))
        score_matrix_2 = IllustrationMatching.load_if_exists(join(save_dir, "score_matrix_2.npy"))
        score_matrix = IllustrationMatching.load_if_exists(join(save_dir, "score_matrix.npy"))
        normalized_matrix = IllustrationMatching.load_if_exists(join(save_dir, "normalized_matrix.npy"))
        propagation_matrix = IllustrationMatching.load_if_exists(join(save_dir, "propagation_matrix.npy"))
        return score_matrix_1, score_matrix_2, score_matrix, normalized_matrix, propagation_matrix

    @staticmethod
    def save_results(save_dir, name, score_matrix):
        if save_dir is not None:
            np.save(join(save_dir, name), score_matrix)

    def run(self, save_dir=None):
        if save_dir is not None:
            Path(save_dir).mkdir(parents=False, exist_ok=True)

        score_matrix_1, score_matrix_2, score_matrix, normalized_matrix, propagation_matrix = IllustrationMatching.get_saved_results(save_dir)

        # if computing was already completed
        if propagation_matrix is not None:
            print("Matrices have been already computed.")
            return score_matrix, normalized_matrix, propagation_matrix

        images1 = IllustrationMatching.get_images_list(self.folder1)
        images2 = IllustrationMatching.get_images_list(self.folder2)
        feature_sizes = self.feature_matching.get_sizes(20, 2)
        descriptors1 = self.feature_matching.compute_multi_scale_descriptors(images1, feature_sizes)
        descriptors2 = self.feature_matching.compute_multi_scale_descriptors(images2, feature_sizes)

        # initialize values
        if score_matrix_1 is None:
            score_matrix_1 = np.full((len(descriptors1), len(descriptors2)), np.NINF)
        if score_matrix_2 is None:
            score_matrix_2 = np.full((len(descriptors1), len(descriptors2)), np.NINF)

        total_iterations = len(descriptors1) * len(descriptors2) * 2
        progress_bar = tqdm(total=total_iterations)
        iteration = 0
        one_percent = math.ceil(total_iterations / 100)
        for i in range(len(descriptors1)):
            feats1 = self.feature_matching.get_feats_tensors(descriptors1[i])
            for j in range(len(descriptors2)):
                progress_bar.update(1)
                iteration += 1
                if score_matrix_1[i, j] != np.NINF:
                    continue
                feat2 = torch.from_numpy(descriptors2[j][len(feature_sizes) // 2])
                score_matrix_1[i, j] = self.compute_score(feats1, feat2)
                if iteration % one_percent == 0:
                    IllustrationMatching.save_results(save_dir, "score_matrix_1.npy", score_matrix_1)

        IllustrationMatching.save_results(save_dir, "score_matrix_1.npy", score_matrix_1)
        for j in range(len(descriptors2)):
            feats2 = self.feature_matching.get_feats_tensors(descriptors2[j])
            for i in range(len(descriptors1)):
                progress_bar.update(1)
                iteration += 1
                if score_matrix_2[i, j] != np.NINF:
                    continue
                feat1 = torch.from_numpy(descriptors1[i][len(feature_sizes) // 2])
                score_matrix_2[i, j] = self.compute_score(feats2, feat1)

                if iteration % one_percent == 0:
                    IllustrationMatching.save_results(save_dir, "score_matrix_2.npy", score_matrix_2)
        IllustrationMatching.save_results(save_dir, "score_matrix_2.npy", score_matrix_2)
        progress_bar.close()

        score_matrix = score_matrix_1 + score_matrix_2
        IllustrationMatching.save_results(save_dir, "score_matrix.npy", score_matrix)
        normalized_matrix = normalize_score_matrix(score_matrix, )
        IllustrationMatching.save_results(save_dir, "normalized_matrix.npy", normalized_matrix)
        print("Information propagation...")
        propagation_matrix = propagate_matches(normalized_matrix, std=5, alpha=0.25)
        IllustrationMatching.save_results(save_dir, "propagation_matrix.npy", propagation_matrix)
        return score_matrix, normalized_matrix, propagation_matrix


if __name__ == "__main__":
    ms1_name = "P2"
    ms2_name = "P3"

    manuscripts_location = "D:/Stage/tmp_manuscripts/paper_manuscripts/{}/illustration"
    M1_path = manuscripts_location.format(ms1_name)
    M2_path = manuscripts_location.format(ms2_name)

    # compute scores
    illustration_matching = IllustrationMatching(M1_path, M2_path, fast=True)

    score_matrix, normalized_matrix, propagation_matrix = illustration_matching.run(save_dir="../saved_runs/{}{}222".format(ms1_name, ms2_name))

    # normalize matrix
    # score_matrix = s1 + s2
    # normalized_matrix = normalize_score_matrix(score_matrix)
    # print("Information propagation...")
    # matrix_after_propagation = propagate_matches(normalized_matrix, std=5, alpha=0.25)
    #
    # # measure performance
    true_matches = load_json_as_list("ground_truth/{}-{}.json".format(ms1_name, ms2_name))
    print("Score matrix recall: {:.1f}%".format(
        get_recall_performance(score_matrix, true_matches) * 100))
    print("Normalized matrix recall: {:.1f}%".format(
        get_recall_performance(normalized_matrix, true_matches) * 100))
    print("Matrix info. propagation recall: {:.1f}%".format(
        get_recall_performance(propagation_matrix, true_matches) * 100))
    exit()
    # parser = argparse.ArgumentParser(description='Find same illustrations in two different manuscripts')
    # parser.add_argument('-d1', '--dir1', nargs='?', type=str, required=True, help='First directory path')
    # parser.add_argument('-d2', '--dir2', nargs='?', type=str, required=True, help='Second directory path')
    # parser.add_argument('-r', '--results_dir', nargs='?', type=str, required=True, help='Results directory')
    # args = parser.parse_args()
    #
    # illustration_matcher(args.manuscript_1, args.manuscript_2, args.results_folder)
