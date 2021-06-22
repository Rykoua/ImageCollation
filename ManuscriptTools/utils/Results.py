import numpy as np
import pathlib

from ManuscriptTools.utils.TableHTML import TableHTML
from ManuscriptTools.utils.score_matrix_func import get_best_matches_axis, get_consistent_matches, \
    get_ordered_scores_matrix
from ManuscriptTools.utils.DirTools import DirTools
import os
from os.path import join


class Results:
    def __init__(self, dir1_path, dir2_path, score_matrix, with_annotation=False):
        self.dir1_path = dir1_path
        self.dir2_path = dir2_path
        self.score_matrix = score_matrix
        self.images = self.get_images()
        self.with_annotation = with_annotation

    def get_images(self):
        return {self.dir1_path: DirTools.list_folder_images(self.dir1_path),
                self.dir2_path: DirTools.list_folder_images(self.dir2_path)}

    def get_image_info(self, dir_path, idx):
        illustration_path = self.images[dir_path][idx]
        if self.with_annotation and DirTools.annotation_exists(dir_path):
            return [DirTools.get_annotation_info(illustration_path), (idx, illustration_path)]
        else:
            return idx, illustration_path

    def save_consistent_match(self, html_file_path, reverse_dirs=False):
        if reverse_dirs:
            dir1_path, dir2_path = self.dir2_path, self.dir1_path
            matches = get_consistent_matches(self.score_matrix.T)
        else:
            dir1_path, dir2_path = self.dir1_path, self.dir2_path
            matches = get_consistent_matches(self.score_matrix)

        table = TableHTML('Consistent matches<br/><br/>Dir1: {}<br/>Dir2: {}'.format(dir1_path, dir2_path))
        table.add_head(["dir1", "dir2"])
        for match in matches:
            table.add_row([self.get_image_info(dir1_path, match[0]),
                           self.get_image_info(dir2_path, match[1])])
        return table.save(html_file_path)

    def save_match_table(self, html_file_path, reverse_dirs=False):
        if reverse_dirs:
            dir1_path, dir2_path = self.dir2_path, self.dir1_path
            matches = get_best_matches_axis(self.score_matrix.T, axis=0)
        else:
            dir1_path, dir2_path = self.dir1_path, self.dir2_path
            matches = get_best_matches_axis(self.score_matrix, axis=0)

        table = TableHTML('Matches<br/><br/>Dir1: {}<br/>Dir2: {}'.format(dir1_path, dir2_path))
        table.add_head(["dir1", "dir2"])
        for match in matches:
            table.add_row([self.get_image_info(dir1_path, match[0]),
                           self.get_image_info(dir2_path, match[1])])
        return table.save(html_file_path)

    def save_nearest_neighbor_table(self, html_file_path, number_nn=20, reverse_dirs=False):
        if reverse_dirs:
            dir1_path, dir2_path = self.dir2_path, self.dir1_path
            nearest_neighbors = get_ordered_scores_matrix(self.score_matrix.T)[:, :number_nn]
        else:
            dir1_path, dir2_path = self.dir1_path, self.dir2_path
            nearest_neighbors = get_ordered_scores_matrix(self.score_matrix)[:, :number_nn]

        table = TableHTML('Nearest Neighbors<br/><br/>Dir1: {}<br/>Dir2: {}'.format(dir1_path, dir2_path))
        table.add_head(["dir1"] + list(np.arange(number_nn)))

        for i in range(nearest_neighbors.shape[0]):
            nn = nearest_neighbors[i]
            table.add_row(
                [self.get_image_info(dir1_path, i)] + [self.get_image_info(dir2_path, k) for k in nn])
        return table.save(html_file_path)

    @staticmethod
    def save_annotation_table(html_file_path, manuscript_illustration_path):
        annotation_dict = DirTools.get_annotation_dict(manuscript_illustration_path)
        table = TableHTML('Annotations<br/>Dir: {}'.format(manuscript_illustration_path))
        table.add_head(["folio", "annotated", "illustrations extracted"])
        for i, (annotation_path, infos) in enumerate(annotation_dict.items()):
            illustrations = infos["illustration"]
            folio_infos = infos["folio"]
            annotation_infos = ("", annotation_path)
            table.add_row([folio_infos, annotation_infos, illustrations])
        return table.save(html_file_path)

    @staticmethod
    def save_illustration_table(html_file_path, manuscript_illustration_path):
        illustration_paths = DirTools.list_folder_images(manuscript_illustration_path)
        table = TableHTML('Illustrations<br/>Dir: {}'.format(manuscript_illustration_path))
        table.add_head(["illustration"])
        for i, illustration_path in enumerate(illustration_paths):
            illustration_info = (i, illustration_path)
            table.add_row([illustration_info])
        return table.save(html_file_path)

    def save_all_pages(self, save_dir, number_nn=20):
        pathlib.Path(save_dir).mkdir(parents=False, exist_ok=True)
        if DirTools.annotation_exists(self.dir1_path):
            Results.save_annotation_table(join(save_dir, "annotations_M1.html"), self.dir1_path)
        if DirTools.annotation_exists(self.dir2_path):
            Results.save_annotation_table(join(save_dir, "annotations_M2.html"), self.dir2_path)
        Results.save_illustration_table(join(save_dir, "illustrations_M1.html"), self.dir1_path)
        Results.save_illustration_table(join(save_dir, "illustrations_M2.html"), self.dir2_path)
        self.save_consistent_match(join(save_dir, "consistent_matches.html"), reverse_dirs=False)
        self.save_match_table(join(save_dir, "matches_M1_M2.html"), reverse_dirs=False)
        self.save_match_table(join(save_dir, "matches_M2_M1.html"), reverse_dirs=True)
        self.save_nearest_neighbor_table(join(save_dir, "nearest_neighbors_M1_M2.html"), number_nn=number_nn,
                                         reverse_dirs=False)
        self.save_nearest_neighbor_table(join(save_dir, "nearest_neighbors_M2_M1.html"), number_nn=number_nn,
                                         reverse_dirs=True)


if __name__ == "__main__":
    ms1_name = "D1"
    ms2_name = "D3"
    score_matrix_path = "D:/GitHub/IllustrationMatcher/saved_runs/{}{}/propagation_matrix.npy".format(ms1_name,
                                                                                                ms2_name)
    score_matrix = np.load(score_matrix_path)
    manuscripts_location = "D:/Stage/tmp_manuscripts/paper_manuscripts/{}/illustration"
    manuscript1_path = manuscripts_location.format(ms1_name)
    manuscript2_path = manuscripts_location.format(ms2_name)
    # compute score matrix
    # images1, images2 = DirTools.get_images_list(dir1_path), DirTools.get_images_list(dir2_path)
    # resnet50_conv4 = get_resnet50_conv4_model()
    # feats1 = compute_features(resnet50_conv4, images1, 320, batch_size=10)
    # feats2 = compute_features(resnet50_conv4, images2, 320, batch_size=10)
    # score_matrix = compute_cosine_matrix(feats1, feats2)
    #
    # create results web pages
    results = Results(manuscript1_path, manuscript2_path, score_matrix, with_annotation=False)
    result_folder = "../web_content/{}{}".format(ms1_name, ms2_name)
    results.save_all_pages(result_folder, number_nn=10)
    exit()