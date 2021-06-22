import argparse

from utils.IllustrationMatching import IllustrationMatching
from utils.get_ground_truth import load_json_as_list
from utils.score_matrix_func import normalize_score_matrix, propagate_matches, get_recall_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find same illustrations in two different manuscripts')
    parser.add_argument('-m1', '--manuscript1', nargs='?', type=str, required=True, help='first manuscript path')
    parser.add_argument('-m2', '--manuscript2', nargs='?', type=str, required=True, help='second manuscript path')
    parser.add_argument('-r', '--results_dir', nargs='?', type=str, required=True, help='results directory')
    parser.add_argument('-gt', '--ground_truth', nargs='?', type=str, required=False, default=None,
                        help='true matches')


    args = parser.parse_args()
    dir1_path = args.manuscript1
    dir2_path = args.manuscript2
    results_dir = args.results_dir
    matches_json_file = args.ground_truth

    # compute scores
    illustration_matching = IllustrationMatching(dir1_path, dir2_path, fast=True)
    s1, s2 = illustration_matching.run(save_dir=args.results_dir)
    score_matrix = s1 + s2

    # normalize matrix
    normalized_matrix = normalize_score_matrix(score_matrix)
    print("Information propagation...")
    matrix_after_propagation = propagate_matches(normalized_matrix, std=5, alpha=0.25)

    # measure performance
    if matches_json_file is not None:
        true_matches = load_json_as_list(matches_json_file)
        print("Score matrix recall: {:.1f}%".format(
            get_recall_performance(score_matrix, true_matches) * 100))
        print("Normalized matrix recall: {:.1f}%".format(
            get_recall_performance(normalized_matrix, true_matches) * 100))
        print("Matrix info. propagation recall: {:.1f}%".format(
            get_recall_performance(matrix_after_propagation, true_matches) * 100))
    exit()