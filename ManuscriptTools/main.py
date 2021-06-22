from utils.Results import Results
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find same illustrations in two different manuscripts')
    parser.add_argument('-m1', '--manuscript1', nargs='?', type=str, required=True,
                        help='first manuscript path')
    parser.add_argument('-m2', '--manuscript2', nargs='?', type=str, required=True,
                        help='second manuscript path')
    parser.add_argument('-s', '--score_matrix', nargs='?', type=str, required=True, help='score matrix file')
    parser.add_argument('-r', '--result_dir', nargs='?', type=str, required=True, help='result directory')
    parser.add_argument('-nn', '--nearest_neighbors_number', default=10, type=int)
    parser.add_argument('--with_annotation', dest='with_annotation', action='store_true')

    args = parser.parse_args()
    manuscript1_path = args.manuscript1
    manuscript2_path = args.manuscript2
    score_matrix_file_path = args.score_matrix
    result_dir_path = args.result_dir
    number_nn = args.nearest_neighbors_number
    with_annotation = args.with_annotation
    score_matrix = np.load(score_matrix_file_path)
    
    results = Results(manuscript1_path, manuscript2_path, score_matrix, with_annotation=with_annotation)
    results.save_all_pages(result_dir_path, number_nn=number_nn)
    print("result web pages saved successfully in the folder {}".format(result_dir_path))
    exit()
