# Illustration Matcher
## Presentation
Given sets of images, this algorithm computes different kind of score matrices that were mentioned in the paper.

- <img src="https://render.githubusercontent.com/render/math?math=S_{trans}"> score matrix. It has been computed using a combination of feature matching and spatial consistency. It is saved in score_matrix.npy.
- <img src="https://render.githubusercontent.com/render/math?math=N_S"> score matrix. It is a simple normalization of the <img src="https://render.githubusercontent.com/render/math?math=S_{trans}"> score matrix. It is saved in normalized_matrix.npy.
- <img src="https://render.githubusercontent.com/render/math?math=P_S"> score matrix. It is an update of the <img src="https://render.githubusercontent.com/render/math?math=N_S"> matrix in which scores are enhanced in a certain way, in order to take into account the illustrations order. It is saved in propagation_matrix.npy.

More details about those three matrices can be found in the paper.

## Command Line

```
python main.py --manuscript1 m1_path --manuscript2 m2_path --results_dir results_folder_path
```

**args**
- `-m1, --manuscript1`: directory where the first manuscript is stored.
- `-m2, --manuscript2`: directory where the second manuscript is stored.
- `-r, --results_dir`: directory in which the score matrices will be stored at the end of the execution of the algorithm.
- `-gt, --ground_truth`: json file containing the list of the true matches, this is an optional argument, if it is specified the algorithm will also return the performance of each score matrix.

You can find for each of the 6 couples of manuscripts mentioned in the paper, the corresponding json file in the ground_truth folder at the root of the repository.

Before the end of the execution, the algorithm will store two matrices <img src="https://render.githubusercontent.com/render/math?math=S_1"> and <img src="https://render.githubusercontent.com/render/math?math=S_2"> in the results folder (score_matrix_1.npy and score_matrix_2.npy) such as:
<div align="center"><img src="./images/Strans2.png" height="15"><br/></div>

They will be initially filled with the value -∞ and will be updated as things progress.
This provides the ability to resume an execution after stopping it.
