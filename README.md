# Image Shape Feature Matrix Factorization (ISFMF)

### Overview
> Recommender system is an information-ﬁltering tool used in solving the problem that the user’s preference in information overload. In recent years, some algorithms have been combined with some side information (i.e., item description documents, user reviews, and social networks), and rating prediction accuracy has been signiﬁcantly improved. However, for fashionable goods, such as apparel and shoes that are important for designing, the contextual information of items is insufﬁcient, and their image shape feature should be considered. Currently, no such recommender system is available to use this feature of image shape. This study proposes a novel probabilistic model using the image shape feature that integrates a convolutional neural network into the probabilistic matrix factorization. The experiment conducted on two real-world datasets corroborates that our model outperforms the other recommendation models. 

### Paper
- Apparel Goods Recommender System-Based Image Shape Features Extracted by a CNN (*IEEE SMC 2018*)
  - Yufeng Duan, Ryosuke Sage

### Installation

#### 1. This model is modiﬁed on the basis of the [ConvMF](http://dm.postech.ac.kr/~cartopy/ConvMF/) and the requirements are as follows:

- Python 3.5
- Keras 2.1.3
- tensorflow 1.7
- CUDA 9.0
- cudnn 7.0.5

#### 2. Datasets
- Our dataset is [here](http://dm.postech.ac.kr/~cartopy/ConvMF/) and unzip them into the folder "data".

#### 3. How to Run

Note: Run `python <install_path>/run.py -h` in bash shell. You will see how to configure several parameters

#### 4. Configuration
You can evaluate our model with different settings in terms of the size of dimension, the value of hyperparameter, the size of image, and etc. Below is a description of all the configurable parameters and their defaults:

Parameter | Default
---       | ---
`-h`, `--help` | {}
`-c <bool>`, `--do_preprocess <bool>` | `False`
`-r <path>`, `--raw_rating_data_path <path>` | {}
`-i <path>`, `--raw_item_document_data_path <path>`| {}
`-m <integer>`, `--min_rating <integer>` | {}
`-l <integer>`, `--max_length_document <integer>` | 300
`-f <float>`, `--max_df <float>` | 0.5
`-s <integer>`, `--vocab_size <integer>` | 8000
`-t <float>`, `--split_ratio <float>` | 0.2
`-d <path>`, `--data_path <path>` | {}
`-a <path>`, `--aux_path <path>` | {}
`-o <path>`, `--res_dir <path>` | {}
`-k <integer>`, `--dimension <integer>` | 50
`-u <float>`, `--lambda_u <float>` | {}
`-v <float>`, `--lambda_v <float>` | {}
`-n <integer>`, `--max_iter <integer>` | 200
`-is <integer>`, `--image_size <integer>` | 130
`-sm <str>`, `--select_model <str>` | {}

1. `do_preprocess`: `True` or `False` in order to preprocess raw data.
2. `raw_rating_data_path`: path to a raw rating data path. The data format should be `user id,item id,rating`.
3. `raw_item_document_data_path`: Path to raw item document data. item document consists of multiple text. data format - item id::text1|text2...".
4. `min_rating`: users who have less than `min_rating` ratings will be removed.
5. `max_length_document`: the maximum length of document of each item.
6. `max_df`: threshold to ignore terms that have a document frequency higher than the given value. i.e. for removing corpus-stop words.
7. `vocab_size`: size of vocabulary.
8. `split_ratio`: 1-ratio, ratio/2 and ratio/2 of the entire dataset will be constructed as training, valid and test set, respectively.
9. `data_path`: path to training, valid and test datasets.
10. `aux_path`: path to R, D_all sets that are generated during the preprocessing step.
11. `res_dir`: path to ConvMF's result
12. `dimension`: the size of latent dimension for users and items.
13. `lambda_u`: parameter of user regularizer.
14. `lambda_v`: parameter of item regularizer.
15. `max_iter`: the maximum number of iteration.
16. `image_size`: size of input images.
17. `select_model`: Choose a model(exmple:ConvMF or PMF or ISFMF).