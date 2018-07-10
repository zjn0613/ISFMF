# Image Shape Feature Matrix Factorization (ISFMF)

### Overview
> Recommender system is an information-ﬁltering tool used in solving the problem that the user’s preference in information overload. In recent years, some algorithms have been combined with some side information (i.e., item description documents, user reviews, and social networks), and rating prediction accuracy has been signiﬁcantly improved. However, for fashionable goods, such as apparel and shoes that are important for designing, the contextual information of items is insufﬁcient, and their image shape feature should be considered. Currently, no such recommender system is available to use this feature of image shape. This study proposes a novel probabilistic model using the image shape feature that integrates a convolutional neural network into the probabilistic matrix factorization. The experiment conducted on two real-world datasets corroborates that our model outperforms the other recommendation models. 

### Paper
- Apparel Goods Recommender System-Based Image Shape Features Extracted by a CNN (*IEEE SMC 2018*)
  - Yufeng Duan, Ryosuke Sage

### Requirements

- Python 3.5
- Keras 2.0

### How to Run

Note: Run `python <install_path>/run.py -h` in bash shell. You will see how to configure several parameters

### Configuration
You can evaluate our model with different settings in terms of the size of dimension, the value of hyperparameter, the number of convolutional kernal, and etc. Below is a description of all the configurable parameters and their defaults:

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
`-e <integer>`, `--emb_dim <integer>` | 200
`-p <path>`, `--pretrain_w2v <path>` | {}
`-g <bool>`, `--give_item_weight <bool>` | `True`
`-k <integer>`, `--dimension <integer>` | 50
`-u <float>`, `--lambda_u <float>` | {}
`-v <float>`, `--lambda_v <float>` | {}
`-n <integer>`, `--max_iter <integer>` | 200
`-w <integer>`, `--num_kernel_per_ws` | 100

1. `do_preprocess`: `True` or `False` in order to preprocess raw data for ConvMF.
2. `raw_rating_data_path`: path to a raw rating data path. The data format should be `user id::item id::rating`.
3. `min_rating`: users who have less than `min_rating` ratings will be removed.
4. `max_length_document`: the maximum length of document of each item.
5. `max_df`: threshold to ignore terms that have a document frequency higher than the given value. i.e. for removing corpus-stop words.
6. `vocab_size`: size of vocabulary.
7. `split_ratio`: 1-ratio, ratio/2 and ratio/2 of the entire dataset will be constructed as training, valid and test set, respectively.
8. `data_path`: path to training, valid and test datasets.
9. `aux_path`: path to R, D_all sets that are generated during the preprocessing step.
10. `res_dir`: path to ConvMF's result
11. `emb_dim`: the size of latent dimension for word vectors.
12. `pretrain_w2v`: path to pretrained word embedding model to initialize word vectors.
13. `give_item_weight` : `True` or `False` to give item weight for R-ConvMF.
14. `dimension`: the size of latent dimension for users and items.
15. `lambda_u`: parameter of user regularizer.
16. `lambda_v`: parameter of item regularizer.
17. `max_iter`: the maximum number of iteration.
18. `num_kernel_per_ws`: the number of kernels per window size for CNN module.

