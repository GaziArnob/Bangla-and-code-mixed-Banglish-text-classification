# step4: TF-IDF baselines on `text_norm`

Split: 80% train / 20% test (stratified by label x group, seed 42); 10% of train held out as validation for choosing C.

| group               |   train |   val |   test |
|:--------------------|--------:|------:|-------:|
| bangla_script       |   31160 |  3463 |   8656 |
| english             |   32678 |  3632 |   9077 |
| mixed_script        |    4084 |   453 |   1135 |
| romanized           |    4953 |   550 |   1376 |
| romanized_codemixed |   13415 |  1490 |   3726 |

## Test results (20% test set)

| model          |   best_C |   n_features |   val_macro_f1 |   test_macro_f1 |   test_accuracy |   neg_precision |   neg_recall |   neg_f1 |   pos_precision |   pos_recall |   pos_f1 |
|:---------------|---------:|-------------:|---------------:|----------------:|----------------:|----------------:|-------------:|---------:|----------------:|-------------:|---------:|
| word_tfidf_lr  |      4   |       115163 |         0.7934 |          0.7982 |          0.8682 |          0.5955 |       0.791  |   0.6794 |          0.9518 |       0.8847 |   0.917  |
| word_tfidf_svm |      0.3 |       115163 |         0.7933 |          0.7996 |          0.8676 |          0.5919 |       0.8068 |   0.6829 |          0.9551 |       0.8807 |   0.9164 |
| char_tfidf_lr  |      4   |       114656 |         0.7806 |          0.786  |          0.8547 |          0.561  |       0.8153 |   0.6647 |          0.9561 |       0.8631 |   0.9073 |
| char_tfidf_svm |      0.3 |       114656 |         0.7794 |          0.7877 |          0.8547 |          0.56   |       0.829  |   0.6684 |          0.9591 |       0.8603 |   0.907  |

## Test macro-F1 per language group

| model          |   bangla_script |   english |   romanized |   romanized_codemixed |   mixed_script |
|:---------------|----------------:|----------:|------------:|----------------------:|---------------:|
| word_tfidf_lr  |          0.7709 |    0.83   |      0.7538 |                0.8019 |         0.8279 |
| word_tfidf_svm |          0.7731 |    0.8317 |      0.7596 |                0.7987 |         0.8292 |
| char_tfidf_lr  |          0.7605 |    0.8138 |      0.7598 |                0.7851 |         0.8111 |
| char_tfidf_svm |          0.7613 |    0.8171 |      0.758  |                0.7876 |         0.8119 |
| n_test         |       8656      | 9077      |   1376      |             3726      |      1135      |
