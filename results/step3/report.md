# step3: TF-IDF baselines on `text_min`

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
| word_tfidf_lr  |      2   |       116723 |         0.7911 |          0.7995 |          0.866  |          0.5862 |       0.821  |   0.684  |          0.958  |       0.8757 |   0.915  |
| word_tfidf_svm |      0.3 |       116723 |         0.7921 |          0.7998 |          0.8676 |          0.5916 |       0.8087 |   0.6833 |          0.9555 |       0.8802 |   0.9163 |
| char_tfidf_lr  |      4   |       130385 |         0.783  |          0.7854 |          0.8546 |          0.561  |       0.8122 |   0.6636 |          0.9554 |       0.8637 |   0.9072 |
| char_tfidf_svm |      0.3 |       130385 |         0.7807 |          0.7873 |          0.8548 |          0.5604 |       0.8255 |   0.6676 |          0.9583 |       0.8611 |   0.9071 |

## Test macro-F1 per language group

| model          |   bangla_script |   english |   romanized |   romanized_codemixed |   mixed_script |
|:---------------|----------------:|----------:|------------:|----------------------:|---------------:|
| word_tfidf_lr  |          0.7735 |    0.8297 |      0.7615 |                0.8016 |         0.8226 |
| word_tfidf_svm |          0.7715 |    0.8312 |      0.7576 |                0.8057 |         0.8286 |
| char_tfidf_lr  |          0.7595 |    0.8134 |      0.7554 |                0.7869 |         0.8097 |
| char_tfidf_svm |          0.7615 |    0.8152 |      0.7551 |                0.7893 |         0.8119 |
| n_test         |       8656      | 9077      |   1376      |             3726      |      1135      |
