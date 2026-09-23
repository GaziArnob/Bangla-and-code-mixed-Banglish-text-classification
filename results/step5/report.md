# Step 5: XLM-R-base vs TF-IDF baselines

All numbers on the same 20% test set (23,970 reviews).

## Overall

| model                 |   macro_f1 |   accuracy |   neg_P |   neg_R |   neg_F1 |   pos_F1 |
|:----------------------|-----------:|-----------:|--------:|--------:|---------:|---------:|
| word_tfidf_svm (min)  |     0.7998 |     0.8676 |  0.5916 |  0.8087 |   0.6833 |   0.9163 |
| word_tfidf_svm (norm) |     0.7996 |     0.8676 |  0.5919 |  0.8068 |   0.6829 |   0.9164 |
| char_tfidf_svm (min)  |     0.7873 |     0.8548 |  0.5604 |  0.8255 |   0.6676 |   0.9071 |
| xlmr (min)            |     0.8137 |     0.8813 |  0.6311 |  0.7896 |   0.7015 |   0.9259 |
| xlmr (norm)           |     0.8207 |     0.8842 |  0.6335 |  0.8177 |   0.7139 |   0.9274 |

## Macro-F1 per language group

| model                 |   bangla_script |   english |   romanized |   romanized_codemixed |   mixed_script |
|:----------------------|----------------:|----------:|------------:|----------------------:|---------------:|
| word_tfidf_svm (min)  |          0.7715 |    0.8312 |      0.7576 |                0.8057 |         0.8286 |
| word_tfidf_svm (norm) |          0.7731 |    0.8317 |      0.7596 |                0.7987 |         0.8292 |
| char_tfidf_svm (min)  |          0.7615 |    0.8152 |      0.7551 |                0.7893 |         0.8119 |
| xlmr (min)            |          0.787  |    0.8509 |      0.7606 |                0.8108 |         0.8445 |
| xlmr (norm)           |          0.7939 |    0.8538 |      0.7765 |                0.8227 |         0.8496 |

## Paired bootstrap differences (1000 resamples, * = p < 0.05)

| comparison                          | ALL      | bangla_script   | english   |   romanized | romanized_codemixed   |   mixed_script |
|:------------------------------------|:---------|:----------------|:----------|------------:|:----------------------|---------------:|
| xlmr (min) - word_tfidf_svm (min)   | +0.0139* | +0.0155*        | +0.0197*  |       0.003 | +0.0051               |         0.0159 |
| xlmr (norm) - word_tfidf_svm (norm) | +0.0211* | +0.0207*        | +0.0221*  |       0.017 | +0.0239*              |         0.0204 |
| xlmr (norm) - xlmr (min)            | +0.0069* | +0.0069*        | +0.0029   |       0.016 | +0.0119*              |         0.0051 |

## XLM-R run details

|                               | xlmr (min)                     | xlmr (norm)                    |
|:------------------------------|:-------------------------------|:-------------------------------|
| fertility_bangla_script       | 2.019                          | 1.956                          |
| fertility_english             | 1.391                          | 1.329                          |
| fertility_romanized           | 1.878                          | 1.803                          |
| fertility_romanized_codemixed | 1.723                          | 1.63                           |
| fertility_mixed_script        | 1.938                          | 1.868                          |
| truncated_test_%              | 0.89                           | 0.86                           |
| unk_rate_test_%               | 0.055                          | 0.056                          |
| best_val_macro_f1             | 0.8141                         | 0.8122                         |
| val_f1_by_epoch               | 0.8141, 0.8074, 0.8106, 0.8141 | 0.8101, 0.8119, 0.8122, 0.8122 |
| train_minutes                 | 26.2                           | 26.4                           |
| gpu                           | Tesla T4                       | Tesla T4                       |

## Full comparison table

|                                                                |     n |   delta |   ci95_lo |   ci95_hi |     p | sig   |
|:---------------------------------------------------------------|------:|--------:|----------:|----------:|------:|:------|
| ('xlmr (norm) - xlmr (min)', 'ALL')                            | 23970 |  0.0069 |    0.0028 |    0.0108 | 0     | *     |
| ('xlmr (norm) - xlmr (min)', 'bangla_script')                  |  8656 |  0.0069 |    0.0001 |    0.0133 | 0.042 | *     |
| ('xlmr (norm) - xlmr (min)', 'english')                        |  9077 |  0.0029 |   -0.004  |    0.0088 | 0.396 |       |
| ('xlmr (norm) - xlmr (min)', 'romanized')                      |  1376 |  0.016  |   -0.0018 |    0.0352 | 0.078 |       |
| ('xlmr (norm) - xlmr (min)', 'romanized_codemixed')            |  3726 |  0.0119 |    0.0009 |    0.0218 | 0.028 | *     |
| ('xlmr (norm) - xlmr (min)', 'mixed_script')                   |  1135 |  0.0051 |   -0.0116 |    0.0214 | 0.514 |       |
| ('xlmr (min) - word_tfidf_svm (min)', 'ALL')                   | 23970 |  0.0139 |    0.0087 |    0.0194 | 0     | *     |
| ('xlmr (min) - word_tfidf_svm (min)', 'bangla_script')         |  8656 |  0.0155 |    0.0078 |    0.0241 | 0     | *     |
| ('xlmr (min) - word_tfidf_svm (min)', 'english')               |  9077 |  0.0197 |    0.0106 |    0.0293 | 0     | *     |
| ('xlmr (min) - word_tfidf_svm (min)', 'romanized')             |  1376 |  0.003  |   -0.02   |    0.0248 | 0.766 |       |
| ('xlmr (min) - word_tfidf_svm (min)', 'romanized_codemixed')   |  3726 |  0.0051 |   -0.0079 |    0.0195 | 0.456 |       |
| ('xlmr (min) - word_tfidf_svm (min)', 'mixed_script')          |  1135 |  0.0159 |   -0.0067 |    0.0376 | 0.164 |       |
| ('xlmr (norm) - word_tfidf_svm (norm)', 'ALL')                 | 23970 |  0.0211 |    0.0162 |    0.0258 | 0     | *     |
| ('xlmr (norm) - word_tfidf_svm (norm)', 'bangla_script')       |  8656 |  0.0207 |    0.0129 |    0.0285 | 0     | *     |
| ('xlmr (norm) - word_tfidf_svm (norm)', 'english')             |  9077 |  0.0221 |    0.0139 |    0.0305 | 0     | *     |
| ('xlmr (norm) - word_tfidf_svm (norm)', 'romanized')           |  1376 |  0.017  |   -0.0033 |    0.0393 | 0.126 |       |
| ('xlmr (norm) - word_tfidf_svm (norm)', 'romanized_codemixed') |  3726 |  0.0239 |    0.0099 |    0.0367 | 0     | *     |
| ('xlmr (norm) - word_tfidf_svm (norm)', 'mixed_script')        |  1135 |  0.0204 |   -0.0017 |    0.0426 | 0.07  |       |
