# Step 7: final tables (20% test set, 23,970 reviews)

Transformer rows: mean ± std over seeds. Macro-F1 is the primary metric; accuracy is supplementary.

## T1. Overall results

|                                | Seeds   | Macro-F1      | Neg P         | Neg R         | Neg F1        | Pos P         | Pos R         | Pos F1        | Accuracy      |
|:-------------------------------|:--------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| Word TF-IDF + LR (minimal)     | -       | 0.799         | 0.586         | 0.821         | 0.684         | 0.958         | 0.876         | 0.915         | 0.866         |
| Word TF-IDF + SVM (minimal)    | -       | 0.800         | 0.592         | 0.809         | 0.683         | 0.955         | 0.880         | 0.916         | 0.868         |
| Char TF-IDF + LR (minimal)     | -       | 0.785         | 0.561         | 0.812         | 0.664         | 0.955         | 0.864         | 0.907         | 0.855         |
| Char TF-IDF + SVM (minimal)    | -       | 0.787         | 0.560         | 0.825         | 0.668         | 0.958         | 0.861         | 0.907         | 0.855         |
| Word TF-IDF + SVM (normalized) | -       | 0.800         | 0.592         | 0.807         | 0.683         | 0.955         | 0.881         | 0.916         | 0.868         |
| Char TF-IDF + SVM (normalized) | -       | 0.788         | 0.560         | 0.829         | 0.668         | 0.959         | 0.860         | 0.907         | 0.855         |
| XLM-R-base (minimal)           | 3       | 0.817 ± 0.003 | 0.631 ± 0.002 | 0.806 ± 0.015 | 0.708 ± 0.006 | 0.956 ± 0.003 | 0.899 ± 0.002 | 0.926 ± 0.001 | 0.882 ± 0.001 |
| XLM-R-base (normalized)        | 3       | 0.820 ± 0.001 | 0.632 ± 0.002 | 0.817 ± 0.001 | 0.713 ± 0.001 | 0.958 ± 0.000 | 0.898 ± 0.001 | 0.927 ± 0.000 | 0.884 ± 0.001 |
| BanglaBERT (minimal)           | 1       | 0.821         | 0.644         | 0.800         | 0.714         | 0.955         | 0.905         | 0.929         | 0.887         |
| BanglaBERT (normalized)        | 1       | 0.826         | 0.658         | 0.792         | 0.719         | 0.953         | 0.912         | 0.932         | 0.891         |

## T2. Macro-F1 per language group

|                                | Bangla script   | English       | Romanized Bangla   | Romanized + English   | Mixed script   |
|:-------------------------------|:----------------|:--------------|:-------------------|:----------------------|:---------------|
| Word TF-IDF + LR (minimal)     | 0.773           | 0.830         | 0.762              | 0.802                 | 0.823          |
| Word TF-IDF + SVM (minimal)    | 0.771           | 0.831         | 0.758              | 0.806                 | 0.829          |
| Char TF-IDF + LR (minimal)     | 0.760           | 0.813         | 0.755              | 0.787                 | 0.810          |
| Char TF-IDF + SVM (minimal)    | 0.761           | 0.815         | 0.755              | 0.789                 | 0.812          |
| Word TF-IDF + SVM (normalized) | 0.773           | 0.832         | 0.760              | 0.799                 | 0.829          |
| Char TF-IDF + SVM (normalized) | 0.761           | 0.817         | 0.758              | 0.788                 | 0.812          |
| XLM-R-base (minimal)           | 0.791 ± 0.004   | 0.853 ± 0.002 | 0.762 ± 0.007      | 0.817 ± 0.005         | 0.843 ± 0.002  |
| XLM-R-base (normalized)        | 0.793 ± 0.001   | 0.855 ± 0.002 | 0.776 ± 0.003      | 0.820 ± 0.003         | 0.846 ± 0.004  |
| BanglaBERT (minimal)           | 0.794           | 0.850         | 0.779              | 0.835                 | 0.849          |
| BanglaBERT (normalized)        | 0.803           | 0.850         | 0.792              | 0.831                 | 0.847          |
| n (test)                       | 8,656           | 9,077         | 1,376              | 3,726                 | 1,135          |

## T3. Effect of Banglish normalization (normalized − minimal, macro-F1)

|                            | Overall        | Bangla script   | English        | Romanized Bangla   | Romanized + English   | Mixed script   |
|:---------------------------|:---------------|:----------------|:---------------|:-------------------|:----------------------|:---------------|
| Word TF-IDF + SVM (1 seed) | -0.000         | +0.002          | +0.001         | +0.002             | -0.007                | +0.001         |
| XLM-R-base (3 seeds)       | +0.003 ± 0.004 | +0.001 ± 0.005  | +0.002 ± 0.001 | +0.014 ± 0.004     | +0.003 ± 0.008        | +0.003 ± 0.003 |
| BanglaBERT (1 seed)        | +0.004         | +0.010          | +0.000         | +0.014             | -0.004                | -0.003         |
