# Step 1: Dataset audit

Total rows: 119,859

## Language group x label

| group               |   negative |   positive |   total |   neg_% |   share_% |
|:--------------------|-----------:|-----------:|--------:|--------:|----------:|
| bangla_script       |       8720 |      34559 |   43279 |    20.1 |      36.1 |
| english             |       6106 |      39285 |   45391 |    13.5 |      37.9 |
| romanized           |       1372 |       5510 |    6882 |    19.9 |       5.7 |
| romanized_codemixed |       3731 |      14904 |   18635 |    20   |      15.5 |
| mixed_script        |       1244 |       4428 |    5672 |    21.9 |       4.7 |
| total               |      21173 |      98686 |  119859 |    17.7 |     100   |

## Language group x split

| group               |   train |   val |   test |   total |
|:--------------------|--------:|------:|-------:|--------:|
| bangla_script       |   30187 |  6428 |   6664 |   43279 |
| english             |   31913 |  6817 |   6661 |   45391 |
| romanized           |    4820 |  1045 |   1017 |    6882 |
| romanized_codemixed |   12987 |  2835 |   2813 |   18635 |
| mixed_script        |    3994 |   854 |    824 |    5672 |
| total               |   83901 | 17979 |  17979 |  119859 |

## Split x label

| split   |   negative |   positive |   total |   neg_% |
|:--------|-----------:|-----------:|--------:|--------:|
| train   |      14821 |      69080 |   83901 |    17.7 |
| val     |       3176 |      14803 |   17979 |    17.7 |
| test    |       3176 |      14803 |   17979 |    17.7 |
| total   |      21173 |      98686 |  119859 |    17.7 |

## Star rating -> label

|   label_raw |   negative |   positive |
|------------:|-----------:|-----------:|
|           1 |      11577 |          0 |
|           2 |       3494 |          0 |
|           3 |       6102 |          0 |
|           4 |          0 |      10449 |
|           5 |          0 |      88237 |

## Tokens per review

| group               |   mean |   50% |   95% |   max |
|:--------------------|-------:|------:|------:|------:|
| bangla_script       |   16.6 |    12 |  45   |   261 |
| english             |   14   |     9 |  40   |   225 |
| romanized           |    7.5 |     6 |  17   |    59 |
| romanized_codemixed |   17.8 |    14 |  44   |   202 |
| mixed_script        |   30.5 |    23 |  86.4 |   227 |

## Script-based group vs dataset flags

| group               |   rom=False,cm=False |   rom=False,cm=True |   rom=True,cm=False |   rom=True,cm=True |
|:--------------------|---------------------:|--------------------:|--------------------:|-------------------:|
| bangla_script       |                43278 |                   1 |                   0 |                  0 |
| english             |                45391 |                   0 |                   0 |                  0 |
| romanized           |                    0 |                   0 |                6882 |                  0 |
| romanized_codemixed |                    0 |                6562 |                   0 |              12073 |
| mixed_script        |                  906 |                4644 |                   3 |                119 |

## Data quality

- Empty texts: 0
- Duplicate rows (same normalized text): 11 (0.0%)
- Distinct texts appearing in more than one split: 1
- Test rows whose exact text also appears in train: 1
- Distinct texts with conflicting labels: 0

Most frequent texts:

| text                                                                                                     |   count |
|:---------------------------------------------------------------------------------------------------------|--------:|
| #name?                                                                                                   |      12 |
| ordered 3 radhuni garam masala got 2 radhuni and the other one was পিওরো i like delivery man's behaviour |       1 |
| perfect one as i wanted. thanks                                                                          |       1 |
| this is my third time purchase.i love this.very cute and small.                                          |       1 |
| nice one , value for money.                                                                              |       1 |
| অনেক অনেক কিউট ব্যাগটা।ধন্যবাদ সেলারকে।                                                                     |       1 |
| বিল্ড কোয়ালিটি খুব ভাল না হলেও চেয়ারটা কাজের আছে।                                                          |       1 |
| ভালা🥰🥰❤️                                                                                               |       1 |
| ভুব ভালো ছিলো, ধন্যবাদ সেলার                                                                               |       1 |
| কি চাইলাম আর কি দিলো। যত্তসব ফালতু প্রোডাক্ট। কালার চাইলাম কি আর দিলো কি। কোন সাইজ নাই, প্রচুর ময়লা।            |       1 |

## Noise present in text_raw (rows)

|               |   rows |
|:--------------|-------:|
| html_tags     |      1 |
| html_entities |      0 |
| urls          |     19 |
| mentions      |      6 |
| hashtags      |    195 |
| newlines      |  15561 |
| has_emoji     |  19050 |
