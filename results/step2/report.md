# Step 2: Preprocessing

- Rows in: 119,859; dropped `#NAME?` rows: 11; rows out: 119,848
- Lexicon: 67 canonical words, 138 variants

## How often each rule fired

|                              |   rows |
|:-----------------------------|-------:|
| html_tags_removed            |      0 |
| html_entities_unescaped      |      0 |
| urls_to_<URL>                |     19 |
| mentions_to_<USER>           |      6 |
| hashtags_kept_as_words       |    184 |
| whitespace_collapsed         |  24377 |
| bengali_digits_converted     |   7892 |
| punct_runs_collapsed         |  18309 |
| elongations_reduced          |    638 |
| lexicon_replacements(tokens) |  29459 |

## Effect of normalization per language group

| group               |   rows |   reviews_changed_by_min_% |   reviews_changed_by_norm_% |   latin_word_tokens |   spelling_variants_replaced |   replaced_%_of_latin_tokens |   word_vocab_min |   word_vocab_norm |   vocab_reduction_% |
|:--------------------|-------:|---------------------------:|----------------------------:|--------------------:|-----------------------------:|-----------------------------:|-----------------:|------------------:|--------------------:|
| bangla_script       |  43279 |                       54   |                        31.6 |                   0 |                            0 |                          0   |            30389 |             29827 |                 1.8 |
| english             |  45387 |                       15.4 |                        22.7 |              527275 |                         1342 |                          0.3 |            18546 |             18349 |                 1.1 |
| romanized           |   6879 |                        8.8 |                        66.9 |               43060 |                         6162 |                         14.3 |             7172 |              6927 |                 3.4 |
| romanized_codemixed |  18631 |                       22.9 |                        73.8 |              272051 |                        21570 |                          7.9 |            21908 |             21454 |                 2.1 |
| mixed_script        |   5672 |                       69.8 |                        49.7 |               23234 |                          366 |                          1.6 |            17145 |             16822 |                 1.9 |

## Top 25 lexicon replacements

| variant   | canonical   |   count |   count_in_english_group |
|:----------|:------------|--------:|-------------------------:|
| onk       | onek        |    3512 |                        0 |
| vlo       | valo        |    2361 |                        0 |
| akta      | ekta        |    1101 |                        0 |
| tnx       | thanks      |    1097 |                      346 |
| cilo      | chilo       |    1041 |                        0 |
| silo      | chilo       |     910 |                        0 |
| bhalo     | valo        |     823 |                        0 |
| amr       | amar        |     732 |                        0 |
| ai        | ei          |     678 |                        0 |
| gula      | gulo        |     609 |                        0 |
| aktu      | ektu        |     531 |                        0 |
| onak      | onek        |     518 |                        0 |
| ata       | eta         |     497 |                        0 |
| paici     | paisi       |     451 |                       38 |
| balo      | valo        |     437 |                       47 |
| akdom     | ekdom       |     436 |                       35 |
| jai       | jay         |     428 |                        0 |
| bt        | but         |     424 |                       98 |
| besi      | beshi       |     412 |                        0 |
| hoi       | hoy         |     410 |                        0 |
| tik       | thik        |     352 |                        0 |
| shundor   | sundor      |     345 |                        0 |
| kub       | khub        |     326 |                       14 |
| peyeci    | peyechi     |     315 |                       26 |
| jinish    | jinis       |     299 |                       10 |

## Variants that also occur in the English-only group (collision check)

| variant   | canonical   |   count |   count_in_english_group |
|:----------|:------------|--------:|-------------------------:|
| tnx       | thanks      |    1097 |                      346 |
| paici     | paisi       |     451 |                       38 |
| balo      | valo        |     437 |                       47 |
| akdom     | ekdom       |     436 |                       35 |
| bt        | but         |     424 |                       98 |
| kub       | khub        |     326 |                       14 |
| peyeci    | peyechi     |     315 |                       26 |
| jinish    | jinis       |     299 |                       10 |
| paichi    | paisi       |     293 |                       17 |
| peyesi    | peyechi     |     269 |                       22 |
| sellar    | seller      |     268 |                       84 |
| jmn       | jemon       |     222 |                       14 |
| motamoti  | motamuti    |     210 |                       16 |
| tmn       | temon       |     202 |                       17 |
| onujai    | onujayi     |     187 |                        7 |
| vhalo     | valo        |     184 |                       18 |
| vloi      | valoi       |     182 |                       27 |
| hoice     | hoise       |     181 |                        6 |
| tnq       | thanks      |     180 |                       36 |
| thnx      | thanks      |     180 |                       99 |
| khob      | khub        |     167 |                       15 |
| daraj     | daraz       |     161 |                       87 |
| hisabe    | hisebe      |     159 |                        1 |
| jamon     | jemon       |     148 |                        8 |
| selar     | seller      |     148 |                       36 |
| dice      | dise        |     138 |                        8 |
| tamon     | temon       |     129 |                        3 |
| shobai    | sobai       |     126 |                        1 |
| lagce     | lagse       |     126 |                        1 |
| amk       | amake       |     121 |                        1 |
| jnno      | jonno       |     115 |                        5 |
| bhai      | vai         |     113 |                        2 |
| kinto     | kintu       |     112 |                        3 |
| sondor    | sundor      |     110 |                        8 |
| celo      | chilo       |     108 |                        6 |
| babohar   | bebohar     |     102 |                        1 |
| selo      | chilo       |      96 |                        7 |
| nh        | na          |      95 |                        3 |
| akto      | ektu        |      91 |                        4 |
| hoyese    | hoyeche     |      85 |                        2 |
| sobay     | sobai       |      77 |                        9 |
| hishebe   | hisebe      |      75 |                        2 |
| mutamuti  | motamuti    |      75 |                        6 |
| thx       | thanks      |      74 |                       38 |
| gece      | geche       |      72 |                        1 |
| posondo   | pochondo    |      71 |                        1 |
| shathe    | sathe       |      69 |                        2 |
| akti      | ekti        |      69 |                        3 |
| khusi     | khushi      |      67 |                        3 |
| hocce     | hocche      |      66 |                        1 |
| onnek     | onek        |      65 |                        2 |
| anek      | onek        |      63 |                        1 |
| osadaron  | osadharon   |      62 |                        4 |
| karap     | kharap      |      59 |                        1 |
| thanx     | thanks      |      55 |                       41 |
| khuv      | khub        |      55 |                        2 |
| kubi      | khubi       |      48 |                        2 |
| pocondo   | pochondo    |      44 |                        1 |
| sundr     | sundor      |      41 |                        5 |
| hishabe   | hisebe      |      37 |                        1 |
| sundar    | sundor      |      31 |                        4 |
| khobi     | khubi       |      25 |                        2 |
| vhaloi    | valoi       |      24 |                        2 |
| baloi     | valoi       |      22 |                        7 |
| chaichi   | chaisi      |      21 |                        1 |
| thnq      | thanks      |      21 |                        8 |
| tiktak    | thikthak    |      19 |                        2 |
| bhaiya    | vaiya       |      19 |                        1 |
| tamoni    | temoni      |      18 |                        4 |
| ayta      | eita        |      15 |                        1 |
| peyachi   | peyechi     |      12 |                        1 |
| jamonta   | jemonta     |      11 |                        2 |
| khoob     | khub        |       7 |                        1 |
| hoyeci    | hoyeche     |       1 |                        1 |
