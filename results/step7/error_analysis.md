# Step 7: error analysis — best model = BanglaBERT (normalized), test set n=23,970

Best-model errors: 2,622 (10.9%).

## Error rate by gold class

| label    |     n |   errors |   error_% |
|:---------|------:|---------:|----------:|
| negative |  4234 |      879 |      20.8 |
| positive | 19736 |     1743 |       8.8 |

## Error rate by language group

|                     |    n |   errors |   error_% |
|:--------------------|-----:|---------:|----------:|
| Bangla script       | 8656 |     1209 |      14   |
| English             | 9077 |      658 |       7.2 |
| Romanized Bangla    | 1376 |      201 |      14.6 |
| Romanized + English | 3726 |      428 |      11.5 |
| Mixed script        | 1135 |      126 |      11.1 |

## Error rate by star rating

|   stars |     n |   errors |   error_% |
|--------:|------:|---------:|----------:|
|       1 |  2259 |      210 |       9.3 |
|       2 |   748 |      132 |      17.6 |
|       3 |  1227 |      537 |      43.8 |
|       4 |  2083 |      410 |      19.7 |
|       5 | 17653 |     1333 |       7.6 |

## Error rate by review length (whitespace tokens)

| len_bucket   |    n |   errors |   error_% |
|:-------------|-----:|---------:|----------:|
| 1-3          | 2379 |      251 |      10.6 |
| 4-7          | 7243 |      687 |       9.5 |
| 8-15         | 7785 |      860 |      11   |
| 16-30        | 4659 |      588 |      12.6 |
| 31+          | 1904 |      236 |      12.4 |

## Error rate with / without a contrast marker (but, kintu, tobe, কিন্তু, তবে …)

| has_contrast   |     n |   errors |   error_% |
|:---------------|------:|---------:|----------:|
| False          | 20354 |     1813 |       8.9 |
| True           |  3616 |      809 |      22.4 |

## Error rate with / without a standalone negation token (not, na, nai, না, নাই, নেই …)

| has_negation   |     n |   errors |   error_% |
|:---------------|------:|---------:|----------:|
| False          | 20658 |     1827 |       8.8 |
| True           |  3312 |      795 |      24   |

## Consensus errors (wrong in all 16 runs: 8 TF-IDF, 6 XLM-R, 2 BanglaBERT)

1,327 reviews (5.5% of test; 50.6% of best-model errors). These are the strongest candidates for label noise (rating disagrees with text).

By star rating:

|   label_raw |   n |
|------------:|----:|
|           1 |  74 |
|           2 |  42 |
|           3 | 225 |
|           4 | 189 |
|           5 | 797 |

By group:

| group               |   n |
|:--------------------|----:|
| bangla_script       | 671 |
| english             | 320 |
| romanized           | 100 |
| romanized_codemixed | 182 |
| mixed_script        |  54 |

Confident best-model errors: gold negative but P(pos) > 0.95: 224; gold positive but P(pos) < 0.05: 411.

`annotation_sample.csv` has 200 random consensus errors with empty columns for a manual check.


## Provisional label-noise check (60 random consensus errors)

Read by Claude, not a human annotator. Treat as an estimate until a human checks `annotation_sample.csv`.

| category | n | % |
|:--|--:|--:|
| A_label_contradicts_text | 40 | 67% |
| B_3star_boundary | 8 | 13% |
| C_mixed_or_ambiguous | 11 | 18% |
| D_genuine_model_error | 1 | 2% |

Extrapolated: about 885 test reviews (3.7% of test, 34% of best-model errors) have a star rating that clearly contradicts the text. Another ~13% of consensus errors are 3-star boundary cases. Only ~2% of consensus errors are clear model mistakes.

## Random consensus errors

- [5★ gold=positive] একদম বাজে ২ নম্বার জিনিস এর আগেও আমি দারাজ থেকে ডায়পার নিয়েছি এটাই নিয়েছি খুব ভালো ছিলো।কিন্তু এটা খুব বাজে ১ বার হিসু করলেই ভিজে থাকে ডায়পার।ফালতু পুরা টাক
- [4★ gold=positive] product r gaye expire date and price lekha chilo na.
- [5★ gold=positive] য়েমনটা ভেবেছি তেমনটা না,,,,,সেটিংস ঠিক তাকে না,,,জানি না কতূিন টিকবে।
- [4★ gold=positive] দাম হিসেবে কোয়ান্টিটি কম মনে হলো। 🫤
- [5★ gold=positive] bottle ar cap ta amon keno????
- [5★ gold=positive] অর্ডার করেছিলাম চারটি ভিন্ন ধরনের গাড়ি কিন্তু পেয়েছি একই রকমের গাড়ি। যার মানে হচ্ছে আমি প্রতারিত হয়েছি। তাই প্রতারিত হতে না চাইলে অনলাইনে অর্ডার না করাই উত্
- [4★ gold=positive] মোটামুটি চলে। আমাদের দেশেই এই দামে এর চেয়ে ভালো পাওয়া যায়।
- [5★ gold=positive] vai product order dilam konta ...r apnara dilen konta.....Keno sudhu sudhu apps er nam kharap koren..buyer ra jeta cay seta dile apnader problem kothay???...ple
- [4★ gold=positive] ব্লুটুথ রেঞ্জ এক্কেবারে বাজে,, মোবাইল দুই মিটার দূরে রাখলে কানেকশন পায় না
- [3★ gold=negative] good produtch
- [5★ gold=positive] বাজে পন্য নিচেরটা কাঠের তৈরি দেখতেও ভালো না
- [5★ gold=positive] its pin. not the color I wanted.
- [4★ gold=positive] ওয়াইফাই রিপিটার হিসেবে এটা কেউ নিবেন না। রিপিটার হিসেবে এটা কাজ করে না। নেট খুবই স্লো হয়েযায় রিপিটার হিসেবে ইউজ করলে।
- [5★ gold=positive] price besi nise
- [5★ gold=positive] ২ নাম্বার জিনিস
- [5★ gold=positive] আমি দুঃখিত যে, রিভিউ লিখতে এসে আমাকে একটা অভিযোগ লিখতে হচ্ছে। অর্ডার করেছিলাম ক্লাসিক, পেলাম স্ট্রবেরি ফ্লেভারেরটা। তা-ও আবার পুরো বক্স। এটা তো মেনে নেয়া যায় 
- [2★ gold=negative] This Product Are Very Good
- [5★ gold=positive] আমি টিভিটা কিনছিলাম মোটামুটি ১ বছর চলছে। ঠিক ১ বছরের মাথায় ইউটিউব সফটওয়্যার এ সমস্যা। আখন তারা বলে সফটওয়্যার এর সমস্যা তারা ঠিক করে না তারা হার্ডওয়্যার সমস্
- [4★ gold=positive] মোটামুটি চলে, বাট খুব ভালো বলবো না
- [4★ gold=positive] box ta te opposite side E Onek boro daag chilo 🚩bhalo Lage nai faltu fake price onno shob jaigai 300 taka ei Khane 500
- [3★ gold=negative] valoi cholcha
- [1★ gold=negative] alhamdulla same to same paici
- [1★ gold=negative] The watch is not very good but it’s very beautiful
- [5★ gold=positive] পছন্দ হই নাই,,খেলনা হিসেবে ব্যবহার করা যেতে পারে
- [5★ gold=positive] I wanted blue, but they gave me black 😩

## Random best-model errors per group

### Bangla script

- [2★ gold=negative, pred=positive] ভাল ব‌ল্লে ভুল হ‌বে নাই থে‌কে ভাল
- [1★ gold=negative, pred=positive] ভালো মন্দ জানি না,এখনো ব্যাবহার করিনি,দেখে মনে হচ্ছে ভাল হবে।
- [5★ gold=positive, pred=negative] মোটামুটি আছে একরকম।
- [5★ gold=positive, pred=negative] জিনিসটা ভালো, কিন্তু তোলায় ফাটা আছে, চেকিং করে দিতে হোত।
- [4★ gold=positive, pred=negative] মোটমুটি ভালো তবে স্টেপ লক থাকলে আরো ভালো হতো 🫥 টাইম ২৪ 🤦‍♂️ ১২ ঘন্টার হলে ভালো হতো
- [5★ gold=positive, pred=negative] প্রোডাক্ট ভালো। কিন্তু কালেকশন পয়েন্টের ওই লোকের ব্যবহার ভালো না😠আচরণ সন্তোষজনক! ওর ব্যবহারে মনে হলো তার কাছে খুব দায় পড়ছি😠

### English

- [3★ gold=negative, pred=positive] parking was very bad iam disappointed and there is no rapping 😔. but product was so good and accurate
- [3★ gold=negative, pred=positive] Print Quality Not As Good As Expected But They Packed The Book With Care, That’s for Sure.
- [5★ gold=positive, pred=negative] product original.....isn’t so effective for me.....and noticeable
- [5★ gold=positive, pred=negative] Product Expitred Date very close. Today july 11 but expired date was 10/20. Please select a lilte wide range expired date product.
- [5★ gold=positive, pred=negative] not so good with quality as I expected.
- [5★ gold=positive, pred=negative] didn't worked instantly.but definitely reduced the amount of the cockroachs

### Romanized Bangla

- [3★ gold=negative, pred=positive] afsos jaitar jonno nicilam oitai pailam na bluetooth acee kintu kaj kore na🤦‍♂️🤦‍♂️
- [3★ gold=negative, pred=positive] valoi chaila nita paran
- [5★ gold=positive, pred=negative] aita ato boro j adjust korte gele beke jachsevalo na.
- [5★ gold=positive, pred=negative] faltu ekta jinis dilen. eta cholai na.
- [5★ gold=positive, pred=negative] amar hoy nai ami bolce akta dica arak ta
- [4★ gold=positive, pred=negative] ghran vhalo lage nai.

### Romanized + English

- [1★ gold=negative, pred=positive] valo ase, kom price e pawa gese
- [3★ gold=negative, pred=positive] product ta valoi cilo.Smell ta mutamuti. Shate akta note book r 2 ta kham dice.
- [5★ gold=positive, pred=negative] It's not the right colour i ordered te pink and sky blue colour
- [5★ gold=positive, pred=negative] It is not a original product.thats why i return it.i thik daraz will refund me or sand me original product.thank you
- [1★ gold=negative, pred=positive] product valoi, kintu amake a v3 na dia v2 dica
- [5★ gold=positive, pred=negative] valo na..fake asce..j colour order dicilam ota dey ni

### Mixed script

- [5★ gold=positive, pred=negative] সিটিসি চা। কিন্তু খেতে লুজ লিফ ফার্স্ট ফ্লাশের মতই হালকা। সাথে Spicy smell-taste আছে। মূল লক্ষনীয় ব্যাপার এটাই। ফার্মান্টেটেড। মিল্ক দিয়ে না খাওয়াই ভাল। বরং 
- [3★ gold=negative, pred=positive] তিন(০৩) মাসে এই সাবান গুলো ( Godrej No.1 ) বিভিন্ন ফেলেভার এবং ছোট বড়ো সাইজে , প্রায় ২০০ পিচের উপরে কিনেছি, যথেষ্ট ভালো,""ধন্যবাদ দারাজ এবং অন্য সবাইকে"".....
- [1★ gold=negative, pred=positive] olover good podak.. Or good prise so you disesan..😒 ok thanks... সবাই একটু ভেবে কিনবেন.. 😏😏
- [4★ gold=positive, pred=negative] product টি মোটামুটি ভালো তবে ভিতরে একটু মরিচা পড়া ছিল। যাই হোক তা নিয়ে আমার কোন সমস্যা নেই কিন্তু Daraz থেকে এটা প্রসেসিং হয়ে ডেলিভারি দিতে অনেক দেরি করেছে। 
- [3★ gold=negative, pred=positive] Delivery was fast. দেখে ভালই মনে হচ্ছে। ২ মিনিটের মত পানিতে ভিজিয়ে রাখছি কিন্তু পানি ডুকেনি। ⚠️দুঃখের বিষয় হইলো ঘড়িটা একটা বক্স দেয় নি। ⚠️রিসিভ ভাউচার সাথে 
- [4★ gold=positive, pred=negative] সবগুলো সাইজে ছোট, সাইজ 6.7 x 8.75 Inch. কুয়ালিটি বেশ ভালো সাইজগুলো ঠিক থাকলে পারফেক্ট হতো
