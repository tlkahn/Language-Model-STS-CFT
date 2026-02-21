# STS Evaluation Results

**Model:** MiniCPM-2B-dpo-bf16 + LoRA adapter (`20260221004650`)
**Date:** 2026-02-21
**GPU:** NVIDIA A10 (22 GiB)
**MTEB version:** 1.7.4
**W&B run:** [sts-eval](https://wandb.ai/yvvai/LM-STS-CFT/runs/jzlv7nc9)

## Task Descriptions

| Task | Test Pairs | Domain | Description |
|------|-----------|--------|-------------|
| BIOSSES | 100 | Biomedical | Sentence pairs from PubMed articles, scored 0-4 by expert annotators for biomedical semantic similarity. |
| SICK-R | 9,927 | Image/video captions | Pairs from Flickr 8K captions and MSR-Video descriptions, systematically transformed to test compositional semantics (negation, passivization, lexical substitution); scored 1-5 for relatedness. |
| STS12 | 3,108 | News, captions, MT, glosses | SemEval 2012 pilot task; pairs from newswire paraphrases (MSRpar), video descriptions (MSRvid), MT evaluation output, and WordNet/OntoNotes glosses. |
| STS13 | 1,500 | News, glosses, MT | SemEval 2013; pairs from news headlines, FrameNet-WordNet gloss mappings, OntoNotes-WordNet glosses, and MT output. |
| STS14 | 3,750 | News, captions, forums, tweets | SemEval 2014; pairs from headlines, image descriptions, DEFT forums, DEFT news, OntoNotes-WordNet glosses, and tweet-news pairs. |
| STS15 | 3,000 | News, captions, Q&A, student answers | SemEval 2015; pairs from headlines, image captions, Stack Exchange answers, BEETLE tutoring answers, and belief annotations. |
| STS16 | 1,186 | Q&A, news, plagiarism, MT | SemEval 2016; pairs from Stack Exchange Q&A, news headlines, plagiarism detection corpora, and MT post-editing data. |
| STS17 | 250 (en-en) | NLI captions | SemEval 2017; English pairs sourced from SNLI corpus, part of a broader multilingual/crosslingual evaluation. |
| STS22 | 1,930 (en) | News articles | SemEval 2022; paragraph-level news article similarity, annotated across geographic focus, entities, narrative, and overall similarity. |
| STSBenchmark | 1,379 | News, captions, forums | Curated selection of English STS pairs from SemEval 2012-2017, combining headlines, image/video captions, and user forum text into a single standardized benchmark. |

## Results (Cosine Similarity)

| Task | Spearman | Pearson | Eval Time |
|------|----------|---------|-----------|
| BIOSSES | 0.7409 | 0.7678 | 9s |
| SICK-R | 0.7578 | 0.8157 | 13m 23s |
| STS12 | 0.6496 | 0.7518 | 4m 13s |
| STS13 | 0.8052 | 0.8034 | 2m 1s |
| STS14 | 0.7037 | 0.7489 | 5m 2s |
| STS15 | 0.8172 | 0.8149 | 4m 3s |
| STS16 | 0.8207 | 0.8137 | 1m 36s |
| STS17 (en-en) | 0.8821 | 0.8809 | 21s |
| STS22 (en) | 0.2895 | 0.2669 | 47s |
| STSBenchmark | 0.8187 | 0.8346 | 1m 50s |
| **Average** | **0.7285** | **0.7499** | |

Note: The eval script reported avg Spearman of 0.7642 over 8 tasks because STS17 and STS22 use nested language splits (`en-en`, `en`) that the parser did not extract. The 10-task average above includes those manually.

## Observations

- **STSBenchmark** (the standard headline benchmark) scored **0.8187** Spearman, a strong result for a 2B-parameter model with LoRA fine-tuning.
- **STS22** scored 0.29 — this is a known-hard cross-domain paragraph-level benchmark; low scores are typical for non-specialized models.
- **STS17** scored highest at 0.88, indicating strong performance on multilingual-origin English pairs.
- **STS12** scored lowest (excluding STS22) at 0.65, consistent with its older, more heterogeneous sentence pairs.
- Total eval time: ~33 minutes at ~25 sent/s (single-sentence encoding on A10).
