---
title: "Improving Text Embeddings with Large Language Models"
source: "https://arxiv.org/html/2401.00368v3"
author:
published:
created: 2026-02-20
description:
tags:
 - "clippings"
---
Liang Wang, Nan Yang, Xiaolong Huang,
Linjun Yang, Rangan Majumder, Furu Wei
Microsoft Corporation
{wangliang,nanya,xiaolhu,yang.linjun,ranganm,fuwei}@microsoft.com

###### Abstract

In this paper, we introduce a novel and simple method for obtaining high-quality text embeddings using only synthetic data and less than $1$ k training steps. Unlike existing methods that often depend on multi-stage intermediate pre-training with billions of weakly-supervised text pairs, followed by fine-tuning with a few labeled datasets, our method does not require building complex training pipelines or relying on manually collected datasets that are often constrained by task diversity and language coverage. We leverage proprietary LLMs to generate diverse synthetic data for hundreds of thousands of text embedding tasks across $93$ languages. We then fine-tune open-source decoder-only LLMs on the synthetic data using standard contrastive loss. Experiments demonstrate that our method achieves strong performance on highly competitive text embedding benchmarks without using any labeled data. Furthermore, when fine-tuned with a mixture of synthetic and labeled data, our model sets new state-of-the-art results on the BEIR and MTEB benchmarks.

Improving Text Embeddings with Large Language Models

Liang Wang, Nan Yang, Xiaolong Huang,Linjun Yang, Rangan Majumder, Furu Wei Microsoft Corporation {wangliang,nanya,xiaolhu,yang.linjun,ranganm,fuwei}@microsoft.com

## 1 Introduction

Text embeddings are vector representations of natural language that encode its semantic information. They are widely used in various natural language processing (NLP) tasks, such as information retrieval (IR), question answering, semantic textual similarity, bitext mining, item recommendation, etc. In the field of IR, the first-stage retrieval often relies on text embeddings to efficiently recall a small set of candidate documents from a large-scale corpus using approximate nearest neighbor search techniques. Embedding-based retrieval is also a crucial component of retrieval-augmented generation (RAG) [^24], which is an emerging paradigm that enables large language models (LLMs) to access dynamic external knowledge without modifying the model parameters. Source attribution of generated text is another important application of text embeddings [^16] that can improve the interpretability and trustworthiness of LLMs.

Previous studies have demonstrated that weighted average of pre-trained word embeddings [^38] is a strong baseline for measuring semantic similarity. However, these methods fail to capture the rich contextual information of natural language. With the advent of pre-trained language models [^12], Sentence-BERT [^40] and SimCSE [^15] have been proposed to learn text embeddings by fine-tuning BERT on natural language inference (NLI) datasets. To further enhance the performance and robustness of text embeddings, state-of-the-art methods like E5 [^49] and BGE [^52] employ a more complex multi-stage training paradigm that first pre-trains on billions of weakly-supervised text pairs, and then fine-tunes on several high-quality labeled datasets.

Existing multi-stage approaches suffer from several drawbacks. Firstly, they entail a complex multi-stage training pipeline that demands substantial engineering efforts to curate large amounts of relevance pairs. Secondly, they rely on manually collected datasets that are often constrained by the diversity of tasks and the coverage of languages. For instance, Instructor [^43] is only trained on instructions from $330$ English datasets, whereas BGE [^52] only focuses on high-resource languages such as English and Chinese. Moreover, most existing methods employ BERT-style encoders as the backbone, neglecting the recent advances of training better LLMs and related techniques such as context length extension [^41].

In this paper, we propose a novel method for text embeddings that leverages LLMs to overcome the limitations of existing approaches. We use proprietary LLMs to generate synthetic data for a diverse range of text embedding tasks in $93$ languages, covering hundreds of thousands of embedding tasks. Specifically, we use a two-step prompting strategy that first prompts the LLMs to brainstorm a pool of candidate tasks, and then prompts the LLMs to generate data conditioned on a given task from the pool. To cover various application scenarios, we design multiple prompt templates for each task type and combine the generated data from different templates to boost diversity. For the text embedding models, we opt for fine-tuning powerful open-source LLMs rather than small BERT-style models. Since LLMs such as Mistral [^21] have been extensively pre-trained on web-scale data, contrastive pre-training that proves to be important for BERT models [^49] offers little additional benefit.

We demonstrate that Mistral-7B, when fine-tuned solely on synthetic data, attains competitive performance on the BEIR [^45] and MTEB [^31] benchmarks. This is particularly intriguing considering that this setting does not involve any labeled data. When fine-tuned on a mixture of synthetic and labeled data, our model achieves new state-of-the-art results, surpassing previous methods by a significant margin (+ $2\%$ ). The entire training process requires less than $1$ k steps.

Moreover, we empirically validate that our model can effectively perform personalized passkey retrieval for inputs up to $32$ k tokens by altering the rotation base of the position embeddings, extending the context length beyond the conventional $512$ token limit. Regarding its multilinguality, our model excels on high-resource languages. However, for low-resource languages, there is still room for improvement as current open-source LLMs are not adequately pre-trained on them.

## 2 Related Work

![Refer to caption](https://arxiv.org/html/2401.00368v3/x1.png)

Figure 1: An example two-step prompt template for generating synthetic data with GPT-4. We first prompt GPT-4 to brainstorm a list of potential retrieval tasks, and then generate (query, positive, hard negative) triplets for each task. “ { … } ” denotes a placeholder that will be replaced by sampling from a predefined set of values. Full prompts are available in Appendix C.

Text Embeddings are continuous low-dimensional representations of text and have been extensively applied to various downstream tasks such as information retrieval, question answering, and retrieval-augmented generation (RAG). Early work on text embeddings includes latent semantic indexing [^11] and weighted average of word embeddings [^28]. More recent methods exploit supervision from natural language inference [^4] and labeled query-document pairs, such as the MS-MARCO passage ranking dataset [^6], to train text embeddings [^40]. However, labeled data are often limited in terms of task diversity and language coverage. To address this challenge, methods like Contriever [^20], OpenAI Embeddings [^33], E5 [^49], and BGE [^52] adopt a multi-stage training paradigm. They first pre-train on large-scale weakly-supervised text pairs using contrastive loss and then fine-tune on small-scale but high-quality datasets. In this paper, we demonstrate that it is possible to obtain state-of-the-art text embeddings with single-stage training.

Synthetic Data Synthetic data generation is a widely studied topic in information retrieval research, with various methods proposed to enhance retrieval systems with artificially created data. For instance, Doc2query [^36], InPars [^3], and Promptagator [^9] generate synthetic queries for unlabeled documents, which are then leveraged for document expansion or model training. GPL [^48] employs a cross-encoder to produce pseudo-labels for query-document pairs. Similarly, Query2doc [^51] generates pseudo-documents for query expansion by few-shot prompting LLMs. Unlike these methods, our approach does not rely on any unlabeled documents or queries and thus can generate more diverse synthetic data.

Another related line of work focuses on knowledge distillation from black-box LLMs by training on synthetic data generated from them. DINO [^42] generates synthetic text pairs for semantic textual similarity. Unnatural Instructions [^18] is a synthetic instruction following dataset by prompting existing LLMs. Orca [^32] and Phi [^17] propose to train better small language models by using high-quality synthetic data from GPT-3.5/4 [^37].

Large Language Models With the popularization of ChatGPT, large language models (LLMs) have demonstrated remarkable capabilities in instruction following and few-shot in-context learning [^5]. However, the most advanced LLMs such as GPT-4 [^37] are proprietary and have little technical details disclosed. To bridge the gap between proprietary and open-source LLMs, several notable efforts have been made, such as LLaMA-2 [^47] and Mistral [^21] models. A major limitation of LLMs is that they lack awareness of recent events and private knowledge. This issue can be partly mitigated by augmenting LLMs with information retrieved from external sources, a technique known as retrieval-augmented generation (RAG). On the other hand, LLMs can also serve as foundation models to enhance text embeddings. RepLLaMA [^27] proposes to fine-tune LLaMA-2 with bi-encoder architecture for ad-hoc retrieval. SGPT [^30], GTR [^35], and Udever [^55] demonstrate the scaling law of text embeddings empirically, but their performance still falls behind small bidirectional encoders such as E5 [^49] and BGE [^52]. In this paper, we present a novel approach to train state-of-the-art text embeddings by exploiting the latest advances of LLMs and synthetic data.

## 3 Method

### 3.1 Synthetic Data Generation

Utilizing synthetic data generated by advanced LLMs such as GPT-4 presents a compelling opportunity, especially in terms of enhancing diversity across a multitude of tasks and languages. Such diversity is essential for developing robust text embeddings that can perform well across different tasks, be it semantic retrieval, textual similarity, or clustering.

To generate diverse synthetic data, we propose a simple taxonomy that categorizes embedding tasks into several groups, and then apply different prompt templates to each group.

Asymmetric Tasks This category comprises tasks where the query and document are semantically related but are not paraphrases of each other. Depending on the length of the query and document, we further divide asymmetric tasks into four subgroups: short-long match, long-short match, short-short match, and long-long match. For instance, short-long match tasks involve a short query and a long document, which is a typical scenario in commercial search engines. For each subgroup, we design a two-step prompt template that first prompts LLMs brainstorm a list of tasks, and then generates a concrete example conditioned on the task definition. In Figure [1](https://arxiv.org/html/2401.00368v3/2401.00368v3#S2.F1), we show an example prompt for the short-long match subgroup. The full output is available in Table [16](https://arxiv.org/html/2401.00368v3/2401.00368v3#A4.T16). The outputs from GPT-4 are mostly coherent and of high quality. In our preliminary experiments, we also attempted to generate the task definition and query-document pairs using a single prompt, but the data diversity was not as satisfactory as the proposed two-step approach.

Symmetric Tasks Symmetric tasks involve queries and documents that have similar semantic meanings but different surface forms. We examine two application scenarios: monolingual semantic textual similarity (STS) and bitext retrieval. We design two distinct prompt templates for each scenario, tailored to their specific objectives. Since the task definition is straightforward, we omit the brainstorming step for symmetric tasks.

To further boost the diversity of the prompts and thus the synthetic data, we incorporate several placeholders in each prompt template, whose values are randomly sampled at runtime. For example, in Figure [1](https://arxiv.org/html/2401.00368v3/2401.00368v3#S2.F1), the value of “ *{ query\_length }* ” is sampled from the set “ *{ less than 5 words, 5-10 words, at least 10 words }* ”.

To generate multilingual data, we sample the value of “ *{ language }* ” from the language list of XLM-R [^7], giving more weight to high-resource languages. Any generated data that does not conform to the predefined JSON format are discarded during the parsing process. We also remove duplicates based on exact string matching.

### 3.2 Training

Given a relevant query-document pair ( $q^{+},d^{+}$ ), we first apply the following instruction template to the original query $q^{+}$ to generate a new one $q^{+}_{\text{inst}}$ :

$q^{+}_{\text{inst}}=\text{Instruct: \{task\_definition\}}\ \backslash n\ \text{Query:\ }\{q^{+}\}$ (1)

where “ *{ task\_definition }* ” is a placeholder for a one-sentence description of the embedding task. For generated synthetic data, we use the outputs from the brainstorming step. For other datasets, such as MS-MARCO, we manually craft the task definitions and apply them to all the queries in the dataset. We do not modify the document side with any instruction prefix. In this way, the document index can be prebuilt, and we can customize the task to perform by changing only the query side.

Given a pretrained LLM, we append an \[EOS\] token to the end of the query and document, and then feed them into the LLM to obtain the query and document embeddings ( $\mathbf{h}_{q^{+}_{\text{inst}}},\mathbf{h}_{d^{+}}$ ) by taking the last layer \[EOS\] vector. To train the embedding model, we adopt the standard InfoNCE loss $\mathbb{L}$ over the in-batch negatives and hard negatives:

$\min\ \ \mathbb{L}=-\log\frac{\phi(q^{+}_{\text{inst}},d^{+})}{\phi(q^{+}_{\text{inst}},d^{+})+\displaystyle\sum_{n_{i}\in\mathbb{N}}(\phi(q^{+}_{\text{inst}},n_{i}))}$ (2)

where $\mathbb{N}$ denotes the set of all negatives, and $\phi(q,d)$ is a function that computes the matching score between query $q$ and document $d$ . In this paper, we adopt the temperature-scaled cosine similarity function as follows:

$\phi(q,d)=\text{exp}(\frac{1}{\tau}\cos(\mathbf{h}_{q},\mathbf{h}_{d}))$ (3)

$\tau$ is a temperature hyper-parameter, which is fixed to $0.02$ in our experiments.

## 4 Experiments

### 4.1 Statistics of the Synthetic Data

![Refer to caption](https://arxiv.org/html/2401.00368v3/x2.png)

Figure 2: Task type and language statistics of the generated synthetic data (see Section 3.1 for task type definitions). The “Others” category contains the remaining languages from the XLM-R language list.

| # of datasets $\rightarrow$             | Class. | Clust. | PairClass. | Rerank | Retr. | STS  | Summ. | Avg  |
| --------------------------------------- | ------ | ------ | ---------- | ------ | ----- | ---- | ----- | ---- |
|                                         | 12     | 11     | 3          | 4      | 15    | 10   | 1     | 56   |
| *Unsupervised Models*                   |        |        |            |        |       |      |       |      |
| Glove [^38]                             | 57.3   | 27.7   | 70.9       | 43.3   | 21.6  | 61.9 | 28.9  | 42.0 |
| SimCSE ${}_{\text{bert-unsup}}$ [^15]   | 62.5   | 29.0   | 70.3       | 46.5   | 20.3  | 74.3 | 31.2  | 45.5 |
| *Supervised Models*                     |        |        |            |        |       |      |       |      |
| SimCSE ${}_{\text{bert-sup}}$ [^15]     | 67.3   | 33.4   | 73.7       | 47.5   | 21.8  | 79.1 | 23.3  | 48.7 |
| Contriever [^20]                        | 66.7   | 41.1   | 82.5       | 53.1   | 41.9  | 76.5 | 30.4  | 56.0 |
| GTR ${}_{\text{xxl}}$ [^35]             | 67.4   | 42.4   | 86.1       | 56.7   | 48.5  | 78.4 | 30.6  | 59.0 |
| Sentence-T5 ${}_{\text{xxl}}$ [^34]     | 73.4   | 43.7   | 85.1       | 56.4   | 42.2  | 82.6 | 30.1  | 59.5 |
| E5 ${}_{\text{large-v2}}$ [^49]         | 75.2   | 44.5   | 86.0       | 56.6   | 50.6  | 82.1 | 30.2  | 62.3 |
| GTE ${}_{\text{large}}$ [^26]           | 73.3   | 46.8   | 85.0       | 59.1   | 52.2  | 83.4 | 31.7  | 63.1 |
| BGE ${}_{\text{large-en-v1.5}}$ [^52]   | 76.0   | 46.1   | 87.1       | 60.0   | 54.3  | 83.1 | 31.6  | 64.2 |
| *Ours*                                  |        |        |            |        |       |      |       |      |
| E5 ${}_{\text{mistral-7b}}$ + full data | 78.5   | 50.3   | 88.3       | 60.2   | 56.9  | 84.6 | 31.4  | 66.6 |
| w/ synthetic data only                  | 78.2   | 50.5   | 86.0       | 59.0   | 46.9  | 81.2 | 31.9  | 63.1 |
| w/ synthetic + msmarco                  | 78.3   | 49.9   | 87.1       | 59.5   | 52.2  | 81.2 | 32.7  | 64.5 |

Table 1: Results on the MTEB benchmark [^31] (56 datasets in the English subset). The numbers are averaged for each category. Please refer to Table [17](https://arxiv.org/html/2401.00368v3/2401.00368v3#A4.T17) for the scores per dataset.

Figure [2](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.F2) presents the statistics of our generated synthetic data. We manage to generate $500$ k examples with $150$ k unique instructions using Azure OpenAI Service <sup>1</sup> <sup>1</sup> 1 [https://oai.azure.com/](https://oai.azure.com/), among which $25\%$ are generated by *GPT-35-Turbo* and others are generated by *GPT-4*. The total token consumption is about $180$ M. The predominant language is English, with coverage extending to a total of $93$ languages. For the bottom $75$ low-resource languages, there are about $1$ k examples per language on average. Please see Table [16](https://arxiv.org/html/2401.00368v3/2401.00368v3#A4.T16) in the appendix for examples of synthetic data.

In terms of data quality, we find that a portion of *GPT-35-Turbo* outputs do not strictly follow the guidelines specified in the prompt templates. Nevertheless, the overall quality remains acceptable, and preliminary experiments have demonstrated the benefits of incorporating this data subset.

### 4.2 Model Fine-tuning and Evaluation

| High-resource Languages                 |      |      |      | Low-resource Languages |      |      |      |      |
| --------------------------------------- | ---- | ---- | ---- | ---------------------- | ---- | ---- | ---- | ---- |
| en                                      | fr   | es   | ru   | te                     | hi   | bn   | sw   |      |
| BM25 [^57]                              | 35.1 | 18.3 | 31.9 | 33.4                   | 49.4 | 45.8 | 50.8 | 38.3 |
| mDPR [^57]                              | 39.4 | 43.5 | 47.8 | 40.7                   | 35.6 | 38.3 | 44.3 | 29.9 |
| mE5 ${}_{\text{base}}$ [^50]            | 51.2 | 49.7 | 51.5 | 61.5                   | 75.2 | 58.4 | 70.2 | 71.1 |
| mE5 ${}_{\text{large}}$ [^50]           | 52.9 | 54.5 | 52.9 | 67.4                   | 84.6 | 62.0 | 75.9 | 74.9 |
| E5 ${}_{\text{mistral-7b}}$ + full data | 57.3 | 55.2 | 52.2 | 67.7                   | 73.9 | 52.1 | 70.3 | 68.4 |

Table 2: nDCG@10 on the dev set of the MIRACL dataset for both high-resource and low-resource languages. We select the $4$ high-resource languages and the $4$ low-resource languages according to the number of candidate documents. The numbers for BM25 and mDPR come from [^57]. For the complete results on all $16$ languages, please see Table [6](https://arxiv.org/html/2401.00368v3/2401.00368v3#A1.T6).

![Refer to caption](https://arxiv.org/html/2401.00368v3/x3.png)

Figure 3: Effects of contrastive pre-training. Detailed numbers are in Appendix Table 7.

The pretrained Mistral-7b [^21] checkpoint is fine-tuned for $1$ epoch using the loss in Equation [2](https://arxiv.org/html/2401.00368v3/2401.00368v3#S3.E2). We follow the training recipe from RankLLaMA [^27] and utilize LoRA [^19] with rank $16$ . To further reduce GPU memory requirement, techniques including gradient checkpointing, mixed precision training, and DeepSpeed ZeRO-3 are applied.

For the training data, we utilize both the generated synthetic data and a collection of $13$ public datasets, yielding approximately $1.8$ M examples after sampling. More details are available in Appendix [A](https://arxiv.org/html/2401.00368v3/2401.00368v3#A1). To provide a fair comparison with some previous work, we also report results when the only labeled supervision is the MS-MARCO passage ranking [^6] dataset.

We evaluate the trained model on the MTEB benchmark [^31]. Note that the retrieval category in MTEB corresponds to the $15$ publicly available datasets in the BEIR benchmark [^45]. Evaluation of one model takes about $3$ days on $8$ V100 GPUs due to the need to encode a large number of documents. Although our model can accommodate sequence length beyond $512$ , we only evaluate on the first $512$ tokens for efficiency. Official metrics are reported for each category. For more details about the evaluation protocol, please refer to the original papers [^31].

### 4.3 Main Results

In Table [1](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.T1), our model “E5 ${}_{\text{mistral-7b}}$ + full data” attains the highest average score on the MTEB benchmark, outperforming the previous state-of-the-art model by $2.4$ points. In the “w/ synthetic data only” setting, no labeled data is used for training, and yet the performance remains quite competitive. We posit that generative language modeling and text embeddings are the two sides of the same coin, with both tasks requiring the model to have a deep understanding of the natural language. Given an embedding task definition, a truly robust LLM should be able to generate training data on its own and then be transformed into an embedding model through light-weight fine-tuning. Our experiments shed light on the potential of this direction, and more research is needed to fully explore it.

| Model                                   | BEIR | MTEB |
|-----------------------------------------|------|------|
| OpenAI text-embedding-3-large           | 55.4 | 64.6 |
| Cohere-embed-english-v3.0               | 55.0 | 64.5 |
| voyage-lite-01-instruct                 | 55.6 | 64.5 |
| UAE-Large-V1                            | 54.7 | 64.6 |
| E5 ${}_{\text{mistral-7b}}$ + full data | 56.9 | 66.6 |

Table 3: Comparison with commercial models and the model that tops the MTEB leaderboard (as of 2023-12-22) [^25]. “BEIR” is the average nDCG@10 score over $15$ public datasets in the BEIR benchmark [^45]. “MTEB” is the average score over $56$ datasets in the English subset of the MTEB benchmark [^31]. For the commercial models listed here, little details are available on their model architectures and training data.

![Refer to caption](https://arxiv.org/html/2401.00368v3/x4.png)

Figure 4: Illustration of the personalized passkey retrieval task adapted from 29. The “ ” and “ ” are repeats of “ The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ” In addition, each document has a unique person name and a random passkey inserted at a random position. The task is to retrieve the document that contains the given person’s passkey from 100 candidates.

![Refer to caption](https://arxiv.org/html/2401.00368v3/x5.png)

Figure 5: Accuracy of personalized passkey retrieval as a function of input context length. For each context length, we randomly generate 50 queries and compute the top-1 accuracy.

In Table [3](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.T3), we also present a comparison with several commercial text embedding models. However, due to the lack of transparency and documentation about these models, a fair comparison is not feasible. We focus especially on the retrieval performance on the BEIR benchmark, since retrieval-augmented generation is an emerging technique to enhance LLM with external knowledge and proprietary data. As Table [3](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.T3) shows, our model outperforms the current commercial models by a significant margin.

### 4.4 Multilingual Retrieval

To assess the multilingual capabilities of our model, we conduct an evaluation on the MIRACL dataset [^57], which comprises human-annotated queries and relevance judgments across $18$ languages. The validation set contains labels for $16$ languages. As shown in Table [2](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.T2), our model surpasses mE5 ${}_{\text{large}}$ on high-resource languages, notably on English. Nevertheless, for low-resource languages, our model remains suboptimal compared to mE5 ${}_{\text{base}}$ . We attribute this to the fact that Mistral-7B is predominantly pre-trained on English data, and we anticipate that future multilingual LLMs will leverage our method to bridge this gap.

|                             | \| BUCC 2018 \| \| --- \| \| 4 langs \| | \| Tatoeba \| \| --- \| \| 112 langs \| | 
| --------------------------- | --------------------------------------- | --------------------------------------- | 
| mContriever                 | 93.7                                    | 37.7                                    | 
| LaBSE                       | 98.8                                    | 81.1                                    | 
| mE5 ${}_{\text{base}}$      | 98.1                                    | 68.1                                    | 
| mE5 ${}_{\text{large}}$     | 98.6                                    | 75.7                                    | 
| E5 ${}_{\text{mistral-7b}}$ | 98.9                                    | 70.1                                    | 

Table 4: Bitext mining results. BUCC 2018 [^59] contains $4$ high-resource languages. Tatoeba [^2] consists of $112$ English-centric language pairs.

To evaluate our model’s cross-lingual retrieval capability, we report Bitext mining results in Table [4](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.T4). For baselines including mContriever [^20], LaBSE [^14], and mE5 [^50], we evaluate the results using publicly available checkpoints. Our observations indicate that, similar to the MIRACL retrieval, E5 ${}_{\text{mistral-7b}}$ excels in bitext mining for high-resource languages only.

| Datasets                    | Class. | Clust. | PairClass. | Rerank | Retr. | STS  | Summ. | Avg                     |
| --------------------------- | ------ | ------ | ---------- | ------ | ----- | ---- | ----- | ----------------------- |
| E5 ${}_{\text{mistral-7b}}$ | 78.3   | 49.9   | 87.1       | 59.5   | 52.2  | 81.2 | 32.7  | 64.5                    |
| w/ LLaMA-2 7b init.         | 76.2   | 48.1   | 85.1       | 58.9   | 49.6  | 81.2 | 30.8  | 62.9 ${}^{\text{-1.6}}$ |
| w/ msmarco data only        | 71.6   | 47.1   | 86.1       | 58.8   | 54.4  | 79.5 | 31.7  | 62.7 ${}^{\text{-1.8}}$ |
| *pooling type*              |        |        |            |        |       |      |       |                         |
| w/ mean pool                | 77.0   | 48.9   | 86.1       | 59.2   | 52.4  | 81.4 | 30.8  | 64.1 ${}^{\text{-0.4}}$ |
| w/ weighted mean            | 77.0   | 49.0   | 86.1       | 59.2   | 52.0  | 81.4 | 30.2  | 64.0 ${}^{\text{-0.5}}$ |
| *LoRA rank*                 |        |        |            |        |       |      |       |                         |
| w/ r= 8 8 8 8               | 78.4   | 50.3   | 87.1       | 59.3   | 53.0  | 81.0 | 31.7  | 64.8 ${}^{\text{+0.3}}$ |
| w/ r= 32 32 32 32           | 78.4   | 50.3   | 87.4       | 59.5   | 52.2  | 81.2 | 30.6  | 64.6 ${}^{\text{+0.1}}$ |
| *instruction type*          |        |        |            |        |       |      |       |                         |
| w/o instruction             | 72.3   | 47.1   | 82.6       | 56.3   | 48.2  | 76.7 | 30.7  | 60.3 ${}^{\text{-4.2}}$ |
| w/ task type prefix         | 71.1   | 46.5   | 79.7       | 54.0   | 52.7  | 73.8 | 30.0  | 60.3 ${}^{\text{-4.2}}$ |

Table 5: Results on the MTEB benchmark with various hyperparameters. The first row corresponds to the default setting, which employs last-token pooling, LoRA rank $16$ , and natural language instructions. Unless otherwise stated, all models are trained on the synthetic and MS-MARCO passage ranking data.

## 5 Analysis

### 5.1 Is Contrastive Pre-training Necessary?

Weakly-supervised contrastive pre-training is one of the key factors behind the success of existing text embedding models. For instance, Contriever [^20] treats random cropped spans as positive pairs for pre-training, while E5 [^49] and BGE [^52] collect and filter text pairs from various sources.

This section re-evaluates the necessity of contrastive pre-training for LLMs, particularly those that have been pre-trained on trillions of tokens. Figure [3](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.F3) shows that contrastive pre-training benefits XLM-R ${}_{\text{large}}$ , enhancing its retrieval performance by $8.2$ points when fine-tuned on the same data, which aligns with prior findings. However, for Mistral-7B based models, contrastive pre-training has negligible impact on the model quality. This implies that extensive auto-regressive pre-training enables LLMs to acquire good text representations, and only minimal fine-tuning is required to transform them into effective embedding models.

### 5.2 Extending to Long Text Embeddings

Existing evaluation datasets for text embedding models are typically short, to evaluate the long-context capability of our model, we introduce a novel synthetic task called *personalized passkey retrieval*, which is illustrated in Figure [4](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.F4). This task requires encoding the passkey information in a long context into the embeddings. We compare the performance of different variants by changing the sliding window size and the RoPE rotation base [^44] in Figure [5](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.F5). The results show that the default configuration with $4$ k sliding window attains $100\%$ accuracy within $4$ k tokens, but the accuracy deteriorates quickly as the context length grows. Naively extending the sliding window size to $32$ k results in worse performance. By changing the RoPE rotation base to $10^{5}$ , the model can achieve over $90\%$ accuracy within $32$ k tokens. However, this entails a minor trade-off in performance for shorter contexts. A potential avenue for future research is to efficiently adapt the model to longer contexts through lightweight post-training [^58].

### 5.3 Analysis of Training Hyperparameters

Table [5](https://arxiv.org/html/2401.00368v3/2401.00368v3#S4.T5) presents the results under different configurations. We notice that the Mistral-7B initialization holds an advantage over LLaMA-2 7B, in line with the findings from Mistral-7B technical report [^21]. The choice of pooling types and LoRA ranks does not affect the overall performance substantially, hence we adhere to the default setting despite the marginal superiority of LoRA rank $8$ . On the other hand, the way of adding instructions has a considerable impact on the performance. We conjecture that natural language instructions better inform the model regarding the embedding task at hand, and thus enable the model to generate more discriminative embeddings. Our framework also provides a way to customize the behavior of text embeddings through instructions without the need to fine-tune the model or re-build document index.

## 6 Conclusion

This paper shows that the quality of text embeddings can be substantially enhanced by exploiting LLMs. We prompt proprietary LLMs such as GPT-4 to generate diverse synthetic data with instructions in many languages. Combined with the strong language understanding capability of the Mistral model, we establish new state-of-the-art results for nearly all task categories on the competitive MTEB benchmark. The training process is much more streamlined and efficient than existing multi-stage approaches, thereby obviating the need for intermediate pre-training.

For future work, we aim to further improve the multilingual performance of our model and explore the possibility of using open-source LLMs to generate synthetic data.

## Limitations

In comparison to the mainstream BERT-style encoders, the employment of LLMs, such as Mistral-7B, for text embeddings results in a significantly increased inference cost. The development of more advanced GPUs and better kernel implementations may enhance the efficiency of the inference process. With regards to storage cost, our model is comparatively more expensive, with embeddings of $4096$ dimensions. Early successes in reducing embedding dimensions while maintaining competitive performance have been demonstrated through techniques such as Matryoshka representation learning [^23].

For synthetic data generation, we rely on manual prompt engineering to elicit high-quality outputs from proprietary LLMs. Automatic prompt optimization presents a promising avenue for improving the quality of synthetic data.

## Acknowledgements

We would like to thank anonymous reviewers for their valuable comments, and ACL 2024 and ACL Rolling Review organizers for their efforts. Opinions expressed in this paper are solely those of the authors and do not represent the views of their employers.

## References

[^1]: Sanjeev Arora, Yingyu Liang, and Tengyu Ma. 2017.[A simple but tough-to-beat baseline for sentence embeddings](https://openreview.net/forum?id=SyK00v5xx).In *5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings*. OpenReview.net.

[^2]: Mikel Artetxe and Holger Schwenk. 2019.Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and beyond.*Transactions of the Association for Computational Linguistics*, 7:597–610.

[^3]: Luiz Henrique Bonifacio, Hugo Abonizio, Marzieh Fadaee, and Rodrigo Nogueira. 2022.Inpars: Unsupervised dataset generation for information retrieval.*Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*.

[^4]: Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015.[A large annotated corpus for learning natural language inference](https://doi.org/10.18653/v1/D15-1075).In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pages 632–642, Lisbon, Portugal. Association for Computational Linguistics.

[^5]: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.[Language models are few-shot learners](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html).In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*.

[^6]: Daniel Fernando Campos, Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, Li Deng, and Bhaskar Mitra. 2016.[Ms marco: A human generated machine reading comprehension dataset](https://arxiv.org/abs/1611.09268).*ArXiv preprint*, abs/1611.09268.

[^7]: Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020.[Unsupervised cross-lingual representation learning at scale](https://doi.org/10.18653/v1/2020.acl-main.747).In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 8440–8451, Online. Association for Computational Linguistics.

[^8]: Alexis Conneau, Douwe Kiela, Holger Schwenk, Loïc Barrault, and Antoine Bordes. 2017.[Supervised learning of universal sentence representations from natural language inference data](https://doi.org/10.18653/v1/D17-1070).In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pages 670–680, Copenhagen, Denmark. Association for Computational Linguistics.

[^9]: Zhuyun Dai, Vincent Y Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith Hall, and Ming-Wei Chang. 2022.Promptagator: Few-shot dense retrieval from 8 examples.In *The Eleventh International Conference on Learning Representations*.

[^10]: DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, and tomtung. 2017.[Quora question pairs](https://kaggle.com/competitions/quora-question-pairs).

[^11]: Scott Deerwester, Susan T Dumais, George W Furnas, Thomas K Landauer, and Richard Harshman. 1990.Indexing by latent semantic analysis.*Journal of the American society for information science*, 41(6):391–407.

[^12]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.[BERT: Pre-training of deep bidirectional transformers for language understanding](https://doi.org/10.18653/v1/N19-1423).In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

[^13]: Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019.[ELI5: Long form question answering](https://doi.org/10.18653/v1/P19-1346).In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 3558–3567, Florence, Italy. Association for Computational Linguistics.

[^14]: Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022.Language-agnostic bert sentence embedding.In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 878–891.

[^15]: Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021.[SimCSE: Simple contrastive learning of sentence embeddings](https://doi.org/10.18653/v1/2021.emnlp-main.552).In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

[^16]: Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023.[Enabling large language models to generate text with citations](https://arxiv.org/abs/2305.14627).*ArXiv preprint*, abs/2305.14627.

[^17]: Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio Cesar Teodoro Mendes, Allison Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero C. Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, S. Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuan-Fang Li. 2023.[Textbooks are all you need](https://arxiv.org/abs/2306.11644).*ArXiv preprint*, abs/2306.11644.

[^18]: Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick. 2022.[Unnatural instructions: Tuning language models with (almost) no human labor](https://arxiv.org/abs/2212.09689).*ArXiv preprint*, abs/2212.09689.

[^19]: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022.[Lora: Low-rank adaptation of large language models](https://openreview.net/forum?id=nZeVKeeFYf9).In *The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net.

[^20]: Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021.[Towards unsupervised dense information retrieval with contrastive learning](https://arxiv.org/abs/2112.09118).*ArXiv preprint*, abs/2112.09118.

[^21]: Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023.[Mistral 7b](https://arxiv.org/abs/2310.06825).*ArXiv preprint*, abs/2310.06825.

[^22]: Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.[Dense passage retrieval for open-domain question answering](https://doi.org/10.18653/v1/2020.emnlp-main.550).In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781, Online. Association for Computational Linguistics.

[^23]: Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, and Ali Farhadi. 2022.Matryoshka representation learning.In *Neural Information Processing Systems*.

[^24]: Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020.[Retrieval-augmented generation for knowledge-intensive NLP tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*.

[^25]: Xianming Li and Jing Li. 2023.[Angle-optimized text embeddings](https://arxiv.org/abs/2309.12871).*ArXiv preprint*, abs/2309.12871.

[^26]: Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023.[Towards general text embeddings with multi-stage contrastive learning](https://arxiv.org/abs/2308.03281).*ArXiv preprint*, abs/2308.03281.

[^27]: Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2023.[Fine-tuning llama for multi-stage text retrieval](https://arxiv.org/abs/2310.08319).*ArXiv preprint*, abs/2310.08319.

[^28]: Tomas Mikolov, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. 2013.Efficient estimation of word representations in vector space.In *ICLR*.

[^29]: Amirkeivan Mohtashami and Martin Jaggi. 2023.[Landmark attention: Random-access infinite context length for transformers](https://arxiv.org/abs/2305.16300).*ArXiv preprint*, abs/2305.16300.

[^30]: Niklas Muennighoff. 2022.[Sgpt: Gpt sentence embeddings for semantic search](https://arxiv.org/abs/2202.08904).*ArXiv preprint*, abs/2202.08904.

[^31]: Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. 2023.[MTEB: Massive text embedding benchmark](https://aclanthology.org/2023.eacl-main.148).In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pages 2014–2037, Dubrovnik, Croatia. Association for Computational Linguistics.

[^32]: Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Hassan Awadallah. 2023.[Orca: Progressive learning from complex explanation traces of gpt-4](https://arxiv.org/abs/2306.02707).*ArXiv preprint*, abs/2306.02707.

[^33]: Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas A. Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David P. Schnurr, Felipe Petroski Such, Kenny Sai-Kin Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, and Lilian Weng. 2022.[Text and code embeddings by contrastive pre-training](https://arxiv.org/abs/2201.10005).*ArXiv preprint*, abs/2201.10005.

[^34]: Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. 2022a.[Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models](https://doi.org/10.18653/v1/2022.findings-acl.146).In *Findings of the Association for Computational Linguistics: ACL 2022*, pages 1864–1874, Dublin, Ireland. Association for Computational Linguistics.

[^35]: Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022b.[Large dual encoders are generalizable retrievers](https://aclanthology.org/2022.emnlp-main.669).In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 9844–9855, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

[^36]: Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019.[Document expansion by query prediction](https://arxiv.org/abs/1904.08375).*ArXiv preprint*, abs/1904.08375.

[^37]: OpenAI. 2023.[Gpt-4 technical report](https://arxiv.org/abs/2303.08774).*ArXiv preprint*, abs/2303.08774.

[^38]: Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014.[GloVe: Global vectors for word representation](https://doi.org/10.3115/v1/D14-1162).In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1532–1543, Doha, Qatar. Association for Computational Linguistics.

[^39]: Yifu Qiu, Hongyu Li, Yingqi Qu, Ying Chen, QiaoQiao She, Jing Liu, Hua Wu, and Haifeng Wang. 2022.[DuReader-retrieval: A large-scale Chinese benchmark for passage retrieval from web search engine](https://aclanthology.org/2022.emnlp-main.357).In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 5326–5338, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

[^40]: Nils Reimers and Iryna Gurevych. 2019.[Sentence-BERT: Sentence embeddings using Siamese BERT-networks](https://doi.org/10.18653/v1/D19-1410).In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

[^41]: Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, I. Evtimov, Joanna Bitton, Manish P Bhatt, Cristian Cantón Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre D’efossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2023.[Code llama: Open foundation models for code](https://arxiv.org/abs/2308.12950).*ArXiv preprint*, abs/2308.12950.

[^42]: Timo Schick and Hinrich Schütze. 2021.Generating datasets with pretrained language models.In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 6943–6951.

[^43]: Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. 2023.[One embedder, any task: Instruction-finetuned text embeddings](https://doi.org/10.18653/v1/2023.findings-acl.71).In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 1102–1121, Toronto, Canada. Association for Computational Linguistics.

[^44]: Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024.Roformer: Enhanced transformer with rotary position embedding.*Neurocomputing*, 568:127063.

[^45]: Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021.Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models.In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*.

[^46]: James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018.[FEVER: a large-scale dataset for fact extraction and VERification](https://doi.org/10.18653/v1/N18-1074).In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 809–819, New Orleans, Louisiana. Association for Computational Linguistics.

[^47]: Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023.[Llama 2: Open foundation and fine-tuned chat models](https://arxiv.org/abs/2307.09288).*ArXiv preprint*, abs/2307.09288.

[^48]: Kexin Wang, Nandan Thakur, Nils Reimers, and Iryna Gurevych. 2022a.[GPL: Generative pseudo labeling for unsupervised domain adaptation of dense retrieval](https://doi.org/10.18653/v1/2022.naacl-main.168).In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2345–2360, Seattle, United States. Association for Computational Linguistics.

[^49]: Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022b.[Text embeddings by weakly-supervised contrastive pre-training](https://arxiv.org/abs/2212.03533).*ArXiv preprint*, abs/2212.03533.

[^50]: Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024.Multilingual e5 text embeddings: A technical report.*arXiv preprint arXiv:2402.05672*.

[^51]: Liang Wang, Nan Yang, and Furu Wei. 2023.[Query2doc: Query expansion with large language models](https://doi.org/10.18653/v1/2023.emnlp-main.585).In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 9414–9423, Singapore. Association for Computational Linguistics.

[^52]: Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighof. 2023.[C-pack: Packaged resources to advance general chinese embedding](https://arxiv.org/abs/2309.07597).*ArXiv preprint*, abs/2309.07597.

[^53]: Xiaohui Xie, Qian Dong, Bingning Wang, Feiyang Lv, Ting Yao, Weinan Gan, Zhijing Wu, Xiangsheng Li, Haitao Li, Yiqun Liu, et al. 2023.[T2ranking: A large-scale chinese benchmark for passage ranking](https://arxiv.org/abs/2304.03679).*ArXiv preprint*, abs/2304.03679.

[^54]: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018.[HotpotQA: A dataset for diverse, explainable multi-hop question answering](https://doi.org/10.18653/v1/D18-1259).In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

[^55]: Xin Zhang, Zehan Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, and Min Zhang. 2023a.[Language models are universal embedders](https://arxiv.org/abs/2310.08232).*ArXiv preprint*, abs/2310.08232.

[^56]: Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin. 2021.[Mr. TyDi: A multi-lingual benchmark for dense retrieval](https://doi.org/10.18653/v1/2021.mrl-1.12).In *Proceedings of the 1st Workshop on Multilingual Representation Learning*, pages 127–137, Punta Cana, Dominican Republic. Association for Computational Linguistics.

[^57]: Xinyu Crystina Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. 2023b.Miracl: A multilingual retrieval dataset covering 18 diverse languages.*Transactions of the Association for Computational Linguistics*, 11:1114–1131.

[^58]: Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. 2023.[Pose: Efficient context window extension of llms via positional skip-wise training](https://arxiv.org/abs/2309.10400).In *The Twelfth International Conference on Learning Representations*.

[^59]: Pierre Zweigenbaum, Serge Sharoff, and Reinhard Rapp. 2018.Overview of the third bucc shared task: Spotting parallel sentences in comparable corpora.In *Proceedings of 11th Workshop on Building and Using Comparable Corpora*, pages 39–42.
