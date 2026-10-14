<!-- page 1 -->

# **RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics** 

Weiliang Chen 

Department of Computer Science Texas Tech University Lubbock, Texas, USA weilchen@ttu.edu 

## **Abstract** 

As AI agents increasingly rely on external tools, accurate retrieval is essential for reliable behavior. Existing methods optimize semantic relevance but largely ignore tool availability during execution. To address this gap, we propose **RATED** ( **R** eliability- **A** ware **T** ool retrieval via **E** xecution **D** ynamics), a retrieval framework that combines robust semantic matching with dynamic reliability adaptation. RATED employs a frozen dual-encoder with contrastive fine-tuning for semantic matching, and a lightweight contextual bandit module that re-ranks candidates based on real-time execution feedback. This design enables rapid adaptation to runtime tool failures without expensive model retraining. We introduce **EcoAgent** , a benchmark of 5,172 real-world tools and over 210,000 query–tool pairs for offline retrieval and dynamic reliability evaluation. Experimental results demonstrate that RATED achieves a 52.2% relative MRR improvement over the ToolLLM API retriever on real-world queries, and reduces fault selection rate from 86.2% to 2.9% under severe API outage scenarios with a 30% failure rate, while incurring negligible latency overhead of less than 1 ms over encode-only inference. Code and dataset are available at https://github.com/Elliotes/EcoAgent. 

## **CCS Concepts** 

### • **Information systems** → **Retrieval models and ranking** ; • **Computing methodologies** → **Artificial intelligence** . 

## **Keywords** 

AI agents; Tool retrieval; Contextual bandits; Contrastive learning 

**ACM Reference Format:** 

Weiliang Chen and Song Liao. 2026. RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics. In _Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 07–11, 2026, Rome, Italy._ ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3799682.3839887 

## **1 Introduction** 

Large Language Models (LLMs) have demonstrated impressive capabilities in natural language processing but are inherently unable to execute real-world actions [10, 17]. To overcome this limitation, recent systems integrate LLMs with external tools and APIs [15, 24], yielding autonomous AI agents that fulfill actionable requests, such 

This work is licensed under a Creative Commons Attribution 4.0 International License. _CIKM ’26, Rome, Italy_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2539-5/2026/11 

https://doi.org/10.1145/3799682.3839887 

Song Liao Department of Computer Science Texas Tech University Lubbock, Texas, USA song.liao@ttu.edu 





**Figure 1: Tool repository growth and reliability gap on the left, and skewed tag distribution on the right.** 

as querying live databases or managing enterprise workflows [5]. The fundamental bottleneck for these systems has shifted from text generation to accurately identifying and retrieving the correct entry from vast, growing tool repositories [19]. To address the tool retrieval challenge, existing research has focused on optimizing semantic relevance for Large Language Models (LLMs). Foundational works like ToolLLM [21] and ToolExpNet [30] treat tool retrieval as a static ranking task. They use dense retrieval or experience networks to map queries to tool documentation based entirely on semantic relevance. While approaches such as TRFeedback [29] and ToolRL [20] incorporate execution feedback, they still aim to optimize how the LLM plans and invokes tools in a static environment. However, real AI agent tool ecosystems are far from static and highly unreliable. We crawl tool catalogs from LangChain [8], CrewAI [4], LlamaIndex [14], and n8n [16] to reveal the severity of this issue. As Figure 1 shows, despite rapid growth, over 50% of these tools exhibit critical flaws like broken links and missing authentication. Many tools are functionally similar, so retrieval cannot rely on semantics alone to identify which ones will execute successfully. To overcome this challenge, the system must adapt dynamically during execution. When a tool fails at runtime, the system must promptly adjust candidate ordering between requests. ToolRL cannot achieve this prompt adjustment because it uses Group Relative Policy Optimization (GRPO) [25] to update the LLM from batched trajectories and still targets the planner. Contextual bandits [11] effectively address this problem. They condition each decision on the current query and update from each individual execution outcome. Moreover, they observe feedback only for the selected tool and make immediate adjustments without retraining the semantic retriever. 

In this paper, we propose **RATED** ( **R** eliability- **A** ware **T** ool retrieval via **E** xecution **D** ynamics), a retrieval framework that dynamically adapts to runtime tool failures without human intervention. First, we introduce **EcoAgent** , a benchmark of 5,172 real-world agent tools and over 210,000 query–tool pairs. Second, RATED employs a dual-encoder fine-tuned with a contrastive objective for 



<!-- page 2 -->

CIKM ’26, November 07–11, 2026, Rome, Italy 

Weiliang Chen and Song Liao 

semantic matching, together with a lightweight contextual bandit that re-ranks the top- _𝐾_ candidates using a prior based on metadata and execution feedback, enabling the system to avoid failing tools. Experiments show that RATED alone keeps fault selection below 3% and recovers at both failure rates. Under a 30% failure injection protocol, RATED reduces fault selection rate to 2.9%, compared with 86.2% for the ToolLLM API retriever, while adding only 0.172 ms latency per step over encode-only inference. 

## **2 The EcoAgent Benchmark** 

We construct EcoAgent, a benchmark comprising two complementary datasets. EcoAgent complements existing tool-use datasets with metadata reliability scores for every tool and with an evaluation protocol for injected execution failures. EcoAgent-Syn supports robust training with synthetic pairs, and EcoAgent-Real supports rigorous evaluation using real-world queries. We aggregate 5,172 real-world tool definitions from four open-source AI agent platforms, namely LangChain, CrewAI, LlamaIndex, and n8n. Metadata are normalized into a shared schema with functional descriptions, names, endpoints, and authentication fields, which supports retrieval text construction and reliability scoring. To operationalize tool quality, each tool receives an automated metadata reliability score between 0 and 1 from three weighted signals, namely name–description consistency, source credibility, and documentation quality, which Section 3.3 defines. The benchmark does not rely on manual per-tool ratings. Tools with scores of at least 0.8 are marked as reliable, identifying 2,940 tools, or 56.8% of the repository. The encoder is trained only on EcoAgent-Syn, while retrieval and dynamic experiment results are validated on the EcoAgent-Real test split, with hyperparameters selected on the validation split. 

We build EcoAgent-Syn with Gemma-3-1b [7]. For each tool, the model receives the tool name, functional description, and metadata as context and generates 10–20 task-oriented queries. Queries are restricted to 5–120 words, and near-duplicates with Jaccard similarity above 0.7 are removed to preserve intent diversity. Figure 2 shows the prompt template, which we also release with the dataset. To improve robustness against spurious correlations, each query is paired with hard negatives from SBERT [22] retrieval and random negatives. EcoAgent-Syn links 69,501 unique synthetic queries to 5,172 ground-truth tools, with 70,122 positive pairs and 140,244 negative pairs. For evaluation under realistic language distributions, we build EcoAgent-Real from ToolBench [21]. Because the original dataset lacks ground-truth mappings to our tool repository, we implement a semi-automated labeling pipeline. GTE-Large [12] retrieves the top-5 candidates per query, discards pairs with cosine similarity below 0.7, and Gemma-3-1b verifies the remainder, retaining only pairs with functional match scores of at least 0.85. Human experts review a stratified sample of 200 queries and agree with the LLM verifier on 94% of cases. After filtering and deduplication, EcoAgent-Real contains 6,379 unique queries and 9,946 verified query–tool pairs. 

## **3 Methodology** 

## **3.1 Problem Setup** 

We formulate agent tool retrieval as sequential decision making over a shared repository. Let U = { _𝑎_ 1 _, . . . ,𝑎𝑀_ } denote the tool 

|**Prompt for Query Synthesis**|
|---|
|**[Task]**You are generating diverse user queries for an AI agent tool recommenda-<br>tion system.|
|**[Inputs]**{tool_name}the tool name |{tool_description}its functional de-<br>scription |{num_queries}how many queries to write|
|**[Requirements]**|
|(1) Each query should be 5-120 words.<br>(2) Vary the intent types: direct action requests, troubleshooting questions, how-<br>to questions.|
|(3) Use natural, conversational language.<br>(4) DO NOT copy phrases verbatim from the tool description.<br>(5) Make queries task-oriented and practical.|
|**[Output]**One query per line, no numbering or bullets.|



**Figure 2: Prompt template that generates the EcoAgent-Syn queries.** 

pool, where _𝑀_ =5 _,_ 172 in EcoAgent. At step _𝑡_ , the agent issues query _𝑞𝑡_ . The retriever returns a candidate set A _𝑡_ ⊂U with |A _𝑡_ | = _𝐾_ , and the policy selects one tool _𝑎𝑡_ ∈A _𝑡_ for execution. The objective is to maximize the probability of successful execution. Unlike static retrieval with a fixed mapping _𝑓_ ( _𝑞,𝑎_ ) →{0 _,_ 1}, production agents must cope with tools that fail at runtime, where tool utility depends on semantic relevance _𝑆_ sem ( _𝑞𝑡,𝑎_ ) and time-varying reliability. After execution, only the chosen tool yields binary feedback _𝑦𝑡_ ∈{0 _,_ 1}, yielding partial feedback that we address with contextual bandits [1]. Figure 3 summarizes the RATED pipeline. 

## **3.2 Semantic Perception Module** 

The semantic perception module implements the first stage of RATED. Given _𝑞𝑡_ , it scores all tools in U by _𝑆_ sem and returns the top- _𝐾_ candidate set A _𝑡_ defined in Section 3.1. The semantic encoder is based on DRAGON [13], fine-tuned on EcoAgent-Syn and frozen at deployment. New tools can be added by encoding their documentation with _𝑓𝑇_ without retraining the encoders. 

_3.2.1 DRAGON Dual-Encoder Architecture._ DRAGON provides separate query encoder _𝑓𝑄_ and tool encoder _𝑓𝑇_ , which suits settings where user queries and tool documentation follow different lexical patterns. For query _𝑞_ and tool _𝑎_ , we take CLS representations and L2-normalize them. The semantic relevance score is the dot product of the normalized vectors. 





_3.2.2 Contrastive Fine-tuning on EcoAgent-Syn._ Although DRAGON is effective for open-domain retrieval, agent tool descriptions use specialized API terminology. We fine-tune _𝑓𝑄_ and _𝑓𝑇_ on EcoAgent-Syn with an in-batch InfoNCE objective [27]. For a batch of _𝐵_ positive pairs {( _𝑞𝑖,𝑎𝑖_<sup>+)}</sup> _𝑖_<sup>_𝐵_</sup> =1<sup>, we optimize</sup> 



where _𝜏_ is a temperature hyperparameter. Hard negatives from EcoAgent-Syn construction in Section 2 are used in training. Inbatch negatives provide additional contrast during optimization. 



<!-- page 3 -->

CIKM ’26, November 07–11, 2026, Rome, Italy 

RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics 



<!-- Start of picture text -->
Query Encoder  Tool Encoder  For each tool candidate Execute Tool<br>(Transformer)  Shared (Transformer)  RelevanceSemantic BanditScore StaticPrior ExecutionSuccess<br>space Latency /<br>EcoAgent-Syn  Schema Error<br>Construction fusion<br>(Query-Tool Pair &  Ranked  a 2 SecurityAlert<br>List<br>Augmentation)<br>Observe Reward<br>Query Embedding Tool Embedding 1 (success)         0 (fail)<br>Top-k Semantic Retrieval<br>Top-k<br>Q T Contrastive  All Tool candidates Dynamic Update LinUCB<br>User query Learning<br>(e.g.“What is the  Q N Fine-tuning<br>weather like in LA?”)<br>Feedback<br><!-- End of picture text -->

**Figure 3: RATED architecture with DRAGON retrieval, bandit re-ranking, and execution feedback.** 

## **3.3 Reliability-Aware Re-ranking** 

The second stage re-ranks A _𝑡_ from Section 3.2 using operational signals while the semantic encoders remain frozen. At step _𝑡_ , the context is the query embedding x _𝑡_ = e _𝑞𝑡_ ∈ R<sup>_𝑑_</sup> , the policy selects _𝑎𝑡_ ∈A _𝑡_ , and observes reward _𝑦𝑡_ ∈{0 _,_ 1} only after execution, where _𝑦𝑡_ =1 denotes success and _𝑦𝑡_ =0 denotes errors or timeouts. 

Tools with little execution history are initialized from catalog metadata. We define a static prior _𝑟_ static ( _𝑎_ ) ∈[0 _,_ 1] from three signals used in benchmark construction in Section 2, namely name– description consistency _𝑟_ cons ( _𝑎_ ), source credibility _𝑤_ src ( _𝑎_ ), and documentation quality _𝑟_ qual ( _𝑎_ ). 





where<sup>�</sup> _𝑖_<sup>_𝜆_</sup> _𝑖_<sup>=1. These signals implement the catalog reliability</sup> score in Section 2, and all three read catalog fields only, so no manual rating is needed. For _𝑟_ cons, _𝑓_ Enc is the SBERT encoder of Section 2, which embeds the tool name and its description, and we take the cosine of the two normalized vectors. For _𝑤_ src, each platform carries a fixed weight, namely 1.0 for an official listing, 0.9 for LangChain, CrewAI, and LlamaIndex, 0.8 for n8n, and 0.6 otherwise. For _𝑟_ qual, we standardize the description length and the tag-string length against their medians and pass 0 _._ 6 _𝑧_ desc + 0 _._ 4 _𝑧_ tag through a logistic function. The 0.8 threshold is used for benchmark statistics, while _𝑟_ static initializes the bandit. Each arm bias is initialized as b _𝑎_ ← _𝑟_ static ( _𝑎_ ) · x0, which covers the cold-start case, where a new or rarely used tool carries no execution history and is therefore scored from its catalog metadata. 

At runtime, the system must adapt to dynamic failures such as API outages. Standard LinUCB [11] weights all past observations equally and therefore adapts slowly when arm quality shifts over time. Alternatively, a global multi-armed bandit (Global MAB) [2] tracks shared or context-free arm statistics. It is cheaper to maintain but cannot model payoffs that depend on the current query context. To overcome these limitations, RATED adopts discounted LinUCB [6] with a separate linear model per tool. This choice addresses non-stationarity by down-weighting older rewards before each update, ensuring that recent failures change tool scores more 

**Algorithm 1** Reliability-aware adaptation via discounted LinUCB **Input:** query stream { _𝑞𝑡_ }<sup>_𝑇_</sup> _𝑡_ =1<sup>, frozen encoders</sup><sup>_𝑓𝑄_and</sup><sup>_𝑓𝑇_, discount</sup> _𝛾_ , exploration _𝛼_ , fusion weights _𝑤_ 1 _,𝑤_ 2 _,𝑤_ 3 

**Output:** tool _𝑎𝑡_ executed at each step, with (A _𝑎,_ b _𝑎_ ) updated online 

- 1: e _𝑎_ ← norm( _𝑓𝑇_ ( _𝑎_ )) for all _𝑎_ ∈U {offline, reused at every step} 

2: A _𝑎_ ← I _𝑑_ and b _𝑎_ ← _𝑟_ static ( _𝑎_ ) x0 {cold start from metadata} 

3: **for** _𝑡_ = 1 **to** _𝑇_ **do** 



- 11: _𝑎𝑡_ ← arg max _𝑎_ ∈A _𝑡 𝑆_ final ( _𝑎_ ) 

- 12: execute _𝑎𝑡_ and observe _𝑦𝑡_ ∈{0 _,_ 1} 

- 13: A _𝑎𝑡_ ← _𝛾_ A _𝑎𝑡_ + x _𝑡_ x _𝑡_<sup>⊤{kept diagonal, so scoring is</sup><sup>_𝑂_(</sup><sup>_𝑑_)}</sup> 14: b _𝑎𝑡_ ← _𝛾_ b _𝑎𝑡_ + _𝑦𝑡_ x _𝑡_ 

15: **end for** 

quickly. Each tool _𝑎_ maintains A _𝑎_ ∈ R<sup>_𝑑_×</sup><sup>_𝑑_</sup> and b _𝑎_ ∈ R<sup>_𝑑_</sup> . The reliability score combines exploitation and exploration. 



where _𝜽_<sup>ˆ</sup> _𝑎_ = A _𝑎_<sup>−1b</sup><sup>_𝑎_. After executing</sup><sup>_𝑎𝑡_and observing</sup><sup>_𝑦𝑡_, we update</sup> with discount _𝛾_ ∈(0 _,_ 1). 





Discounting emphasizes recent feedback and encourages timely switches to alternatives. 

## **3.4 Hybrid Fusion for Tool Ranking** 

RATED fuses semantic, bandit, and metadata signals into one score over A _𝑡_ . For query _𝑞𝑡_ and candidate _𝑎_ ∈A _𝑡_ , we define 

_𝑆_ final ( _𝑞𝑡,𝑎_ ) = _𝑤_ 1 _𝑆_ sem ( _𝑞𝑡 ,𝑎_ ) + _𝑤_ 2 _𝑆_ bandit (x _𝑡,𝑎_ ) + _𝑤_ 3 _𝑟_ static ( _𝑎_ ) (9) 



<!-- page 4 -->

CIKM ’26, November 07–11, 2026, Rome, Italy 

Weiliang Chen and Song Liao 

**Table 1: Retrieval performance on EcoAgent-Syn and EcoAgent-Real.** 

|Method||EcoAgent-|Syn||EcoAgent-|Real|
|---|---|---|---|---|---|---|
||MRR|Recall@1|Recall@5|MRR|Recall@1|Recall@5|
|TF-IDF|0.680|0.607|0.769|0.274|0.116|0.326|
|BM25|0.235|0.178|0.300|0.169|0.078|0.194|
|OpenAI Ada-002|0.790|0.736|0.857|0.670|0.430|0.750|
|BGE-M3|0.765|0.702|0.840|0.511|0.289|0.579|
|GTE-Large|0.772|0.713|0.842|0.515|0.294|0.581|
|E5-Large-v2|0.746|0.674|0.830|0.597|0.466|0.756|
|DRAGON (pretrained)|0.751|0.690|0.823|0.518|0.407|0.654|
|SPLADE-v3|0.754|0.689|0.834|0.626|0.491|0.789|
|BGE-Reranker|0.596|0.483|0.735|0.642|0.491|0.840|
|ToolLLM API Retriever|0.724|0.666|0.795|0.406|0.222|0.409|
|RATED (w/o Contrastive)|0.751|0.690|0.823|0.518|0.407|0.654|
|RATED (Full)|**0.845**|**0.804**|**0.893**|0.618|0.516|0.740|



where _𝑤_ 1+ _𝑤_ 2+ _𝑤_ 3 = 1. Weights are tuned on the validation split and fixed for testing in Section 4. At each step, the system retrieves top- _𝐾_ tools by _𝑆_ sem, ranks A _𝑡_ by _𝑆_ final, executes the highest-scoring tool, and updates bandit parameters from _𝑦𝑡_ . Algorithm 1 summarizes one step of this loop. 

## **4 Experiments** 

## **4.1 Experimental Setup** 

All experiments run on a single NVIDIA GeForce RTX 5070 GPU (12 GB VRAM). DRAGON is fine-tuned on EcoAgent-Syn for one epoch with batch size 8, learning rate 1 × 10<sup>−5</sup> , 1,000 warmup steps, _𝜏_ = 0 _._ 05, mixed precision, and maximum sequence length 256. At inference, the system retrieves top- _𝐾_ =10 tools by _𝑆_ sem and re-ranks them with Eq. 9. The bandit uses _𝛼_ = 0 _._ 1, _𝛾_ = 0 _._ 95, and 2,000 warm-up pairs, with diagonal covariance updates for _𝑂_ ( _𝑑_ ) scoring at each step. Fusion weights tuned on the EcoAgent-Real validation split are ( _𝑤_ 1 _,𝑤_ 2 _,𝑤_ 3) = (0 _._ 75 _,_ 0 _._ 15 _,_ 0 _._ 10). Static weights ( _𝜆_ 1 _, 𝜆_ 2 _, 𝜆_ 3) in Eq. 5 are (0 _._ 5 _,_ 0 _._ 3 _,_ 0 _._ 2), and biases are initialized with x0 = 1/√ _𝑑_ . 

## **4.2 Evaluation Metrics** 

Offline retrieval is measured with MRR, Recall@1, and Recall@5 on the EcoAgent-Real test split. EcoAgent-Syn uses the same metrics for in-domain comparison. Dynamic reliability is evaluated over _𝑇_ =10 _,_ 000 sequential selections. At step _𝑡_ , the policy selects _𝑎𝑡_ for query _𝑞𝑡_ with ground truth _𝑎𝑡_<sup>∗. The indicator Status(</sup><sup>_𝑎,𝑡_)∈{0</sup><sup>_,_1}</sup> denotes whether tool _𝑎_ is available under simulated failure injection. To simulate a persistent outage, the simulator ranks the _𝑁_ distinct ground-truth tools of the evaluation stream by frequency, skips the top 10%, and marks the next _𝜌𝑁_ tools as failing, where _𝜌_ is the failure rate. The failing set is fixed before the run, and its tools stay unavailable from failure onset to the end. Metrics are averaged over the window after failure onset Tpost = {2 _,_ 001 _, . . . ,_ 5 _,_ 000}, which contains _𝑇_ post=3 _,_ 000 steps. 





**Table 2: Dynamic reliability evaluation under injected tool failures.** 

|Method|Failure|Rate 20%||Failure|Rate 30%||
|---|---|---|---|---|---|---|
||Post-Fail SR (%)↑|FSR (%)↓|AS↓|Post-Fail SR (%)↑|FSR (%)↓|AS↓|
|BM25|18.8|20.0|_Failed_|10.2|30.0|_Failed_|
|ToolLLM API Retriever|17.4|79.6|_Failed_|10.5|86.2|_Failed_|
|OpenAI Ada-002|19.4|80.4|_Failed_|11.1|88.8|_Failed_|
|DRAGON (pretrained)|78.1|18.1|_Failed_|75.7|21.2|_Failed_|
|SPLADE-v3|16.0|7.7|_Failed_|15.8|9.1|_Failed_|
|BGE-Reranker|83.2|17.5|_Failed_|81.0|20.5|_Failed_|
|Health Check Blacklist|82.6|1.4|2,620|80.2|1.9|_Failed_|
|Retry (_𝑘_=3)|84.8|16.4|70|82.9|19.1|_Failed_|
|ToolRL-like Batch GRPO|80.3|5.4|2,620|79.5|6.1|1|
|RATED (w/o Feedback)|81.1|16.5|_Failed_|78.7|19.4|_Failed_|
|RATED (w/ Global MAB)|78.1|10.1|3,852|74.3|13.2|2,718|
|**RATED (Full)**|**83.4**|**2.5**|**1,603**|**82.1**|**2.9**|**0**|



where FSR is the fault selection rate. Adaptation steps (AS) measure recovery speed using 



computed from failure onset with a sliding window of _𝑊_ =200 steps, peak success rate before failure SRpeak, and _𝜃_ =0 _._ 9 at 20% failure rate or _𝜃_ =0 _._ 8 at 30%. Static rankers without online feedback never reach the recovery target and are marked _Failed_ . 

## **4.3 Baselines** 

Retrieval baselines include lexical matchers TF-IDF [26] and BM25 [23], dense encoders OpenAI Ada-002 [18], BGE-M3 [3], GTE-Large [12], and E5-Large-v2 [28], and neural IR models DRAGON [13], which is not fine-tuned on EcoAgent-Syn, and SPLADE-v3 [9]. We also include a BGE-Reranker-v2-m3 [3] cross-encoder that re-ranks DRAGON top-50 candidates, and the ToolLLM API retriever [21]. For dynamic evaluation, two reactive baselines share the DRAGON ranker: Health Check Blacklist permanently excludes any tool after one failed execution, and Retry ( _𝑘_ =3) tries up to three candidates within one step and keeps no memory across steps. A batch GRPO proxy similar to ToolRL [25] updates weights periodically rather than online. RATED uses DRAGON fine-tuned on EcoAgent-Syn as the backbone and is evaluated as RATED (Full), RATED (w/o Contrastive), RATED (w/o Feedback), and RATED (w/ Global MAB). 

## **4.4 Main Results** 

Table 1 reports offline retrieval on EcoAgent-Syn and the held-out EcoAgent-Real test split. RATED (Full) ranks first on Syn across all metrics, and RATED (w/o Contrastive) matches pretrained DRAGON on both benchmarks, which isolates the gain from contrastive fine-tuning. On EcoAgent-Real, RATED (Full) improves MRR by 52.2% over the ToolLLM API retriever but stays below Ada-002 and SPLADE-v3. High offline ranking accuracy on real queries therefore does not guarantee robust tool selection when tools fail at runtime. 

Table 2 reports Post-Fail SR, FSR, and adaptation steps under 20% and 30% failure injection. Static retrievers, including ToolLLM and Ada-002, frequently select unavailable tools and never recover within the evaluation horizon. DRAGON (pretrained) and BGEReranker reduce FSR but remain unable to adapt online. Relative 



<!-- page 5 -->

CIKM ’26, November 07–11, 2026, Rome, Italy 

RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics 



**Figure 4: Hyperparameter sensitivity at 20% failure injection.** 

to RATED (w/o Feedback), RATED (Full) lowers FSR to 2.5% and 2.9% while it maintains Post-Fail SR above 82%, which is a 7 _._ 3× reduction over DRAGON (pretrained) and a 30× reduction over ToolLLM. Batch GRPO updates similar to ToolRL retain higher FSR, and health check blacklisting attains low FSR but recovers slowly. RATED (w/ Global MAB) stays inferior to the full model. These results support decoupling semantic matching from reliability estimation updated online from execution feedback. 

## **4.5 Hyperparameter Sensitivity** 

Figure 4 reports Post-Fail SR and FSR under 20% failure injection for two grid sweeps over fusion and bandit hyperparameters. Panel (a) varies _𝑤_ 1 and _𝑤_ 3 with bandit settings fixed. Post-Fail SR rises with _𝑤_ 1 and peaks at _𝑤_ 1 = 0 _._ 75 with _𝑤_ 3 = 0 _._ 10, whereas FSR grows slowly over the same range and stays in the low single-digit range throughout. We therefore select _𝑤_ 1 = 0 _._ 75 and _𝑤_ 3 = 0 _._ 10 on the validation split, which trades about one point of FSR for the highest Post-Fail SR. Panel (b) varies _𝛾_ and _𝛼_ with fusion weights fixed at the validated setting. Post-Fail SR is highest at _𝛾_ = 0 _._ 95, where the three _𝛼_ curves nearly coincide, and FSR again grows slowly with _𝛾_ . We therefore set _𝛾_ = 0 _._ 95 and _𝛼_ = 0 _._ 1 on the validation split. 

## **4.6 Efficiency Analysis** 

We profile end-to-end inference at batch size 1 on the same hardware as Section 4. Table 3 reports average latency, P95 latency, and incremental overhead relative to DRAGON encoding only. The encode-only baseline averages 3.223 ms. RATED (w/o Feedback) applies only the static reliability prior in Eq. 5 and adds 0.025 ms. Contextual bandit re-ranking incurs modest additional cost. RATED (w/ Global MAB) adds 0.108 ms, and RATED (Full) adds 0.172 ms, reaching 3.395 ms average and 4.391 ms P95 latency. These overheads are far below batch GRPO updates similar to ToolRL at 17.839 ms average, which require periodic policy synchronization. Therefore, RATED is fast enough for interactive agent loops and still reduces fault selection. 

## **5 Conclusion** 

We introduced RATED, a reliability-aware tool retrieval framework that combines DRAGON encoders fine-tuned with contrastive learning and contextual bandit re-ranking from execution feedback. On 

**Table 3: Inference latency at batch size 1 (ms).** 

|Method|Avg. Latency|P95 Latency|ΔOverhead|
|---|---|---|---|
|DRAGON encode only|3.223|4.269|–|
|ToolRL-like Batch GRPO|17.839|28.921|+14.616|
|RATED (w/o Feedback)|3.248|4.274|+0.025|
|RATED (w/ Global MAB)|3.331|4.351|+0.108|
|**RATED (Full)**|**3.395**|**4.391**|**+0.172**|



EcoAgent, RATED achieves strong in-domain retrieval and remains competitive on real-world queries. Under injected tool failures, it reduces fault selection to 2.5–2.9% FSR, 7 _._ 3× lower than static DRAGON, with only 0.172 ms added latency over encode-only inference. These results highlight the critical importance of incorporating execution feedback into retrieval systems for autonomous agents operating in dynamic environments, and demonstrate that RATED provides an efficient solution for reliable tool discovery in agent systems. Evaluating this cold-start path on newly added tools and validating RATED in live agent runtimes remain future work. 

## **Acknowledgments** 

This work is supported by the National Science Foundation (NSF) under award number 2623261. 

## **GenAI Usage Disclosure** 

We use Gemma-3-1b to generate the synthetic queries of EcoAgentSyn and to verify candidate query–tool pairs during EcoAgent-Real labeling, and we release the prompt templates with the dataset. We used OpenAI’s ChatGPT to improve the clarity and grammar of sentences during the writing process. 

## **References** 

- [1] Naoki Abe, Alan W. Biermann, and Philip M. Long. 2003. Reinforcement Learning with Immediate Rewards and Linear Hypotheses. _Algorithmica_ 37, 4 (Dec. 2003), 263–293. doi:10.1007/s00453-003-1038-1 

- [2] Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. 2002. Finite-time Analysis of the Multiarmed Bandit Problem. _Machine Learning_ 47 (05 2002), 235–256. doi:10.1023/A:1013689704352 

- [3] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2025. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL] https://arxiv.org/abs/2402.03216 

- [4] CrewAI. 2024. CrewAI. GitHub repository. https://github.com/joaomdmoura/ crewAI 

- [5] Xiaofei Dong, Xueqiang Zhang, Weixin Bu, Dan Zhang, and Feng Cao. 2024. A Survey of LLM-based Agents: Theories, Technologies, Applications and Suggestions. In _2024 3rd International Conference on Artificial Intelligence, Internet of Things and Cloud Computing Technology (AIoTC)_ . IEEE, Shenyang, China, 407–413. doi:10.1109/AIoTC63215.2024.10748304 

- [6] Aurélien Garivier and Eric Moulines. 2011. On upper-confidence bound policies for switching bandit problems. In _Proceedings of the 22nd International Conference on Algorithmic Learning Theory_ (Espoo, Finland) _(ALT’11)_ . Springer-Verlag, Berlin, Heidelberg, 174–188. 

- [7] Gemini Team. 2025. Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805 [cs.CL] https://arxiv.org/abs/2312.11805 

- [8] LangChain AI. 2023. LangChain. GitHub repository. https://github.com/ langchain-ai/langchain 

- [9] Claude Lassance, Hervé Déjean, Thibault Formal, and Stéphane Clinchant. 2024. SPLADE-v3: New baselines for SPLADE. arXiv:2403.06789 [cs.IR] https://arxiv. org/abs/2403.06789 

- [10] Jiawei Li, Yang Gao, Yizhe Yang, Yu Bai, Xiaofeng Zhou, Yinghao Li, Huashan Sun, Yuhang Liu, Xingpeng Si, Yuhao Ye, Yixiao Wu, Yiguan Lin, Bin Xu, Bowen Ren, Chong Feng, and Heyan Huang. 2025. Fundamental Capabilities and Applications 



<!-- page 6 -->

CIKM ’26, November 07–11, 2026, Rome, Italy 

Weiliang Chen and Song Liao 

of Large Language Models: A Survey. _ACM Comput. Surv._ 58, 2, Article 38 (Sept. 2025), 42 pages. doi:10.1145/3735632 

- [11] Lihong Li, Wei Chu, John Langford, and Robert E. Schapire. 2010. A ContextualBandit Approach to Personalized News Article Recommendation. In _Proceedings of the 19th International Conference on World Wide Web (WWW ’10)_ . ACM, New York, NY, USA, 661–670. doi:10.1145/1772690.1772758 

- [12] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards General Text Embeddings with Multi-stage Contrastive Learning. arXiv:2308.03281 [cs.CL] https://arxiv.org/abs/2308.03281 

- [13] Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2023. How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval. In _Findings of the Association for Computational Linguistics: EMNLP 2023_ . Association for Computational Linguistics, Singapore, 6385–6400. https://aclanthology.org/2023.findingsemnlp.424 

- [14] LlamaIndex. 2024. LlamaIndex. GitHub repository. https://github.com/runllama/llama_index 

- [15] Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, Edouard Grave, Yann LeCun, and Thomas Scialom. 2023. Augmented Language Models: a Survey. arXiv:2302.07842 [cs.CL] https://arxiv.org/abs/2302. 07842 

- [16] n8n. 2024. n8n. Workflow automation platform. https://n8n.io/ 

- [17] Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. 2025. A Comprehensive Overview of Large Language Models. _ACM Trans. Intell. Syst. Technol._ 16, 5, Article 106 (Aug. 2025), 72 pages. doi:10.1145/3744746 

- [18] OpenAI. 2022. New and improved embedding model. https://openai.com/index/ new-and-improved-embedding-model/. Accessed: 2026-01-23. 

- [19] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. 2023. Gorilla: Large Language Model Connected with Massive APIs. arXiv:2305.15334 [cs.CL] https://arxiv.org/abs/2305.15334 

- [20] Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek HakkaniTür, Gokhan Tur, and Heng Ji. 2025. ToolRL: Reward is All Tool Learning Needs. arXiv:2504.13958 [cs.LG] https://arxiv.org/abs/2504.13958 

- [21] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing 

Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2023. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs. arXiv:2307.16789 [cs.AI] https://arxiv.org/abs/2307.16789 

- [22] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv:1908.10084 [cs.CL] https://arxiv.org/abs/ 1908.10084 

- [23] S. E. Robertson and S. Walker. 1994. Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval. In _Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval_ (Dublin, Ireland) _(SIGIR ’94)_ . Springer-Verlag, Berlin, Heidelberg, 232–241. 

- [24] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761 [cs.CL] https://arxiv.org/abs/2302.04761 

- [25] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300 [cs.CL] https://arxiv.org/abs/2402.03300 

- [26] Karen Spärck Jones. 1972. A statistical interpretation of term specificity and its application in retrieval. _Journal of documentation_ 28, 1 (1972), 11–21. 

- [27] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2019. Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748 [cs.LG] https://arxiv.org/ abs/1807.03748 

- [28] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2024. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533 [cs.CL] https://arxiv.org/abs/2212. 03533 

- [29] Qiancheng Xu, Yongqi Li, Heming Xia, and Wenjie Li. 2024. Enhancing Tool Retrieval with Iterative Feedback from Large Language Models. arXiv:2406.17465 [cs.CL] https://arxiv.org/abs/2406.17465 

- [30] Zijing Zhang, Zhanpeng Chen, He Zhu, Ziyang Chen, Nan Du, and Xiaolong Li. 2025. ToolExpNet: Optimizing Multi-Tool Selection in LLMs with Similarity and Dependency-Aware Experience Networks. In _Findings of the Association for Computational Linguistics: ACL 2025_ , Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 15706–15722. doi:10.18653/v1/2025.findings-acl.811 

