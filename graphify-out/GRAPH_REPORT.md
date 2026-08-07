# Graph Report - .  (2026-07-14)

## Corpus Check
- 248 files · ~414,701 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1422 nodes · 2303 edges · 119 communities (98 shown, 21 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 118 edges (avg confidence: 0.71)
- Token cost: 454,422 input · 0 output

## Community Hubs (Navigation)
- Selective Forwarding Attack Runner
- DDoS Live Flood Evaluation
- DRAG Log Client (Off-chain)
- Selective Forwarding Attack Core
- LLM Service API Server
- Data Poisoning Attack
- Smart Contract Dependencies
- SSM Score Attack & Defense
- Collate Functions (RoRA)
- Selective Forwarding Defense (Sim)
- On-chain Reliability Score Updates
- Membership Inference Attack
- DRAG Data Poisoning Demo
- Monte Carlo Shapley Base
- MIA Defense Evaluator
- MC-SHAP Model Wrappers
- SHAP Visualization
- DDoS Attack Simulation
- MIA Decision-Match Scoring
- DDoS & SFA Sim READMEs
- MC-SHAP Embeddings & Init
- On-chain Record Queries
- Live Network Peers (SFA Sim)
- Model Trainer
- On-chain Log Record Reads
- DDoS Defense (Sim)
- MIA Weight Tuning
- MC-SHAP Text Splitters
- Data Source Retriever (FAISS)
- LLM Service Configuration
- Accuracy Metric
- Loss Metric
- Metric Base Class
- HuggingFace Model Wrapper
- Reranker (BM25 + Dense)
- DDoS Attack Runner
- SSM Vulnerability Report
- System Architecture Diagram
- SFA Attack (Sim)
- Data Source API Server
- Token-level SHAP Analysis
- LLM Service Tests
- Thesis CIA-Triad Framework
- SFA Sim Config & Runner
- OpenAI Embeddings (SHAP)
- SFA Sim Evaluation Runner
- Python Client (DragScores)
- Mock Network Simulation
- Security Analysis Report (Unified)
- KB & DDoS Report Findings
- KB Extraction Attack
- MIA Attack Runner
- DRAG vs Reliable-dRAG Context
- Client Local Test Examples
- Source Query & Signing
- Feedback Update Tests
- Client Hardhat Utilities
- DDoS Attack Report Findings
- MC-SHAP Splitter Base
- vLLM Model Wrapper
- RoRA Trainer
- MIA/KB Evaluation Bugs
- SSM Chain & Gray-hole Report
- KB Defense & Data Source READMEs
- On-chain Score Record Creation
- Contract Status Utilities
- Reliability Score Screenshot
- MIA Report Revision 7
- PubMedQA Corpus Builder
- MIA Defense Runner
- Contract Gas Tests
- Data Source Service Tests
- Result Curve Plotting
- Defense Design Notes
- All-Services Test
- On-chain Score Updates (Batch)
- Reliability History Visualization
- HTML Reports Index
- Contract Container Entrypoint
- Contract Container Healthcheck
- Python Client README
- Client Full Test Script
- MIA Response Sanitization Notes
- MIA Dataset Notes
- Contract Deployment Module
- Data Source Docker Script
- SFA Patch Script
- Live RAG Network Note
- Mock RAG Network Note
- DDoS Overload Table Note
- DDoS Wave Lifecycle Note
- LivePeer Note
- MockPeer Note
- DDoS Mock/Live Mode Config
- Framework LLM Service Node

## God Nodes (most connected - your core abstractions)
1. `DragLogSol` - 42 edges
2. `DragLogClient` - 23 edges
3. `LiveRAGNetwork` - 21 edges
4. `SelectiveForwardingDefense` - 21 edges
5. `BaseSHAP` - 21 edges
6. `DragScoresClient` - 21 edges
7. `DDoSAttack` - 19 edges
8. `run_detection()` - 19 edges
9. `run_mitigation()` - 19 edges
10. `run_stealthy()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `dRAG Decentralized RAG System` --semantically_similar_to--> `Reliable-dRAG`  [INFERRED] [semantically similar]
  README.md → .claude/CLAUDE.md
- `KB extraction rate-limit confound bugfix (session write-up)` --semantically_similar_to--> `TrafficFlood real concurrent HTTP flood`  [INFERRED] [semantically similar]
  html_reports/kb_report.html → reports/ddos_attack.md
- `sfa container localhost networking bug (AUC degenerate at 0.50)` --semantically_similar_to--> `Corpus-drift configuration bug (sources_0.jsonl SQuAD to PubMedQA)`  [INFERRED] [semantically similar]
  mia_report.html → reports/kb.md
- `SFA Updated Implementation Report (attack/selective_forward, 2026-07-02)` --semantically_similar_to--> `SFA Security Analysis Report (selective_forward_sim module)`  [EXTRACTED] [semantically similar]
  sfa_report_updated.html → reports/SFA_Security_Analysis_Report.md
- `get_onchain_reliability_scores()` --calls--> `DragScoresClient`  [INFERRED]
  attack/selective_forward/run_attack.py → drag_python_client/drag_python_client/client.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **DRAG-to-Reliable-dRAG Generalizability Evaluation** — claude_claude_claude_cia_triad_framework, claude_claude_claude_drag, claude_claude_claude_reliable_drag [INFERRED 0.85]
- **Shared Mock/Live Dual-Mode Simulation Pattern** — attack_ddos_sim_readme_ddosattack, attack_selective_forward_sim_readme_selectiveforwardingattack, defense_sfa_sim_defense_readme_selectiveforwardingdefense [INFERRED 0.85]
- **SSM Vulnerability-to-Contract-Fix Mapping** — defense_ssm_defense_readme_contract_hardening, attack_reports_ssm_score_report_vuln01, attack_reports_ssm_score_report_vuln02, attack_reports_ssm_score_report_vuln03 [EXTRACTED 1.00]
- **CIA Triad Attack Framework (SSM-Score / MIA / SFA)** — reports_security_analysis_report_ssm_score, reports_mia_security_analysis_report_mia, reports_sfa_security_analysis_report_sfa [EXTRACTED 1.00]
- **Live DDoS Evaluation Pipeline (flood, scoring, bugfix)** — reports_ddos_attack_traffic_flood, reports_ddos_attack_nlg_metrics, reports_ddos_attack_role_token_bug [EXTRACTED 0.90]
- **MIA Composite Weight Re-Tuning (Revision 7)** — reports_mia_security_analysis_report_decision_match, reports_mia_security_analysis_report_gated_composite, reports_mia_security_analysis_report_ablation_composite, reports_mia_security_analysis_report_revision7 [EXTRACTED 0.90]
- **LLM Service query-to-answer pipeline** — figures_framework_query, figures_framework_reliability_guided_source_sampling, figures_framework_documents_retrieval, figures_framework_reliability_guided_reranking, figures_framework_llm, figures_framework_answer, figures_framework_sentence_importance_evaluation [EXTRACTED 1.00]
- **Blockchain-based reliability scoring feedback loop** — figures_framework_feedback_log, figures_framework_smart_contract, figures_framework_scoreboard, figures_framework_verify_signatures_step, figures_framework_decentralized_data_sources [EXTRACTED 1.00]
- **Source Scores Dashboard: subplots, legend, and tooltip form one screenshot** — figures_screen_shot_source_scores_chart, figures_screen_shot_usefulness_scores_plot, figures_screen_shot_reliability_scores_plot, figures_screen_shot_sources_legend, figures_screen_shot_query_1017_tooltip [EXTRACTED 1.00]

## Communities (119 total, 21 thin omitted)

### Community 0 - "Selective Forwarding Attack Runner"
Cohesion: 0.07
Nodes (58): _answer_in_context(), _apply_attack(), _build_real_sources(), check_blockchain_status(), _check_sources_reachable(), _detector_observe(), _detector_report(), _evaluate() (+50 more)

### Community 1 - "DDoS Live Flood Evaluation"
Cohesion: 0.05
Nodes (53): FloodStats, attack/ddos_sim/live_flood.py  A *real* congestion-based DDoS against the actual, Starts `workers_per_source` daemon threads per targeted source, each     firing, TrafficFlood, average_metrics(), best_token_prf1(), bleu(), exact_match() (+45 more)

### Community 2 - "DRAG Log Client (Off-chain)"
Cohesion: 0.07
Nodes (37): analyze_log_file(), doc_to_sha256(), DragLogClient, log_record_to_dict(), log_record_to_json(), LogRecord, LogRecordHistory, LogRecordInput (+29 more)

### Community 3 - "Selective Forwarding Attack Core"
Cohesion: 0.06
Nodes (19): _Block, check_blockchain_status(), naive_route(), Any, attack/selective_forward/selective_forward_attack.py  Drop-in replacement for th, Public helper — returns the current SSM chain integrity report.     Called by pa, Applies a probabilistic (10-30 %) gray-hole selective forwarding attack     to a, Monkey-patch `sources` — each element must have a `.query(question, k)` method (+11 more)

### Community 4 - "LLM Service API Server"
Cohesion: 0.07
Nodes (38): analyze_with_mc_shapley(), analyze_with_rora(), compute_importance_scores(), _format_sse_event(), get_score_events(), get_scores_from_blockchain(), health_check(), health_check_data_sources() (+30 more)

### Community 5 - "Data Poisoning Attack"
Cohesion: 0.08
Nodes (24): DataPoisoningAttack, Any, Data Poisoning Attack on the Reliable-dRAG system.  Adapted from demo/attack/dat, Call /reset on every data source to restore clean state., Query /info on all data sources and return stats., Evaluate attack success by comparing accuracy before and after.          Paramet, Fetch reliability/usefulness scores from the LLM service., Create a poisoned version of a JSONL document. (+16 more)

### Community 6 - "Smart Contract Dependencies"
Cohesion: 0.05
Nodes (36): chai, dotenv, dependencies, dotenv, ethers, @openzeppelin/contracts, devDependencies, chai (+28 more)

### Community 7 - "SSM Score Attack & Defense"
Cohesion: 0.11
Nodes (19): is_correct(), _load_corpus_contexts(), load_squad_eval(), main(), measure_accuracy(), query_rag(), attack/ssm_score/run_attack.py SSM-Score manipulation attack for Reliable-dRAG., Passages actually served by the Docker data sources (all three sources     share (+11 more)

### Community 8 - "Collate Functions (RoRA)"
Cohesion: 0.11
Nodes (20): CollateFn, ABC, Any, Text, A processor that takes an input and construct it into a format that will become, output_format should be one of         ['g', 'l', 's', 'gs', 'ls', 'gls', 'n], generate_no_more_than_ngrams(), Any (+12 more)

### Community 9 - "Selective Forwarding Defense (Sim)"
Cohesion: 0.11
Nodes (9): Any, defense/sfa_sim_defense/selective_forwarding_defense.py  Countermeasure for atta, One-sided binomial significance test: is this peer's miss rate         significa, One-sided binomial p-value: is miss_rate > honest_miss_rate?, Maximum number of peers auto-blacklisting may exclude at once.         Uncapped, Up to `redundancy_k` peers the network hasn't already visited this         query, SelectiveForwardingDefense, EMA-based peer reputation tracking (+1 more)

### Community 10 - "On-chain Reliability Score Updates"
Cohesion: 0.12
Nodes (14): Create a new usefulness record.                  Args:             log_id (str):, Build a transaction for contract interaction.                  Args:, Create a general record with specified type.                  Args:, Update a general record.                  Args:             record_id (str): The, Update the score of any record.                  Args:             record_id (st, Send a signed transaction and wait for receipt.                  Args:, Check if a record exists.                  Args:             record_id (str): Th, Create a new reliability record.                  Args:             data_source_ (+6 more)

### Community 11 - "Membership Inference Attack"
Cohesion: 0.13
Nodes (19): _consistency_score(), _cosine_similarity(), _join_pubmedqa_context(), _load_corpus_contexts(), load_membership_documents(), MIAAttack, Any, ndarray (+11 more)

### Community 12 - "DRAG Data Poisoning Demo"
Cohesion: 0.12
Nodes (16): BaseAttack, Datapoint, DataPoisoningAttack, Any, Data Poisoning Attack on Distributed RAG Systems.          This attack injects, Create semantically similar questions with wrong answers., Perturb question slightly., Get answer from same topic for maximum confusion. (+8 more)

### Community 13 - "Monte Carlo Shapley Base"
Cohesion: 0.13
Nodes (13): BaseSHAP, Any, Generate text from prompt          Args:             prompt: Text prompt, Base class for SHAP implementations, Print debug messages if debug mode is enabled, Calculate baseline model response, Generate random combinations efficiently using binary representation, Get model responses for combinations                  Args:             content: (+5 more)

### Community 14 - "MIA Defense Evaluator"
Cohesion: 0.13
Nodes (21): _auc_or_half(), calibrate_target_length(), _compute_metrics(), _detect_decision_token(), MIADefenseEvaluator, normalize_length(), obfuscate_decision(), obfuscate_decision_content() (+13 more)

### Community 15 - "MC-SHAP Model Wrappers"
Cohesion: 0.12
Nodes (15): initialize_model(), Initialize the model based on configuration., default_output_handler(), LocalModel, ModelBase, OpenAIModel, ABC, Queue (+7 more)

### Community 16 - "SHAP Visualization"
Cohesion: 0.12
Nodes (13): PixelSHAPVisualizer, ndarray, Plot horizontal bar chart of object importance ranking with color gradient and o, Displays an image with bounding boxes and segmentation masks      Parameters:, Create a fixed-size thumbnail with a frame around it                  Args:, Creates a sketch-like binary border mask around the object, Helper method to place labels on objects while avoiding overlaps, Helper method to add a legend showing object importance ranking (+5 more)

### Community 17 - "DDoS Attack Simulation"
Cohesion: 0.16
Nodes (9): apply_ddos_wave(), DDoSAttack, OverloadState, Any, attack/ddos_sim/ddos_attack.py  Congestion-based (application-layer) DDoS simula, Install a dynamic overload-table lookup on every peer's `.query()`.         Idem, Execute one attack wave: recover expired peers, select this         wave's targe, One-shot convenience helper: build, attach, and run a single wave. (+1 more)

### Community 18 - "MIA Decision-Match Scoring"
Cohesion: 0.13
Nodes (16): _get_encoder(), _answer_length_ratio(), _certainty_score(), _decision_match_adaptive(), _decision_match_semantic(), min(len(response)/len(gold_answer), 1.0) -- capped so a verbose non-member     a, Redesigned (2nd revision) around PubMedQA's actual answer format.      The origi, Adaptive-attacker variant of `_decision_match()`: scans the *entire*     respons (+8 more)

### Community 19 - "DDoS & SFA Sim READMEs"
Cohesion: 0.11
Nodes (21): DDoSAttack Class (congestion-based DDoS simulation), SFA_Security_Analysis_Report.md (design rationale doc), DragScores View-Call Reads (get_scores_batch), Live-Mode Rate-Limit / 429 Handling Fix, SelectiveForwardingAttack Class, Stealthy drop_rate Design (Uniform(0.10,0.30)), config/ddos_sim.yaml (DDoSAttack config), config/ddos_sim_defense.yaml (DDoSDefense config) (+13 more)

### Community 20 - "MC-SHAP Embeddings & Init"
Cohesion: 0.13
Nodes (11): HuggingFaceEmbeddings, DataFrame, ndarray, Initialize the appropriate model and tokenizer/processor, Base class for text vectorization, Create DataFrame with combination results, Initialize HuggingFace sentence embeddings vectorizer - much simpler implementat, Get embeddings using sentence-transformers - much simpler (+3 more)

### Community 21 - "On-chain Record Queries"
Cohesion: 0.10
Nodes (11): Any, Get all records from the contract.                  Returns:             List[Di, Get all usefulness records.                  Returns:             List[Dict]: Li, Read a reliability record.                  Args:             data_source_id (st, Read a log record.                  Args:             log_id (str): The log ID, Read a feedback record.                  Args:             log_id (str): The log, Get all reliability records.                  Returns:             List[Dict]: L, Get all feedback records.                  Returns:             List[Dict]: List (+3 more)

### Community 22 - "Live Network Peers (SFA Sim)"
Cohesion: 0.14
Nodes (10): get_onchain_reliability_scores(), LivePeer, LiveRAGNetwork, attack/selective_forward_sim/live_network.py  Live counterpart to network_sim.Mo, Fully-connected overlay over the real docker-compose data sources     (source_0/, Total 429s observed across all peers -- a *subset* of         error_total(). Rep, Total non-clean-200 outcomes across all peers -- timeouts,         connection er, Read real reliability scores from the deployed DragScores contract     (view cal (+2 more)

### Community 23 - "Model Trainer"
Cohesion: 0.19
Nodes (13): DataLoader, Any, Module, Text, Save the metrics to a file., Save the model if its performance is the top-k best so far., Given a batch of data, compute the loss and return the outputs., Given a batch of data, compute the loss and return the outputs. (+5 more)

### Community 24 - "On-chain Log Record Reads"
Cohesion: 0.11
Nodes (11): DragLogSol, Draglog Solidity Contract Interaction Class  This module provides a Python inter, Get all record IDs from the contract.                  Returns:             List, Get the total number of records in the contract.                  Returns:, Check if the contract has any records of the specified type.                  Ar, Read multiple records in batch., Compose on-chain records to a dictionary of scores for each source.         Args, Python class to interact with the Draglog smart contract.          This class pr (+3 more)

### Community 25 - "DDoS Defense (Sim)"
Cohesion: 0.13
Nodes (5): DDoSDefense, Any, defense/ddos_sim_defense/ddos_defense.py  Countermeasure for attack.ddos_sim.DDo, Call once per attack wave (before that wave's queries run) so         deprioriti, Wrap every peer's `.query()` to observe responses, and install         this defe

### Community 26 - "MIA Weight Tuning"
Cohesion: 0.19
Nodes (17): _decision_match(), _normalize_similarity(), _parse_llm_response(), Extract the answer string from whatever the LLM service returns.     Tries a wid, Remap raw cosine similarity from [-1, 1] to [0, 1]. Required so the     gated co, Does the response's leading commitment token match the gold yes/no/maybe     dec, cmd_collect(), cmd_evaluate() (+9 more)

### Community 27 - "MC-SHAP Text Splitters"
Cohesion: 0.12
Nodes (9): initialize_sentence_importance(), Initialize sentence importance evaluation method based on config.          Args:, Split text by pattern (default: space), Split text using HuggingFace tokenizer, Get tokens from prompt, Prepare model arguments for a combination, Get unique key for combination, StringSplitter (+1 more)

### Community 28 - "Data Source Retriever (FAISS)"
Cohesion: 0.21
Nodes (11): initialize_retriever(), Initialize the FastRetriever with documents from the configured path., Document, FastRetriever, _normalize_rows(), Any, ndarray, Retrieve top-k documents.         - Dense search always runs.         - If hybri (+3 more)

### Community 29 - "LLM Service Configuration"
Cohesion: 0.12
Nodes (17): Blockchain Configuration (Hardhat wallet, DragScores address), LLM Model Configuration (Qwen2.5-1.5B-Instruct / gpt-4o-mini), Reranker Settings (hybrid, all-MiniLM-L6-v2), Retrieval & Reranking Settings, Sentence Importance Evaluation Settings (mc_shap/rora switch), llm-service Docker Compose container (port 9000), Blockchain Source Score Integration, Drag LLM Service (+9 more)

### Community 30 - "Accuracy Metric"
Cohesion: 0.15
Nodes (8): ClassificationAccuracy, GenerationAccuracyMetric, Any, PreTrainedTokenizer, Text, accuracy metrics that can be used in the model, Does not convert the calling., pred_tensors = [batch_size, num_classes]         labels: [batch_size]

### Community 31 - "Loss Metric"
Cohesion: 0.14
Nodes (7): AvgLoss, calculate average loss function., Calculate the average loss of the model., Load the model from the given path., Evaluate RORA with a local fine-tuned model., Run self.trainer._eval_step on the given rationale and question.         and com, RORAModel

### Community 32 - "Metric Base Class"
Cohesion: 0.16
Nodes (9): Metric, ABC, Any, Text, Detach all tensors in the outputs., # TODO: think about how to support other types of metrics, Overload the trainer eval_step to also evaluate rora, Define an abstract trainer. (+1 more)

### Community 33 - "HuggingFace Model Wrapper"
Cohesion: 0.14
Nodes (8): HuggingfaceWrapperModule, Huggingface wrapper that has the saving property., Forward generation of the model., Generate from the model., Save the model to the given path., Model, ABC, define a model abstract class of pytorch models that can be trained.

### Community 34 - "Reranker (BM25 + Dense)"
Cohesion: 0.22
Nodes (10): _l2_normalize_rows(), _minmax(), Any, ndarray, Reranker: re-ranks retrieved documents across multiple sources and returns top-k, Re-rank a list of candidate documents.          - query: question text         -, Re-rank candidates while incorporating per-source reliability.          - reliab, Re-rank candidate documents with one of the following strategies:      - method= (+2 more)

### Community 35 - "DDoS Attack Runner"
Cohesion: 0.26
Nodes (14): _chunks_covering(), main(), _print_table(), Any, attack/ddos_sim/run_attack.py  CLI runner for the congestion-based DDoS simulati, Split `total` questions into `parts` near-equal, non-empty batches., run_baseline(), run_ddos_scenario() (+6 more)

### Community 36 - "SSM Vulnerability Report"
Cohesion: 0.16
Nodes (16): DragScores.sol Smart Contract, VULN-01: Exposed Default Hardhat Private Keys, VULN-02: Unconstrained Score Delta, VULN-03: Self-Signature Acceptance, get_onchain_reliability_scores() Targeting Fix, SFAMitigation (routing + redundant probe), DragScores Contract Hardening (allow-list, cooldown, delta cap, bounds), ssm_score_defense.py Off-Chain Monitor (+8 more)

### Community 37 - "System Architecture Diagram"
Cohesion: 0.17
Nodes (16): Answer (Output), Database, Decentralized Data Sources (component group), Decentralized Scoring Management (component group), Documents Retrieval, Feedback Log, Ground Truth (Input), LLM (+8 more)

### Community 38 - "SFA Attack (Sim)"
Cohesion: 0.19
Nodes (8): apply_selective_forwarding(), Any, attack/selective_forward_sim/selective_forwarding_attack.py  Selective Forwardin, Monkey-patch the selected peers' `.query()` to a silent drop., Restore original `.query()` on every compromised peer., One-shot convenience helper: build + apply in a single call., Parameters     ----------     attack_ratio : fraction of peers to compromise (0., SelectiveForwardingAttack

### Community 39 - "Data Source API Server"
Cohesion: 0.14
Nodes (13): get_info(), health_check(), initialize_contract_client(), inject_poison(), load_config(), API server for the data source retrieval service., Initialize the DragScores contract client., Health check endpoint. (+5 more)

### Community 40 - "Token-level SHAP Analysis"
Cohesion: 0.16
Nodes (9): get_text_before_last_underscore(), DataFrame, Plot text visualization with importance colors                  Args:, Helper function to get text before last underscore, Print text with background colors based on importance, Analyze token importance in a prompt                  Args:             prompt:, Analyzes token importance in text prompts using SHAP values, Print text with tokens colored by importance (+1 more)

### Community 41 - "LLM Service Tests"
Cohesion: 0.14
Nodes (13): Test script for the LLM service., Test score_events endpoint that returns score update events from blockchain., Test query_analyze endpoint with SSE streaming., Test data sources health check endpoint., Test query endpoint that returns just the response., Test query_analyze endpoint that returns response with analysis., Test basic health check endpoint., test_health_check() (+5 more)

### Community 42 - "Thesis CIA-Triad Framework"
Cohesion: 0.15
Nodes (13): DRAG vs. Reliable-dRAG Security Comparison Table, SSM-Score Manipulation Attack (report), TriviaQA Dataset (eval questions), Blockchain Reliability Scoring Mechanism (R_i/U_i), CIA-Triad Security Framework (thesis contribution), Data Poisoning Attack (Integrity), Denial of Service (DoS) Attack (Availability), KB Extraction Attack (Confidentiality) (+5 more)

### Community 43 - "SFA Sim Config & Runner"
Cohesion: 0.26
Nodes (11): deep_get(), load_yaml(), Any, attack/selective_forward_sim/config_loader.py  Tiny YAML config loader shared by, Load a YAML file by name (resolved against config/) or absolute path., main(), _print_table(), Any (+3 more)

### Community 44 - "OpenAI Embeddings (SHAP)"
Cohesion: 0.18
Nodes (7): OpenAIEmbeddings, Get embeddings from OpenAI API, Calculate cosine similarity between vectors, Initialize the API client with the given base_url., Generates text based on a prompt with optional vision support., Initialize OpenAI embeddings vectorizer                  Args:             api_k, Exception

### Community 45 - "SFA Sim Evaluation Runner"
Cohesion: 0.26
Nodes (11): _join_pubmedqa_context(), _live_questions(), Must match data/build_pubmedqa_corpus.py's join_context() and     attack/Mia_att, Real questions matched against the documents actually loaded into the     runnin, _average_trials(), main(), _print_summary(), Any (+3 more)

### Community 46 - "Python Client (DragScores)"
Cohesion: 0.21
Nodes (3): DragScoresClient, Any, Search for ScoreRecordUpdated events.                  Args:             source_

### Community 47 - "Mock Network Simulation"
Cohesion: 0.25
Nodes (6): MockPeer, MockRAGNetwork, Random, attack/selective_forward_sim/network_sim.py  In-process simulation of a DRAG-sty, A peer with a synthetic knowledge base: answers with probability     `hit_prob`, Barabasi-Albert overlay of MockPeer nodes with TTL-bounded BFS query     routing

### Community 48 - "Security Analysis Report (Unified)"
Cohesion: 0.22
Nodes (11): Binomial hypothesis test detection theory (gray-hole), Unified Project Security Analysis Report (SSM-Score, MIA, SFA), DragScores.sol reliability/usefulness ledger (as analyzed), Reliability-weighted reranker formula (score = (1-w)*sim + w*R_hat), SSM-Score (Source Selection Manipulation) Attack, Barabasi-Albert scale-free overlay simulation, high_connectivity targeting strategy, Selective Forwarding Attack (gray-hole, simulation module) (+3 more)

### Community 49 - "KB & DDoS Report Findings"
Cohesion: 0.24
Nodes (10): 82.5% block-rate small-corpus caveat, KB extraction rate-limit confound bugfix (session write-up), KB Extraction Report (HTML dashboard), nlg_metrics.py 13-metric NLG scorer, TrafficFlood real concurrent HTTP flood, Chunk Recovery Rate (CRR) metric, Knowledge-Base Extraction Attack & Defense Report, KB Extraction Attack (black-box API enumeration) (+2 more)

### Community 51 - "MIA Attack Runner"
Cohesion: 0.33
Nodes (8): _dry_run(), main(), parse_args(), Any, Namespace, attack/Mia_attack/run_attack.py  Run the MIA (Membership Inference Attack) again, Simulate MIA with random similarity scores.     AUC-ROC should be ≈ 0.50 — confi, save_log()

### Community 52 - "DRAG vs Reliable-dRAG Context"
Cohesion: 0.22
Nodes (9): DRAG (Xu et al., 2025), Reliable-dRAG, arXiv:2511.07577 Paper, Decentralized Blockchain Network Component, Decentralized Data Sources Component, dRAG Decentralized RAG System, LLM Service Component, Sepolia Testnet Deployment (example DragScores) (+1 more)

### Community 53 - "Client Local Test Examples"
Cohesion: 0.33
Nodes (7): get_hardhat_private_keys(), get_project_root(), main(), Returns an owner private key and a list of 10 source private keys from the local, Creates 3 default sources using source names and private keys from drag_data_sou, test_create_default_sources_from_configs(), Path

### Community 54 - "Source Query & Signing"
Cohesion: 0.29
Nodes (6): query(), Query endpoint for document retrieval.          Request body:         {, Validate that the provided scores match the on-chain scores.          Args:, validate_selected_sources(), Produce an off-chain signature compatible with the DragScores contract verificat, sign_message_personal()

### Community 55 - "Feedback Update Tests"
Cohesion: 0.32
Nodes (7): get_current_scores(), query_data_sources(), Test script for feedback_and_update_score_records functionality. Queries all 3 d, Test feedback_and_update_score_records functionality., Get current scores from the contract., Query all 3 data source services and collect signatures., test_feedback_and_update()

### Community 56 - "Client Hardhat Utilities"
Cohesion: 0.36
Nodes (5): ScoreRecordAlreadyExistsError, find_local_ignition_deployed_address(), load_drag_scores_abi(), Load the DragScores ABI from Hardhat artifacts., Try to read Ignition's deployed address for DragScores on the localhost chain (3

### Community 57 - "DDoS Attack Report Findings"
Cohesion: 0.25
Nodes (8): exact_match leaked role-token bugfix (session write-up), DDoS Attack Report (HTML dashboard), Bernoulli-trial drop-probability model, Graph-based cascade propagation to neighbours, DDoSAttack congestion simulation (wave-based), DDoSDefense reputation/backoff mechanism, DDoS Attack Security Analysis Report, Leaked chat-template role-token parsing bug (exact_match=0 confound)

### Community 58 - "MC-SHAP Splitter Base"
Cohesion: 0.29
Nodes (4): Queue, Base class for text splitting, Initialize TokenSHAP                  Args:             model: Model to analyze, Splitter

### Community 60 - "RoRA Trainer"
Cohesion: 0.48
Nodes (4): Any, Module, Text, RORATrainer

### Community 61 - "MIA/KB Evaluation Bugs"
Cohesion: 0.33
Nodes (7): sfa container localhost networking bug (AUC degenerate at 0.50), MIA Dry-Run & Docker Networking Report (2026-07-02), Three-Seed Thesis Protocol (seeds 0,1,2), Corpus-drift configuration bug (sources_0.jsonl SQuAD to PubMedQA), MIA Security Analysis Report (Revision 7), Membership Inference Attack (MIA) against PubMedQA corpus, PubMedQA (pqa_labeled) corpus

### Community 62 - "SSM Chain & Gray-hole Report"
Cohesion: 0.29
Nodes (7): _SSMChain in-process SHA-256 hash-chained ledger, SQuAD-based attack scripts dependencies (MIA/SFA/KB standalone sims), Hash-chained blockchain SSM ledger, EWMA + binomial anomaly detector (SFADetector), Stealthy probabilistic gray-hole drop (10-30% per node), Suspicion-aware mitigation (SFAMitigation), SFA Updated Implementation Report (attack/selective_forward, 2026-07-02)

### Community 64 - "KB Defense & Data Source READMEs"
Cohesion: 0.33
Nodes (6): QueryDiversityThrottle Class, Topic-Balanced Extraction Strategy (attack side), FastRetriever (SentenceTransformer + FAISS), Flask API Server (drag_data_source/app/server.py), POST /query Endpoint, sentence-transformers / faiss-cpu Dependency

### Community 65 - "On-chain Score Record Creation"
Cohesion: 0.33
Nodes (3): Create multiple records with scores in batch.                  Args:, Create scores for multiple sources., Get contract events using the working method (from_block/to_block keyword argume

### Community 66 - "Contract Status Utilities"
Cohesion: 0.33
Nodes (3): Get network information.                  Returns:             Dict: Network inf, Get contract statistics.                  Returns:             Dict: Contract st, Print comprehensive contract status.

### Community 67 - "Reliability Score Screenshot"
Cohesion: 0.40
Nodes (6): Hover Tooltip for Query 1017 (arachidonic acid question, Correctness=False), Reliability Scores Subplot (bottom panel, Query ID vs Score), Reliable-dRAG Blockchain R_i / U_i Reliability Scoring Mechanism (concept), "Source Scores Over Queries" Plotly Dashboard Screenshot, Sources Legend (sources_sources_0/20/40/60/80/100), Usefulness Scores Subplot (top panel, Query ID vs Score)

### Community 68 - "MIA Report Revision 7"
Cohesion: 0.40
Nodes (6): Ablation composite (decision-dominant, sim/certainty zeroed), Multi-probe consistency score signal, decision_match signal (yes/no/maybe decision-commitment check), Gated 4-signal Composite Scoring Formula, obfuscate_decision_content() content-level defense, Revision 7 — empirical weight re-tuning under train/test split

### Community 69 - "PubMedQA Corpus Builder"
Cohesion: 0.50
Nodes (4): join_context(), main(), data/build_pubmedqa_corpus.py  One-off generator: builds data/polluted_token/sou, Single shared join function -- mia_attack.py must use this exact     logic when

### Community 70 - "MIA Defense Runner"
Cohesion: 0.50
Nodes (4): main(), parse_args(), Namespace, defense/mia_defense/run_defense.py Evaluate the MIA response-sanitization defens

### Community 72 - "Data Source Service Tests"
Cohesion: 0.40
Nodes (3): Test script for the data source service., Test health check endpoint., test_health_check()

### Community 73 - "Result Curve Plotting"
Cohesion: 0.50
Nodes (4): load_jsonl(), plot_scores(), Plot usefulness and reliability scores for each source across queries., Load data from JSONL file.

### Community 74 - "Defense Design Notes"
Cohesion: 0.50
Nodes (4): Why Not Confidence-Score Noise Injection (rationale), decision_match Gating Design (Revision 4), Gated Composite Membership Score (decision_match-gated), MIADefenseEvaluator Class

### Community 75 - "All-Services Test"
Cohesion: 0.50
Nodes (3): Test script for all three data source services., Test all three data source services., test_all_services()

### Community 78 - "HTML Reports Index"
Cohesion: 0.50
Nodes (4): Security Evaluation Reports Landing Page, Selective Forwarding Attack Report (HTML dashboard), SFA Security Analysis Report (selective_forward_sim module), Live sweep rate-limit confounder diagnosis and fix (SS12.9)

### Community 81 - "Python Client README"
Cohesion: 0.67
Nodes (3): DragScores Python Client, Off-chain ECDSA signature compatibility (personal_sign / toEthSignedMessageHash), drag_python_client Python dependencies (web3, eth-account, pyyaml)

## Ambiguous Edges - Review These
- `Blockchain Reliability Scoring Mechanism (R_i/U_i)` → `_SSMChain (in-process hash-chained reputation ledger)`  [AMBIGUOUS]
  defense/sfa_defense/README.md · relation: conceptually_related_to
- `decision_match signal (yes/no/maybe decision-commitment check)` → `Multi-probe consistency score signal`  [AMBIGUOUS]
  reports/MIA_Security_Analysis_Report.md · relation: conceptually_related_to
- `Sources Legend (sources_sources_0/20/40/60/80/100)` → `Reliable-dRAG Blockchain R_i / U_i Reliability Scoring Mechanism (concept)`  [AMBIGUOUS]
  figures/screen shot.jpg · relation: conceptually_related_to

## Knowledge Gaps
- **90 isolated node(s):** `entrypoint.sh script`, `PYTHONPATH`, `healthcheck.sh script`, `PYTHONPATH`, `{ buildModule }` (+85 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Blockchain Reliability Scoring Mechanism (R_i/U_i)` and `_SSMChain (in-process hash-chained reputation ledger)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `decision_match signal (yes/no/maybe decision-commitment check)` and `Multi-probe consistency score signal`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Sources Legend (sources_sources_0/20/40/60/80/100)` and `Reliable-dRAG Blockchain R_i / U_i Reliability Scoring Mechanism (concept)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `DragScoresClient` connect `Python Client (DragScores)` to `Selective Forwarding Attack Runner`, `Data Source API Server`, `SSM Score Attack & Defense`, `Client Local Test Examples`, `Live Network Peers (SFA Sim)`, `Feedback Update Tests`, `Client Hardhat Utilities`, `Source Query & Signing`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `HuggingFaceEmbeddings` connect `MC-SHAP Embeddings & Init` to `MIA Decision-Match Scoring`, `vLLM Model Wrapper`, `MC-SHAP Model Wrappers`?**
  _High betweenness centrality (0.131) - this node is a cross-community bridge._
- **Why does `initialize_sentence_importance()` connect `MC-SHAP Text Splitters` to `LLM Service API Server`, `MC-SHAP Embeddings & Init`, `Loss Metric`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **What connects `entrypoint.sh script`, `PYTHONPATH`, `healthcheck.sh script` to the rest of the system?**
  _90 weakly-connected nodes found - possible documentation gaps or missing edges._