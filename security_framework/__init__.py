"""
FedRAG Security Evaluation Framework
=====================================
A generalised, plug-and-play security evaluation layer for FedRAG
(and architecturally similar federated / distributed RAG systems).

Attacks implemented
-------------------
- DataPoisoningAttack        — inject malicious (query, response) pairs
- MembershipInferenceAttack  — infer whether a query appears in the store
- KnowledgeExtractionAttack  — recover knowledge-store nodes via probing
- NodeAvailabilityAttack     — simulate removal / Byzantine / DDoS / Sybil

Defenses implemented
--------------------
- ClientDataPoisoningDefense  — inspect and quarantine poisoned clients
- ScoreMaskingKnowledgeStore  — hide retrieval scores from the attacker
- CrossPeerValidation         — majority-vote across peer answers
- ClientQueryRateLimiter      — cap queries per client
- ExtractionAnomalyDetector   — flag high-volume, broad-topic probing
- ResponsePerturbation        — add noise to raw retrieved answers

Quick start — single pipeline
------------------------------
>>> from security_framework import FedRAGSimulator, SecurityPipeline
>>> sim   = FedRAGSimulator(num_clients=10, num_examples=200, seed=0)
>>> pipe  = SecurityPipeline(sim)
>>> report = pipe.run_all()
>>> report.print_summary()
>>> report.save("results/")

Quick start — global system impact analysis
-------------------------------------------
>>> from security_framework import FedRAGSimulator
>>> from security_framework.global_impact import GlobalImpactEvaluator
>>> from security_framework.global_reporter import GlobalImpactReporter
>>>
>>> sim      = FedRAGSimulator(num_clients=10, num_examples=200, seed=0)
>>> evaluator = GlobalImpactEvaluator(sim)
>>> report    = evaluator.run()
>>> reporter  = GlobalImpactReporter(report)
>>> reporter.print_summary()
>>> reporter.save("results/global/")
"""

from security_framework.pipeline import SecurityPipeline
from security_framework.reporter import SecurityReport
from security_framework.simulator import FedRAGSimulator

__all__ = [
    "FedRAGSimulator",
    "SecurityPipeline",
    "SecurityReport",
]
