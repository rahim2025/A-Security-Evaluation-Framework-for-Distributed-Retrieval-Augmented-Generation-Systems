# Selective Forwarding Attack — Original Module (`selective_forward`)

This is the first Selective Forwarding Attack (SFA) implementation in the
repo: a stealthy probabilistic gray-hole attack (`SelectiveForwardingAttack`)
plus `SFADetector` (EWMA + binomial anomaly detection), `SFAMitigation`
(suspicion-aware re-routing + blacklist), and a hash-chained SSM ledger
(`check_blockchain_status`) for reputation-score integrity.

**Status: kept deliberately, not legacy debris.** `attack/selective_forward_sim`
(+ `defense/sfa_sim_defense`) is the newer, config-driven, larger-scale
implementation and is what `reports/SFA_Security_Analysis_Report.md`
empirically evaluates in depth. This module is retained as the design
source that report ports concepts from directly — most notably
`SFADetector`'s binomial-test anomaly detection (cited in §2.6/§12.7 of
that report) and the `STEALTHY_LO`/`STEALTHY_HI` stealthy drop-rate range
that `selective_forward_sim` matches exactly. It is not evaluated to the
same empirical depth as the newer module (no multi-trial sweeps, no
mock/live parity harness) — see `reports/SFA_Security_Analysis_Report.md`
§15 for the explicit scope note.

If you're looking for current attack/defense numbers, use
`attack/selective_forward_sim` and `defense/sfa_sim_defense` instead.
