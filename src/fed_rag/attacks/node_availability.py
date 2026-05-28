"""Node availability attack utilities for FedRAG.

These attacks model availability threats against a federated or distributed
RAG deployment.  A ``node`` can be a DRAG peer, a FedRAG client, or any
entity that participates in retrieval / answer generation.  The attack
operates on an abstract sequence of nodes so that the same code can be reused
across architectures.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class NodeAvailabilityResult:
    """Results from a node-availability attack."""

    attack_type: str
    attack_ratio: float
    total_nodes: int
    affected_nodes: list[int]
    availability_before: float
    availability_after: float
    strategy: str
    iterations: list[dict[str, Any]] = field(default_factory=list)


class NodeAvailabilityAttack:
    """Simulate node-availability attacks on a set of RAG nodes.

    Attack types:
        - ``node_removal``: permanently disable selected nodes.
        - ``byzantine``: mark nodes as Byzantine (they return corrupted answers).
        - ``partition``: split the node set into disconnected partitions.
        - ``ddos``: overload selected nodes so they drop queries.
        - ``sybil``: inject fake nodes into the network.
    """

    def __init__(
        self,
        attack_type: str = "node_removal",
        attack_ratio: float = 0.3,
        attack_iterations: int = 1,
        seed: int | None = None,
    ) -> None:
        if attack_type not in {
            "node_removal",
            "byzantine",
            "partition",
            "ddos",
            "sybil",
        }:
            raise ValueError(f"Unsupported attack_type: {attack_type}")
        self.attack_type = attack_type
        self.attack_ratio = attack_ratio
        self.attack_iterations = max(1, int(attack_iterations))
        self.seed = seed
        self._rng = random.Random(seed)

    def execute(
        self,
        nodes: Sequence[Any],
        strategy: str = "random",
        target_ids: Sequence[int] | None = None,
    ) -> NodeAvailabilityResult:
        """Run the configured attack against *nodes*.

        Args:
            nodes: sequence of node objects (length = total node count).
            strategy: how to pick victims — ``random``, ``high_connectivity``,
                ``high_data``, or ``specific``.
            target_ids: explicit victim IDs when *strategy* is ``specific``.

        Returns:
            ``NodeAvailabilityResult`` with affected node IDs and availability.
        """
        total = len(nodes)
        availability_before = self._availability(nodes)
        affected: set[int] = set()
        iterations: list[dict[str, Any]] = []

        for iteration in range(self.attack_iterations):
            if self.attack_type == "node_removal":
                victims = self._select_targets(
                    total, strategy, target_ids, nodes
                )
                affected.update(victims)
                iterations.append(
                    {
                        "iteration": iteration + 1,
                        "removed": list(victims),
                        "total_removed": len(affected),
                    }
                )
            elif self.attack_type == "byzantine":
                victims = self._select_targets(
                    total, strategy, target_ids, nodes
                )
                affected.update(victims)
                iterations.append(
                    {
                        "iteration": iteration + 1,
                        "byzantine": list(victims),
                        "total_byzantine": len(affected),
                    }
                )
            elif self.attack_type == "partition":
                p1, p2 = self._partition_nodes(total)
                iterations.append(
                    {
                        "iteration": iteration + 1,
                        "partition_1": p1,
                        "partition_2": p2,
                    }
                )
            elif self.attack_type == "ddos":
                victims = self._select_targets(
                    total, strategy, target_ids, nodes
                )
                affected.update(victims)
                iterations.append(
                    {
                        "iteration": iteration + 1,
                        "ddos_targets": list(victims),
                        "total_ddos": len(affected),
                    }
                )
            elif self.attack_type == "sybil":
                num_sybil = max(1, int(total * self.attack_ratio))
                iterations.append(
                    {
                        "iteration": iteration + 1,
                        "sybil_injected": num_sybil,
                    }
                )

        availability_after = self._post_attack_availability(
            total, affected, self.attack_type
        )

        return NodeAvailabilityResult(
            attack_type=self.attack_type,
            attack_ratio=self.attack_ratio,
            total_nodes=total,
            affected_nodes=sorted(affected),
            availability_before=availability_before,
            availability_after=availability_after,
            strategy=strategy,
            iterations=iterations,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_targets(
        self,
        total: int,
        strategy: str,
        target_ids: Sequence[int] | None,
        nodes: Sequence[Any],
    ) -> list[int]:
        num_attack = max(1, int(total * self.attack_ratio))

        if strategy == "specific" and target_ids:
            pool = [i for i in target_ids if 0 <= i < total]
            return self._rng.sample(pool, min(num_attack, len(pool))) if pool else []

        if strategy == "high_data" and nodes:
            indexed = sorted(
                range(total),
                key=lambda i: getattr(nodes[i], "data_size", 0),
                reverse=True,
            )
            return indexed[:num_attack]

        if strategy == "high_connectivity" and nodes:
            indexed = sorted(
                range(total),
                key=lambda i: getattr(nodes[i], "connectivity", 0),
                reverse=True,
            )
            return indexed[:num_attack]

        # Default: random
        return self._rng.sample(range(total), min(num_attack, total))

    def _partition_nodes(self, total: int) -> tuple[list[int], list[int]]:
        split_at = max(1, min(int(total * self.attack_ratio), total - 1))
        return list(range(split_at)), list(range(split_at, total))

    def _availability(self, nodes: Sequence[Any]) -> float:
        active = sum(1 for n in nodes if n is not None)
        return active / len(nodes) if nodes else 0.0

    def _post_attack_availability(
        self, total: int, affected: set[int], attack_type: str
    ) -> float:
        if attack_type in {"node_removal", "ddos"}:
            active = total - len(affected)
        elif attack_type == "byzantine":
            active = total  # nodes still respond, just maliciously
        elif attack_type == "partition":
            active = total  # network split but nodes alive
        elif attack_type == "sybil":
            num_sybil = max(1, int(total * self.attack_ratio))
            active = total  # original nodes still alive
            total = total + num_sybil
        else:
            active = total
        return active / total if total else 0.0
