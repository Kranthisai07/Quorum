"""Quorum: a shared-memory coordination layer for concurrent AI agents.

Multiple agents work one large task at the same time against a single shared
memory in CockroachDB. Quorum's job is to make their contention *safe and
visible*: claim conflicts, semantic conflicts, and invalidation cascades are
resolved with serializable transactions and a distributed vector index rather
than with hope.
"""

__version__ = "0.1.0"
