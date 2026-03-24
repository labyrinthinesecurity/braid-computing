#!/usr/bin/env python3
"""
Load 6_topologies.json which contains 1000 generated SCCs and produce their corresponding adjacency matrixes in a CSV (1000adjacency_matrixes.csv) of ratchet topologies whose
SCCID column matches burau.py's "ratchet ID" numbering exactly.

37 columns:
  scc_id        - ratchet ID as assigned by burau.py (0-based)
  adj_RC        - 36 adjacency matrix entries (R,C in 0..5), row-major,
                  1 if edge R→C exists, 0 otherwise

Usage:
  python3 this_script.py              # reads 6_topologies.json in cwd
  python3 this_script.py path/to/6_topologies.json
"""

import csv, itertools, json, random, sys
from collections import defaultdict

INPUT  = sys.argv[1] if len(sys.argv) > 1 else "6_topologies.json"
OUTPUT = "1000adjacency_matrixes.csv"
N      = 6
SEED   = 42

# ── Helpers (inlined from analyze_scc_collisions.py) ─────────────────────────

def is_fully_bidir(edges):
    return all(is_bidir for _, _, is_bidir in edges)

def is_strongly_connected(n, edges):
    adj, adj_rev = defaultdict(set), defaultdict(set)
    for src, dst, _ in edges:
        adj[src].add(dst)
        adj_rev[dst].add(src)
    def bfs(start, graph):
        seen, q = {start}, [start]
        while q:
            v = q.pop(0)
            for u in graph[v]:
                if u not in seen:
                    seen.add(u); q.append(u)
        return seen
    return len(bfs(0, adj)) == n and len(bfs(0, adj_rev)) == n

# ── Load topologies ───────────────────────────────────────────────────────────

with open(INPUT) as f:
    raw = json.load(f)

# burau.py converts to tuples (line 1159)
topos = [[tuple(e) for e in topo] for topo in raw]

fully_bidir  = [t for t in topos if     is_fully_bidir(t) and is_strongly_connected(N, t)]
has_directed = [t for t in topos if not is_fully_bidir(t) and is_strongly_connected(N, t)]

# ── Replicate burau.py's generate_labeled_sccs_stratified (seed=42) ───────────
# Purpose: reproduce the exact ordering of type_topos['ratchet'] before
# the second shuffle.

all_scalars       = list(itertools.product(range(N), repeat=N))
nonconstant       = [s for s in all_scalars if len(set(s)) > 1]
constant          = [s for s in all_scalars if len(set(s)) == 1]

random.seed(SEED)

# 1. oscillation_pool  (111 bidir topos × 270 scalars each → 29,970 entries)
#    Each call to random.sample advances global RNG state.
scalars_per_topo = max(1, 30000 // len(fully_bidir))   # → 270
for topo in fully_bidir:
    random.sample(nonconstant, min(len(nonconstant), scalars_per_topo))

# 2. ratchet_pool  (one random.choice per directed topo → 1,000 entries)
ratchet_pool = []
for topo in has_directed:
    random.choice(all_scalars)          # advances state; WAR not needed here
    ratchet_pool.append(topo)

per_type = min(30000 // 3, len(has_directed))   # → 1,000

# 3. Sample passes that advance RNG before the ratchet sample
reduction_pool_size = len(fully_bidir) * len(constant)  # 111×6 = 666
random.sample(range(reduction_pool_size), min(reduction_pool_size, per_type))
random.sample(range(len(fully_bidir) * scalars_per_topo),
              min(len(fully_bidir) * scalars_per_topo, per_type))

# 4. The ratchet sample — this is what burau.py stores as type_topos['ratchet']
ratchet_ordered = random.sample(ratchet_pool, min(len(ratchet_pool), per_type))

# ── Replicate the second shuffle (rng = random.Random(42)) ───────────────────
# The for-loop in burau.py iterates type_topos in insertion order:
#   'reduction' (666 entries), 'oscillation' (1000 entries), 'ratchet' (1000)
# Only the last type's indices are applied (Python indentation quirk).
# rng advances through all three before producing the final ratchet order.

rng = random.Random(SEED)
rng.shuffle(list(range(min(reduction_pool_size, per_type))))  # reduction
rng.shuffle(list(range(min(len(fully_bidir) * scalars_per_topo, per_type))))  # oscillation
idx = list(range(len(ratchet_ordered)))
rng.shuffle(idx)

final_ratchets = [ratchet_ordered[i] for i in idx]

# ── Build adjacency matrices and write CSV ────────────────────────────────────

def adj_row(topo):
    m = [[0]*N for _ in range(N)]
    for src, dst, _ in topo:
        m[src][dst] = 1
    return [m[r][c] for r in range(N) for c in range(N)]

header = ["scc_id"] + [f"adj_{r}{c}" for r in range(N) for c in range(N)]

with open(OUTPUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for scc_id, topo in enumerate(final_ratchets):
        w.writerow([scc_id] + adj_row(topo))

print(f"Written {len(final_ratchets)} ratchets to {OUTPUT}", file=sys.stderr)

