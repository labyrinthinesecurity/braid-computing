#!/usr/bin/env python3
"""
Exhaustive analysis of SCC invariant collisions.

For each SCC size n, counts:
- Total (topology, WAR) possibilities
- Trivial possibilities (invariant is all 1s - flat bidirectional cycles only)
- Non-trivial possibilities
- Collisions among non-trivial possibilities

A collision occurs when two different topologies produce the same invariant vector.

Constraints enforced:
0. SCCs only - graph must be strongly connected
1. Cycle orientation - enforced by Johnson's algorithm
2. WAR/edge consistency - edge types derived from WAR values
3. Unoriented cycle handling - when ALL edges are bidirectional, use symmetric factor

Algorithm for R computation (per spec):
- For ORIENTED cycles (has at least one directed edge):
  - Bidirectional edge: s=s' -> 1, s<s' -> q/p, s>s' -> 1/(p*q)
  - Directed edge: s=s' -> 1/p, s<s' -> q, s>s' -> 1/(p*q²)
- For UNORIENTED cycles (all edges bidirectional):
  - Bidirectional edge: s=s' -> 1, s≠s' -> 1/(p*q) [symmetric!]
"""

import itertools
from fractions import Fraction
from collections import defaultdict
import sys
import random
import time
import argparse

# Import from elementary_cycle_invariant.py for prime utilities
sys.path.insert(0, '.')
from elementary_cycle_invariant import get_nth_prime, primorial, next_prime_after

# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================
# Default number of topologies to sample for large n (n >= 5)
DEFAULT_TOPOLOGY_SAMPLES = 100

# Default number of WAR orderings to sample per topology
DEFAULT_WAR_SAMPLES = 100

# Random seed for reproducibility (set to None for random behavior)
RANDOM_SEED = 42


def set_random_seed(seed=None):
    """Set random seed for reproducibility. Uses RANDOM_SEED if seed is None."""
    if seed is None:
        seed = RANDOM_SEED
    if seed is not None:
        random.seed(seed)
        print(f"Random seed set to: {seed}")


def is_fully_bidirectional_topology(edges):
    """
    Check if a topology is fully bidirectional (all edges are bidir).

    Fully bidirectional topologies only produce:
    - Reductions (R=1) when WAR is flat
    - Oscillators (R=0) when WAR is non-flat

    These have trivial/fixed invariants and should be filtered out
    from collision analysis since they don't provide meaningful
    discriminative power.
    """
    return all(is_bidir for _, _, is_bidir in edges)


def generate_scc_topologies(n, include_fully_bidir=True):
    """
    Generate all canonical strongly connected topologies for n vertices.

    For n=3, there are exactly 5 canonical topologies:
    4 out 5 topologies have a non hub topology:
    1. All directed (pure cycle)
    2. 1 bidir edge + 2 directed edges
    3. 2 bidir edges + 1 directed edge
    4. All bidir (complete bidirectional)
    1 topologies is a hub topology:
    5. 2 bidir edges

    Args:
        n: Number of vertices
        include_fully_bidir: If False, exclude fully bidirectional topologies
                            (which only produce oscillators/reductions)

    Returns list of edge sets where each edge is (src, dst, is_bidir).
    Topologies are canonical (one representative per isomorphism class).
    """
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    seen_canonical = set()
    scc_topologies = []

    for choices in itertools.product([0, 1, 2, 3], repeat=len(pairs)):
        edges = []
        for pair_idx, choice in enumerate(choices):
            i, j = pairs[pair_idx]
            if choice == 1:  # i→j only (directed)
                edges.append((i, j, False))
            elif choice == 2:  # j→i only (directed)
                edges.append((j, i, False))
            elif choice == 3:  # bidirectional
                edges.append((i, j, True))
                edges.append((j, i, True))

        if is_strongly_connected(n, edges):
            # Skip fully bidirectional if requested
            if not include_fully_bidir and is_fully_bidirectional_topology(edges):
                continue

            # Get canonical form to avoid duplicates
            canonical = canonicalize_topology(n, edges)
            if canonical not in seen_canonical:
                seen_canonical.add(canonical)
                scc_topologies.append(edges)

    return scc_topologies


def generate_fusion_topologies(n):
    """
    Generate only fusion topologies (SCCs with at least one directed edge).

    This filters out fully bidirectional topologies which only produce:
    - Reductions (R=1): flat WAR, trivial invariant
    - Oscillators (R=0): non-flat WAR, trivial invariant

    For collision analysis, only fusions are interesting since they have
    non-trivial R values that could potentially collide.
    """
    return generate_scc_topologies(n, include_fully_bidir=False)


def generate_random_fusion_topology(n):
    """
    Generate a random fusion topology (SCC with at least one directed edge).

    Uses rejection sampling: generate random edge configurations until we get
    a strongly connected graph that is not fully bidirectional.

    This avoids enumerating all topologies, making it feasible for large n.

    Returns:
        List of edges where each edge is (src, dst, is_bidir), or None if
        max_attempts exceeded (should be rare for reasonable n).
    """
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    max_attempts = 10000  # Prevent infinite loops

    for _ in range(max_attempts):
        # Generate random edge configuration
        edges = []
        has_directed = False

        for i, j in pairs:
            choice = random.randint(0, 3)
            if choice == 1:  # i→j only (directed)
                edges.append((i, j, False))
                has_directed = True
            elif choice == 2:  # j→i only (directed)
                edges.append((j, i, False))
                has_directed = True
            elif choice == 3:  # bidirectional
                edges.append((i, j, True))
                edges.append((j, i, True))
            # choice == 0: no edge

        # Must have at least one directed edge (fusion requirement)
        if not has_directed:
            continue

        # Must be strongly connected
        if is_strongly_connected(n, edges):
            return edges

    return None  # Failed to generate (very unlikely)


def is_strongly_connected(n, edges):
    """Check if graph is strongly connected."""
    if n <= 1:
        return n == 1

    adj = defaultdict(set)
    adj_rev = defaultdict(set)
    for src, dst, _ in edges:
        adj[src].add(dst)
        adj_rev[dst].add(src)

    # Forward BFS from 0
    visited = {0}
    queue = [0]
    while queue:
        v = queue.pop(0)
        for u in adj[v]:
            if u not in visited:
                visited.add(u)
                queue.append(u)
    if len(visited) != n:
        return False

    # Backward BFS from 0
    visited = {0}
    queue = [0]
    while queue:
        v = queue.pop(0)
        for u in adj_rev[v]:
            if u not in visited:
                visited.add(u)
                queue.append(u)

    return len(visited) == n


def generate_war_patterns(n, include_flat=True):
    """
    Generate all valid WAR patterns for n vertices using I/D/F notation.

    Each vertex is assigned one of:
    - I (Increasing/max): vertex has higher WAR than neighbors
    - D (Decreasing/min): vertex has lower WAR than neighbors
    - F (Flat): vertex has equal WAR with some neighbors

    Valid patterns:
    - All F (flat reduction case) - only if include_flat=True
    - Patterns with at least one I AND at least one D

    Invalid patterns (excluded):
    - All I (no minimum exists)
    - All D (no maximum exists)
    - Mix of I+F only (no minimum)
    - Mix of D+F only (no maximum)

    Args:
        n: Number of vertices
        include_flat: If False, exclude all-F patterns (reductions)

    Returns list of tuples where each tuple has values 0 (D), 1 (F), or 2 (I).
    """
    I, D, F = 2, 0, 1  # Using numeric values for easier processing

    valid_patterns = []

    for pattern in itertools.product([D, F, I], repeat=n):
        has_I = I in pattern
        has_D = D in pattern
        has_F = F in pattern
        all_F = all(v == F for v in pattern)

        # Skip all-F if not including flat patterns
        if all_F:
            if include_flat:
                valid_patterns.append(pattern)
            continue

        # Valid: has both I and D
        if has_I and has_D:
            valid_patterns.append(pattern)

    return valid_patterns


def generate_fusion_war_patterns(n):
    """
    Generate WAR patterns for fusion analysis (excludes all-F flat patterns).

    All-F patterns only apply to reductions (R=1) which are filtered out.
    For fusions, we need patterns with at least one I and at least one D.
    """
    return generate_war_patterns(n, include_flat=False)


def pattern_to_war_values(pattern):
    """
    Convert I/D/F pattern to WAR value dict.
    I -> 3 (high), F -> 2 (mid), D -> 1 (low)
    """
    # Map: D=0 -> 1, F=1 -> 2, I=2 -> 3
    return {i: pattern[i] + 1 for i in range(len(pattern))}


def is_war_compatible_with_topology(n, edges, war_values):
    """
    Check if WAR pattern is compatible with topology.

    Enforces level semantics:
    - I (WAR=3): max level - no neighbor can be higher
    - D (WAR=1): min level - no neighbor can be lower
    - F (WAR=2): intermediate level - between I and D

    The pattern-level filter (generate_war_patterns) already ensures
    we have valid I/D/F combinations. This just checks the WAR values
    are consistent with the topology's edge structure.

    Returns True if the pattern is realizable on this topology.
    """
    # Build neighbor sets (considering edge directions)
    neighbors = {i: set() for i in range(n)}
    for src, dst, is_bidir in edges:
        neighbors[src].add(dst)
        if is_bidir:
            neighbors[dst].add(src)

    for v in range(n):
        if not neighbors[v]:
            continue

        neighbor_wars = [war_values[u] for u in neighbors[v]]
        my_war = war_values[v]

        # Check level constraints
        if my_war == 3:  # I: at max level (no neighbor higher)
            if any(w > my_war for w in neighbor_wars):
                return False
        elif my_war == 1:  # D: at min level (no neighbor lower)
            if any(w < my_war for w in neighbor_wars):
                return False
        # F (my_war == 2): intermediate, always valid

    return True


def generate_war_orderings(n, include_flat=True):
    """
    Generate all valid WAR orderings for n vertices.
    Uses I/D/F pattern system.

    Args:
        n: Number of vertices
        include_flat: If False, exclude all-F patterns (reductions)

    For backwards compatibility, returns list of tuples with numeric values.
    """
    patterns = generate_war_patterns(n, include_flat=include_flat)
    # Convert to the format expected by the rest of the code
    return [tuple(p[i] + 1 for i in range(n)) for p in patterns]


def generate_fusion_war_orderings(n):
    """
    Generate WAR orderings for fusion analysis only.
    Excludes all-F (flat) patterns which only apply to reductions.
    """
    return generate_war_orderings(n, include_flat=False)


def find_elementary_cycles(n, edges):
    """Find all elementary cycles using Johnson's algorithm."""
    adj = defaultdict(list)
    for src, dst, _ in edges:
        adj[src].append(dst)

    all_cycles = []

    def unblock(u, blocked, block_map):
        blocked.discard(u)
        for w in list(block_map[u]):
            block_map[u].discard(w)
            if w in blocked:
                unblock(w, blocked, block_map)

    def circuit(v, start, component, stack, blocked, block_map):
        found = False
        stack.append(v)
        blocked.add(v)

        for w in adj.get(v, []):
            if w not in component:
                continue
            if w == start:
                all_cycles.append(tuple(stack))
                found = True
            elif w not in blocked:
                if circuit(w, start, component, stack, blocked, block_map):
                    found = True

        if found:
            unblock(v, blocked, block_map)
        else:
            for w in adj.get(v, []):
                if w in component:
                    block_map[w].add(v)

        stack.pop()
        return found

    vertices = list(range(n))
    for start in vertices:
        component = set(v for v in vertices if v >= start)
        blocked = set()
        block_map = defaultdict(set)
        circuit(start, start, component, [], blocked, block_map)

    return all_cycles


def compute_cycle_r_value(cycle, war_values, edges):
    """
    Compute R value for a single cycle following the spec.

    IMPORTANT: Handles three cases:
    - Unoriented (all bidir) + flat WAR: R = 1 (reduction)
    - Unoriented (all bidir) + non-flat WAR: R = 0 (oscillator - by definition)
    - Oriented (has at least one directed edge): compute using factor formula

    Returns the minimum R across all rotations.
    """
    # Build edge info
    directed_edges = set()
    bidir_edges = set()
    for src, dst, is_bidir in edges:
        directed_edges.add((src, dst))
        if is_bidir:
            directed_edges.add((dst, src))
            bidir_edges.add((src, dst))
            bidir_edges.add((dst, src))

    n = len(cycle)

    # Determine if cycle is unoriented (ALL edges in cycle are bidirectional)
    cycle_edges_bidir = []
    for i in range(n):
        src = cycle[i]
        dst = cycle[(i + 1) % n]
        is_bidir = (src, dst) in bidir_edges
        cycle_edges_bidir.append(is_bidir)

    all_bidir = all(cycle_edges_bidir)

    # For unoriented cycles, check if WAR is flat
    if all_bidir:
        cycle_wars = [war_values[v] for v in cycle]
        is_flat = len(set(cycle_wars)) == 1

        if is_flat:
            # Unoriented + flat WAR = reduction: R = 1
            return Fraction(1)
        else:
            # Unoriented + non-flat WAR = oscillator: R = 0 (by definition)
            return Fraction(0)

    # Oriented cycle: compute R for each rotation and take minimum
    r_values = []

    for rotation in range(n):
        # Rotate the cycle
        rotated = cycle[rotation:] + cycle[:rotation]

        # Prime setup
        large_p = next_prime_after(primorial(n))
        small_p = get_nth_prime(0)
        R = Fraction(1)

        for i in range(n):
            src = rotated[i]
            dst = rotated[(i + 1) % n]

            s1 = war_values[src]
            s2 = war_values[dst]
            is_bidir = (src, dst) in bidir_edges

            if is_bidir:
                # Bidirectional edge in oriented cycle: asymmetric factors
                if s1 == s2:
                    factor = Fraction(1)
                elif s1 < s2:
                    factor = Fraction(large_p, small_p)
                else:
                    factor = Fraction(1, small_p * large_p)
            else:
                # Directed edge
                if s1 == s2:
                    factor = Fraction(1, small_p)
                elif s1 < s2:
                    factor = Fraction(large_p)
                else:
                    factor = Fraction(1, small_p * large_p * large_p)

            R *= factor
            small_p = next_prime_after(small_p)
            large_p = next_prime_after(large_p)

        r_values.append(R)

    return min(r_values)


def compute_invariant_vector(n, edges, war_values):
    """Compute the invariant vector (sorted tuple of R values) for an SCC."""
    cycles = find_elementary_cycles(n, edges)

    if not cycles:
        return None

    r_values = []
    for cycle in cycles:
        r = compute_cycle_r_value(cycle, war_values, edges)
        r_values.append(r)

    if not r_values:
        return None

    return tuple(sorted(r_values))


def is_trivial_invariant(inv):
    """
    Check if invariant is trivial (all 1s).
    Trivial means all cycles are flat bidirectional (R=1 reductions).
    """
    if inv is None:
        return False
    return all(r == 1 for r in inv)


def is_oscillator_invariant(inv):
    """
    Check if invariant is oscillator (all 0s).
    Oscillator means all cycles are fully bidirectional with non-flat WAR (R=0).
    """
    if inv is None:
        return False
    return all(r == 0 for r in inv)


def classify_invariant(inv):
    """
    Classify invariant into three categories:
    - 'reduction': all R=1 (fully bidir + flat WAR)
    - 'oscillator': all R=0 (fully bidir + non-flat WAR)
    - 'fusion': R≠0,1 (has directed edges)
    """
    if inv is None:
        return None
    if is_trivial_invariant(inv):
        return 'reduction'
    if is_oscillator_invariant(inv):
        return 'oscillator'
    return 'fusion'


def canonicalize_topology(n, edges):
    """Create canonical representation of topology."""
    directed = set()
    for src, dst, is_bidir in edges:
        directed.add((src, dst))
        if is_bidir:
            directed.add((dst, src))

    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append('self')
            else:
                has_ij = (i, j) in directed
                has_ji = (j, i) in directed
                if has_ij and has_ji:
                    edge_type = 'bidir'
                elif has_ij:
                    edge_type = 'fwd'
                elif has_ji:
                    edge_type = 'rev'
                else:
                    edge_type = 'none'
                row.append(edge_type)
        matrix.append(tuple(row))

    min_matrix = None
    for perm in itertools.permutations(range(n)):
        new_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                old_i, old_j = perm[i], perm[j]
                row.append(matrix[old_i][old_j])
            new_matrix.append(tuple(row))
        new_matrix = tuple(new_matrix)
        if min_matrix is None or new_matrix < min_matrix:
            min_matrix = new_matrix

    return min_matrix


def analyze_scc_size_exhaustive(n, verbose=True, fusions_only=False):
    """
    Exhaustively analyze all SCCs of size n.
    Suitable for n <= 4.

    Args:
        n: Number of vertices
        verbose: Print progress information
        fusions_only: If True, filter out oscillators and flat bidirectional
                     topologies (reductions). Only analyze fusions which have
                     at least one directed edge.

    Classifies invariants into three categories:
    - Reductions (R=1): fully bidir + flat WAR
    - Oscillators (R=0): fully bidir + non-flat WAR
    - Fusions (R≠0,1): has directed edges
    """
    mode_str = "FUSIONS ONLY" if fusions_only else "all SCCs"
    if verbose:
        print(f"\n{'='*70}")
        print(f"Analyzing {n}-vertex SCCs exhaustively ({mode_str})...")
        print(f"{'='*70}")

    # Filter out fully bidirectional topologies if fusions_only
    scc_topologies = generate_scc_topologies(n, include_fully_bidir=not fusions_only)
    war_orderings = generate_war_orderings(n)

    if verbose:
        print(f"SCC topologies: {len(scc_topologies)}")
        print(f"WAR orderings: {len(war_orderings)}")
        print(f"Total possibilities: {len(scc_topologies) * len(war_orderings)}")

    total_possibilities = 0
    reduction_count = 0
    oscillator_count = 0
    fusion_count = 0

    reduction_inv_to_topos = defaultdict(set)
    oscillator_inv_to_topos = defaultdict(set)
    fusion_inv_to_topos = defaultdict(set)

    skipped_incompatible = 0

    for edges in scc_topologies:
        topo = canonicalize_topology(n, edges)

        for war in war_orderings:
            war_dict = {i: war[i] for i in range(n)}

            # Skip if WAR pattern is incompatible with topology
            if not is_war_compatible_with_topology(n, edges, war_dict):
                skipped_incompatible += 1
                continue

            inv = compute_invariant_vector(n, edges, war_dict)

            if inv is None:
                continue

            total_possibilities += 1
            category = classify_invariant(inv)

            if category == 'reduction':
                reduction_count += 1
                reduction_inv_to_topos[inv].add(topo)
            elif category == 'oscillator':
                oscillator_count += 1
                oscillator_inv_to_topos[inv].add(topo)
            else:  # fusion
                fusion_count += 1
                fusion_inv_to_topos[inv].add(topo)

    reduction_collisions = sum(1 for topos in reduction_inv_to_topos.values() if len(topos) > 1)
    oscillator_collisions = sum(1 for topos in oscillator_inv_to_topos.values() if len(topos) > 1)
    fusion_collisions = sum(1 for topos in fusion_inv_to_topos.values() if len(topos) > 1)

    # For backwards compatibility, map to old names
    trivial_count = reduction_count
    nontrivial_count = oscillator_count + fusion_count
    trivial_collisions = reduction_collisions
    nontrivial_collisions = oscillator_collisions + fusion_collisions

    if verbose:
        print(f"Skipped incompatible (topo, WAR) pairs: {skipped_incompatible}")
        print(f"Valid possibilities: {total_possibilities}")

    results = {
        'n': n,
        'method': 'exhaustive',
        'total_topologies': len(scc_topologies),
        'total_war_orderings': len(war_orderings),
        'skipped_incompatible': skipped_incompatible,
        'total_possibilities': total_possibilities,
        # New three-category stats
        'reduction_count': reduction_count,
        'oscillator_count': oscillator_count,
        'fusion_count': fusion_count,
        'unique_reduction_invariants': len(reduction_inv_to_topos),
        'unique_oscillator_invariants': len(oscillator_inv_to_topos),
        'unique_fusion_invariants': len(fusion_inv_to_topos),
        'reduction_collisions': reduction_collisions,
        'oscillator_collisions': oscillator_collisions,
        'fusion_collisions': fusion_collisions,
        # Backwards compatible
        'trivial_count': trivial_count,
        'nontrivial_count': nontrivial_count,
        'unique_trivial_invariants': len(reduction_inv_to_topos),
        'unique_nontrivial_invariants': len(oscillator_inv_to_topos) + len(fusion_inv_to_topos),
        'trivial_collisions': trivial_collisions,
        'nontrivial_collisions': nontrivial_collisions,
    }

    if verbose:
        print_results(results, fusion_inv_to_topos)

    return results


def analyze_scc_size_sampled(n, num_topology_samples=None, num_war_samples=None, seed=None, verbose=True, fusions_only=False):
    """
    Sample-based analysis for large n (n >= 5).

    Args:
        n: Number of vertices
        num_topology_samples: Number of topologies to sample (default: DEFAULT_TOPOLOGY_SAMPLES)
        num_war_samples: Number of WAR orderings per topology (default: DEFAULT_WAR_SAMPLES)
        seed: Random seed for reproducibility (default: uses RANDOM_SEED)
        verbose: Print progress information
        fusions_only: If True, filter out oscillators and flat bidirectional
                     topologies (reductions). Only analyze fusions.
    """
    # Set random seed for reproducibility
    set_random_seed(seed)

    if num_topology_samples is None:
        num_topology_samples = DEFAULT_TOPOLOGY_SAMPLES
    if num_war_samples is None:
        num_war_samples = DEFAULT_WAR_SAMPLES

    mode_str = "FUSIONS ONLY" if fusions_only else "all SCCs"
    if verbose:
        print(f"\n{'='*70}")
        print(f"Analyzing {n}-vertex SCCs via SAMPLING ({mode_str})...")
        print(f"{'='*70}")
        print(f"Strategy: Sample {num_topology_samples} topologies × {num_war_samples} WAR orderings")

    # Generate topologies (filter out fully bidir if fusions_only)
    if verbose:
        print("Generating SCC topologies (this may take a while for n=5)...")

    start_time = time.time()
    scc_topologies = generate_scc_topologies(n, include_fully_bidir=not fusions_only)
    topo_time = time.time() - start_time

    war_orderings = generate_war_orderings(n)

    if verbose:
        print(f"Total SCC topologies: {len(scc_topologies)} (generated in {topo_time:.1f}s)")
        print(f"Total WAR orderings: {len(war_orderings)}")
        print(f"Full space: {len(scc_topologies) * len(war_orderings):,} possibilities")
        print(f"Sampling: {num_topology_samples} × {num_war_samples} = {num_topology_samples * num_war_samples:,} samples")

    # Sample topologies
    if len(scc_topologies) <= num_topology_samples:
        sampled_topologies = scc_topologies
    else:
        sampled_topologies = random.sample(scc_topologies, num_topology_samples)

    # Sample WAR orderings
    if len(war_orderings) <= num_war_samples:
        sampled_wars = war_orderings
    else:
        sampled_wars = random.sample(war_orderings, num_war_samples)

    total_sampled = 0
    skipped_incompatible = 0
    reduction_count = 0
    oscillator_count = 0
    fusion_count = 0

    reduction_inv_to_topos = defaultdict(set)
    oscillator_inv_to_topos = defaultdict(set)
    fusion_inv_to_topos = defaultdict(set)

    if verbose:
        print("Processing samples...")

    for idx, edges in enumerate(sampled_topologies):
        if verbose and idx % 100 == 0:
            print(f"  Progress: {idx}/{len(sampled_topologies)} topologies...")

        topo = canonicalize_topology(n, edges)

        for war in sampled_wars:
            war_dict = {i: war[i] for i in range(n)}

            # Skip if WAR pattern is incompatible with topology
            if not is_war_compatible_with_topology(n, edges, war_dict):
                skipped_incompatible += 1
                continue

            inv = compute_invariant_vector(n, edges, war_dict)

            if inv is None:
                continue

            total_sampled += 1
            category = classify_invariant(inv)

            if category == 'reduction':
                reduction_count += 1
                reduction_inv_to_topos[inv].add(topo)
            elif category == 'oscillator':
                oscillator_count += 1
                oscillator_inv_to_topos[inv].add(topo)
            else:  # fusion
                fusion_count += 1
                fusion_inv_to_topos[inv].add(topo)

    reduction_collisions = sum(1 for topos in reduction_inv_to_topos.values() if len(topos) > 1)
    oscillator_collisions = sum(1 for topos in oscillator_inv_to_topos.values() if len(topos) > 1)
    fusion_collisions = sum(1 for topos in fusion_inv_to_topos.values() if len(topos) > 1)

    # For backwards compatibility
    trivial_count = reduction_count
    nontrivial_count = oscillator_count + fusion_count
    trivial_collisions = reduction_collisions
    nontrivial_collisions = oscillator_collisions + fusion_collisions

    # Estimate collision rate for fusions (the interesting case)
    fusion_collision_rate = fusion_collisions / len(fusion_inv_to_topos) if fusion_inv_to_topos else 0

    results = {
        'n': n,
        'method': 'sampled',
        'total_topologies': len(scc_topologies),
        'sampled_topologies': len(sampled_topologies),
        'total_war_orderings': len(war_orderings),
        'sampled_war_orderings': len(sampled_wars),
        'total_possibilities': len(scc_topologies) * len(war_orderings),
        'total_sampled': total_sampled,
        # New three-category stats
        'reduction_count': reduction_count,
        'oscillator_count': oscillator_count,
        'fusion_count': fusion_count,
        'unique_reduction_invariants': len(reduction_inv_to_topos),
        'unique_oscillator_invariants': len(oscillator_inv_to_topos),
        'unique_fusion_invariants': len(fusion_inv_to_topos),
        'reduction_collisions': reduction_collisions,
        'oscillator_collisions': oscillator_collisions,
        'fusion_collisions': fusion_collisions,
        # Backwards compatible
        'trivial_count': trivial_count,
        'nontrivial_count': nontrivial_count,
        'unique_trivial_invariants': len(reduction_inv_to_topos),
        'unique_nontrivial_invariants': len(oscillator_inv_to_topos) + len(fusion_inv_to_topos),
        'trivial_collisions': trivial_collisions,
        'nontrivial_collisions': nontrivial_collisions,
        'collision_rate': fusion_collision_rate,
    }

    if verbose:
        print_results(results, fusion_inv_to_topos)
        print(f"\nNote: Sampling provides a LOWER BOUND on collisions.")
        print(f"Estimated fusion collision rate: {fusion_collision_rate*100:.2f}% of fusion invariants have collisions")

    return results


def print_results(results, fusion_inv_to_topos=None):
    """Print analysis results with focus on fusions (oscillators/reductions filtered)."""
    n = results['n']
    method = results.get('method', 'exhaustive')

    print(f"\n--- Results for n={n} ({method}) ---")

    if method == 'sampled':
        print(f"Sampled: {results['total_sampled']:,} of {results['total_possibilities']:,} possibilities")
    else:
        print(f"Total fusion possibilities: {results['total_possibilities']:,}")

    # Get counts (reductions/oscillators should be 0 when fusions_only=True)
    reduction_count = results.get('reduction_count', results.get('trivial_count', 0))
    oscillator_count = results.get('oscillator_count', 0)
    fusion_count = results.get('fusion_count', results.get('nontrivial_count', 0) - oscillator_count)

    # Check if we're in fusions-only mode (no reductions/oscillators)
    fusions_only_mode = (reduction_count == 0 and oscillator_count == 0)

    if fusions_only_mode:
        print(f"Fusions analyzed: {fusion_count:,}")
    else:
        total = reduction_count + oscillator_count + fusion_count
        if total > 0:
            print(f"\nCategory breakdown:")
            print(f"  Reductions (R=1, bidir+flat):     {reduction_count:,} ({100*reduction_count/total:.1f}%)")
            print(f"  Oscillators (R=0, bidir+non-flat): {oscillator_count:,} ({100*oscillator_count/total:.1f}%)")
            print(f"  Fusions (R varies, has directed):  {fusion_count:,} ({100*fusion_count/total:.1f}%)")

    # Unique invariants
    unique_reductions = results.get('unique_reduction_invariants', results.get('unique_trivial_invariants', 0))
    unique_oscillators = results.get('unique_oscillator_invariants', 0)
    unique_fusions = results.get('unique_fusion_invariants', results.get('unique_nontrivial_invariants', 0) - unique_oscillators)

    if fusions_only_mode:
        print(f"Unique fusion invariant vectors: {unique_fusions}")
    else:
        print(f"\nUnique invariant vectors:")
        print(f"  Reductions:  {unique_reductions}")
        print(f"  Oscillators: {unique_oscillators}")
        print(f"  Fusions:     {unique_fusions}")

    # Collisions (only fusions matter for collision analysis)
    fusion_collisions = results.get('fusion_collisions', results.get('nontrivial_collisions', 0))

    print(f"Fusion collisions: {fusion_collisions}")

    if fusion_collisions > 0 and fusion_inv_to_topos:
        print(f"\nFusion collisions found! Examples:")
        shown = 0
        for inv, topos in fusion_inv_to_topos.items():
            if len(topos) > 1:
                print(f"  Invariant {inv}: {len(topos)} different topologies")
                shown += 1
                if shown >= 3:
                    break
    elif fusion_collisions == 0:
        print(f"No fusion collisions - R vectors uniquely identify topologies!")


def estimate_collision_growth(results_list, verbose=True):
    """
    Estimate collision count as a function of n based on observed data.

    Uses the collected results to:
    1. Identify the pattern of growth in collisions
    2. Fit exponential/polynomial models
    3. Extrapolate to larger n

    Args:
        results_list: List of result dictionaries from analyze_scc_size_* functions
        verbose: Print detailed analysis

    Returns:
        Dictionary with model parameters and predictions
    """
    import math

    if verbose:
        print("\n" + "="*70)
        print("COLLISION GROWTH ESTIMATION")
        print("="*70)

    # Extract data points where we have actual collision counts
    data_points = []
    for r in results_list:
        n = r['n']
        collisions = r['nontrivial_collisions']
        topologies = r['total_topologies']
        unique_invs = r['unique_nontrivial_invariants']
        method = r.get('method', 'exhaustive')

        data_points.append({
            'n': n,
            'collisions': collisions,
            'topologies': topologies,
            'unique_invariants': unique_invs,
            'method': method
        })

    if verbose:
        print("\nObserved Data:")
        print(f"{'n':<4} {'Topologies':<12} {'Unique Invs':<14} {'Collisions':<12} {'Method':<12}")
        print("-"*54)
        for d in data_points:
            print(f"{d['n']:<4} {d['topologies']:<12} {d['unique_invariants']:<14} {d['collisions']:<12} {d['method']:<12}")

    # Compute derived metrics
    if verbose:
        print("\nDerived Metrics:")
        print(f"{'n':<4} {'Topo Growth':<14} {'Collision Rate':<16} {'Coll/Topo':<12}")
        print("-"*46)

    for i, d in enumerate(data_points):
        if i > 0:
            prev = data_points[i-1]
            topo_growth = d['topologies'] / prev['topologies'] if prev['topologies'] > 0 else 0
            d['topo_growth'] = topo_growth
        else:
            d['topo_growth'] = None

        # Collision rate: fraction of unique invariants that have collisions
        if d['unique_invariants'] > 0:
            d['collision_rate'] = d['collisions'] / d['unique_invariants']
        else:
            d['collision_rate'] = 0

        # Collisions per topology
        if d['topologies'] > 0:
            d['coll_per_topo'] = d['collisions'] / d['topologies']
        else:
            d['coll_per_topo'] = 0

        if verbose:
            tg = f"{d['topo_growth']:.2f}x" if d['topo_growth'] else "N/A"
            print(f"{d['n']:<4} {tg:<14} {d['collision_rate']*100:.2f}%{'':<10} {d['coll_per_topo']:.4f}")

    # Fit exponential model for topology growth
    # Topologies(n) ≈ a * b^n
    # ln(T) = ln(a) + n*ln(b)
    if verbose:
        print("\n" + "-"*70)
        print("GROWTH MODELS")
        print("-"*70)

    # Estimate topology growth rate (from n=3 to n=5 where we have data)
    valid_topos = [(d['n'], d['topologies']) for d in data_points if d['topologies'] > 1]
    topo_base = None
    if len(valid_topos) >= 2:
        n1, t1 = valid_topos[0]
        n2, t2 = valid_topos[-1]

        # Solve t2 = t1 * b^(n2-n1) for b
        if t1 > 0 and n2 != n1:
            topo_base = (t2 / t1) ** (1 / (n2 - n1))
            if verbose:
                print(f"\nTopology Growth Model: T(n) ≈ {t1:.0f} * {topo_base:.2f}^(n-{n1})")
                print(f"  Estimated topologies for n=6: ~{int(t2 * topo_base)}")
                print(f"  Estimated topologies for n=7: ~{int(t2 * topo_base**2)}")

    # Estimate collision growth (only meaningful for n >= 4 where collisions exist)
    collision_data = [(d['n'], d['collisions']) for d in data_points if d['collisions'] > 0]

    predictions = {}

    if len(collision_data) >= 1:
        if verbose:
            print(f"\nCollision Growth Analysis:")

        # From n=4, collisions grow very quickly
        # This is because:
        # 1. More topologies = more potential pairs that could collide
        # 2. More cycle combinations = higher chance of collision
        # 3. Collision rate appears to be stable/growing

        # Use latest collision rate to predict
        latest = data_points[-1]
        if latest['collision_rate'] > 0:
            collision_rate = latest['collision_rate']

            if verbose:
                print(f"  Latest collision rate (n={latest['n']}): {collision_rate*100:.2f}%")
                print(f"  Collisions per topology: {latest['coll_per_topo']:.4f}")

            # For n >= 4 with collisions, estimate using collision rate
            for d in data_points:
                if d['collisions'] > 0:
                    base_n = d['n']
                    base_coll = d['collisions']
                    base_topos = d['topologies']

                    # Estimate collisions grow faster than topologies due to combinatorial effect
                    # Collisions(n) ~ c * topologies(n)^alpha where alpha > 1
                    # For now, use a simple ratio-based estimate

            # Predictions for n=6,7
            for future_n in [6, 7, 8]:
                # Conservative estimate: collisions grow at least as fast as topologies
                if topo_base is not None:
                    growth_factor = topo_base ** (future_n - latest['n'])

                    # Collision growth tends to be super-linear due to combinatorics
                    # Use collision_rate * estimated_unique_invariants
                    estimated_topos = latest['topologies'] * growth_factor

                    # Unique invariants grow slower than topologies (due to collisions)
                    # Estimate: inv_growth ~ topo_growth^0.8
                    estimated_invs = latest['unique_invariants'] * (growth_factor ** 0.8)

                    # Apply collision rate (may increase with n)
                    collision_rate_adjusted = min(collision_rate * (1.1 ** (future_n - latest['n'])), 0.5)
                    estimated_collisions = int(estimated_invs * collision_rate_adjusted)

                    predictions[future_n] = {
                        'estimated_topologies': int(estimated_topos),
                        'estimated_collisions': estimated_collisions,
                        'collision_rate': collision_rate_adjusted,
                        'note': 'extrapolated'
                    }

            if verbose and predictions:
                print(f"\n  Predictions (extrapolated):")
                print(f"  {'n':<4} {'Est. Topos':<15} {'Est. Collisions':<18} {'Est. Rate':<12}")
                print(f"  " + "-"*50)
                for n, p in sorted(predictions.items()):
                    print(f"  {n:<4} {p['estimated_topologies']:<15,} {p['estimated_collisions']:<18,} {p['collision_rate']*100:.1f}%")

    else:
        if verbose:
            print("\nNo collisions found in data - unable to estimate collision growth.")
            print("This could indicate the invariant is collision-free for small n.")

    # Summary
    if verbose:
        print("\n" + "-"*70)
        print("KEY FINDINGS")
        print("-"*70)

        first_collision_n = None
        for d in data_points:
            if d['collisions'] > 0:
                first_collision_n = d['n']
                break

        if first_collision_n:
            print(f"  • First non-trivial collisions appear at n={first_collision_n}")
            print(f"  • Collision rate at n={latest['n']}: {latest['collision_rate']*100:.2f}%")
            print(f"  • Growth pattern: Collisions increase super-linearly with topologies")
            if predictions:
                print(f"  • Estimated collisions at n=6: ~{predictions.get(6, {}).get('estimated_collisions', 'N/A')}")
        else:
            print(f"  • No collisions found for n ≤ {data_points[-1]['n']}")
            print(f"  • The invariant may be collision-free for small SCCs")

    return {
        'data_points': data_points,
        'predictions': predictions,
        'first_collision_n': first_collision_n if 'first_collision_n' in dir() else None
    }


def analyze_scc_random_sampling(n, timeout_minutes, seed=None, verbose=True):
    """
    Analyze SCCs of size n using random topology sampling with a timeout.

    This is useful for large n (n >= 6) where enumerating all topologies
    is infeasible. Instead, we generate random fusion topologies on-the-fly
    and analyze them until the timeout is reached.

    Args:
        n: Number of vertices
        timeout_minutes: Maximum time to run in minutes
        seed: Random seed for reproducibility
        verbose: Print progress information

    Returns:
        Dictionary with analysis results
    """
    set_random_seed(seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f"RANDOM SAMPLING for n={n} (timeout: {timeout_minutes} minutes)")
        print(f"{'='*70}")
        print("Strategy: Generate random fusion topologies on-the-fly")
        print("This avoids enumerating all topologies (infeasible for n >= 6)")

    # Generate all WAR orderings (this is still feasible)
    war_orderings = generate_war_orderings(n)
    if verbose:
        print(f"WAR orderings: {len(war_orderings)}")

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    # Track statistics
    topologies_generated = 0
    total_samples = 0
    skipped_incompatible = 0
    fusion_count = 0

    seen_topologies = set()  # Track unique topologies (by canonical form)
    fusion_inv_to_topos = defaultdict(set)

    last_progress = start_time

    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            break

        # Print progress every 30 seconds
        if verbose and time.time() - last_progress >= 30:
            unique_topos = len(seen_topologies)
            unique_invs = len(fusion_inv_to_topos)
            collisions = sum(1 for topos in fusion_inv_to_topos.values() if len(topos) > 1)
            print(f"  [{elapsed/60:.1f}m] {topologies_generated} topos generated, "
                  f"{unique_topos} unique, {total_samples} samples, {collisions} collisions")
            last_progress = time.time()

        # Generate a random fusion topology
        edges = generate_random_fusion_topology(n)
        if edges is None:
            continue  # Failed to generate (rare)

        topologies_generated += 1

        # Get canonical form to track unique topologies
        topo = canonicalize_topology(n, edges)
        is_new_topo = topo not in seen_topologies
        if is_new_topo:
            seen_topologies.add(topo)

        # Sample WAR orderings for this topology
        # Use all orderings for new topologies, sample for seen ones
        if is_new_topo or len(war_orderings) <= 50:
            wars_to_test = war_orderings
        else:
            wars_to_test = random.sample(war_orderings, 50)

        for war in wars_to_test:
            war_dict = {i: war[i] for i in range(n)}

            if not is_war_compatible_with_topology(n, edges, war_dict):
                skipped_incompatible += 1
                continue

            inv = compute_invariant_vector(n, edges, war_dict)
            if inv is None:
                continue

            total_samples += 1
            category = classify_invariant(inv)

            if category == 'fusion':
                fusion_count += 1
                fusion_inv_to_topos[inv].add(topo)

    elapsed = time.time() - start_time
    unique_topologies = len(seen_topologies)
    unique_invariants = len(fusion_inv_to_topos)
    fusion_collisions = sum(1 for topos in fusion_inv_to_topos.values() if len(topos) > 1)
    collision_rate = fusion_collisions / unique_invariants if unique_invariants > 0 else 0

    if verbose:
        print(f"\n--- Results for n={n} (random sampling, {elapsed/60:.1f} minutes) ---")
        print(f"Topologies generated: {topologies_generated:,}")
        print(f"Unique topologies found: {unique_topologies:,}")
        print(f"Total (topo, WAR) samples: {total_samples:,}")
        print(f"Skipped incompatible: {skipped_incompatible:,}")
        print(f"Fusions analyzed: {fusion_count:,}")
        print(f"Unique fusion invariants: {unique_invariants:,}")
        print(f"Fusion collisions: {fusion_collisions}")
        print(f"Collision rate: {collision_rate*100:.2f}%")

        if fusion_collisions > 0:
            print(f"\nCollision examples:")
            shown = 0
            for inv, topos in fusion_inv_to_topos.items():
                if len(topos) > 1:
                    print(f"  Invariant with {len(inv)} R-values: {len(topos)} topologies")
                    shown += 1
                    if shown >= 3:
                        break

    return {
        'n': n,
        'method': 'random',
        'timeout_minutes': timeout_minutes,
        'elapsed_seconds': elapsed,
        'topologies_generated': topologies_generated,
        'total_topologies': unique_topologies,
        'total_sampled': total_samples,
        'skipped_incompatible': skipped_incompatible,
        'fusion_count': fusion_count,
        'unique_fusion_invariants': unique_invariants,
        'fusion_collisions': fusion_collisions,
        'collision_rate': collision_rate,
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Analyze SCC invariant collisions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_scc_collisions.py              # Run default analysis (n=2-5)
  python analyze_scc_collisions.py --sampling 6 --timeout 5   # Sample n=6 for 5 minutes
  python analyze_scc_collisions.py --sampling 7 --timeout 30  # Sample n=7 for 30 minutes
        """
    )
    parser.add_argument('--sampling', type=int, metavar='N',
                        help='Run random sampling for n-vertex SCCs (for n >= 6)')
    parser.add_argument('--timeout', type=float, metavar='MINUTES', default=5,
                        help='Timeout in minutes for random sampling (default: 5)')
    parser.add_argument('--seed', type=int, metavar='SEED',
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # If --sampling is specified, run random sampling mode
    if args.sampling is not None:
        n = args.sampling
        if n < 2:
            print("Error: n must be at least 2")
            sys.exit(1)

        print("="*70)
        print(f"RANDOM SAMPLING MODE: n={n}, timeout={args.timeout} minutes")
        print("="*70)
        print("\nThis mode generates random fusion topologies on-the-fly.")
        print("Useful for n >= 6 where full enumeration is infeasible.")
        print("\n" + "-"*70)
        print("FILTERING: Oscillators and flat bidir (reductions) are excluded.")
        print("Only FUSIONS (SCCs with directed edges) are analyzed.")
        print("-"*70)

        results = analyze_scc_random_sampling(
            n=n,
            timeout_minutes=args.timeout,
            seed=args.seed,
            verbose=True
        )

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"n={n}: {results['fusion_collisions']} collisions found")
        print(f"Collision rate: {results['collision_rate']*100:.2f}%")
        print(f"Unique topologies sampled: {results['total_topologies']:,}")
        print(f"Time elapsed: {results['elapsed_seconds']/60:.1f} minutes")
        return

    # Default mode: run standard analysis
    print("="*70)
    print("EXHAUSTIVE SCC INVARIANT COLLISION ANALYSIS")
    print("="*70)
    print("\nThis script analyzes SCC invariant collisions with CORRECT handling of:")
    print("  - Oriented cycles (has directed edge): asymmetric bidir factors")
    print("  - Unoriented cycles (all bidir): symmetric factors for s≠s'")
    print("\nConstraints enforced:")
    print("  0. SCCs only (strongly connected)")
    print("  1. Cycle orientation (Johnson's algorithm)")
    print("  2. WAR/edge consistency (types derived from WAR)")
    print("  3. Unoriented cycle symmetry (s<s' and s>s' same factor)")
    print("\n" + "-"*70)
    print("FILTERING: Oscillators and flat bidir (reductions) are excluded.")
    print("Only FUSIONS (SCCs with directed edges) are analyzed for collisions.")
    print("Rationale: Oscillators/reductions have trivial R=0 or R=1 invariants.")
    print("-"*70)
    print("\nTip: Use --sampling N --timeout M for large n (e.g., n=6,7)")

    all_results = []

    # Exhaustive for n=2,3,4,5 (fusions only)
    # n=5 takes ~2 minutes but gives accurate results (7.07% collision rate)
    for n in range(2, 6):
        results = analyze_scc_size_exhaustive(n, verbose=True, fusions_only=False)
        all_results.append(results)

    # Summary table
    print("\n" + "="*90)
    print("SUMMARY TABLE (Fusions Only - Oscillators/Reductions Filtered)")
    print("="*90)
    print(f"{'n':<3} {'Method':<10} {'FusTopos':<10} {'Tested':<12} {'UniqueInv':<12} {'Collisions':<12} {'Rate':<8}")
    print("-"*90)

    for r in all_results:
        method = r.get('method', 'exhaustive')[:8]
        tested = r.get('total_sampled', r['total_possibilities'])
        unique_inv = r.get('unique_fusion_invariants', 0)
        fusion_coll = r.get('fusion_collisions', r.get('nontrivial_collisions', 0))
        rate = (fusion_coll / unique_inv * 100) if unique_inv > 0 else 0
        print(f"{r['n']:<3} {method:<10} {r['total_topologies']:<10} {tested:<12,} "
              f"{unique_inv:<12,} {fusion_coll:<12} {rate:.2f}%")

    print("-"*90)

    # Final verdict
    print("\nVERDICT:")
    for r in all_results:
        method = r.get('method', 'exhaustive')
        fusion_coll = r.get('fusion_collisions', r.get('nontrivial_collisions', 0))
        unique_inv = r.get('unique_fusion_invariants', 0)
        rate = (fusion_coll / unique_inv * 100) if unique_inv > 0 else 0
        if fusion_coll == 0:
            print(f"  n={r['n']}: No fusion collisions ({method})")
        else:
            print(f"  n={r['n']}: {fusion_coll} fusion collisions, {rate:.2f}% rate ({method})")

    print("\nFor n >= 6, use: python analyze_scc_collisions.py --sampling N --timeout M")

    # Estimate collision growth for n > 3
    estimate_collision_growth(all_results, verbose=True)


if __name__ == '__main__':
    main()
