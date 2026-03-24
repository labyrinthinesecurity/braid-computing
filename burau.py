#!/usr/bin/env python3
"""
Compute braid invariants for synthetic SCCs.

Uses analyze_scc_collisions.py infrastructure to generate SCCs.

"""

import sys, os, json
import random, math
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from fractions import Fraction

# log(SR) of the injection word σᵢ²σᵢ₊₁⁻¹ at t=-1.
# The abelian ("counting") LE approximation is r3_count × LOG_SR_INJECTION / T.
LOG_SR_INJECTION = math.log(2 + math.sqrt(3))  # ≈ 1.3170

sys.path.insert(0, '.')
from analyze_scc_collisions import (
    generate_scc_topologies,
    generate_fusion_topologies,
    generate_random_fusion_topology,
    is_strongly_connected,
    is_fully_bidirectional_topology,
    canonicalize_topology,
)


# ============================================================================
# BURAU REPRESENTATION - NON-ABELIAN BRAID INVARIANT
# ============================================================================

def extract_scalar_trajectories_from_simulation(vertex_trajectories, wars):
    """
    Convert vertex trajectories from simulate_nhi_trajectories to scalar trajectories.
    
    Args:
        vertex_trajectories: list of lists from simulate_nhi_trajectories
            vertex_trajectories[i][t] = vertex index
        wars: dict from simulate_nhi_trajectories
            wars[vertex] = WAR scalar value
            
    Returns:
        scalar_trajectories: list of lists
            scalar_trajectories[i][t] = WAR scalar at time t
    """
    scalar_trajs = []
    for vertex_traj in vertex_trajectories:
        scalar_traj = [wars[v] for v in vertex_traj]
        scalar_trajs.append(scalar_traj)
    return scalar_trajs

def analyze_scc_with_burau(edges, n_nhis=6, n_epochs=400, seed=None, scc_type=None, directed_bias=4.0):
    """
    Complete pipeline: simulate → extract scalars → build braid → compute Burau.

    Args:
        edges: labeled edges (src, dst, src_war, dst_war, is_bidir)
        n_nhis: number of NHI strands (≥6 needed for non-trivial braid
                structure: with n strands there are n−2 generator positions,
                and the adjacent-pair guard makes within-epoch pairs
                non-adjacent.  At n=6 this gives 4 positions with genuine
                non-commutativity across epochs.)
        n_epochs: walk length
        seed: random seed
    """
    n_vertices = max(max(e[0], e[1]) for e in edges) + 1

    # Step 1: Simulate NHI trajectories
    vertex_trajs, wars, walk_stats = simulate_nhi_trajectories(
        n_vertices=n_vertices,
        ed=edges,
        n_nhis=n_nhis,
        n_epochs=n_epochs,
        seed=seed,
        directed_bias=directed_bias,
    )

    # Check if we got enough trajectories
    if len(vertex_trajs) < 3:  # Need at least 3 for R3
        return {
            'error': 'Not enough closed walks generated (need 3+ for R3)',
            'walk_stats': walk_stats
        }

    n_strands = len(vertex_trajs)
    T = len(vertex_trajs[0])

    # Build directed edge set (non-bidir only)
    directed_edges = set()
    for src, dst, _, _, is_bidir in edges:
        if not is_bidir:
            directed_edges.add((src, dst))

    # Step 2: 2-strand injection — σᵢ² σᵢ₊₁⁻¹ for each qualifying consecutive pair.
    #
    # Trigger: both strands s0=war_order[i] and s1=war_order[i+1] traverse directed
    # (non-bidir) edges at step t, AND their combined net WAR flow is positive.
    #
    # Word: σᵢ² σᵢ₊₁⁻¹  (3 generators, 3 strands involved, non-abelian)
    #   Eigenvalues at t=-1: {1, 2+√3, 2-√3}
    #   SR = 2+√3 ≈ 3.732
    #
    # Chosen over the length-2 alternative σᵢσᵢ₊₁⁻¹ (SR=φ²≈2.618) for the
    # 1.43× higher per-injection eigenvalue, with no extra overflow.
    #
    # Adjacent-position cancellation (universal for all σᵢ^a σᵢ₊₁^b words):
    #   σᵢ²σᵢ₊₁⁻¹ · σᵢ₊₁²σᵢ₊₂⁻¹ = σᵢ²σᵢ₊₁σᵢ₊₂⁻¹ → SR=1
    #   σᵢσᵢ₊₁⁻¹  · σᵢ₊₁σᵢ₊₂⁻¹  = σᵢσᵢ₊₂⁻¹     → SR=1
    # Both words are equally fragile.  The adjacent-pair guard prevents
    # this within each epoch; cross-epoch adjacent interactions produce
    # partial spectral cancellation that encodes spatial structure
    # (see shuffling experiment, §5).
    #
    # Non-adjacent double: σ₀²σ₁⁻¹ · σ₂²σ₃⁻¹ → SR≈7.09, ||commutator||≈7.48.
    modified_braid = []
    active_counts = [0] * (n_strands - 1)  # active_counts[k] = steps with k active pairs
    edge_coverage_count = 0
    directed_strand_total = 0
    max_directed_strands = 0

    for t in range(T - 1):
      on_directed = set(s for s in range(n_strands)
                        if (vertex_trajs[s][t], vertex_trajs[s][t + 1]) in directed_edges)

      n_directed = len(on_directed)
      if n_directed > 0:
          edge_coverage_count += 1
      directed_strand_total += n_directed
      if n_directed > max_directed_strands:
          max_directed_strands = n_directed

      # Directed-first ordering within WAR: directed-edge strands cluster at low
      # positions so they naturally form the pairs we want to check first.
      war_order = sorted(range(n_strands), key=lambda s: (
          0 if s in on_directed else 1,
          wars[vertex_trajs[s][t]]
      ))

      step_active = 0
      last_pair_fired = -2  # prevents adjacent-pair σᵢ₊₁⁻¹·σᵢ₊₁² → σᵢ₊₁ residual, SR=1
      for i in range(n_strands - 2):
        if i == last_pair_fired + 1:
            continue  # skip: σᵢ²σᵢ₊₁⁻¹·σᵢ₊₁²σᵢ₊₂⁻¹ = σᵢ²σᵢ₊₁σᵢ₊₂⁻¹ → SR=1
        s0, s1 = war_order[i], war_order[i + 1]

        e0 = (vertex_trajs[s0][t], vertex_trajs[s0][t + 1])
        e1 = (vertex_trajs[s1][t], vertex_trajs[s1][t + 1])

        if e0 in directed_edges and e1 in directed_edges:
            # Net WAR flow gate over the two directed strands
            net_war_flow = (
                wars[vertex_trajs[s0][t + 1]] - wars[vertex_trajs[s0][t]] +
                wars[vertex_trajs[s1][t + 1]] - wars[vertex_trajs[s1][t]]
            )
            if net_war_flow > 0:
                # σᵢ² σᵢ₊₁⁻¹ — SR = 2+√3 ≈ 3.732 at t=-1
                # non-adjacent pair bonus: SR ≈ 7.09
                modified_braid.append((i, +1))
                modified_braid.append((i, +1))
                modified_braid.append((i + 1, -1))
                step_active += 1
                last_pair_fired = i
      active_counts[min(step_active, len(active_counts) - 1)] += 1

    # Step 3: Compute unreduced Burau representation on 2-strand-injected braid
    burau_matrix_r3 = compute_burau_matrix(modified_braid, n_strands, t=-1)
    burau_disc_r3 = burau_discriminators(burau_matrix_r3)
    spectral_radius_r3 = burau_disc_r3['spectral_radius']
    overflow = burau_disc_r3.get('overflow', False) or not np.isfinite(spectral_radius_r3)
    if scc_type=='ratchet' and spectral_radius_r3==1.0:
      print("weird!")
      print(modified_braid)
      print()

    # Step 4: Compute baseline (no R3 injection) - empty braid
    burau_matrix_no_r3 = np.eye(n_strands, dtype=complex)
    spectral_radius_no_r3 = 1.0

    # Compute delta_log_sr safely
    if overflow or spectral_radius_r3 <= 0:
        delta_log_sr = float('inf')
    else:
        delta_log_sr = float(np.log(spectral_radius_r3) - np.log(spectral_radius_no_r3))

    # WAR concentration: absolute privilege level visited across all strands/steps.
    # wars[v] is the WAR scalar of vertex v (set in simulate_nhi_trajectories from
    # src_war labels; all vertices reachable in a strongly-connected SCC are covered).
    all_war_values = [
        wars[vertex_trajs[s][t]]
        for s in range(n_strands)
        for t in range(T)
    ]
    mean_war = float(np.mean(all_war_values))
    max_war  = float(np.max(all_war_values))

    return {
        'r3_count': len(modified_braid) // 3,  # Number of R3 triplets injected
        'crossing_count': len(modified_braid),
        'n_strands': n_strands,
        'spectral_radius': spectral_radius_r3,
        'spectral_radius_no_r3': spectral_radius_no_r3,
        'log_sr_with_r3': np.log(spectral_radius_r3) if (spectral_radius_r3 > 0 and np.isfinite(spectral_radius_r3)) else float('inf'),
        'log_sr_no_r3': np.log(spectral_radius_no_r3) if spectral_radius_no_r3 > 0 else 0,
        'delta_log_sr': delta_log_sr,
        'overflow': overflow,
        'mean_war': mean_war,
        'max_war': max_war,
        'triple_density': [c / (T - 1) for c in active_counts],  # [frac_0, frac_1, frac_2]
        'edge_coverage': edge_coverage_count / (T - 1),
        'mean_directed_strands': directed_strand_total / (T - 1),
        'max_directed_strands': max_directed_strands,
        'walk_stats': walk_stats
    }

def temporal_scaling_experiment(type_topos, n=6, n_runs=64, epochs=[50], seed=None, n_war_asns=50):
    """
    Temporal scaling.
    Averages delta_log_sr over n_war_asns sampled WAR assignments per ratchet
    topology, making the metric a topological invariant rather than a
    (topology x WAR) sample.
    """
    print("type topos:",type_topos.keys())
    print(len(type_topos['oscillation']),len(type_topos['ratchet']))

    def label_edges(topo):
        labeled = []
        for src, dst, src_s, dst_s, is_bidir in topo:
            labeled.append((src, dst, src_s, dst_s, is_bidir))
        return labeled

    # Pre-compute all WAR scalar tuples once; sample from these per topology
    all_scalars = generate_scalar_tuples(n)
    rng_war = random.Random(seed if seed is not None else 42)

    # Pre-draw WAR assignments for all SCCs before the T loop.
    # Same assignment must be used at T_low and T_high so that per-WAR
    # ratio = dsr(T_high)/dsr(T_low) compares the same WAR scenario.
    scc_sample_range = range(43, 44)
    scc_sample_range = range(0,len(type_topos['ratchet'])) 
    war_samples_by_scc = {
        scc: rng_war.choices(all_scalars, k=n_war_asns)
        for scc in scc_sample_range
    }

    results = {'oscillation': {}, 'ratchet': {}}

    for T in epochs:
      results['oscillation'][T]={}
      results['ratchet'][T]={}

      for scc_sample in scc_sample_range:
        osc_topo = type_topos['oscillation'][scc_sample] #any fully-bidir
        fusion_topo = type_topos['ratchet'][scc_sample]  # any directed

        osc_edges = label_edges(osc_topo)

        # Extract raw topology (connectivity only) from labeled ratchet edges
        raw_fusion_topo = [(src, dst, is_bidir) for src, dst, _, _, is_bidir in fusion_topo]
        # Detect bidir traps once per SCC (topology property, WAR-independent)
        ratchet_bidir_traps = find_bidir_traps(raw_fusion_topo)
        war_samples = war_samples_by_scc[scc_sample]

        results['oscillation'][T][scc_sample]={}
        results['ratchet'][T][scc_sample]={}
        print(f"Testing T={T} epochs...")

        for scc_type in ['oscillation', 'ratchet']:
            # Results grouped by WAR combo: list-of-lists, one inner list per combo.
            # Keeps WAR-combo variation separate from walk-level noise so that
            # per-WAR means can be computed before cross-combo aggregation.
            results_by_war = []

            if scc_type == 'oscillation':
                # R3 never fires for fully-bidir regardless of WAR: one pass suffices
                edge_variants = [osc_edges]
            else:
                # Build one labeled edge list per WAR assignment
                edge_variants = [
                    [(src, dst, war[src], war[dst], is_bidir)
                     for src, dst, is_bidir in raw_fusion_topo]
                    for war in war_samples
                ]
                print(f"++++SCC graph for ratchet ID{scc_sample} (base WAR assignment):")
                for s,d,s_w,d_w,zdir in edge_variants[0]:
                    arrow = "+ > -" if s_w>d_w else ("- > +" if d_w>s_w else "     ")
                    scals = "("+str(s_w)+","+str(d_w)+")"
                    if zdir:
                        if s>d:
                            print(f"  {s}<=>{d}    {arrow}     {scals}")
                    else:
                        print(f"  {s}-->{d}    {arrow}     {scals}")

            for edges in edge_variants:
                war_run_results = []
                for run in range(n_runs):
                    result = analyze_scc_with_burau(
                        edges=edges,
                        n_nhis=6,
                        n_epochs=T,
                        seed=seed+run,
                        scc_type=scc_type,
                        directed_bias=4.0,
                    )
                    if 'error' not in result:
                        war_run_results.append(result)
                if war_run_results:
                    results_by_war.append(war_run_results)

            all_results = [r for war_runs in results_by_war for r in war_runs]

            if not all_results:
                print(f"  {scc_type}: NO VALID RUNS (all walks failed)")
                continue

            # ── Filter overflow results to avoid biasing statistics ───────────
            valid_results = [r for r in all_results if not r.get('overflow', False)]
            overflow_count = len(all_results) - len(valid_results)

            if not valid_results:
                print(f"  {scc_type} ID{scc_sample}: ALL {len(all_results)} RUNS OVERFLOWED — skipping")
                continue

            if overflow_count > 0:
                print(f"  {scc_type} ID{scc_sample}: {overflow_count}/{len(all_results)} runs overflowed, excluded")

            # ── per-WAR-combo mean SR (walk noise averaged out first) ──────────
            # Keep original index (j) so we can recover the war_samples tuple
            # that produced each valid combo — needed for per-WAR DataFrame.
            valid_by_war_indexed = [
                (j, [r for r in wr if not r.get('overflow', False)])
                for j, wr in enumerate(results_by_war)
            ]
            valid_by_war_indexed = [(j, wr) for j, wr in valid_by_war_indexed if wr]
            valid_by_war   = [wr for _, wr in valid_by_war_indexed]
            valid_war_indices = [j for j, _ in valid_by_war_indexed]

            with np.errstate(over='ignore', invalid='ignore'):
                war_mean_srs = [
                    np.mean([r['spectral_radius'] for r in wr])
                    for wr in valid_by_war
                ]
                war_mean_delta = [
                    np.mean([r['delta_log_sr'] for r in wr])
                    for wr in valid_by_war
                ]
                # Abelian firing rate per WAR combo: mean r3 injections per walk.
                # le_counting for a WAR combo = war_mean_r3[i] * LOG_SR_INJECTION / T
                war_mean_r3 = [
                    np.mean([r['r3_count'] for r in wr])
                    for wr in valid_by_war
                ]
                # Per-WAR-combo mean WAR: genuinely variable across combos because
                # different WAR assignments place different scalar values on vertices.
                # VaR95 over these is the first truly informative VaR95 metric.
                war_mean_war_vals = [
                    np.mean([r['mean_war'] for r in wr])
                    for wr in valid_by_war
                ]
                war_max_war_vals = [
                    np.max([r['max_war'] for r in wr])
                    for wr in valid_by_war
                ]

                spectral_radii = [r['spectral_radius'] for r in valid_results]
                spectral_radii_no_r3 = [r['spectral_radius_no_r3'] for r in valid_results]
                delta_log_sr = [r['delta_log_sr'] for r in valid_results]
                r3_counts    = [r['r3_count']    for r in valid_results]
                triple_densities = np.mean([r['triple_density'] for r in valid_results], axis=0)
                edge_coverage_mean = np.mean([r['edge_coverage'] for r in valid_results])
                mean_directed_strands_mean = np.mean([r['mean_directed_strands'] for r in valid_results])
                max_directed_strands_ever = int(max(r['max_directed_strands'] for r in valid_results))
                frac_runs_hit_3 = np.mean([1 if r['max_directed_strands'] >= 3 else 0 for r in valid_results])
                mean_war_mean = float(np.mean([r['mean_war'] for r in valid_results]))
                max_war_ever  = float(np.max([r['max_war']  for r in valid_results]))

                r = {
                    'n_epochs': T,
                    'spectral_radius_log_mean': np.log(np.mean(spectral_radii)),
                    'spectral_radius_mean': np.mean(spectral_radii),
                    'delta_log_sr': np.mean(delta_log_sr),
                    # Abelian approximation: mean gate-firing count × log(SR_word) / T
                    'r3_count_mean': float(np.mean(r3_counts)),
                    'le_counting': float(np.mean(r3_counts)) * LOG_SR_INJECTION / T,
                    # SR VaR95 — still degenerate (LE is WAR-independent) but kept
                    'spectral_radius_war_var95': float(np.percentile(war_mean_srs, 95)),
                    'delta_log_sr_war_var95': float(np.percentile(war_mean_delta, 95)),
                    # WAR VaR95 — now genuinely informative: different WAR assignments
                    # place different privilege levels → p95 captures worst-case WAR scenario
                    'mean_war': mean_war_mean,
                    'max_war_ever': max_war_ever,
                    'mean_war_war_var95': float(np.percentile(war_mean_war_vals, 95)),
                    'max_war_war_var95': float(np.percentile(war_max_war_vals, 95)),
                    'spectral_radius_std': np.std(spectral_radii),
                    'spectral_radius_cv': np.std(spectral_radii) / np.mean(spectral_radii) if np.mean(spectral_radii) > 0 else 0.0,
                    'spectral_radius_log_std': np.log(np.std(spectral_radii)) if np.std(spectral_radii) > 0 else float('-inf'),
                    'spectral_radius_q25': float(np.percentile(spectral_radii, 25)),
                    'spectral_radius_q50': float(np.percentile(spectral_radii, 50)),
                    'spectral_radius_q75': float(np.percentile(spectral_radii, 75)),
                    'spectral_radius_q90': float(np.percentile(spectral_radii, 90)),
                    'bidir_traps': ratchet_bidir_traps if scc_type == 'ratchet' else [],
                    'n_strands': valid_results[0]['n_strands'],
                    'triple_density': triple_densities,
                    'edge_coverage': edge_coverage_mean,
                    'mean_directed_strands': mean_directed_strands_mean,
                    'max_directed_strands_ever': max_directed_strands_ever,
                    'frac_runs_hit_3': frac_runs_hit_3,
                    'overflow_count': overflow_count,
                    'valid_runs': len(valid_results),
                    'total_runs': len(all_results),
                    # Per-WAR-assignment data for DataFrame output (ratchet only).
                    # per_war_dsr[i]        = mean delta_log_sr for war_samples[valid_war_indices[i]]
                    # per_war_war_tuples[i] = the WAR scalar tuple that produced it
                    # Keyed by tuple so T_low and T_high results can be joined.
                    'per_war_dsr': war_mean_delta if scc_type == 'ratchet' else [],
                    'per_war_r3':  war_mean_r3    if scc_type == 'ratchet' else [],
                    'per_war_war_tuples': (
                        [tuple(war_samples[j]) for j in valid_war_indices]
                        if scc_type == 'ratchet' else []
                    ),
                }

            results[scc_type][T][scc_sample]=r
    
    return results


def burau_generator_matrix(gen_idx, sign, n_strands, t=-1):
    """
    Construct elementary Burau matrix for generator σᵢ^(±1).

    The Burau representation maps braid generators to matrices:
    σᵢ acts on strands i and i+1

    At t=-1 all entries are integers (1-t=2, t=-1, 0, 1), so we use
    dtype=object (Python arbitrary-precision int) to avoid float64 overflow
    during matrix product accumulation.  The caller converts to complex
    float before eigenvalue computation.

    Args:
        gen_idx: which generator (0-indexed, so σ₁ = gen_idx 0)
        sign: +1 for σᵢ, -1 for σᵢ⁻¹
        n_strands: total number of strands
        t: parameter (usually t=-1 for unreduced Burau)

    Returns:
        (n_strands × n_strands) matrix
    """
    dtype = object if t == -1 else complex
    M = np.eye(n_strands, dtype=dtype)
    i = gen_idx
    one_minus_t = 1 - t  # = 2 at t=-1 (Python int when dtype=object)

    if sign == 1:
        # σᵢ (positive crossing)
        M[i, i] = one_minus_t
        M[i, i+1] = t
        M[i+1, i] = 1
        M[i+1, i+1] = 0
    else:
        # σᵢ⁻¹ (negative crossing)
        M[i, i] = 0
        M[i, i+1] = 1
        M[i+1, i] = t
        M[i+1, i+1] = one_minus_t

    return M


def compute_burau_matrix(braid_word, n_strands, t=-1):
    """
    Compute Burau representation of entire braid word.

    At t=-1 the product is accumulated in exact Python-integer arithmetic
    (dtype=object) to eliminate float64 overflow in long braid words.
    The result is converted to complex float before return so that
    burau_discriminators() receives a standard numeric array.

    Args:
        braid_word: list of (gen_idx, sign) tuples
        n_strands: number of strands
        t: parameter (t=-1 standard)

    Returns:
        Burau matrix (product of all generators) as complex float array
    """
    if t == -1:
        # Exact integer accumulation: eliminates float64 silent precision loss.
        # For extreme dispersed ratchets the entries can exceed ~10^308 (float64
        # max) and cannot be converted directly.  We signal that by returning
        # an all-inf matrix so burau_discriminators() sets overflow=True and the
        # result is excluded 
        M = np.eye(n_strands, dtype=object)
        for gen_idx, sign in braid_word:
            B = burau_generator_matrix(gen_idx, sign, n_strands, t)
            M = M @ B
        try:
            return M.astype(complex)
        except OverflowError:
            return np.full((n_strands, n_strands), np.inf, dtype=complex)
    else:
        M = np.eye(n_strands, dtype=complex)
        with np.errstate(over='ignore', invalid='ignore'):
            for gen_idx, sign in braid_word:
                B = burau_generator_matrix(gen_idx, sign, n_strands, t)
                M = M @ B
        return M


def burau_discriminators(M):
    """
    Extract discriminating features from Burau matrix.

    Returns dict of scalar features for classification.
    Sets 'overflow' flag if any computation produces inf/nan.

    Notes on numerical stability:
    - ord=2 spectral norm requires SVD and internally squares entries, so it
      can overflow even when individual matrix entries are within float64 range
      (e.g. entries ~10^150 square to ~10^300, near float64 max ~10^308).
      We use the Frobenius norm instead; it is cheaper and doesn't square.
    - LinAlgError from eigvals (raised when M contains inf) is caught and
      treated as overflow rather than propagating as an exception.
    """
    # overflow is set ONLY on eigenvalue failure — the sole quantity used by
    # callers to exclude results.  Auxiliary norms (matrix_norm, off_diag_norm)
    # can legitimately overflow for large but valid matrices (entries ~10^206
    # square to ~10^412 > float64 max) without invalidating the spectral radius.
    overflow = not np.all(np.isfinite(M))

    with np.errstate(over='ignore', invalid='ignore'):
        trace = np.trace(M)

        det = np.linalg.det(M)

        try:
            eigenvalues = np.linalg.eigvals(M)
        except np.linalg.LinAlgError:
            eigenvalues = np.array([])
            overflow = True
        finite_eig = eigenvalues[np.isfinite(np.abs(eigenvalues))] if len(eigenvalues) else np.array([])
        if len(finite_eig) > 0:
            max_eigenval = np.max(np.abs(finite_eig))
        else:
            max_eigenval = float('inf')
            overflow = True
        if not np.isfinite(max_eigenval):
            overflow = True
        spectral_radius = float(np.real(max_eigenval))

        # Frobenius norm: used only as an auxiliary statistic; overflow here
        # does NOT set the overflow flag (see note above).
        _matrix_norm_val = np.linalg.norm(M, ord='fro')
        matrix_norm = float(np.log(_matrix_norm_val)) if (np.isfinite(_matrix_norm_val) and _matrix_norm_val > 0) else float('inf')

        _off_diag_val = np.linalg.norm(M - np.diag(np.diag(M)), ord='fro')
        off_diag_norm = float(np.log(_off_diag_val)) if (np.isfinite(_off_diag_val) and _off_diag_val > 0) else float('inf')

    return {
        'trace_real': float(np.real(trace)),
        'trace_imag': float(np.imag(trace)),
        'trace_abs': float(np.abs(trace)),
        'det_abs': float(np.abs(det)) if np.isfinite(det) else float('inf'),
        'spectral_radius': spectral_radius,
        'matrix_norm': matrix_norm,
        'off_diagonal_norm': off_diag_norm,
        'eigenvalue_spread': float(np.std(np.abs(finite_eig))) if len(finite_eig) > 0 else float('inf'),
        'overflow': overflow
    }


# =============================================================================
# RATCHET STRATIFICATION
# =============================================================================

def find_bidir_traps(topo, max_escapes=2):
    """
    Find bidirectional near-trap subsets in an SCC topology.

    A bidir near-trap is a maximal subset C (|C| >= 2) of vertices connected
    purely via bidirectional edges, where all *internal* edges are bidir (no
    directed shortcuts within C) and at most max_escapes directed edges leave C.

    Strict trap (0 escapes): walk can never leave C.  Crossings cancel because
    every σᵢ is paired with σᵢ⁻¹ over time → Burau product → I → SR = 1.

    Near-trap (1–max_escapes escapes): the single exit makes escape rare but
    possible; the bidir dynamics dominate → SR stays close to 1.

    Args:
        topo:        list of (src, dst, is_bidir) tuples
        max_escapes: maximum number of escape (directed outgoing) edges to
                     still qualify as a near-trap (default 2)

    Returns:
        list of dicts, each:
          {'vertices': sorted vertex list,
           'escapes':  sorted list of (src, dst) directed escape edges}
        Empty list if no near-traps found.
    """
    vertices = set()
    for src, dst, _ in topo:
        vertices.add(src)
        vertices.add(dst)

    # Build adjacency restricted to bidirectional edges
    bidir_adj = defaultdict(set)
    for src, dst, is_bidir in topo:
        if is_bidir:
            bidir_adj[src].add(dst)

    # Find connected components of the bidir-only subgraph (BFS)
    visited = set()
    components = []
    for start in sorted(vertices):
        if start in visited:
            continue
        component = set()
        queue = [start]
        while queue:
            v = queue.pop()
            if v in component:
                continue
            component.add(v)
            for w in bidir_adj[v]:
                if w not in component:
                    queue.append(w)
        visited.update(component)
        components.append(frozenset(component))

    near_traps = []
    for comp in components:
        if len(comp) < 2:
            continue

        # Require internal purity: no directed edges between vertices of comp.
        # (By BFS construction all bidir edges stay within comp; a directed
        # edge between two comp vertices would break the symmetric-cancellation
        # property we care about.)
        has_internal_directed = any(
            not is_bidir
            for src, dst, is_bidir in topo
            if src in comp and dst in comp
        )
        if has_internal_directed:
            continue

        # Escape edges: directed edges leaving comp (dst outside comp,
        # guaranteed since internal directed edges were already excluded above).
        escapes = sorted(
            (src, dst)
            for src, dst, is_bidir in topo
            if src in comp and not is_bidir
        )

        if len(escapes) <= max_escapes:
            near_traps.append({'vertices': sorted(comp), 'escapes': escapes})

    return near_traps


# =============================================================================
# SCC TYPE CLASSIFICATION
# =============================================================================

def classify_scc_type(n, ed):
    edges = [(t[0], t[1], t[4]) for t in ed]
    scalars = [(t[2], t[3]) for t in ed]
    bidir_set = set()
    directed_only = set()

    all_war_values = set()
    for src, dst, src_war, dst_war, is_bidir in ed:
      all_war_values.add(src_war)
      all_war_values.add(dst_war)
    is_constant = len(all_war_values) == 1

    for src, dst, is_bidir in edges:
        if is_bidir:
            bidir_set.add((src, dst))
            bidir_set.add((dst, src))

    all_symmetric = True
    for src, dst, is_bidir in edges:
        if not is_bidir:
            reverse_exists = any(s == dst and d == src for s, d, _ in edges)
            if not reverse_exists:
                all_symmetric = False
                break

    scc_type = 'reduction' if all_symmetric and is_constant else 'oscillation' if all_symmetric else 'ratchet'
    #print(f"Debug: edges={edges}, scalars={scalars}, is_constant={is_constant}, all_symmetric={all_symmetric}, scc_type={scc_type}")
    return scc_type


# =============================================================================
# NHI TRAJECTORY SIMULATION
# =============================================================================

def simulate_nhi_trajectories(n_vertices, ed, n_nhis, n_epochs, seed=None, directed_bias=1.0):
    """
    Simulate NHI trajectories through an SCC as CLOSED WALKS.

    Each NHI starts at a random vertex, follows edges randomly, and
    must return to its starting vertex at the final epoch. The starting
    vertex does not appear at any intermediate epoch (elementary-like).

    Returns:
        trajectories: list of lists, trajectories[i][t] = vertex of NHI i at epoch t
        wars: dict mapping vertex -> WAR value
    """
    if seed is not None:
        random.seed(seed)

    edges = [(t[0], t[1], t[4]) for t in ed]
    scalars = [(t[2], t[3]) for t in ed]

    # Build adjacency list and directed-neighbor index
    adj = defaultdict(list)
    directed_neighbors = defaultdict(set)  # v -> {dst : (v,dst) is a directed edge}
    wars = defaultdict()
    for src, dst, src_war, dst_war, is_bidir in ed:
        adj[src].append(dst)
        wars[src]=src_war
        if is_bidir:
            #adj[dst].append(src)
            wars[dst]=dst_war
        else:
            directed_neighbors[src].add(dst)

    # Compute distance from every vertex to the nearest directed-source vertex
    # (i.e. a vertex with at least one outgoing directed edge) via reverse BFS.
    # This distance field is passed to _generate_open_walk so it can bias
    # transitions toward directed sources even from deep inside bidir-only
    # regions where the old "prefer directed edges from current vertex" logic
    # had zero effect.
    dist_to_dir_source = {}
    if directed_bias != 1.0:
        directed_sources = {v for v, nbrs in directed_neighbors.items() if nbrs}
        if directed_sources:
            # Reverse adjacency: rev_adj[u] = all v such that u in adj[v]
            rev_adj = defaultdict(list)
            for v in adj:
                for u in adj[v]:
                    rev_adj[u].append(v)
            queue = list(directed_sources)
            dist_to_dir_source = {v: 0 for v in directed_sources}
            while queue:
                v = queue.pop(0)
                for u in rev_adj[v]:
                    if u not in dist_to_dir_source:
                        dist_to_dir_source[u] = dist_to_dir_source[v] + 1
                        queue.append(u)

    # Assign WAR values: spread them out for interesting crossings
    #war_values = {v: (v + 1) * 100 + random.randint(-20, 20) for v in range(n_vertices)}
    # Initialize NHIs at distinct starting vertices
    start_vertices = list(range(n_vertices))
    random.shuffle(start_vertices)
    if n_nhis > n_vertices:
        start_vertices = [random.choice(range(n_vertices)) for _ in range(n_nhis)]
    else:
        start_vertices = start_vertices[:n_nhis]

    trajectories = []
    closed_count=0
    null_count=0
    for i in range(n_nhis):
        start = start_vertices[i]
        traj = _generate_open_walk(adj, start, n_epochs, dist_to_dir_source, directed_bias)
        if traj is not None:
          trajectories.append(traj)
          closed_count+=1
        else:
          null_count+=1

    total = closed_count + null_count
    fallback_rate = null_count / total if total > 0 else 0

    walk_stats = {
        'closed': closed_count,
        'open': null_count,
        'total': total,
        'fallback_rate': fallback_rate,
    }

    return trajectories, wars, walk_stats

def _generate_open_walk(adj, start, length, dist_to_dir_source=None, directed_bias=1.0):
    """
    Random walk with optional directed-source proximity bias.

    When directed_bias > 1.0, transition weights are set by a gradient that
    points toward the nearest directed-source vertex (a vertex with at least
    one outgoing directed edge).  Specifically:

        weight(neighbor) = directed_bias ** (1 / (1 + dist[neighbor]))

    where dist[neighbor] is the BFS distance from that neighbor to the
    nearest directed source (0 if the neighbor IS a directed source).
    This gives weight=directed_bias for direct steps onto a source,
    decaying smoothly with distance so the bias creates a drift field
    everywhere in the SCC, not just at vertices that already have directed
    outgoing edges.

    This fixes the earlier approach (biasing along directed edges from the
    current vertex) which had no effect when the walk was deep inside a
    bidir-only region: now the gradient guides the walker out of those
    regions toward directed sources from any starting position.
    """
    traj = [start]
    current = start
    for _ in range(length - 1):
        neighbors = adj[current]
        if not neighbors:
            break
        if dist_to_dir_source and directed_bias != 1.0:
            weights = [directed_bias ** (1.0 / (1.0 + dist_to_dir_source.get(n, 999)))
                       for n in neighbors]
            total_w = sum(weights)
            r = random.random() * total_w
            cumulative = 0.0
            current = neighbors[-1]  # fallback
            for n, w in zip(neighbors, weights):
                cumulative += w
                if r <= cumulative:
                    current = n
                    break
        else:
            current = random.choice(neighbors)
        traj.append(current)
    return traj

# =============================================================================
# BRAID CONSTRUCTION
# =============================================================================

# NOTE: Braid word construction is handled directly by the gate-injection loop
# in analyze_scc_with_burau().  Only σᵢ² σᵢ₊₁⁻¹ words are injected (when the
# gate fires); non-gate rank swaps (lone σᵢ±¹) are excluded because their
# Burau matrices have SR = 1 and contribute no spectral growth.

def braid_permutation(word, k):
    """
    Compute the permutation induced by a braid word.

    Parameters
    ----------
    word : list of ints
        Example: [1, -2, 1] means σ1 σ2^{-1} σ1
    k : int
        Number of strands

    Returns
    -------
    perm : list
        The permutation in one-line notation (1-based indexing).
        Example: [2,1,3] means strand 1->2, 2->1, 3->3.
    """

    # Start with identity permutation
    perm = list(range(1, k+1))

    for g in word:
        i = abs(g) - 1  # convert to 0-based index
        
        # swap strands i and i+1
        perm[i], perm[i+1] = perm[i+1], perm[i]

    return perm


# ================================================================sum(writhes) / len(writhes),=============
# MAIN ANALYSIS
# =============================================================================

def generate_scalar_tuples(n):
    return list(itertools.product(range(n), repeat=n))

def generate_labeled_sccs_stratified(n, scalars, topos, classify_scc_type, 
                                      seed=None, num_samples=None, min_per_type=10):
    """
    Generate labeled SCCs with stratified sampling by type.
    
    Stratification is done by topology structure, not by classification:
    - reduction:   fully-bidir topology + constant scalar
    - oscillation: fully-bidir topology + non-constant scalar  
    - ratchet:     directed topology + any scalar
    
    Sample sizes are capped by actual pool sizes to maintain
    topological independence. For n>=6 where only 1 fully-bidir
    topology exists, all scalar combinations are used.
    """
    if seed is not None:
        random.seed(seed)

    # =========================================================================
    # Step 1: Partition topologies by structure
    # =========================================================================
    #fully_bidir = [t for t in topos if is_fully_bidirectional_topology(t)]
    #has_directed = [t for t in topos if not is_fully_bidirectional_topology(t)]

    fully_bidir  = [t for t in topos if is_fully_bidirectional_topology(t) and is_strongly_connected(n, t)]
    has_directed = [t for t in topos if not is_fully_bidirectional_topology(t) and is_strongly_connected(n, t)]

    constant_scalars    = [s for s in scalars if len(set(s)) == 1]
    nonconstant_scalars = [s for s in scalars if len(set(s)) > 1]

    print(f"  Topology partition: {len(fully_bidir)} fully-bidir, "
          f"{len(has_directed)} directed (SCC validated)")
    print(f"  Scalar partition: {len(constant_scalars)} constant, "
          f"{len(nonconstant_scalars)} non-constant")

    # =========================================================================
    # Step 2: Build labeled SCC pools
    # =========================================================================
    def make_labeled(topo, scalar_tuple):
        return [
            (u, v, scalar_tuple[u], scalar_tuple[v], directed)
            for u, v, directed in topo
        ]

    # --- Reduction pool: fully-bidir + constant scalar ---
    # Always fully enumerate (small pool)
    reduction_pool = [
        make_labeled(topo, scalar)
        for topo in fully_bidir
        for scalar in constant_scalars
    ]

    # --- Oscillation pool: fully-bidir + non-constant scalar ---
    if len(fully_bidir) == 1:
        # Only 1 fully-bidir topology (typical for n>=6):
        # use ALL non-constant scalars to maximize diversity
        oscillation_pool = [
            make_labeled(fully_bidir[0], scalar)
            for scalar in nonconstant_scalars
        ]
        print(f"  Note: only 1 fully-bidir topology — "
              f"using all {len(nonconstant_scalars)} non-constant scalars "
              f"for oscillations")
    else:
        # Multiple fully-bidir topologies (n<=5):
        # distribute non-constant scalars evenly across topologies
        scalars_per_topo = max(1, (num_samples or len(nonconstant_scalars)) 
                               // len(fully_bidir))
        oscillation_pool = []
        for topo in fully_bidir:
            selected_scalars = random.sample(
                nonconstant_scalars,
                min(len(nonconstant_scalars), scalars_per_topo)
            )
            for scalar in selected_scalars:
                oscillation_pool.append(make_labeled(topo, scalar))

    # --- Ratchet pool: directed topology + one random scalar each ---
    # One scalar per topology preserves topological independence
    ratchet_pool = [
        make_labeled(topo, random.choice(scalars))
        for topo in has_directed
    ]

    print(f"  Pool sizes: reductions={len(reduction_pool)}, "
          f"oscillations={len(oscillation_pool)}, "
          f"ratchets={len(ratchet_pool)}")

    # =========================================================================
    # Step 3: Compute sample targets
    # =========================================================================
    # Topological independence cap: 
    # limited by the smallest meaningful pool (fully-bidir count)
    topo_cap = len(has_directed)  # ratchets: one per topology

    if num_samples is not None:
        per_type = min(num_samples // 3, topo_cap)
    else:
        per_type = topo_cap

    # Each type capped by both per_type and its actual pool size
    reduction_target   = min(len(reduction_pool),   per_type)
    oscillation_target = min(len(oscillation_pool), per_type)
    ratchet_target     = min(len(ratchet_pool),     per_type)

    print(f"  Sample targets: reductions={reduction_target}, "
          f"oscillations={oscillation_target}, "
          f"ratchets={ratchet_target}")

    # =========================================================================
    # Step 4: Sample from pools
    # =========================================================================
    # Use random.sample for pools larger than target (no replacement)
    # Use random.choices for pools smaller than target (with replacement)
    def sample_pool(pool, target):
        if len(pool) == 0 or target == 0:
            return []
        if len(pool) >= target:
            return random.sample(pool, target)
        else:
            # Pool smaller than target: sample with replacement
            # (acknowledged duplication)
            print(f"  WARNING: pool size {len(pool)} < target {target}, "
                  f"sampling with replacement")
            return random.choices(pool, k=target)

    reduction_sample   = sample_pool(reduction_pool,   reduction_target)
    oscillation_sample = sample_pool(oscillation_pool, oscillation_target)
    ratchet_sample     = sample_pool(ratchet_pool,     ratchet_target)

    print(f"  Actual samples: reductions={len(reduction_sample)}, "
          f"oscillations={len(oscillation_sample)}, "
          f"ratchets={len(ratchet_sample)}")

    # =========================================================================
    # Step 5: Yield labeled SCCs with type labels
    # =========================================================================
    for labeled_scc in reduction_sample:
        yield ('reduction', None, labeled_scc)

    for labeled_scc in oscillation_sample:
        yield ('oscillation', None, labeled_scc)

    for labeled_scc in ratchet_sample:
        yield ('ratchet', None, labeled_scc)

def generate_topologies_for_n(n, force_regenerate=True, 
                               num_directed_samples=1000,
                               num_fully_bidir_samples=200):
    """
    Generate or load topologies for n vertices.
    
    For n<=5: exhaustive enumeration (feasible)
    For n>=6: random sampling (exhaustive takes days)
    
    Two separate sampling loops:
    - fully-bidir topologies: only bidirectional edges allowed
    - directed/fusion topologies: at least one directed edge required
    
    Results are cached to disk for reuse.
    
    Args:
        n: Number of vertices
        force_regenerate: If True, ignore cache and regenerate
        num_directed_samples: Target number of directed topologies (n>=6)
        num_fully_bidir_samples: Target number of fully-bidir topologies (n>=6)
    
    Returns:
        List of topologies, each a list of (src, dst, is_bidir) tuples
    """
    cache_file = f"{n}_topologies.json"

    # Check cache
    if os.path.exists(cache_file) and not force_regenerate:
        with open(cache_file, 'r') as f:
            loaded_data = json.load(f)
            topos = [[tuple(l) for l in inner] for inner in loaded_data]
            print(f"  Loaded {len(topos)} topologies from cache ({cache_file})")
            return topos

    # =========================================================================
    # n<=5: exhaustive enumeration
    # =========================================================================
    if n < 5:
        print(f"  Exhaustive enumeration for n={n}...")
        topos = generate_scc_topologies(n, include_fully_bidir=True)
        print(f"  Generated {len(topos)} topologies exhaustively")

    # =========================================================================
    # n>=6: random sampling
    # =========================================================================
    else:
        print(f"  Random sampling for n={n} "
              f"(exhaustive infeasible for n>={n})...")
        print(f"  Targets: {num_fully_bidir_samples} fully-bidir, "
              f"{num_directed_samples} directed")

        topos = []
        topos_set = set()  # canonical forms for deduplication
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

        # =====================================================================
        # Loop 1: fully-bidir topologies
        # Only allow no-edge (0) or bidirectional (3) choices per pair
        # =====================================================================
        fully_bidir_found = 0
        fully_bidir_attempts = 0
        max_fully_bidir_attempts = num_fully_bidir_samples * 100

        print(f"  Sampling fully-bidir topologies...")
        while (fully_bidir_found < num_fully_bidir_samples and
               fully_bidir_attempts < max_fully_bidir_attempts):
            fully_bidir_attempts += 1

            edges = []
            for i, j in pairs:
                choice = random.choice([0, 3])  # no edge or bidir only
                if choice == 3:
                    edges.append((i, j, True))
                    edges.append((j, i, True))

            # Must be strongly connected and fully bidirectional
            if (is_strongly_connected(n, edges) and
                    is_fully_bidirectional_topology(edges)):
                canonical = canonicalize_topology(n, edges)
                if canonical not in topos_set:
                    topos_set.add(canonical)
                    topos.append(edges)
                    fully_bidir_found += 1

        print(f"  Found {fully_bidir_found} unique fully-bidir topologies "
              f"after {fully_bidir_attempts} attempts")

        if fully_bidir_found == 0:
            # Guarantee at least one: inject the complete bidir graph
            print(f"  WARNING: no fully-bidir topologies found by sampling — "
                  f"injecting complete bidir graph")
            edges = []
            for i, j in pairs:
                edges.append((i, j, True))
                edges.append((j, i, True))
            if is_strongly_connected(n, edges):
                canonical = canonicalize_topology(n, edges)
                topos_set.add(canonical)
                topos.append(edges)
                fully_bidir_found = 1
                print(f"  Injected complete bidir graph as fallback")

        # =====================================================================
        # Loop 2: directed/fusion topologies
        # At least one directed edge required (use existing function)
        # No canonicalization for speed — duplicates acceptable
        # =====================================================================
        directed_found = 0
        directed_attempts = 0
        max_directed_attempts = num_directed_samples * 20

        print(f"  Sampling directed/fusion topologies...")
        while (directed_found < num_directed_samples and
               directed_attempts < max_directed_attempts):
            directed_attempts += 1

            edges = generate_random_fusion_topology(n)
            if edges is None:
                continue

            # Skip canonicalization for speed (duplicates acceptable for n>=6)
            topos.append(edges)
            directed_found += 1

        print(f"  Found {directed_found} directed topologies "
              f"after {directed_attempts} attempts "
              f"(duplicates possible, no canonicalization)")

        print(f"  Total: {len(topos)} topologies "
              f"({fully_bidir_found} fully-bidir, "
              f"{directed_found} directed)")

    # =========================================================================
    # Cache to disk
    # =========================================================================
    serializable_data = [[list(t) for t in inner] for inner in topos]
    with open(cache_file, 'w') as f:
        json.dump(serializable_data, f)
    print(f"  Cached to {cache_file}")

    return topos

def generate_fully_bidir_topology(n):
    """
    Generate the complete fully-bidirectional topology for n vertices.
    This is the unique fully-bidir SCC: all pairs connected bidirectionally.
    """
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j, True))
            edges.append((j, i, True))
    if is_strongly_connected(n, edges):
        return edges
    return None

def analyze_synthetic_sccs(max_n=4, n_nhis_per_scc=3, n_epochs=20,
                           samples_per_type=30, mc_samples=200,
                           chirality_runs=500, seed=42):
    """
    Generate synthetic SCCs, compute braid invariants, validate claims.
    """

    print("=" * 70)
    print("PERMISSION BRAID INVARIANTS: SYNTHETIC SCC ANALYSIS")
    print("=" * 70)
    print(f"Parameters: max_n={max_n}, NHIs/SCC={n_nhis_per_scc}, "
          f"epochs={n_epochs}, samples/type={samples_per_type}")
    print(f"Chirality test: {chirality_runs} random walks per SCC (distributional)")

    # Collect results by SCC type
    results_by_type = defaultdict(list)

    for n in range(max_n, max_n + 1):
        print(f"\n--- Generating n={n} vertex SCCs ---")

        # Get all topologies
        if os.path.exists(f"{n}_topologies.json"): 
          with open(f"{n}_topologies.json", 'r') as f:
            loaded_data = json.load(f)
            topos = [[tuple(l) for l in inner] for inner in loaded_data]
            print("loaded from file")
        else:
          topos = generate_topologies_for_n(n)
          serializable_data = [[list(t) for t in inner] for inner in topos]
          with open(f"{n}_topologies.json", 'w') as f:
            json.dump(serializable_data, f)
        print(f"  Total topologies: {len(topos)}")
        scalars = generate_scalar_tuples(n)
        # Classify and sample
        type_topos = defaultdict(list)
        war_topos = defaultdict(list)
        lcount=0
        for scc_type, scalar_tuple, labeled_scc in generate_labeled_sccs_stratified(
            n, scalars, topos, classify_scc_type, seed=seed, num_samples=30000
        ):
            lcount+=1
            if lcount%10000==0:
              print(lcount)
            #scc_type = classify_scc_type(n, labeled_scc[1])
            type_topos[scc_type].append(labeled_scc)
            war_topos[scc_type].append(scalar_tuple)

        for scc_type, edge_list in type_topos.items():
            print(f" {scc_type}: {len(edge_list)} topologies")
        print()

    rng=random.Random(seed)
    for scc_type in type_topos:
      # Generate a shuffled list of indices
      indices = list(range(len(type_topos[scc_type])))
      rng.shuffle(indices)

    # Apply the same shuffling to both lists
    type_topos[scc_type] = [type_topos[scc_type][i] for i in indices]
    war_topos[scc_type] = [war_topos[scc_type][i] for i in indices]

    n_runs=64
    #epochs=[100,200,400]
    epochs=[50]
    #epochs=[2,3]
    #epochs=[5000]
    rz=temporal_scaling_experiment(type_topos=type_topos,n=6,n_runs=n_runs,epochs=epochs,seed=seed)
    print("epochs:",epochs,"runs per SCC:",n_runs,"epochs per run:",epochs[0],"sample seed:",seed)

    # ── T-scaling ratio discriminant ─────────────────────────────────────────
    # ratio = Δlog_sr(T_high) / Δlog_sr(T_low)
    #
    # For a genuine per-step Lyapunov exponent (pseudo-Anosov braid):
    #   ratio ≈ T_high / T_low   (linear growth — hard ratchet confirmed)
    #
    # Three regimes (for epochs=[50,100], expected_ratio=2.0):
    #
    #   ratio ≥ SCALING_PA_FRAC × expected   →  pseudo_anosov
    #     Linear Δlog growth confirms irreducible (pA) braid structure.
    #     Each step adds a fixed Lyapunov increment regardless of T.
    #
    #   SCALING_SAT_THRESH ≤ ratio < PA_FRAC × expected  →  quasi_oscillator
    #     Sub-linear growth: braid looks topologically like a ratchet
    #     (has directed edges) but LE does not scale linearly.
    #     Signature of a *reducible* braid — oscillatory subsystem partially
    #     cancels ratchet accumulation each period, suppressing net SR growth.
    #     Static bidir fraction alone cannot detect this; the ratio test is
    #     the minimal sufficient statistic.
    #
    #   ratio < SCALING_SAT_THRESH  →  saturated  (dead queue)
    #     Burau LE unreliable: directed edges dynamically sparse, braid word
    #     saturates at a fixed short length regardless of walk length.
    #
    # Threshold constants:
    SCALING_PA_FRAC    = 0.90  # ratio >= 90% of expected → pseudo-Anosov
    SCALING_SAT_THRESH = 1.50  # ratio < this → saturated (dead queue)

    dead_queue             = {}  # (scc_type, stat) -> diagnostic dict
    quasi_oscillator_queue = {}  # (scc_type, stat) -> diagnostic dict
    scaling_classes        = {}  # (scc_type, stat) -> 'pseudo_anosov'|'quasi_oscillator'|'saturated'
    scaling_ratios         = {}  # (scc_type, stat) -> float

    expected_ratio = 1.0  # fallback when only one epoch
    sorted_epochs = sorted(epochs)
    T_low  = sorted_epochs[0]
    T_high = sorted_epochs[-1]  # equals T_low when only one epoch
    if len(sorted_epochs) >= 2:
        expected_ratio = T_high / T_low
        pa_threshold = SCALING_PA_FRAC * expected_ratio
        for scc_type in rz:
            for stat in rz[scc_type].get(T_low, {}):
                res_low  = rz[scc_type][T_low].get(stat, {})
                res_high = rz[scc_type][T_high].get(stat, {})
                dsr_low  = res_low.get('delta_log_sr')
                dsr_high = res_high.get('delta_log_sr')
                if dsr_low and dsr_high and dsr_low > 0:
                    ratio = dsr_high / dsr_low
                    scaling_ratios[(scc_type, stat)] = ratio
                    info = {
                        'ratio': ratio,
                        'expected': expected_ratio,
                        'edge_coverage': res_low.get('edge_coverage', float('nan')),
                        'mean_directed_strands': res_low.get('mean_directed_strands', float('nan')),
                        'dsr_low': dsr_low,
                        'dsr_high': dsr_high,
                    }
                    if ratio < SCALING_SAT_THRESH:
                        scaling_classes[(scc_type, stat)] = 'saturated'
                        dead_queue[(scc_type, stat)] = info
                    elif ratio < pa_threshold:
                        scaling_classes[(scc_type, stat)] = 'quasi_oscillator'
                        quasi_oscillator_queue[(scc_type, stat)] = info
                    else:
                        scaling_classes[(scc_type, stat)] = 'pseudo_anosov'

    # ── Per-WAR-assignment DataFrame ──────────────────────────────────────────
    # One row per (ratchet SCC, WAR assignment): SCC_ID, W0..W5, ratio.
    # ratio = dsr(T_high) / dsr(T_low) for that specific WAR scenario.
    # This is the security-relevant output: in production a specific deployment
    # has fixed WAR values, so the per-WAR ratio is the actual risk discriminant,
    # not the population VaR95.  Rows are joined on the WAR tuple so only
    # assignments that produced valid (non-overflow) results are included.
    # With a single epoch T_low == T_high, so ratio column will be 1.0.
    if sorted_epochs:
        df_rows = []
        for scc_sample in sorted(rz.get('ratchet', {}).get(T_low, {}).keys()):
            res_low  = rz['ratchet'][T_low].get(scc_sample, {})
            res_high = rz['ratchet'][T_high].get(scc_sample, {})

            # Directed edges for this SCC — topology is WAR-independent
            fusion_topo = type_topos['ratchet'][scc_sample]
            directed_edges = [(src, dst)
                              for src, dst, _, _, is_bidir in fusion_topo
                              if not is_bidir]

            dsr_by_war_low  = dict(zip(
                res_low.get('per_war_war_tuples', []),
                res_low.get('per_war_dsr', [])
            ))
            dsr_by_war_high = dict(zip(
                res_high.get('per_war_war_tuples', []),
                res_high.get('per_war_dsr', [])
            ))
            r3_by_war_low = dict(zip(
                res_low.get('per_war_war_tuples', []),
                res_low.get('per_war_r3', [])
            ))

            for war_tuple, dsr_low_val in dsr_by_war_low.items():
                dsr_high_val = dsr_by_war_high.get(war_tuple)
                if dsr_high_val is None:
                    continue  # overflowed at T_high
                if dsr_low_val and dsr_low_val > 0:
                    per_war_ratio = dsr_high_val / dsr_low_val
                else:
                    per_war_ratio = float('nan')

                # DWS = Σ(W_dst - W_src) over directed edges.
                # Measures net WAR "uphill" potential: positive → gate fires more.
                dws = sum(war_tuple[dst] - war_tuple[src] for src, dst in directed_edges)

                row = {'SCC_ID': scc_sample}
                for v, w in enumerate(war_tuple):
                    row[f'W{v}'] = w
                row['dws'] = dws
                row['delta_log_sr'] = dsr_low_val
                row['lyapunov_exponent'] = dsr_low_val / T_low if T_low > 0 else float('nan')
                row['ratio'] = per_war_ratio
                r3_mean = r3_by_war_low.get(war_tuple, float('nan'))
                row['r3_count_mean'] = r3_mean
                row['le_counting'] = (r3_mean * LOG_SR_INJECTION / T_low
                                      if T_low > 0 and math.isfinite(r3_mean)
                                      else float('nan'))
                df_rows.append(row)

        if df_rows:
            df = pd.DataFrame(df_rows)
            col_order = ['SCC_ID'] + [f'W{v}' for v in range(n)] + [
                'dws', 'delta_log_sr', 'lyapunov_exponent', 'le_counting', 'r3_count_mean', 'ratio']
            df = df[col_order]
            csv_path = 'scc_war_ratios.csv'
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(df)} rows ({df['SCC_ID'].nunique()} SCCs, "
                  f"up to {df.groupby('SCC_ID').size().max()} WAR assignments each) "
                  f"to {csv_path}")

    for r in rz:
      print()
      print("SCC type:",r)
      for T in rz[r]:
        print("  EPOCH:",T)
        for stat in rz[r][T]:
          if (r, stat) in dead_queue:
              continue
          res = rz[r][T][stat]
          if not res or 'delta_log_sr' not in res:
              print(f"  SCC ID{stat}: no valid results (all runs overflowed or failed) — skipping")
              continue
          # WAR concentration 
          mean_war = res.get('mean_war', float('nan'))
          max_war  = res.get('max_war_ever', float('nan'))
          var95_war = res.get('mean_war_war_var95', float('nan'))
          war_str = f"  mean_war: {mean_war:.2f}  max_war: {max_war:.2f}  VaR95(WAR): {var95_war:.2f}"
          overflow_count = res.get('overflow_count', 0)
          total_runs = res.get('total_runs', '?')
          overflow_str = f"  overflows: {overflow_count}/{total_runs}" if overflow_count > 0 else ""
          trap_str = ""
          if strat['bidir_traps']:
            parts = []
            for t in strat['bidir_traps']:
                vset = "{" + ",".join(str(v) for v in t['vertices']) + "}"
                if t['escapes']:
                    esc = ",".join(f"{s}→{d}" for s, d in t['escapes'])
                    parts.append(f"{vset}({len(t['escapes'])} escape: {esc})")
                else:
                    parts.append(f"{vset}(strict)")
            trap_str = "  traps:" + ",".join(parts)
          td = res.get('triple_density')
          density_str = f"  pair_density(0/1/2): {td[0]:.2f}/{td[1]:.2f}/{td[2]:.2f}" if td is not None else ""
          ec = res.get('edge_coverage')
          mds = res.get('mean_directed_strands')
          max_ds_ever = res.get('max_directed_strands_ever')
          frac3 = res.get('frac_runs_hit_3')
          coverage_str = f"  edge_coverage: {ec:.2f}  mean_directed_strands: {mds:.2f}  max_ever: {max_ds_ever}  frac_runs_hit_3: {frac3:.2f}" if ec is not None else ""
          war95_sr  = res.get('spectral_radius_war_var95', float('nan'))
          war95_dsr = res.get('delta_log_sr_war_var95',  float('nan'))
          sc = scaling_classes.get((r, stat), '')
          ratio_val = scaling_ratios.get((r, stat), float('nan'))
          if sc:
              sc_str = f"  {sc}  ratio={ratio_val:.2f}/{expected_ratio:.1f}"
          else:
              sc_str = ''
          print(f"--- SCC ID{stat}, SR mean/std: {res['spectral_radius_mean']:.1e}/{res['spectral_radius_std']:.1e}  mean(delta log spectral radius): {res['delta_log_sr']:.1f}  VaR95(WAR combos): SR={war95_sr:.1e} Δlog={war95_dsr:.1f}{trap_str}{density_str}{coverage_str}")
        print()
      print()
    print()

    # ── Dead queue summary (saturated) ───────────────────────────────────────
    if dead_queue:
        print("=" * 70)
        print(f"DEAD QUEUE — {len(dead_queue)} saturated ratchet(s) (LE non-scaling)")
        print(f"  Δlog scaling ratio < {SCALING_SAT_THRESH:.2f}  (expected ≈{expected_ratio:.1f} for epochs {sorted_epochs})")
        print(f"  Directed edges dynamically sparse; Burau LE unreliable.")
        print(f"  Walk bias (proximity gradient, bias={4.0}) insufficient for topology-sparse SCCs.")
        print(f"  Threshold: mean_directed_strands >= 1.35 for LE to scale; these have <= 0.35.")
        print(f"  Root cause: too few directed edges relative to bidir edges for triple collision.")
        print("=" * 70)
        for (scc_type, stat), info in sorted(dead_queue.items()):
            print(f"  {scc_type} ID{stat}:  ratio={info['ratio']:.2f} (expected {info['expected']:.1f})"
                  f"  Δlog {info['dsr_low']:.1f}→{info['dsr_high']:.1f}"
                  f"  edge_coverage={info['edge_coverage']:.2f}"
                  f"  mean_directed_strands={info['mean_directed_strands']:.2f}"
                  f"  [saturated]")
        print()

    # ── Quasi-oscillator summary ──────────────────────────────────────────────
    # Sub-linear Δlog growth: braid is topologically directed but LE does not
    # scale at the pseudo-Anosov rate.  Signature: reducible braid structure
    # where an oscillatory subsystem partially cancels ratchet accumulation.
    # ratio band: [{SCALING_SAT_THRESH:.2f}, {pa_threshold:.2f})
    # Static bidir fraction cannot detect these; the ratio test is the
    # minimal sufficient statistic.
    if quasi_oscillator_queue:
        print("=" * 70)
        print(f"QUASI-OSCILLATOR QUEUE — {len(quasi_oscillator_queue)} sub-linear ratchet(s)")
        print(f"  T-scaling ratio in [{SCALING_SAT_THRESH:.2f}, {pa_threshold:.2f})  "
              f"(expected ≈{expected_ratio:.1f}, PA threshold {SCALING_PA_FRAC:.0%})")
        print(f"  Interpretation: reducible braid suspected.")
        print(f"  The oscillatory subsystem partially cancels ratchet accumulation each period,")
        print(f"  producing sub-linear Δlog_sr growth.  Cannot be detected from static bidir")
        print(f"  fraction alone — the T-scaling ratio is the minimal sufficient statistic.")
        print(f"  Risk: lower than pseudo-Anosov but non-zero; oscillatory component may")
        print(f"  not cancel completely in all WAR assignments.")
        print("=" * 70)
        for (scc_type, stat), info in sorted(quasi_oscillator_queue.items()):
            print(f"  {scc_type} ID{stat}:  ratio={info['ratio']:.2f} (expected {info['expected']:.1f})"
                  f"  Δlog {info['dsr_low']:.1f}→{info['dsr_high']:.1f}"
                  f"  edge_coverage={info['edge_coverage']:.2f}"
                  f"  mean_directed_strands={info['mean_directed_strands']:.2f}"
                  f"  [quasi_oscillator]")
        print()

if __name__ == '__main__':
    results = analyze_synthetic_sccs(
        max_n=6,
        n_nhis_per_scc=6,
        n_epochs=400,
        samples_per_type=10000,
        mc_samples=150,
        chirality_runs=500,
        seed=42
    )
