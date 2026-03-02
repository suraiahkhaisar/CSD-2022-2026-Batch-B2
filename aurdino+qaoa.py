"""
Quantum-Assisted Delivery Route Optimization with Alternate Route Switching
Complete simulation system with clustering, QAOA optimization, and animation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Arrow, Polygon, FancyArrowPatch
from sklearn.cluster import KMeans
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator
from scipy.spatial import ConvexHull
import random
import time
import math
from collections import defaultdict, Counter
import socket # NETWORK BRIDGE: for phone bridge communication
import msvcrt  # ARDUINO INTEGRATION: for non-blocking key check on Windows

# ============================================================================
# CONFIGURATION
# ============================================================================

GRID_SIZE = 100  # 100x100 grid for city map
NUM_CUSTOMERS = 12  # Number of delivery points
NUM_CLUSTERS = 3  # Number of clusters for scalability (can be changed)
NUM_ALTERNATE_ROUTES = 5  # Number of alternate routes to generate

# Vehicle animation settings
# Tuned for snappy, app-like motion while keeping visuals stable.
VEHICLE_SPEED = 8.0   # Restored to original speed for smoother motion
VEHICLE_SIZE = 2.0    # Smaller vehicle icon size
OBSTACLE_WARNING_DISTANCE = 3.0  # Distance before obstacle to show warning

# ============================================================================
# ANIMATION ENHANCEMENT CONFIGURATION (Optional, togglable)
# ============================================================================
# Enable/disable advanced visualization modules (set to False for calmer view)
ENABLE_QAOA_PANEL = False          # Disabled for cleaner UI
ENABLE_OBSTACLE_ANIMATION = True    # Keep this for visual rerouting
ENABLE_JUNCTION_EVENTS = False      # Disabled to avoid stuttering
ENABLE_ROUTE_TRANSITIONS = True     # Keep for smooth switching
ENABLE_STATUS_PANELS = True         # Keep main status
ENABLE_METRIC_INSETS = False         # Disabled to speed up rendering

# ---------------------------------------------------------------------------
# GLOBAL HISTORIES FOR OPTIONAL GAP ANALYSIS (non-disruptive to main logic)
# ---------------------------------------------------------------------------

# GAP-1: Track cluster feasibility and load-balancing behaviour across runs
CLUSTER_FEASIBILITY_HISTORY = []
1
# GAP-4: Track penalty tuning behaviour and stability across QAOA executions
PENALTY_HISTORY = []

# ============================================================================
# ARDUINO INTEGRATION START (PHONE BRIDGE MODE)
# ============================================================================
#
# Arduino acts purely as an EXECUTION layer:
# - It mirrors motion decisions made by the QAOA-based routing engine.
# - All "intelligence" (clustering, QUBO, QAOA solving, route choice) stays
#   in this Python code. Arduino only receives compact commands like:
#     'S' (start), 'X' (stop), 'L', 'R', 'F' (left / right / forward).
#

# ============================================================================
# DIRECT BLUETOOTH CONFIGURATION (MAC ADDRESS)
# ============================================================================
HC05_MAC = "00:00:13:01:81:A3"  # <--- UPDATE THIS with your HC-05 MAC Address
BT_PORT = 1                     # Default RFCOMM port for HC-05
ARDUINO_SER = None 

def init_arduino_serial():
    """Initialize Direct Bluetooth (RFCOMM) connection."""
    global ARDUINO_SER
    print(f"[BT] Connecting directly to HC-05 at {HC05_MAC}...")
    try:
        # Create a Bluetooth RFCOMM socket
        client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        client.settimeout(10.0)
        client.connect((HC05_MAC, BT_PORT))
        
        ARDUINO_SER = client
        print(f"[BT] Successfully connected! Robot is now linked to QAOA via MAC.")
    except Exception as e:
        ARDUINO_SER = None
        print(f"[BT] Connection failed: {e}")
        print("Note: Ensure your laptop Bluetooth is ON and the HC-05 is not connected to your phone.")

def arduino_send(cmd):
    """
    Send a single-character command via Bluetooth.
    """
    global ARDUINO_SER
    if not ARDUINO_SER:
        return
    try:
        msg = str(cmd).encode("utf-8")
        ARDUINO_SER.send(msg)
        print(f"[CMD] Sent to Robot: {cmd}") # Debug log
    except Exception as e:
        print(f"[BT] Send failed: {e}")

def read_arduino_feedback():
    """
    Non-blocking read of Robot feedback (like OBSTACLE).
    """
    global ARDUINO_SER
    if not ARDUINO_SER:
        return None
    try:
        ARDUINO_SER.setblocking(False)
        data = ARDUINO_SER.recv(1024).decode("utf-8", errors="ignore").strip()
        if data:
            print(f"[ROBOT] {data}")
            if "OBSTACLE" in data.upper():
                return "OBSTACLE"
    except BlockingIOError:
        pass
    except Exception as e:
        pass
    return None

# ============================================================================
# ARDUINO INTEGRATION END
# ============================================================================

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_depot_and_customers(num_customers, grid_size=GRID_SIZE):
    """Generate depot and customer locations on 2D grid"""
    np.random.seed(42)
    
    # Depot at center
    depot = np.array([grid_size / 2, grid_size / 2])
    
    # Generate customer locations (avoiding depot area)
    customers = []
    for _ in range(num_customers):
        x = np.random.uniform(10, grid_size - 10)
        y = np.random.uniform(10, grid_size - 10)
        customers.append([x, y])
    
    customers = np.array(customers)
    
    print(f"Generated depot at: {depot}")
    print(f"Generated {len(customers)} customers")
    
    return depot, customers

# ============================================================================
# CLASSICAL LAYER: CLUSTERING
# ============================================================================

def cluster_customers(customers, n_clusters, max_cluster_size=10):
    """
    GAP 1: SCALING / FEASIBILITY COLLAPSE
    Cluster customers ensuring each cluster ≤10 customers for scalability
    Uses candidate-route encoding instead of full edge/node encoding
    """
    if len(customers) <= n_clusters:
        # Too few customers, return single cluster
        labels = np.zeros(len(customers), dtype=int)
        cluster_centers = np.mean(customers, axis=0).reshape(1, -1)

        # Minimal feasibility record for degenerate case (keeps API identical)
        feasibility_metrics = {
            'max_cluster_size': len(customers),
            'min_cluster_size': len(customers),
            'clusters_within_limit': 1 if len(customers) <= max_cluster_size else 0,
            'total_clusters': 1 if len(customers) > 0 else 0,
            'feasibility_rate': 1.0 if len(customers) <= max_cluster_size and len(customers) > 0 else 0.0,
        }
        _record_cluster_feasibility(feasibility_metrics, n_clusters, max_cluster_size)
        return [list(range(len(customers)))], customers, labels, cluster_centers, feasibility_metrics
    
    # Ensure clusters don't exceed max size
    # Adjust number of clusters if needed
    min_clusters_needed = int(np.ceil(len(customers) / max_cluster_size))
    if n_clusters < min_clusters_needed:
        print(f"⚠️  Adjusting cluster count: {n_clusters} -> {min_clusters_needed} to meet ≤{max_cluster_size} customers/cluster constraint")
        n_clusters = min_clusters_needed
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(customers)
    cluster_centers = kmeans.cluster_centers_
    
    clusters = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0].tolist()
        if len(cluster_indices) > 0:
            clusters.append(cluster_indices)
    
    # GAP 1: Verify cluster size constraint
    cluster_sizes = [len(c) for c in clusters]
    max_size = max(cluster_sizes) if cluster_sizes else 0
    min_size = min(cluster_sizes) if cluster_sizes else 0
    avg_cluster_size = len(customers) / n_clusters if n_clusters > 0 else 0

    feasibility_metrics = {
        'max_cluster_size': max_size,
        'min_cluster_size': min_size,
        'clusters_within_limit': sum(1 for c in clusters if len(c) <= max_cluster_size),
        'total_clusters': len(clusters),
        'feasibility_rate': sum(1 for c in clusters if len(c) <= max_cluster_size) / len(clusters) if clusters else 0,
    }
    
    # ----------------------------------------------------------------------
    # GAP-1: Adaptive cluster resizing and load-balancing diagnostics
    # ----------------------------------------------------------------------
    if clusters:
        # Simple notion of "load" – currently one unit per customer
        total_load = float(len(customers))
        ideal_size = total_load / len(clusters)
        size_deviation = [len(c) - ideal_size for c in clusters]
        load_imbalance = float(np.std(size_deviation)) if len(size_deviation) > 1 else 0.0

        # Suggest an adaptive resize that keeps clusters within max_cluster_size
        # without changing the behaviour of the current run.
        min_clusters_needed = int(np.ceil(total_load / max_cluster_size)) if max_cluster_size > 0 else len(clusters)
        suggested_n_clusters = max(1, min_clusters_needed)

        if suggested_n_clusters > n_clusters:
            resize_action = 'increase'
            resize_reason = (
                f'increase clusters from {n_clusters} to ≈{suggested_n_clusters} '
                f'to keep sizes ≤{max_cluster_size}'
            )
        elif suggested_n_clusters < n_clusters:
            resize_action = 'decrease'
            resize_reason = (
                f'clusters appear under-loaded; consider reducing from {n_clusters} '
                f'towards ≈{suggested_n_clusters} for better utilization'
            )
        else:
            resize_action = 'keep'
            resize_reason = 'current number of clusters is consistent with size limits'

        feasibility_metrics.update({
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': avg_cluster_size,
            'load_imbalance_std': load_imbalance,
            'adaptive_resize': {
                'current_n_clusters': n_clusters,
                'suggested_n_clusters': suggested_n_clusters,
                'action': resize_action,
                'reason': resize_reason,
            },
        })
    else:
        feasibility_metrics.update({
            'cluster_sizes': [],
            'avg_cluster_size': 0.0,
            'load_imbalance_std': 0.0,
            'adaptive_resize': {
                'current_n_clusters': n_clusters,
                'suggested_n_clusters': n_clusters,
                'action': 'keep',
                'reason': 'no clusters formed',
            },
        })
    
    print(f"\n{'='*70}")
    print(f"CLUSTERING ANALYSIS - {n_clusters} Clusters")
    print(f"{'='*70}")
    print(f"Total customers: {len(customers)}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Average cluster size: {avg_cluster_size:.1f} customers")
    print(f"Largest cluster: {max_size} customers")
    print(f"Smallest cluster: {min_size} customers")
    print(f"\nCluster Details:")
    for i, cluster in enumerate(clusters):
        # Calculate cluster spread (average distance from center)
        cluster_customers = customers[cluster]
        center = cluster_centers[i]
        distances = [np.linalg.norm(c - center) for c in cluster_customers]
        avg_distance = np.mean(distances) if distances else 0
        
        print(f"  Cluster {i+1}:")
        print(f"    Size: {len(cluster)} customers")
        print(f"    Indices: {cluster}")
        print(f"    Center: ({cluster_centers[i][0]:.2f}, {cluster_centers[i][1]:.2f})")
        print(f"    Avg distance from center: {avg_distance:.2f}")
    
    print(f"{'='*70}")
    
    # GAP 1: Log cluster size and feasibility
    print(f"\n📊 GAP 1 - SCALING METRICS:")
    print(f"  Max cluster size: {feasibility_metrics['max_cluster_size']} (limit: {max_cluster_size})")
    print(f"  Clusters within limit: {feasibility_metrics['clusters_within_limit']}/{feasibility_metrics['total_clusters']}")
    print(f"  Feasibility rate: {feasibility_metrics['feasibility_rate']*100:.1f}%")
    print(f"  Load imbalance (std dev): {feasibility_metrics['load_imbalance_std']:.2f}")
    adaptive = feasibility_metrics.get('adaptive_resize', {})
    if adaptive:
        print(f"  Adaptive resize suggestion: {adaptive.get('action', 'keep')}"
              f" → target clusters ≈ {adaptive.get('suggested_n_clusters', n_clusters)}")
        print(f"    Rationale: {adaptive.get('reason', '')}")
    if feasibility_metrics['max_cluster_size'] > max_cluster_size:
        print(f"  ⚠️  WARNING: Some clusters exceed limit - feasibility may degrade")
    print(f"{'='*70}\n")
    
    # Persist non-disruptive feasibility snapshot for longitudinal analysis
    _record_cluster_feasibility(feasibility_metrics, n_clusters, max_cluster_size)
    
    return clusters, customers, labels, cluster_centers, feasibility_metrics


def _record_cluster_feasibility(feasibility_metrics, n_clusters, max_cluster_size):
    """
    Optional GAP-1 helper: keep a longitudinal record of cluster feasibility.

    This does NOT change clustering behaviour; it merely appends diagnostics to
    the global CLUSTER_FEASIBILITY_HISTORY for later research analysis.
    """
    snapshot = {
        'timestamp': time.time(),
        'n_clusters': n_clusters,
        'max_cluster_size_limit': max_cluster_size,
        'metrics': dict(feasibility_metrics),
    }
    try:
        CLUSTER_FEASIBILITY_HISTORY.append(snapshot)
    except Exception:
        # History is best-effort; never break the main pipeline.
        pass

def explain_clustering_effects(n_clusters, num_customers):
    """Explain what happens with different cluster sizes"""
    print(f"\n{'='*70}")
    print("CLUSTERING EFFECTS EXPLANATION")
    print(f"{'='*70}")
    print(f"\nCurrent Configuration:")
    print(f"  Number of customers: {num_customers}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Average customers per cluster: {num_customers / n_clusters:.1f}")
    
    print(f"\n📊 What Clustering Does:")
    print(f"  1. Groups nearby customers together")
    print(f"  2. Reduces problem complexity for quantum solver")
    print(f"  3. Creates manageable sub-problems")
    
    print(f"\n🔍 Effects of Changing Cluster Size:")
    
    if n_clusters == 1:
        print(f"  → Single cluster: All customers in one group")
        print(f"    • Problem size: Large (all {num_customers} customers)")
        print(f"    • Quantum solver: May struggle with large problem")
        print(f"    • Solution time: Longer")
    elif n_clusters < num_customers / 3:
        print(f"  → Few clusters ({n_clusters}): Large groups")
        print(f"    • Problem size: Medium-Large")
        print(f"    • Quantum solver: Moderate difficulty")
        print(f"    • Solution time: Medium")
    elif n_clusters < num_customers / 2:
        print(f"  → Moderate clusters ({n_clusters}): Balanced groups")
        print(f"    • Problem size: Medium")
        print(f"    • Quantum solver: Optimal balance")
        print(f"    • Solution time: Fast")
    else:
        print(f"  → Many clusters ({n_clusters}): Small groups")
        print(f"    • Problem size: Small per cluster")
        print(f"    • Quantum solver: Easy to solve")
        print(f"    • Solution time: Very fast")
        print(f"    • Note: May create too many small routes")
    
    print(f"\n💡 Recommendation:")
    optimal = max(2, int(num_customers / 4))
    if n_clusters == optimal:
        print(f"  ✓ Current cluster size ({n_clusters}) is optimal!")
    elif n_clusters < optimal:
        print(f"  → Consider increasing to {optimal} clusters for better performance")
    else:
        print(f"  → Consider decreasing to {optimal} clusters for better route efficiency")
    
    print(f"{'='*70}\n")

# ============================================================================
# DISTANCE CALCULATION
# ============================================================================

def calculate_distance_matrix(depot, customers):
    """Calculate Euclidean distance matrix"""
    n = len(customers) + 1  # +1 for depot
    
    # Combine depot and customers
    points = np.vstack([depot.reshape(1, -1), customers])
    
    # Calculate distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
    
    return dist_matrix, points

# ============================================================================
# CLASSICAL LAYER: HEURISTIC ROUTE GENERATION
# ============================================================================

def nearest_neighbor_route(dist_matrix, start=0):
    """Generate route using nearest neighbor heuristic"""
    n = len(dist_matrix)
    unvisited = set(range(1, n))  # Exclude depot
    route = [start]  # Start at depot
    current = start
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    route.append(start)  # Return to depot
    return route

def random_route(n):
    """Generate random route (for alternate routes)"""
    route = [0]  # Start at depot
    customers = list(range(1, n))
    random.shuffle(customers)
    route.extend(customers)
    route.append(0)  # Return to depot
    return route

def perturb_route(route, perturbation_rate=0.3):
    """
    GAP 2: SENSITIVITY TO INITIAL SOLUTION
    Create slight perturbation of a good route
    """
    n = len(route)
    if n <= 3:
        return route
    
    perturbed = route.copy()
    num_swaps = max(1, int((n - 2) * perturbation_rate))  # Don't swap depot
    
    for _ in range(num_swaps):
        # Swap two non-depot positions
        i = random.randint(1, n - 2)
        j = random.randint(1, n - 2)
        if i != j:
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
    
    return perturbed


def _two_opt_swap(route, i, k):
    """Internal helper: perform a 2-opt swap between indices i and k (inclusive)."""
    new_route = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
    return new_route


def local_2opt_improvement(route, dist_matrix, max_iterations=20):
    """
    Optional GAP-2 enhancement: apply a lightweight 2-opt local search
    to increase structural diversity of candidate routes.

    This function is PURE and never called from the main pipeline unless
    explicitly enabled by flags in generate_candidate_routes.
    """
    best_route = route[:]
    best_cost = calculate_route_cost(best_route, dist_matrix)
    n = len(best_route)

    improved = True
    it = 0
    while improved and it < max_iterations:
        improved = False
        it += 1
        for i in range(1, n - 2):  # keep depot endpoints intact
            for k in range(i + 1, n - 1):
                candidate = _two_opt_swap(best_route, i, k)
                cost = calculate_route_cost(candidate, dist_matrix)
                if cost + 1e-9 < best_cost:
                    best_cost = cost
                    best_route = candidate
                    improved = True
        # Break early if no better move found
    return best_route, best_cost


def compute_route_objectives(route, dist_matrix, points):
    """
    GAP-2: multi-objective view of a route.

    Returns:
        dict with:
          - 'distance': total geometric distance
          - 'num_hops': number of edges
          - 'spread': spatial spread of visited customers
    """
    distance = calculate_route_cost(route, dist_matrix)
    num_hops = max(0, len(route) - 1)

    # Spatial spread: average distance of customers from their centroid
    node_indices = [idx for idx in route if idx != 0]
    if node_indices:
        pts = points[node_indices]
        center = pts.mean(axis=0)
        spread = float(np.mean([np.linalg.norm(p - center) for p in pts]))
    else:
        spread = 0.0

    return {
        'distance': distance,
        'num_hops': num_hops,
        'spread': spread,
    }


def pareto_prune_candidates(candidates, dist_matrix, points, max_keep=None):
    """
    GAP-2: Pareto-based pruning on (distance, num_hops, spread).

    This is an OPTIONAL post-processing layer which can reduce the number of
    candidates while preserving a diverse Pareto front. It is only activated
    via flags in generate_candidate_routes and leaves the default behaviour
    unchanged when disabled.
    """
    if not candidates:
        return candidates, []

    metrics = []
    for name, route in candidates:
        obj = compute_route_objectives(route, dist_matrix, points)
        metrics.append((name, route, obj))

    pareto_set = []
    for i, (name_i, route_i, obj_i) in enumerate(metrics):
        dominated = False
        for j, (_, _, obj_j) in enumerate(metrics):
            if i == j:
                continue
            # j dominates i if it is no worse in all objectives and better in at least one
            if (
                obj_j['distance'] <= obj_i['distance'] + 1e-9
                and obj_j['num_hops'] <= obj_i['num_hops']
                and obj_j['spread'] >= obj_i['spread'] - 1e-9
                and (
                    obj_j['distance'] < obj_i['distance'] - 1e-9
                    or obj_j['num_hops'] < obj_i['num_hops']
                    or obj_j['spread'] > obj_i['spread'] + 1e-9
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto_set.append((name_i, route_i, obj_i))

    # Optionally limit size but keep most distance-efficient solutions
    pareto_set.sort(key=lambda x: x[2]['distance'])
    if max_keep is not None and len(pareto_set) > max_keep:
        pareto_set = pareto_set[:max_keep]

    pruned_candidates = [(name, route) for (name, route, _) in pareto_set]
    return pruned_candidates, pareto_set

def generate_candidate_routes(
    dist_matrix,
    num_routes=5,
    multi_start=True,
    enhanced_diversity=False,
    enable_dominance_filter=False,
    enable_pareto_pruning=False,
    points=None,
):
    """
    GAP 2: SENSITIVITY TO INITIAL SOLUTION
    Generate multiple candidate routes using different methods:
    - Nearest-neighbor
    - Random shuffles
    - Perturbations of good routes
    """
    n = len(dist_matrix)
    candidates = []
    route_costs = []
    
    # Method 1: Nearest neighbor (main route)
    nn_route = nearest_neighbor_route(dist_matrix)
    nn_cost = calculate_route_cost(nn_route, dist_matrix)
    candidates.append(('nearest_neighbor', nn_route))
    route_costs.append(nn_cost)
    
    # Method 2: Random routes
    num_random = max(2, num_routes // 2)
    for i in range(num_random):
        rand_route = random_route(n)
        rand_cost = calculate_route_cost(rand_route, dist_matrix)
        candidates.append((f'random_{i+1}', rand_route))
        route_costs.append(rand_cost)
    
    # Method 3: Perturbations of best route (GAP 2)
    if multi_start and len(candidates) > 0:
        best_route = candidates[0][1]  # Use nearest neighbor as base
        num_perturbations = num_routes - len(candidates)
        for i in range(num_perturbations):
            perturbed = perturb_route(best_route, perturbation_rate=0.2 + i*0.1)
            pert_cost = calculate_route_cost(perturbed, dist_matrix)
            candidates.append((f'perturbed_{i+1}', perturbed))
            route_costs.append(pert_cost)
    
    # Optional GAP-2: enhanced structural diversity via local 2-opt
    if enhanced_diversity and candidates:
        enriched = []
        enriched_costs = []
        for name, route in candidates:
            improved_route, improved_cost = local_2opt_improvement(route, dist_matrix, max_iterations=10)
            enriched.append((f"{name}_2opt", improved_route))
            enriched_costs.append(improved_cost)

        candidates.extend(enriched)
        route_costs.extend(enriched_costs)

        print(f"\n📊 GAP 2 - ENHANCED DIVERSITY:")
        print(f"  Base candidate routes: {len(candidates) - len(enriched)}")
        print(f"  2-opt enhanced routes: {len(enriched)}")
        print(f"  Total candidates after enhancement: {len(candidates)}")

    # Optional GAP-2: Pareto-based pruning / dominance filtering
    pareto_front = None
    if (enable_dominance_filter or enable_pareto_pruning) and points is not None and candidates:
        before = len(candidates)
        candidates, pareto_front = pareto_prune_candidates(
            candidates,
            dist_matrix,
            points,
            max_keep=len(candidates) if not enable_pareto_pruning else None,
        )
        # Recompute route_costs to stay consistent with final candidate set
        route_costs = [calculate_route_cost(route, dist_matrix) for _, route in candidates]

        print(f"\n📊 GAP 2 - PARETO / DOMINANCE FILTERING:")
        print(f"  Candidates before filtering: {before}")
        print(f"  Candidates after filtering: {len(candidates)}")
        if pareto_front is not None:
            print(f"  Pareto front size: {len(pareto_front)}")
    
    # GAP 2: Calculate variance
    if len(route_costs) > 1:
        cost_variance = np.var(route_costs)
        cost_std = np.std(route_costs)
        cost_mean = np.mean(route_costs)
    else:
        cost_variance = 0
        cost_std = 0
        cost_mean = route_costs[0] if route_costs else 0
    
    print(f"\n📊 GAP 2 - MULTI-START METRICS:")
    print(f"  Number of candidate routes: {len(candidates)}")
    print(f"  Cost mean: {cost_mean:.2f}")
    print(f"  Cost std: {cost_std:.2f}")
    print(f"  Cost variance: {cost_variance:.2f}")
    print(f"  Cost range: {max(route_costs) - min(route_costs):.2f}")
    
    return candidates, {'variance': cost_variance, 'std': cost_std, 'mean': cost_mean, 'costs': route_costs}

# ============================================================================
# ROUTE VALIDATION
# ============================================================================

def validate_route(route, n_customers):
    """Validate that route visits all customers"""
    customers_visited = set(route[1:-1])  # Exclude depot at start/end
    expected_customers = set(range(1, n_customers + 1))
    
    if customers_visited == expected_customers:
        return True, "Valid"
    else:
        missing = expected_customers - customers_visited
        return False, f"Missing customers: {missing}"

def classical_feasibility_filter(candidate_routes, dist_matrix, n_customers):
    """
    GAP 3: TIME-WINDOW & CONTINUOUS CONSTRAINT VIOLATIONS
    Classical rule-based feasibility filter before quantum execution
    Removes infeasible routes to ensure QAOA only sees feasible candidates
    """
    feasible_routes = []
    infeasible_routes = []
    
    for name, route in candidate_routes:
        # Check 1: All customers visited
        is_valid, msg = validate_route(route, n_customers)
        
        # Check 2: Route starts and ends at depot
        starts_at_depot = route[0] == 0
        ends_at_depot = route[-1] == 0
        
        # Check 3: No immediate return to depot (except at end)
        no_immediate_return = True
        for i in range(len(route) - 1):
            if route[i] == 0 and route[i+1] == 0 and i < len(route) - 2:
                no_immediate_return = False
                break
        
        # Check 4: Reasonable route length (not too long)
        route_length = len(route)
        reasonable_length = route_length <= n_customers + 5  # Allow some redundancy
        
        if is_valid and starts_at_depot and ends_at_depot and no_immediate_return and reasonable_length:
            feasible_routes.append((name, route))
        else:
            infeasible_routes.append((name, route, {
                'valid': is_valid,
                'starts_depot': starts_at_depot,
                'ends_depot': ends_at_depot,
                'no_return': no_immediate_return,
                'reasonable': reasonable_length
            }))
    
    feasibility_ratio = len(feasible_routes) / len(candidate_routes) if candidate_routes else 0
    
    print(f"\n📊 GAP 3 - FEASIBILITY FILTER METRICS:")
    print(f"  Total candidate routes: {len(candidate_routes)}")
    print(f"  Feasible routes: {len(feasible_routes)}")
    print(f"  Infeasible routes (pruned): {len(infeasible_routes)}")
    print(f"  Feasibility ratio: {feasibility_ratio*100:.1f}%")
    if infeasible_routes:
        print(f"  Pruned routes:")
        for name, route, reasons in infeasible_routes[:3]:  # Show first 3
            print(f"    - {name}: {reasons}")
    
    return feasible_routes, feasibility_ratio

def calculate_route_cost(route, dist_matrix):
    """Calculate total distance of a route"""
    total_cost = 0
    for i in range(len(route) - 1):
        total_cost += dist_matrix[route[i], route[i+1]]
    return total_cost

# ============================================================================
# QUANTUM LAYER: QUBO FORMULATION
# ============================================================================

def create_route_selection_qubo(candidate_routes, dist_matrix, cluster_size=None):
    """
    Create QUBO for selecting best route from candidates
    Binary variable x_i = 1 if route i is selected
    """
    num_routes = len(candidate_routes)
    
    # Calculate costs for each route
    route_costs = []
    for name, route in candidate_routes:
        cost = calculate_route_cost(route, dist_matrix)
        route_costs.append(cost)
        print(f"  {name} route cost: {cost:.2f}")
    
    # Normalize costs (0 to 1 scale)
    max_cost = max(route_costs)
    min_cost = min(route_costs)
    if max_cost > min_cost:
        normalized_costs = [(c - min_cost) / (max_cost - min_cost) for c in route_costs]
    else:
        normalized_costs = [0] * num_routes
    
    # Create QUBO: minimize cost, with penalty for selecting multiple routes
    qp = QuadraticProgram()
    
    # Binary variables for each route
    for i in range(num_routes):
        qp.binary_var(f'x_{i}')
    
    # GAP 4: PENALTY TUNING INSTABILITY
    # Automatic penalty tuning with alpha sweep
    def tune_penalty(route_costs, alpha_values=[1.0, 1.5, 2.0, 2.5], cluster_size=None):
        """Auto-tune penalty by sweeping alpha values"""
        if not route_costs:
            return 100, 1.5
        
        base_penalty = np.mean(route_costs)  # Use average instead of max for normalization
        best_alpha = 1.5
        best_penalty = base_penalty * best_alpha
        
        # Penalty should increase smoothly with cluster size
        if cluster_size:
            cluster_factor = 1.0 + (cluster_size - 1) * 0.1  # Slight increase per cluster
            base_penalty *= cluster_factor
        
        # Test different alpha values
        test_results = []
        for alpha in alpha_values:
            penalty = base_penalty * alpha
            # Heuristic: prefer penalties that are 1.2-2.0x the max cost
            max_cost = max(route_costs)
            penalty_ratio = penalty / max_cost if max_cost > 0 else 1
            score = 1.0 / (1.0 + abs(penalty_ratio - 1.5))  # Prefer ratio around 1.5
            test_results.append((alpha, penalty, score))
        
        # Select best alpha
        best_alpha, best_penalty, _ = max(test_results, key=lambda x: x[2])
        
        return best_penalty, best_alpha
    
    # Get cluster size from context if available
    cluster_size = None  # Will be passed from caller
    penalty, alpha = tune_penalty(route_costs, cluster_size=cluster_size)
    base_penalty = np.mean(route_costs) if route_costs else 100
    
    # Calculate penalty details (GAP 4)
    penalty_info = {
        'base_penalty': base_penalty,
        'penalty_multiplier': alpha,
        'final_penalty': penalty,
        'min_cost': min_cost,
        'max_cost': max_cost,
        'cost_range': max_cost - min_cost if max_cost > min_cost else 0,
        'cluster_size': cluster_size,
        'alpha_tested': [1.0, 1.5, 2.0, 2.5],
        'selected_alpha': alpha
    }
    
    print(f"\n{'='*70}")
    print("PENALTY CALCULATION")
    print(f"{'='*70}")
    print(f"Route Costs:")
    for i, cost in enumerate(route_costs):
        print(f"  Route {i+1}: {cost:.2f}")
    print(f"\nPenalty Parameters:")
    print(f"  Minimum cost: {min_cost:.2f}")
    print(f"  Maximum cost: {max_cost:.2f}")
    print(f"  Cost range: {penalty_info['cost_range']:.2f}")
    print(f"  Base penalty: {base_penalty:.2f}")
    print(f"  Penalty multiplier (alpha): {penalty_info['penalty_multiplier']}")
    print(f"  Final penalty value: {penalty:.2f}")
    if cluster_size:
        print(f"  Cluster size: {cluster_size} (penalty adjusted)")
    print(f"  Normalized costs: {[f'{nc:.3f}' for nc in normalized_costs]}")
    print(f"\n📊 GAP 4 - PENALTY TUNING:")
    print(f"  Alpha values tested: {penalty_info['alpha_tested']}")
    print(f"  Selected alpha: {penalty_info['selected_alpha']}")
    print(f"  Penalty normalization: Based on average cost")
    print(f"{'='*70}\n")
    
    # Objective function: minimize cost + penalty for multiple selections
    linear_coef = {}
    for i in range(num_routes):
        # Scale normalized cost (multiply by 100 for better resolution)
        linear_coef[f'x_{i}'] = normalized_costs[i] * 100
    
    # Quadratic penalty: if multiple routes selected, add large penalty
    quadratic_coef = {}
    for i in range(num_routes):
        for j in range(i + 1, num_routes):
            # Large penalty if both routes selected
            quadratic_coef[(f'x_{i}', f'x_{j}')] = penalty
    
    qp.minimize(linear=linear_coef, quadratic=quadratic_coef)
    
    # Constraint: select exactly one route
    qp.linear_constraint(
        linear={f'x_{i}': 1 for i in range(num_routes)},
        sense='==',
        rhs=1,
        name='select_one'
    )
    
    return qp, route_costs, penalty_info


def record_penalty_feedback(penalty_info, comp_metrics, route_costs):
    """
    GAP-4: feedback-based penalty tracking and stability-aware diagnostics.

    This does NOT retroactively change the penalty used in the current run.
    Instead, it records how the chosen penalty interacted with QAOA stability
    (e.g., consensus_fraction) so that future experiments can adapt α offline.
    """
    if penalty_info is None:
        return

    snapshot = {
        'timestamp': time.time(),
        'penalty_info': dict(penalty_info),
        'route_cost_stats': {
            'mean': float(np.mean(route_costs)) if route_costs else 0.0,
            'std': float(np.std(route_costs)) if route_costs else 0.0,
            'min': float(min(route_costs)) if route_costs else 0.0,
            'max': float(max(route_costs)) if route_costs else 0.0,
        },
        'stability': {},
    }

    if comp_metrics:
        stability = {}
        if 'consensus_fraction' in comp_metrics:
            stability['consensus_fraction'] = float(comp_metrics['consensus_fraction'])
        if 'successes' in comp_metrics and 'num_starts' in comp_metrics and comp_metrics['num_starts'] > 0:
            stability['success_rate'] = float(comp_metrics['successes'] / comp_metrics['num_starts'])
        snapshot['stability'] = stability

    try:
        PENALTY_HISTORY.append(snapshot)
    except Exception:
        # History is advisory only – never disrupt main computation.
        pass

# ============================================================================
# QUANTUM OPTIMIZATION: MULTI-START QAOA
# ============================================================================

def solve_with_qaoa(qp, num_starts=3, cluster_size=None):
    """Solve QUBO using multi-start QAOA with computation tracking."""
    best_result = None
    best_cost = float('inf')
    
    computation_metrics = {
        'num_starts': num_starts,
        'problem_size': qp.get_num_binary_vars(),
        'cluster_size': cluster_size,
        'start_times': [],
        'end_times': [],
        'iterations': [],
        'successes': 0,
        'failures': 0,
        # GAP-5: track which route each start prefers for stability analysis
        'selected_routes': [],
    }
    
    print(f"\n{'='*70}")
    print(f"QUANTUM OPTIMIZATION - QAOA")
    print(f"{'='*70}")
    if cluster_size:
        print(f"Cluster Size: {cluster_size}")
    print(f"Problem Size: {computation_metrics['problem_size']} binary variables")
    print(f"Number of Starts: {num_starts}")
    print(f"Running QAOA with {num_starts} random starts...")
    print(f"{'='*70}")
    
    for start in range(num_starts):
        start_time = time.time()
        computation_metrics['start_times'].append(start_time)
        try:
            # Set random seed for each start
            np.random.seed(42 + start)
            
            # Create QAOA solver
            estimator = StatevectorEstimator()
            max_iter = 50
            qaoa = QAOA(estimator=estimator, optimizer=COBYLA(maxiter=max_iter), reps=2)
            optimizer = MinimumEigenOptimizer(min_eigen_solver=qaoa)
            
            # Solve
            result = optimizer.solve(qp)
            end_time = time.time()
            computation_metrics['end_times'].append(end_time)
            computation_metrics['iterations'].append(max_iter)
            
            # Evaluate solution
            if result and hasattr(result, 'x'):
                # Find which route was selected
                selected_route_idx = None
                for i in range(len(result.x)):
                    if result.x[i] > 0.5:  # Binary threshold
                        selected_route_idx = i
                        break
                
                if selected_route_idx is not None:
                    # Get objective value
                    obj_value = result.fval if hasattr(result, 'fval') else float('inf')
                    elapsed = end_time - start_time
                    computation_metrics['successes'] += 1

                    # GAP-5: record which route index this start selected
                    computation_metrics['selected_routes'].append(int(selected_route_idx))
                    
                    if obj_value < best_cost:
                        best_cost = obj_value
                        best_result = (selected_route_idx, result)
                    
                    print(f"  Start {start+1}: ✓ Success - Route {selected_route_idx+1}, Cost: {obj_value:.2f}, Time: {elapsed:.3f}s")
                else:
                    computation_metrics['failures'] += 1
                    print(f"  Start {start+1}: ✗ Failed - No valid route selected")
        
        except Exception as e:
            computation_metrics['failures'] += 1
            end_time = time.time()
            computation_metrics['end_times'].append(end_time)
            print(f"  Start {start+1}: ✗ Error - {e}")
            continue
    
    # Calculate total computation time
    if computation_metrics['end_times']:
        total_time = max(computation_metrics['end_times']) - min(computation_metrics['start_times'])
        avg_time = total_time / num_starts if num_starts > 0 else 0
        computation_metrics['total_time'] = total_time
        computation_metrics['avg_time'] = avg_time
        
        print(f"\n{'='*70}")
        print("COMPUTATION METRICS")
        print(f"{'='*70}")
        print(f"Total computation time: {total_time:.3f} seconds")
        print(f"Average time per start: {avg_time:.3f} seconds")
        print(f"Successful starts: {computation_metrics['successes']}/{num_starts}")
        print(f"Failed starts: {computation_metrics['failures']}/{num_starts}")
        print(f"Success rate: {(computation_metrics['successes']/num_starts*100):.1f}%")
        # GAP-5: basic stability statistics over starts
        if computation_metrics['selected_routes']:
            counts = Counter(computation_metrics['selected_routes'])
            consensus_idx, consensus_support = counts.most_common(1)[0]
            consensus_fraction = consensus_support / max(1, len(computation_metrics['selected_routes']))
            computation_metrics['consensus_route_idx'] = int(consensus_idx)
            computation_metrics['consensus_support'] = int(consensus_support)
            computation_metrics['consensus_fraction'] = float(consensus_fraction)
            print(f"Consensus route index: {consensus_idx} "
                  f"(support {consensus_support}/{len(computation_metrics['selected_routes'])}, "
                  f"{consensus_fraction*100:.1f}% agreement)")
        if cluster_size:
            print(f"Cluster size impact: {computation_metrics['problem_size']} variables")
        print(f"{'='*70}\n")
    
    return best_result, computation_metrics

# ============================================================================
# ROUTE MANAGEMENT
# ============================================================================

def repair_route(route, n_customers):
    """
    GAP 5: POST-HOC REPAIR LIMITATION
    Repair infeasible routes by adding missing customers
    """
    customers_visited = set(route[1:-1])
    expected_customers = set(range(1, n_customers + 1))
    missing = expected_customers - customers_visited
    
    if not missing:
        return route, False  # Already feasible
    
    # Insert missing customers using nearest insertion
    repaired = route.copy()
    for missing_cust in missing:
        # Find best insertion position
        best_pos = 1
        best_cost_increase = float('inf')
        
        for pos in range(1, len(repaired)):
            # Calculate cost increase if inserted here
            if pos < len(repaired) - 1:
                # Cost: from prev to missing + from missing to next - from prev to next
                cost_increase = 0  # Simplified
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_pos = pos
        
        repaired.insert(best_pos, missing_cust)
    
    return repaired, True  # Was repaired

def repair_and_reinsert_loop(candidate_routes, dist_matrix, n_customers, max_iterations=3):
    """
    GAP 5: POST-HOC REPAIR LIMITATION
    Repair-and-reinsert loop: repair infeasible routes and feed back as candidates
    """
    repaired_routes = []
    repair_metrics = {
        'iterations': 0,
        'routes_repaired': 0,
        'improvements': []
    }
    
    for iteration in range(max_iterations):
        iteration_repaired = 0
        new_candidates = []
        
        for name, route in candidate_routes:
            is_valid, msg = validate_route(route, n_customers)
            if not is_valid:
                repaired, was_repaired = repair_route(route, n_customers)
                if was_repaired:
                    new_candidates.append((f'{name}_repaired_iter{iteration}', repaired))
                    iteration_repaired += 1
                    repair_metrics['routes_repaired'] += 1
            else:
                new_candidates.append((name, route))
        
        if iteration_repaired == 0:
            break  # No more repairs needed
        
        repair_metrics['iterations'] = iteration + 1
        repair_metrics['improvements'].append({
            'iteration': iteration + 1,
            'repaired': iteration_repaired,
            'total_candidates': len(new_candidates)
        })
        
        candidate_routes = new_candidates
    
    print(f"\n📊 GAP 5 - REPAIR METRICS:")
    print(f"  Repair iterations: {repair_metrics['iterations']}")
    print(f"  Total routes repaired: {repair_metrics['routes_repaired']}")
    for imp in repair_metrics['improvements']:
        print(f"  Iteration {imp['iteration']}: {imp['repaired']} routes repaired")
    
    return candidate_routes, repair_metrics

def hybrid_orchestrator(customers, depot, n_clusters, dist_matrix, points):
    """
    GAP 6: LIMITED HYBRID ORCHESTRATION
    Orchestrator that decides which parts run classically vs quantum
    Clear separation of responsibilities
    """
    print(f"\n{'='*70}")
    print("GAP 6 - HYBRID ORCHESTRATOR")
    print(f"{'='*70}")
    
    orchestrator_log = {
        'classical_tasks': [],
        'quantum_tasks': [],
        'decisions': []
    }
    
    n_customers = len(customers)
    
    # DECISION 1: Clustering (Classical)
    print("\n[ORCHESTRATOR] Decision: Clustering → CLASSICAL")
    orchestrator_log['classical_tasks'].append('clustering')
    orchestrator_log['decisions'].append({
        'task': 'clustering',
        'method': 'classical',
        'reason': 'Deterministic, fast, no quantum advantage needed'
    })
    clusters, clustered_customers, cluster_labels, cluster_centers, feasibility_metrics = cluster_customers(
        customers, n_clusters, max_cluster_size=10
    )
    
    # DECISION 2: Candidate Route Generation (Classical)
    print("[ORCHESTRATOR] Decision: Candidate Generation → CLASSICAL")
    orchestrator_log['classical_tasks'].append('candidate_generation')
    orchestrator_log['decisions'].append({
        'task': 'candidate_generation',
        'method': 'classical',
        'reason': 'Heuristic methods, multiple starts for diversity'
    })
    candidate_routes, multi_start_metrics = generate_candidate_routes(dist_matrix, num_routes=8, multi_start=True)
    
    # DECISION 3: Feasibility Filtering (Classical)
    print("[ORCHESTRATOR] Decision: Feasibility Filtering → CLASSICAL")
    orchestrator_log['classical_tasks'].append('feasibility_filter')
    orchestrator_log['decisions'].append({
        'task': 'feasibility_filter',
        'method': 'classical',
        'reason': 'Rule-based constraints, fast validation'
    })
    feasible_routes, feasibility_ratio = classical_feasibility_filter(
        candidate_routes, dist_matrix, n_customers
    )
    
    # DECISION 4: Route Selection (Quantum)
    print("[ORCHESTRATOR] Decision: Route Selection → QUANTUM (QAOA)")
    orchestrator_log['quantum_tasks'].append('route_selection')
    orchestrator_log['decisions'].append({
        'task': 'route_selection',
        'method': 'quantum',
        'reason': 'Discrete optimization, quantum advantage for selection'
    })
    
    if not feasible_routes:
        print("⚠️  No feasible routes after filtering, using all candidates")
        feasible_routes = candidate_routes
    
    # DECISION 5: Repair (Classical)
    print("[ORCHESTRATOR] Decision: Route Repair → CLASSICAL")
    orchestrator_log['classical_tasks'].append('repair')
    orchestrator_log['decisions'].append({
        'task': 'repair',
        'method': 'classical',
        'reason': 'Deterministic constraint fixing'
    })
    final_routes, repair_metrics = repair_and_reinsert_loop(
        feasible_routes, dist_matrix, n_customers, max_iterations=2
    )
    
    print(f"\n📊 ORCHESTRATOR SUMMARY:")
    print(f"  Classical tasks: {len(orchestrator_log['classical_tasks'])}")
    print(f"  Quantum tasks: {len(orchestrator_log['quantum_tasks'])}")
    print(f"  Total decisions: {len(orchestrator_log['decisions'])}")
    print(f"{'='*70}\n")
    
    return final_routes, orchestrator_log, {
        'feasibility': feasibility_metrics,
        'multi_start': multi_start_metrics,
        'feasibility_ratio': feasibility_ratio,
        'repair': repair_metrics
    }, clusters, cluster_labels, cluster_centers

def select_routes_quantum(candidate_routes, dist_matrix, cluster_size=None):
    """Select main and alternate routes using quantum optimization"""
    print("\n" + "="*70)
    print("QUANTUM ROUTE SELECTION")
    print("="*70)
    
    # Create QUBO with cluster size for penalty tuning
    qp, route_costs, penalty_info = create_route_selection_qubo(candidate_routes, dist_matrix, cluster_size)
    
    # Solve with QAOA (GAP 2: multiple starts with different seeds)
    result, comp_metrics = solve_with_qaoa(qp, num_starts=5, cluster_size=cluster_size)
    
    if result is None:
        print("QAOA failed, using classical selection")
        # Fallback: select route with minimum cost
        main_idx = np.argmin(route_costs)
        routes = {
            'main': candidate_routes[main_idx][1],
            'alternates': [candidate_routes[i][1] for i in range(len(candidate_routes)) if i != main_idx]
        }
        comp_metrics = None
    else:
        main_idx, _ = result
        routes = {
            'main': candidate_routes[main_idx][1],
            'alternates': [candidate_routes[i][1] for i in range(len(candidate_routes)) if i != main_idx]
        }
        print(f"\n✓ Selected main route: {main_idx+1} (cost: {route_costs[main_idx]:.2f})")
    
        # GAP-5: derive a simple confidence score from multi-start agreement
        confidence_score = None
        if comp_metrics and comp_metrics.get('selected_routes'):
            total_successes = len(comp_metrics['selected_routes'])
            main_support = comp_metrics['selected_routes'].count(main_idx)
            if total_successes > 0:
                confidence_score = main_support / total_successes
                comp_metrics['confidence_score'] = float(confidence_score)
                print(f"QAOA main-route confidence: {confidence_score*100:.1f}% "
                      f"({main_support}/{total_successes} successful starts)")
    
    # Record GAP-4 feedback for penalty tuning diagnostics
    record_penalty_feedback(penalty_info, comp_metrics, route_costs)

    # Validate and rank routes
    n_customers = len(dist_matrix) - 1
    validated_routes = []
    
    # Validate main route
    is_valid, msg = validate_route(routes['main'], n_customers)
    main_cost = calculate_route_cost(routes['main'], dist_matrix)
    validated_routes.append(('main', routes['main'], main_cost, is_valid, msg))
    print(f"Main route validation: {msg}")
    
    # Validate alternate routes
    for i, alt_route in enumerate(routes['alternates'][:NUM_ALTERNATE_ROUTES]):
        is_valid, msg = validate_route(alt_route, n_customers)
        alt_cost = calculate_route_cost(alt_route, dist_matrix)
        validated_routes.append((f'alternate_{i+1}', alt_route, alt_cost, is_valid, msg))
        print(f"Alternate route {i+1} validation: {msg}, cost: {alt_cost:.2f}")
    
    # Sort by cost
    validated_routes.sort(key=lambda x: x[2])
    
    return validated_routes, comp_metrics, penalty_info

# ============================================================================
# ANIMATION SYSTEM
# ============================================================================

# ============================================================================
# ANIMATION ENHANCEMENT MODULES (Optional Extensions)
# ============================================================================

class QAOAAnimationPanel:
    """
    # ANIMATION: QAOA Optimization Progress Panel
    Shows multi-start cost bars, best candidate highlighting, and QAOA selection progress.
    """
    def __init__(self, ax, num_starts=5):
        self.ax = ax
        self.num_starts = num_starts
        self.cost_history = [[] for _ in range(num_starts)]  # Track costs per start
        self.best_candidate_idx = 0
        self.bars = None
        self.best_indicator = None
        self.initialized = False
    
    def initialize(self, initial_costs):
        """Initialize the panel with starting costs."""
        if not ENABLE_QAOA_PANEL:
            return
        self.ax.set_xlim(0, self.num_starts)
        self.ax.set_ylim(0, max(initial_costs) * 1.2 if initial_costs else 100)
        self.ax.set_xlabel('QAOA Start', fontsize=9)
        self.ax.set_ylabel('Route Cost', fontsize=9)
        self.ax.set_title('QAOA Multi-Start Progress', fontsize=10, fontweight='bold')
        self.bars = self.ax.bar(range(self.num_starts), initial_costs, 
                               color=['#4CAF50' if i == self.best_candidate_idx else '#2196F3' 
                                      for i in range(self.num_starts)],
                               alpha=0.7, edgecolor='black', linewidth=1)
        self.initialized = True
    
    def update(self, costs, best_idx):
        """Update cost bars and highlight best candidate."""
        if not ENABLE_QAOA_PANEL or not self.initialized:
            return
        self.best_candidate_idx = best_idx
        for i, (bar, cost) in enumerate(zip(self.bars, costs)):
            bar.set_height(cost)
            bar.set_color('#4CAF50' if i == best_idx else '#2196F3')
            bar.set_alpha(0.9 if i == best_idx else 0.6)
        self.ax.set_ylim(0, max(costs) * 1.2 if costs else 100)
        self.ax.figure.canvas.draw_idle()


class ObstacleAnimationManager:
    """
    # ANIMATION: Obstacle Appearance Animation
    Animates obstacles with fade-in and blink effects.
    """
    def __init__(self):
        self.obstacle_alpha = 0.0
        self.obstacle_blink_phase = 0.0
        self.obstacle_appearing = False
        self.appearance_duration = 30  # frames
    
    def start_appearance(self):
        """Start obstacle appearance animation."""
        if not ENABLE_OBSTACLE_ANIMATION:
            self.obstacle_alpha = 1.0
            return
        self.obstacle_appearing = True
        self.obstacle_alpha = 0.0
        self.obstacle_blink_phase = 0.0
    
    def update(self, frame):
        """Update obstacle animation state."""
        if not ENABLE_OBSTACLE_ANIMATION:
            self.obstacle_alpha = 1.0
            return
        
        if self.obstacle_appearing:
            # Fade-in
            progress = min(1.0, frame / self.appearance_duration)
            self.obstacle_alpha = progress
            if progress >= 1.0:
                self.obstacle_appearing = False
        
        # Blink effect
        self.obstacle_blink_phase = (self.obstacle_blink_phase + 0.2) % (2 * math.pi)
    
    def get_alpha(self):
        """Get current alpha with blink effect."""
        if not ENABLE_OBSTACLE_ANIMATION:
            return 1.0
        blink_factor = 0.7 + 0.3 * (1 + math.sin(self.obstacle_blink_phase)) / 2
        return self.obstacle_alpha * blink_factor
    
    def get_linewidth(self):
        """Get line width with pulse effect."""
        if not ENABLE_OBSTACLE_ANIMATION:
            return 5.0
        pulse = 1.0 + 0.3 * math.sin(self.obstacle_blink_phase)
        return 5.0 * pulse


class JunctionEventManager:
    """
    # ANIMATION: Junction Event Visualization
    Pauses at junctions, highlights candidate routes, animates transitions.
    """
    def __init__(self):
        self.current_junction = None
        self.junction_pause_frames = 0
        self.candidate_routes_shown = False
        self.transition_progress = 0.0
        self.transitioning = False
        self.old_route = None
        self.new_route = None
    
    def trigger_junction_event(self, junction_pos, candidate_routes):
        """Trigger junction pause and show candidate routes."""
        if not ENABLE_JUNCTION_EVENTS:
            return
        self.current_junction = junction_pos
        self.junction_pause_frames = 20  # Pause for 20 frames
        self.candidate_routes_shown = True
        self.candidate_routes = candidate_routes
    
    def update(self):
        """Update junction event state."""
        if not ENABLE_JUNCTION_EVENTS:
            return
        if self.junction_pause_frames > 0:
            self.junction_pause_frames -= 1
    
    def start_route_transition(self, old_route, new_route):
        """Start smooth transition between routes."""
        if not ENABLE_ROUTE_TRANSITIONS:
            return
        self.transitioning = True
        self.transition_progress = 0.0
        self.old_route = old_route
        self.new_route = new_route
    
    def update_transition(self, speed=0.05):
        """Update transition progress."""
        if not ENABLE_ROUTE_TRANSITIONS or not self.transitioning:
            return
        self.transition_progress = min(1.0, self.transition_progress + speed)
        if self.transition_progress >= 1.0:
            self.transitioning = False
    
    def get_interpolated_route(self):
        """Get interpolated route between old and new."""
        if not ENABLE_ROUTE_TRANSITIONS or not self.transitioning:
            return None
        # Simple linear interpolation (can be enhanced with easing)
        if self.old_route is None or self.new_route is None:
            return None
        # For now, return new route when transition complete
        return self.new_route if self.transition_progress >= 1.0 else self.old_route


class StatusPanelManager:
    """
    # ANIMATION: Embedded Status Panels
    Shows current cost, penalty value, cluster info.
    """
    def __init__(self, ax):
        self.ax = ax
        self.text_artists = {}
    
    def update_panel(self, metrics_dict):
        """Update status panel with current metrics."""
        if not ENABLE_STATUS_PANELS:
            return
        
        y_start = GRID_SIZE - 5
        y_spacing = 8
        x_pos = GRID_SIZE - 35
        
        panel_items = [
            ('Current Cost', metrics_dict.get('current_cost', 'N/A')),
            ('Penalty', metrics_dict.get('penalty', 'N/A')),
            ('Clusters', metrics_dict.get('num_clusters', 'N/A')),
            ('QAOA Conf', f"{metrics_dict.get('confidence', 0)*100:.0f}%"),
        ]
        
        for i, (label, value) in enumerate(panel_items):
            y_pos = y_start - i * y_spacing
            text = f"{label}: {value}"
            if (label, value) in self.text_artists:
                self.text_artists[(label, value)].set_text(text)
            else:
                artist = self.ax.text(x_pos, y_pos, text, fontsize=8,
                                    bbox=dict(boxstyle='round', facecolor='white', 
                                            alpha=0.85, edgecolor='gray', linewidth=1),
                                    zorder=100)
                self.text_artists[(label, value)] = artist


class MetricInsetManager:
    """
    # ANIMATION: Optional Graph Insets
    Small subplots showing trending metrics (cost variance, etc.).
    """
    def __init__(self, fig):
        self.fig = fig
        self.insets = {}
        self.metric_history = defaultdict(list)
    
    def add_inset(self, name, position, xlabel, ylabel, title):
        """Add a new metric inset subplot."""
        if not ENABLE_METRIC_INSETS:
            return None
        ax = self.fig.add_axes(position)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=8, fontweight='bold')
        self.insets[name] = ax
        return ax
    
    def update_metric(self, name, value, max_points=100):
        """Update metric history and redraw inset."""
        if not ENABLE_METRIC_INSETS or name not in self.insets:
            return
        self.metric_history[name].append(value)
        if len(self.metric_history[name]) > max_points:
            self.metric_history[name].pop(0)
        
        ax = self.insets[name]
        ax.clear()
        ax.set_xlabel(ax.get_xlabel(), fontsize=7)
        ax.set_ylabel(ax.get_ylabel(), fontsize=7)
        ax.set_title(ax.get_title(), fontsize=8, fontweight='bold')
        ax.plot(self.metric_history[name], color='#2196F3', linewidth=1.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_points)


class VehicleState:
    """Lightweight vehicle state for smooth movement along a single route."""
    def __init__(self, name, route_nodes, color, alt_route=None):
        self.name = name
        self.route_nodes = route_nodes  # ordered node indices
        self.alt_route = alt_route      # fallback ordered node indices
        self.color = color
        self.path = []                  # expanded waypoints (grid)
        self.segment_index = 0
        self.pos = None
        self.done = False
        self.blocked = False
        self.switch_logged = False
        self.node_index = 0
        self.visited_nodes = set()
        self.trail = []
        self.pause_frames = 0  # short pause when obstacle is detected for clarity
        # Path following buffers
        self.path = []
        self.dense_path = []
        self.step_idx = 0


class RouteSimulation:
    def __init__(
        self,
        depot,
        customers,
        routes,
        dist_matrix,
        points,
        cluster_labels=None,
        cluster_centers=None,
        end_point=None,
        enable_confidence_visuals=False,
        confidence_score=None,
    ):
        self.depot = depot
        self.customers = customers
        self.routes = routes
        self.dist_matrix = dist_matrix
        self.points = points
        self.cluster_labels = cluster_labels
        self.cluster_centers = cluster_centers
        self.end_point = end_point
        end_idx = len(points) - 1  # default destination index fallback

        # GAP-5: optional confidence-aware visuals
        self.enable_confidence_visuals = enable_confidence_visuals
        self.confidence_score = confidence_score
        
        # ANIMATION: Initialize optional animation managers (will be set after figure creation)
        self.obstacle_anim = ObstacleAnimationManager() if ENABLE_OBSTACLE_ANIMATION else None
        self.junction_events = JunctionEventManager() if ENABLE_JUNCTION_EVENTS else None
        self.status_panel = None  # Will be initialized after figure creation
        self.qaoa_panel = None  # Will be initialized after figure creation
        self.metric_insets = None  # Will be initialized after figure creation

        # ARDUINO INTEGRATION: track whether we have already sent a STOP ('X')
        # to the Arduino once all vehicles have finished.
        self.arduino_stop_sent = False
        
        # ANIMATION: Route transition state
        self.route_transition_active = False
        self.transition_old_route = None
        self.transition_new_route = None
        self.transition_progress = 0.0
        
        # Realistic cluster colors - professional and distinct
        self.cluster_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749', '#8B5A3C', '#4A90A4']
        
        # Find main route
        self.main_route = None
        self.alternate_routes = []
        self.selected_alternate_index = None  # index into self.alternate_routes when auto-rerouting
        for route_info in routes:
            if route_info[0] == 'main':
                self.main_route = route_info[1]
            elif 'alternate' in route_info[0]:
                self.alternate_routes.append(route_info[1])
        
        # Ensure route includes destination if specified
        if self.main_route:
            # Normalize base route; cluster-coverage enforcement will run
            # after the final destination index is known.
            self.current_route = self._normalize_route(self.main_route.copy(), end_idx)
            # If end_point is specified and not in route, add it at the end
            if self.end_point is not None:
                # Find closest point to end_point
                end_idx = len(points) - 1  # Assume last point or find closest
                min_dist = float('inf')
                for i, point in enumerate(points):
                    dist = np.linalg.norm(point - self.end_point)
                    if dist < min_dist:
                        min_dist = dist
                        end_idx = i
                
                # If route doesn't end at destination, modify it
                if self.current_route[-1] != end_idx:
                    # Remove final depot return if exists, add destination
                    if self.current_route[-1] == 0:  # If ends at depot
                        self.current_route = self.current_route[:-1]  # Remove depot
                    self.current_route.append(end_idx)  # Add destination
        else:
            self.current_route = []
        
        # Normalize route to avoid repeated depot/destination crossings
        self.destination_idx = end_idx if self.end_point is not None else 0
        # Final normalization + safety: ensure clusters are still visited before destination
        self.current_route = self._normalize_route(self.current_route, self.destination_idx)
        self.current_route = self._ensure_clusters_before_destination(self.current_route)
        
        # Time windows for customers (synthetic for visualization)
        self.time_windows = self._generate_time_windows(len(self.customers))
        self.time_log = []
        
        # Build vehicles (at least 3) using main and alternates
        self.vehicles = self._build_vehicle_states()
        
        # Obstacles
        self.obstacle_detected = False
        self.obstacle_segment = None  # (p1, p2) in xy
        self.obstacle_edge = None
        # UI for alternate routes when obstacle appears
        self.alt_routes_ui = []
        self.reroute_active = False
        self.reroute_timer = 0
        # When True, a persistent note is drawn on the UI listing the
        # available alternate routes and highlighting the one being followed.
        self.show_alt_routes_ui = False
        # Track which alternate route we are actively following (for UI + draw)
        self.active_alternate_route = None
        self.active_alternate_cost = None
        
        # Initialize junction detection
        self._init_junctions()
        self.obstacle_frame = 25  # Re-enable automatic visual obstacle trigger
        
        # Movement trail for visualization (per vehicle)
        self.trail_length = 12  # Keep recent positions for clarity
        
        # Setup figure
        if ENABLE_QAOA_PANEL or ENABLE_METRIC_INSETS:
            # Create figure with subplots for panels/insets
            self.fig = plt.figure(figsize=(16, 10))
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal')
        self.ax.set_title('Delivery Route Optimization - Quantum VRP', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X Coordinate', fontsize=10)
        self.ax.set_ylabel('Y Coordinate', fontsize=10)
        # Subtle grid for better readability
        self.ax.grid(False)
        
        # ANIMATION: Initialize status panel and QAOA panel after figure creation
        if ENABLE_STATUS_PANELS:
            self.status_panel = StatusPanelManager(self.ax)
        
        if ENABLE_QAOA_PANEL:
            qaoa_ax = self.fig.add_axes([0.02, 0.02, 0.25, 0.25])  # Bottom left
            self.qaoa_panel = QAOAAnimationPanel(qaoa_ax, num_starts=5)
        
        if ENABLE_METRIC_INSETS:
            self.metric_insets = MetricInsetManager(self.fig)
            # Add cost variance inset
            self.metric_insets.add_inset('cost_variance', [0.75, 0.02, 0.22, 0.15],
                                       'Frame', 'Variance', 'Cost Variance Over Time')
        
        # Draw static elements
        self.draw_static_elements()

        # Emoji-based vehicle artist (high zorder for visibility)
        start_pos = self.vehicles[0].path[0] if self.vehicles and self.vehicles[0].path else self.depot
        # keep a persistent artist so blit works; large emoji for visibility
        self.vehicle_artist = self.ax.text(start_pos[0], start_pos[1], "🚗", fontsize=22,
                                           ha="center", va="center", zorder=10, color='#1b5e20',
                                           fontweight='bold')
        # Obstacle line artist
        self.block_line, = self.ax.plot([], [], color='red', linewidth=4, zorder=6, label='Blocked road')
        # Persistent status text
        self.status_artist = self.ax.text(5, GRID_SIZE - 5, "", fontsize=10,
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ------------------------------------------------------------------ #
    # Helpers for realism (time windows, vehicles, roads)
    # ------------------------------------------------------------------ #
    def _generate_time_windows(self, n_customers):
        """Create simple time windows for visualization."""
        slots = [(9, 11), (10, 12), (11, 13), (12, 14)]
        return [slots[i % len(slots)] for i in range(n_customers)]

    def _build_vehicle_states(self):
        """Create a single vehicle (main + alternate route fallback)."""
        base_routes = []
        # Use current_route which has been adjusted for destination
        if self.current_route:
            base_routes.append(self.current_route)
        elif self.main_route:
            base_routes.append(self.main_route)
            
        base_routes.extend(self.alternate_routes)
        if not base_routes:
            # Ensure all customers are included in default route
            all_customers = list(range(1, len(self.points)))
            base_routes = [[0] + all_customers + [0]]

        primary = self._normalize_route(base_routes[0], self.destination_idx)
        primary = self._ensure_clusters_before_destination(primary)
        
        # Ensure route visits all customers from all clusters (add missing ones if needed)
        expected_customers = set(range(1, len(self.customers) + 1))
        visited_customers = set([n for n in primary if n != 0 and n < len(self.points)])
        missing_customers = expected_customers - visited_customers
        
        # Also ensure we visit at least one customer from each cluster if clusters exist
        if self.cluster_labels is not None and len(self.cluster_labels) > 0:
            cluster_customer_map = defaultdict(list)
            for i, cluster_id in enumerate(self.cluster_labels):
                customer_idx = i + 1  # Customer indices start from 1 (0 is depot)
                if customer_idx < len(self.points):
                    cluster_customer_map[cluster_id].append(customer_idx)
            
            # Check each cluster has at least one visited customer
            for cluster_id, customers_in_cluster in cluster_customer_map.items():
                cluster_visited = any(c in visited_customers for c in customers_in_cluster)
                if not cluster_visited and customers_in_cluster:
                    # Add the first customer from this unvisited cluster
                    missing_customers.add(customers_in_cluster[0])
        
        if missing_customers:
            # Insert missing customers using nearest insertion
            for missing in missing_customers:
                best_pos = 1
                best_cost_increase = float('inf')
                for i in range(1, len(primary)):
                    # Simplified cost calculation
                    if i < len(primary):
                        cost_inc = 0  # Will be handled by route optimization
                        if cost_inc < best_cost_increase:
                            best_cost_increase = cost_inc
                            best_pos = i
                primary.insert(best_pos, missing)
        
        alt = self._normalize_route(base_routes[1], self.destination_idx) if len(base_routes) > 1 else primary
        alt = self._ensure_clusters_before_destination(alt)

        v = VehicleState("Vehicle 1", primary, '#228B22', alt_route=alt)
        v.path = self._expand_route_to_roads(v.route_nodes)
        # If the expanded path is too short, create a direct depot->destination leg
        if len(v.path) < 2 and len(v.route_nodes) > 1:
            start_pt = self.points[v.route_nodes[0]]
            end_pt = self.points[v.route_nodes[-1]]
            v.path = self._dedup_waypoints(self._manhattan_path(start_pt, end_pt))
        v.dense_path = self._densify_path(v.path)
        v.pos = np.array(v.dense_path[0]) if v.dense_path else self.depot.copy()
        v.trail = []
        v.visited_customers = set()  # Track visited customers for visualization
        return [v]

    def _expand_route_to_roads(self, route_nodes):
        """Convert node list to Manhattan road waypoints (no diagonal)."""
        waypoints = []
        for i in range(len(route_nodes) - 1):
            p1 = self.points[route_nodes[i]]
            p2 = self.points[route_nodes[i + 1]]
            segment = self._manhattan_path(p1, p2)
            if not waypoints:
                waypoints.extend(segment)
            else:
                # avoid duplicate join point
                waypoints.extend(segment[1:])
        return self._dedup_waypoints(waypoints)

    def _manhattan_path(self, p1, p2):
        """Return orthogonal polyline between two points."""
        mid = np.array([p2[0], p1[1]])
        return [p1.copy(), mid, p2.copy()]

    def _dedup_waypoints(self, pts):
        """Remove consecutive duplicate waypoints to prevent zero-length segments."""
        if not pts:
            return []
        deduped = [pts[0]]
        for pt in pts[1:]:
            if not np.allclose(pt, deduped[-1]):
                deduped.append(pt)
        # If path collapses to a single point, duplicate to allow movement step
        if len(deduped) == 1:
            deduped.append(deduped[0].copy())
        return deduped

    def _densify_path(self, waypoints, step=1.0):
        """Create a dense polyline so the vehicle moves smoothly along the route."""
        dense = []
        if not waypoints:
            return dense
        pts = self._dedup_waypoints(waypoints)
        dense.append(np.array(pts[0]))
        for i in range(len(pts) - 1):
            p1 = np.array(pts[i])
            p2 = np.array(pts[i + 1])
            vec = p2 - p1
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                continue
            n_steps = max(1, int(dist / step))
            for k in range(1, n_steps + 1):
                dense.append(p1 + vec * (k / n_steps))
        return dense

    def _direction_to_arduino_cmd(self, prev_dir, new_dir):
        """
        ARDUINO INTEGRATION: Translate 2D movement into 'L', 'R', or 'F'.
        Arduino acts only as an execution layer for routing decisions.
        """
        if prev_dir is None:
            return "F"
        
        # Cross product to determine turn direction
        cp = prev_dir[0] * new_dir[1] - prev_dir[1] * new_dir[0]
        # Dot product for the angle (cosine)
        dp = np.dot(prev_dir, new_dir)
        
        # If angle change > 70 degrees (cos < 0.34), signal a turn.
        # This prevents jitter on minor corrections and only signals real 90-deg turns.
        if dp < 0.34:
            return "L" if cp > 0 else "R"
        return "F"
    
    def _init_junctions(self):
        """Initialize junction detection on the road grid."""
        self.junction_nodes = []
        self.junction_degrees = {}
        road_spacing = 10
        
        # Build road network graph
        road_nodes = {}
        road_adj = defaultdict(set)
        
        # Create nodes at road intersections
        for x in range(0, GRID_SIZE + 1, road_spacing):
            for y in range(0, GRID_SIZE + 1, road_spacing):
                node = (float(x), float(y))
                road_nodes[node] = node
                
                # Connect horizontal neighbors
                if x + road_spacing <= GRID_SIZE:
                    nbr = (float(x + road_spacing), float(y))
                    road_adj[node].add(nbr)
                    road_adj[nbr].add(node)
                
                # Connect vertical neighbors
                if y + road_spacing <= GRID_SIZE:
                    nbr = (float(x), float(y + road_spacing))
                    road_adj[node].add(nbr)
                    road_adj[nbr].add(node)
        
        # Detect junctions (nodes with degree >= 3)
        for node, neighbors in road_adj.items():
            degree = len(neighbors)
            self.junction_degrees[node] = degree
            if degree >= 3:
                self.junction_nodes.append(node)
        
    # ------------------------------------------------------------------ #
    def _draw_traffic_cone(self, pos, alpha_val=1.0, zorder=20):
        """Draw a traffic cone obstacle icon."""
        x, y = pos[0], pos[1]
        cone_height = 3.0
        cone_base = 2.0
        
        # Draw cone body (orange triangle)
        cone_points = [
            [x, y],  # Top point
            [x - cone_base/2, y - cone_height],  # Bottom left
            [x + cone_base/2, y - cone_height]   # Bottom right
        ]
        cone_body = Polygon(cone_points, color='#FF8C00', edgecolor='#FF4500', 
                          linewidth=1.5, zorder=zorder, alpha=alpha_val*0.9)
        self.ax.add_patch(cone_body)
        
        # Draw white stripes on cone
        for i in range(1, 4):
            stripe_y = y - (cone_height * i / 4)
            stripe_width = cone_base * (1 - i * 0.15)
            stripe = Rectangle((x - stripe_width/2, stripe_y - 0.2), stripe_width, 0.4,
                              color='white', zorder=zorder+1, alpha=alpha_val*0.8)
            self.ax.add_patch(stripe)
        
        # Draw base circle
        base_circle = Circle((x, y - cone_height), cone_base/2, color='#FF4500', 
                            edgecolor='#FF0000', linewidth=1, zorder=zorder, alpha=alpha_val*0.9)
        self.ax.add_patch(base_circle)
    
    def _draw_delivery_vehicle(self, pos, angle, zorder=15):
        """Draw realistic delivery vehicle (truck) with direction."""
        x, y = pos[0], pos[1]
        size = 2.5
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Truck body
        body_corners = [
            [-size*1.2, -size*0.7],
            [size*0.8, -size*0.7],
            [size*0.8, size*0.7],
            [-size*1.2, size*0.7]
        ]
        
        rotated_body = []
        for cx, cy in body_corners:
            rx = cx * cos_a - cy * sin_a + x
            ry = cx * sin_a + cy * cos_a + y
            rotated_body.append([rx, ry])
        
        truck_body = Polygon(rotated_body, color='#FFD700', edgecolor='#FF8C00', 
                            linewidth=2, zorder=zorder, alpha=0.95)
        self.ax.add_patch(truck_body)
        
        # Truck cabin
        cabin_corners = [
            [-size*1.2, -size*0.7],
            [-size*0.4, -size*0.7],
            [-size*0.4, size*0.7],
            [-size*1.2, size*0.7]
        ]
        
        rotated_cabin = []
        for cx, cy in cabin_corners:
            rx = cx * cos_a - cy * sin_a + x
            ry = cx * sin_a + cy * cos_a + y
            rotated_cabin.append([rx, ry])
        
        cabin = Polygon(rotated_cabin, color='#FF8C00', edgecolor='#FF4500', 
                       linewidth=2, zorder=zorder+1, alpha=0.95)
        self.ax.add_patch(cabin)
        
        # Wheels
        wheel_positions = [
            [-size*0.7, -size*0.9],
            [-size*0.7, size*0.9],
            [size*0.3, -size*0.9],
            [size*0.3, size*0.9]
        ]
        
        for wx, wy in wheel_positions:
            rx = wx * cos_a - wy * sin_a + x
            ry = wx * sin_a + wy * cos_a + y
            wheel = Circle((rx, ry), size*0.25, color='#1C1C1C', zorder=zorder+2, 
                          edgecolor='#000000', linewidth=1)
            self.ax.add_patch(wheel)

    def draw_static_elements(self):
        """Draw depot, customers, clusters, and routes"""
        # Subtle background grid to give spatial context (light, unobtrusive).
        road_spacing = 10
        for x in range(0, GRID_SIZE + 1, road_spacing):
            self.ax.plot(
                [x, x],
                [0, GRID_SIZE],
                color='#e0e0e0',
                linewidth=0.5,
                zorder=0,
                alpha=0.4,
            )
        for y in range(0, GRID_SIZE + 1, road_spacing):
            self.ax.plot(
                [0, GRID_SIZE],
                [y, y],
                color='#e0e0e0',
                linewidth=0.5,
                zorder=0,
                alpha=0.4,
            )

        # Draw clusters with different colors - visible dashed boundaries
        if self.cluster_labels is not None and self.cluster_centers is not None:
            num_clusters = len(self.cluster_centers)
            
            # Cluster colors matching demo style
            demo_cluster_colors = ['#FFD700', '#00FF00', '#FF0000', '#00BFFF', '#FF00FF', '#FFA500']
            
            # Draw cluster boundaries (convex hull approximation with dashed lines)
            for cluster_id in range(num_clusters):
                cluster_mask = self.cluster_labels == cluster_id
                cluster_customers = self.customers[cluster_mask]
                
                if len(cluster_customers) > 0:
                    cluster_color = demo_cluster_colors[cluster_id % len(demo_cluster_colors)]
                    
                    # Draw cluster boundary using convex hull
                    try:
                        if len(cluster_customers) >= 3:
                            hull = ConvexHull(cluster_customers)
                            boundary_points = cluster_customers[hull.vertices]
                            boundary_points = np.vstack([boundary_points, boundary_points[0]])  # Close the loop
                            
                            # Draw dashed boundary line
                            for i in range(len(boundary_points) - 1):
                                self.ax.plot([boundary_points[i][0], boundary_points[i+1][0]], 
                                           [boundary_points[i][1], boundary_points[i+1][1]], 
                                           color=cluster_color, linestyle='--', linewidth=2.5, 
                                           alpha=0.7, zorder=1.5, dashes=(8, 4))
                    except:
                        # Fallback: draw rectangle around cluster
                        if len(cluster_customers) > 0:
                            min_x, min_y = cluster_customers.min(axis=0) - 5
                            max_x, max_y = cluster_customers.max(axis=0) + 5
                            # Draw dashed rectangle
                            rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                           fill=False, edgecolor=cluster_color, linestyle='--',
                                           linewidth=2.5, alpha=0.7, zorder=1.5)
                            self.ax.add_patch(rect)
                    
                    # Draw cluster label
                    center = self.cluster_centers[cluster_id]
                    self.ax.text(center[0], center[1] - 8, f'Cluster {cluster_id+1}', 
                               ha='center', fontsize=10, fontweight='bold', 
                               color=cluster_color, zorder=2,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       alpha=0.8, edgecolor=cluster_color, linewidth=2))
        
        # Draw depot - realistic warehouse building
        # Reduced depot size for realism
        depot_size = 3.5
        # Warehouse building
        warehouse = Rectangle((self.depot[0] - depot_size, self.depot[1] - depot_size/2), 
                             depot_size*2, depot_size*1.2, 
                             color='#4169E1', zorder=5, 
                             edgecolor='#1E3A8A', linewidth=1.5, alpha=0.9)
        self.ax.add_patch(warehouse)
        # Warehouse roof
        warehouse_roof = Polygon([(self.depot[0] - depot_size, self.depot[1] + depot_size*0.6),
                                (self.depot[0], self.depot[1] + depot_size*1.2),
                                (self.depot[0] + depot_size, self.depot[1] + depot_size*0.6)],
                               color='#8B0000', zorder=6, edgecolor='#1E3A8A', linewidth=1.5, alpha=0.9)
        self.ax.add_patch(warehouse_roof)
        # Clear DEPOT label - smaller
        self.ax.text(self.depot[0], self.depot[1] + 5.5, 'DEPOT (START)', 
                    ha='center', fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1E3A8A', alpha=0.95, edgecolor='white', linewidth=1))
        
        # Draw destination if specified - reduced size
        if self.end_point is not None:
            dest_size = 3.0
            dest_building = Rectangle((self.end_point[0] - dest_size, self.end_point[1] - dest_size/2), 
                                    dest_size*2, dest_size*1.2, 
                                    color='#DC143C', zorder=5, 
                                    edgecolor='#8B0000', linewidth=1.5, alpha=0.9)
            self.ax.add_patch(dest_building)
            dest_roof = Polygon([(self.end_point[0] - dest_size, self.end_point[1] + dest_size*0.6),
                                (self.end_point[0], self.end_point[1] + dest_size*1.2),
                                (self.end_point[0] + dest_size, self.end_point[1] + dest_size*0.6)],
                               color='#8B0000', zorder=6, edgecolor='#8B0000', linewidth=1.5, alpha=0.9)
            self.ax.add_patch(dest_roof)
            self.ax.text(self.end_point[0], self.end_point[1] + 5, 'DESTINATION (END)', 
                        ha='center', fontsize=9, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#8B0000', alpha=0.95, edgecolor='white', linewidth=1))
        
        # Draw customers - realistic delivery points with visit status
        for i, customer in enumerate(self.customers):
            customer_idx = i + 1  # Customer index in points array (1-based)
            
            # Check if customer has been visited
            visited = False
            if self.vehicles and hasattr(self.vehicles[0], 'visited_customers'):
                visited = customer_idx in self.vehicles[0].visited_customers
            
            if self.cluster_labels is not None and i < len(self.cluster_labels):
                cluster_id = self.cluster_labels[i]
                customer_color = self.cluster_colors[cluster_id % len(self.cluster_colors)]
            else:
                customer_color = '#8B4513'  # Brown for unclustered
            
            # Visited customers get green highlight
            if visited:
                customer_color = '#00AA00'  # Green for visited
                edge_color = '#006600'
                alpha_val = 1.0
            else:
                edge_color = '#333333'
                alpha_val = 0.8
            
            # Draw customer as yellow circle with C and number (matching demo style)
            customer_circle = Circle(customer, 2.5, color='#FFD700', 
                                    edgecolor='#FFA500', linewidth=2, zorder=4, alpha=0.9)
            self.ax.add_patch(customer_circle)
            
            # Customer label: C and number inside circle
            label_text = f'C{i+1}'
            if visited:
                label_text = f'✓\nC{i+1}'
            self.ax.text(customer[0], customer[1], label_text, 
                        ha='center', va='center', fontsize=9, fontweight='bold', 
                        color='#000000', zorder=5)
            # Time window label
            tw = self.time_windows[i]
            self.ax.text(customer[0], customer[1] - 6, f'{tw[0]:02d}:00-{tw[1]:02d}:00',
                        ha='center', fontsize=7, color='#0D47A1',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#e8f0fe', alpha=0.8, edgecolor='#90caf9', linewidth=0.6))
        
        # Draw main/alternate road polylines using expanded paths of vehicles
        if self.vehicles:
            main_path = self.vehicles[0].path
            if main_path:
                mx = [p[0] for p in main_path]
                my = [p[1] for p in main_path]
                
                # Determine color and label based on state
                route_color = '#2E7D32' if not getattr(self, 'switched', False) else '#FF8C00' # Green or Dark Orange
                route_label = 'Route 1 (Main)' if not getattr(self, 'switched', False) else 'Active Route (Switched)'
                route_style = '-' if not getattr(self, 'switched', False) else '--'
                
                # GAP-5: Pulse effect on active route for visual prominence (QAOA-selected route)
                base_width = 5.0 + 2.0 * math.sin(getattr(self, 'current_frame', 0) * 0.15)
                # Confidence-aware scaling (QAOA confidence affects route thickness)
                confidence_factor = 1.0
                if getattr(self, 'enable_confidence_visuals', False) and self.confidence_score is not None:
                    # QAOA confidence directly affects route prominence
                    confidence_factor = 0.8 + 0.6 * float(self.confidence_score)
                pulse_width = base_width * confidence_factor
                # Highlight QAOA-selected route with bright, pulsing line
                plot_kwargs = dict(
                    color=route_color,
                    linewidth=pulse_width,
                    zorder=1.5,
                    label=f'QAOA Route (Conf: {self.confidence_score*100:.0f}%)' if self.confidence_score else route_label,
                    linestyle=route_style,
                    alpha=0.9,
                )
                if route_style == '--':
                    # Only supply explicit dash pattern for dashed style
                    plot_kwargs["dashes"] = (6, 3)
                self.ax.plot(mx, my, **plot_kwargs)
                
                # Draw arrows
                for i in range(0, len(main_path) - 1, max(1, len(main_path)//10)):
                    p1, p2 = main_path[i], main_path[i+1]
                    arrow = FancyArrowPatch((p1[0], p1[1]), (p2[0], p2[1]),
                                            arrowstyle='->', mutation_scale=12,
                                            color=route_color, linewidth=1.5, alpha=0.9, zorder=2)
                    self.ax.add_patch(arrow)
        
        # Draw alternate routes when obstacle is detected or recently rerouted
        if self.obstacle_detected or getattr(self, 'reroute_active', False):
            # Use cached alt_routes_ui if available to keep visualization stable
            valid_alts = self.alt_routes_ui if self.alt_routes_ui else self._compute_valid_alternate_routes()
            if valid_alts:
                colors = ['#1565C0', '#D2691E', '#800080', '#008080']
                for idx, (original_i, alt_route, cost) in enumerate(valid_alts):
                    # Convert node route to path
                    alt_path_nodes = self._normalize_route(alt_route, self.destination_idx)
                    alt_path_coords = self._expand_route_to_roads(alt_path_nodes)
                    
                    if alt_path_coords:
                        axp = [p[0] for p in alt_path_coords]
                        ayp = [p[1] for p in alt_path_coords]
                        
                        # Highlight the chosen route if we have already made a selection
                        is_selected = (
                            hasattr(self, "selected_alternate_index")
                            and self.selected_alternate_index is not None
                            and idx == self.selected_alternate_index
                        )
                        linewidth = 6.0 if is_selected else 3.0
                        alpha = 1.0 if is_selected else 0.5
                        route_color = colors[idx % len(colors)]
                        
                        self.ax.plot(
                            axp,
                            ayp,
                            color=route_color,
                            linewidth=linewidth,
                            zorder=1.8 if is_selected else 1.4,
                            linestyle='--',
                            dashes=(4, 4),
                            alpha=alpha,
                            label=f"Alt {original_i+1} (Cost: {cost:.1f})" if idx < 3 else None
                        )
        
        # Create legend with cluster info
        legend_elements = []
        if self.cluster_labels is not None:
            num_clusters = len(set(self.cluster_labels))
            for i in range(num_clusters):
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor=self.cluster_colors[i % len(self.cluster_colors)], 
                                           label=f'Cluster {i+1}'))
        
        route_color = '#2E7D32' if not getattr(self, 'switched', False) else '#FF8C00'
        route_label = 'Main Route' if not getattr(self, 'switched', False) else 'Active Route'
        
        legend_elements.extend([
            plt.Line2D([0], [0], linestyle='-', color=route_color, linewidth=3, label=route_label),
            plt.Line2D([0], [0], linestyle='--', color='#696969', linewidth=2, label='Alternate Route'),
            plt.Line2D([0], [0], linestyle='-', color='#4169E1', linewidth=2, label='Vehicle Path')
        ])
        
        if self.obstacle_detected:
             legend_elements.append(plt.Line2D([0], [0], linestyle='-', color='red', linewidth=4, label='Blocked Road'))
        
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    def simulate_obstacle(self, frame):
        """Simulate obstacle appearing after some time"""
        # Obstacle appears after 30 frames
        if frame == 30 and not self.obstacle_detected:
            # Find an edge in the current route to block
            if len(self.current_route) > 2:
                edge_start = random.randint(0, len(self.current_route) - 2)
                self.obstacle_edge = (self.current_route[edge_start], self.current_route[edge_start + 1])
                self.obstacle_detected = True
                self.paused_for_selection = True
                
                print(f"\n" + "!"*70)
                print("OBSTACLE DETECTED!")
                print("!"*70)
                print(f"Blocked edge: {self.obstacle_edge}")
                print("\nMultiple alternate routes available:")
                
                # Show available alternate routes
                self.show_alternate_routes()
                
                # Ask user to choose
                self.ask_user_route_selection()
    
    def show_alternate_routes(self):
        """Display available alternate routes to user"""
        print("\n" + "-"*70)
        print("AVAILABLE ALTERNATE ROUTES:")
        print("-"*70)
        valid_routes = self._compute_valid_alternate_routes()
        for i, alt_route, route_cost in valid_routes:
            route_str = " -> ".join([f"Node {node}" for node in alt_route])
            print(f"  Route {i+1}: Cost = {route_cost:.2f}")
            print(f"    Path: {route_str}")
        
        print("-"*70)
        return valid_routes
    
    def ask_user_route_selection(self):
        """Ask user to select which route to take"""
        # Kept for compatibility, but routing is now automatic.
        valid_routes = self._compute_valid_alternate_routes()
        
        if not valid_routes:
            print("\n❌ No valid alternate routes found!")
            return
        
        print(f"\nSelect route to continue (1-{len(valid_routes)}):")
        while True:
            try:
                choice = input(f"Enter route number (1-{len(valid_routes)}): ").strip()
                route_num = int(choice)
                if 1 <= route_num <= len(valid_routes):
                    selected_idx, selected_route, selected_cost = valid_routes[route_num - 1]
                    self.user_selected_route = selected_route
                    self.paused_for_selection = False
                    
                    # Find closest point in selected route to current position
                    if self.vehicles:
                        current_pos = self.vehicles[0].pos.copy()
                        min_dist = float('inf')
                        best_idx = 0
                        for i, node in enumerate(selected_route):
                            node_pos = self.points[node]
                            dist = np.linalg.norm(node_pos - current_pos)
                            if dist < min_dist:
                                min_dist = dist
                                best_idx = i
                        
                        self.current_route = selected_route
                        self.current_route_idx = best_idx
                        self.switched = True
                    break
                else:
                    pass
            except ValueError:
                pass
    
    def switch_to_alternate(self):
        """Switch to user-selected alternate route"""
        if self.user_selected_route is not None:
            self.current_route = self.user_selected_route
            self.switched = True
    
    def handle_obstacle_interaction(self):
        """Handle user interaction when obstacle is detected - QAOA Manual Choice"""
        # Pause animation to prevent UI freeze and serial flooding
        if hasattr(self, 'anim') and self.anim:
            self.anim.event_source.stop()
            
        # --- STOP PHYSICAL ROBOT IMMEDIATELY ---
        arduino_send('X') 
        time.sleep(0.5)
        
        # Ensure we have valid routes to show
        valid_routes = self._compute_valid_alternate_routes()
        self.alt_routes_ui = valid_routes
        
        if not valid_routes:
            print("\n❌ No alternate routes available that avoid this obstacle.")
            if hasattr(self, 'anim') and self.anim:
                self.anim.event_source.start()
            return

        print("\n" + "!"*50)
        print("   OBSTACLE DETECTED: QAOA REROUTING REQUIRED")
        print("!"*50)
        print(f"Blocked Edge: {self.obstacle_edge}")
        print("\nAvailable Alternate Routes:")
        for idx, (i, alt_route, cost) in enumerate(valid_routes):
            print(f"  [{idx}] Route {i+1}: Cost = {cost:.2f}")
        
        print("\n>>> PRESS KEY [0-9] on keyboard to select route...")
        
        choice = -1
        start_time = time.time()
        # Wait for key press (manual selection)
        while choice == -1:
            if msvcrt.kbhit():
                try:
                    char = msvcrt.getch().decode('utf-8')
                    val = int(char)
                    if 0 <= val < len(valid_routes):
                        choice = val
                except:
                    pass
            
            # Optional: auto-timeout after 10 seconds to best route
            if time.time() - start_time > 10.0:
                print("\n[TIMEOUT] Auto-selecting most efficient alternate route...")
                choice = 0
                break
            
            time.sleep(0.1)
            
        best_idx, best_route, best_cost = valid_routes[choice]
        self.selected_alternate_index = choice
        self.active_alternate_route = best_route
        self.active_alternate_cost = float(best_cost)
        
        print(f"\n[QAOA] Selected Route {best_idx+1}. Transmitting to Robot...")
        
        # RE-ENABLE ROBOT AFTER SELECTION PAUSE
        arduino_send('S') 
        time.sleep(0.2)
        
        self.switched = True
        self.obstacle_detected = False
        self.obstacle_segment = None
        if self.vehicles:
            self.apply_route_update(self.vehicles[0], best_route)
            
        # Resume animation
        if hasattr(self, 'anim') and self.anim:
            self.anim.event_source.start()
            
    def apply_route_update(self, vehicle, new_route_nodes):
        """Update vehicle path to follow new route from current position"""
        # ANIMATION: Start route transition if enabled
        if ENABLE_ROUTE_TRANSITIONS and hasattr(vehicle, 'route_nodes'):
            old_route = vehicle.route_nodes.copy()
            if self.junction_events:
                self.junction_events.start_route_transition(old_route, new_route_nodes)
            self.route_transition_active = True
            self.transition_old_route = old_route
            self.transition_new_route = new_route_nodes
            self.transition_progress = 0.0
        
        # First, normalize and enforce cluster coverage before destination
        new_route = self._normalize_route(new_route_nodes, self.destination_idx)
        new_route = self._ensure_clusters_before_destination(new_route)

        # Find closest node in new route
        current_pos = vehicle.pos
        best_idx = 0
        min_dist = float('inf')
        
        for i, node in enumerate(new_route):
            node_pos = self.points[node]
            dist = np.linalg.norm(node_pos - current_pos)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        # Construct new path: current_pos -> closest_node -> rest_of_route
        remaining_nodes = new_route[best_idx:]
        
        # If remaining nodes is empty (e.g. at end), just go to destination
        if not remaining_nodes:
            if self.destination_idx is not None:
                remaining_nodes = [self.destination_idx]
            else:
                remaining_nodes = [0]

        # 1. Path from current_pos to first node of remaining_nodes
        first_target_node = remaining_nodes[0]
        first_target_pos = self.points[first_target_node]
        
        connection_path = self._manhattan_path(current_pos, first_target_pos)
        
        # 2. Path for the rest of the route
        route_path = self._expand_route_to_roads(remaining_nodes)
        
        # Combine
        full_path = connection_path + route_path
            
        vehicle.route_nodes = new_route
        vehicle.path = self._dedup_waypoints(full_path)
        vehicle.dense_path = self._densify_path(vehicle.path)
        vehicle.segment_index = 0
        vehicle.switch_logged = True
        vehicle.done = False  # Reset done status
        
        # ARDUINO INTEGRATION: Resume movement with new path
        arduino_send('F')
    
    def _compute_valid_alternate_routes(self):
        """
        Compute alternate routes that avoid the currently blocked edge and
        return them with their total distance cost.
        """
        valid_routes = []
        if not self.alternate_routes or self.obstacle_edge is None:
            return valid_routes
        
        for i, alt_route in enumerate(self.alternate_routes):
            avoids_obstacle = True
            for j in range(len(alt_route) - 1):
                edge = (alt_route[j], alt_route[j + 1])
                if edge == self.obstacle_edge or edge == tuple(reversed(self.obstacle_edge)):
                    avoids_obstacle = False
                    break
            if not avoids_obstacle:
                continue
            
            # Calculate route cost
            route_cost = 0.0
            for k in range(len(alt_route) - 1):
                route_cost += self.dist_matrix[alt_route[k], alt_route[k + 1]]
            valid_routes.append((i, alt_route, route_cost))
        
        return valid_routes

    def animate(self, frame):
        """Animation function with smooth multi-vehicle motion and obstacle switching."""
        self.current_frame = frame
        
        # ANIMATION: Update animation managers
        if self.obstacle_anim:
            self.obstacle_anim.update(frame)
        if self.junction_events:
            self.junction_events.update()
            if self.junction_events.transitioning:
                self.junction_events.update_transition()

        # ARDUINO INTEGRATION: poll Arduino feedback once per frame
        feedback = read_arduino_feedback()
        if feedback == "OBSTACLE":
            print(f"[ARDUINO] Physical Obstacle Detected! Triggering QAOA reroute...")
            self.obstacle_detected = True
            # Force show the obstacle in the simulation at the vehicle's current position
            if self.vehicles:
                v = self.vehicles[0]
                self.obstacle_edge = self._find_current_edge(v)
            self.handle_obstacle_interaction()
        elif feedback == "JUNCTION":
            print(f"[ARDUINO] Junction reached. Triggering logic...")
            self.handle_obstacle_interaction()


        if frame == 0 and self.vehicles:
            # ARDUINO INTEGRATION: Signal start to hardware
            arduino_send('S')
            
            # Reset per-run movement state to avoid stale indices
            v0 = self.vehicles[0]
            v0.segment_index = 0
            v0.done = False
            v0.last_sent_cmd = None
            v0.move_dir = None
            if v0.dense_path:
                v0.pos = np.array(v0.dense_path[0])
                v0.trail = [v0.pos.copy()]
                
        # Visual Obstacle Trigger (QAOA Predictive)
        if (not self.obstacle_detected) and self.vehicles:
            v = self.vehicles[0]
            if len(v.route_nodes) > 2 and v.pos is not None:
                # Find the NEXT edge the vehicle will encounter (ahead of current position)
                current_pos = v.pos
                
                # Find current segment index in dense_path
                current_seg_idx = v.segment_index
                if current_seg_idx < len(v.dense_path) - 1:
                    # Calculate distance to next few segments
                    lookahead_dist = 0
                    lookahead_idx = current_seg_idx
                    
                    # Look ahead only OBSTACLE_WARNING_DISTANCE units (right before vehicle)
                    while lookahead_idx < len(v.dense_path) - 1 and lookahead_dist < OBSTACLE_WARNING_DISTANCE:
                        seg_start = v.dense_path[lookahead_idx]
                        seg_end = v.dense_path[lookahead_idx + 1]
                        seg_len = np.linalg.norm(seg_end - seg_start)
                        lookahead_dist += seg_len
                        lookahead_idx += 1
                    
                    # Trigger obstacle only when vehicle is very close (right before it)
                    if lookahead_dist >= OBSTACLE_WARNING_DISTANCE * 0.3 and frame >= self.obstacle_frame:
                        # Find which route edge corresponds to this path segment
                        # Map dense_path segment back to route_nodes edge
                        best_edge_idx = 0
                        min_dist_to_edge = float('inf')
                        
                        for i in range(len(v.route_nodes) - 1):
                            edge_start = self.points[v.route_nodes[i]]
                            edge_end = self.points[v.route_nodes[i + 1]]
                            
                            # Check distance from current position to this edge
                            # Use point-to-line-segment distance
                            edge_vec = edge_end - edge_start
                            edge_len_sq = np.dot(edge_vec, edge_vec)
                            
                            if edge_len_sq > 1e-6:
                                t = max(0, min(1, np.dot(current_pos - edge_start, edge_vec) / edge_len_sq))
                                closest_on_edge = edge_start + t * edge_vec
                                dist_to_edge = np.linalg.norm(current_pos - closest_on_edge)
                                
                                # Prefer edges right before vehicle (very close, in direction of travel)
                                if dist_to_edge < min_dist_to_edge and t > 0.1:  # Must be ahead, but very close
                                    min_dist_to_edge = dist_to_edge
                                    best_edge_idx = i
                        
                        if best_edge_idx < len(v.route_nodes) - 1:
                            node_a = v.route_nodes[best_edge_idx]
                            node_b = v.route_nodes[best_edge_idx + 1]
                            self.obstacle_edge = (node_a, node_b)
                            
                            # Set obstacle_segment for visualization
                            p1 = self.points[node_a]
                            p2 = self.points[node_b]
                            segment_path = self._manhattan_path(p1, p2)
                            
                            if len(segment_path) >= 2:
                                self.obstacle_segment = (segment_path[0], segment_path[1])
                            else:
                                self.obstacle_segment = (p1, p2)

                            self.obstacle_detected = True
                            
                            # ANIMATION: Start obstacle appearance animation
                            if self.obstacle_anim:
                                self.obstacle_anim.start_appearance()
                            
                            print(f"\n⚠️  OBSTACLE DETECTED AHEAD! Blocking edge: {self.obstacle_edge}")
                            print(f"   Vehicle position: ({current_pos[0]:.1f}, {current_pos[1]:.1f})")
                            print(f"   Distance to obstacle: {min_dist_to_edge:.1f}")
                            # Compute valid alternate routes and show them on the UI
                            # (sorted by cost so the best option is listed first).
                            self.alt_routes_ui = sorted(
                                self._compute_valid_alternate_routes(),
                                key=lambda x: x[2],
                            )
                            if self.alt_routes_ui:
                                # Keep the note visible for the rest of the run.
                                self.show_alt_routes_ui = True
                                print("\nAvailable alternate routes (also shown on UI):")
                                for idx, (i, alt_route, route_cost) in enumerate(self.alt_routes_ui):
                                    route_str = " -> ".join([f"Node {n}" for n in alt_route])
                                    print(f"  Route {i+1}: Cost = {route_cost:.2f}")
                                    print(f"    Path: {route_str}")
                                
                                # Visualize the options on screen before auto-rerouting
                                self.ax.set_title("⚠️ QAOA Rerouting Analysis...", color='red', fontweight='bold')
                                
                                # Automatically select the best alternate route for visual obstacles
                                best_idx, best_route, best_cost = self.alt_routes_ui[0]
                                self.selected_alternate_index = 0
                                self.active_alternate_route = best_route
                                self.active_alternate_cost = float(best_cost)
                                self.switched = True
                                self.apply_route_update(v, best_route)
                                print(f"[QAOA] Visual obstacle bypassed. Rerouted to Route {best_idx+1}.")
                                
                                # Pause vehicle for a moment visually without blocking simulation loop
                                v.pause_frames = 15
                                
                                # Keep visuals (cones/dashed lines) on-screen until the vehicle passes it.
                                self.reroute_active = True
                                self.reroute_timer = 50  # Keep visible for longer
                                
                                # Immediately tell Arduino to move on the new route
                                arduino_send('S') 
                                
        # Reroute visualization timer
        if hasattr(self, 'reroute_active') and self.reroute_active:
            if self.reroute_timer > 0:
                self.reroute_timer -= 1
            else:
                self.reroute_active = False
                self.obstacle_detected = False
                self.obstacle_segment = None
                self.alt_routes_ui = None
                self.ax.set_title('Delivery Route Optimization - Quantum VRP', fontsize=14, fontweight='bold')

        # Base step size for realistic but fast motion
        base_step = VEHICLE_SPEED * 3.0  # Much faster movement

        # Move single vehicle along its dense path
        for v in self.vehicles:
            if v.done or len(v.dense_path) < 2:
                continue

            # Ensure starting position is defined
            if v.pos is None:
                v.pos = np.array(v.dense_path[0])

            # Advance along current segment
            if v.segment_index >= len(v.dense_path) - 1:
                if not v.done:
                    v.done = True
                    print(f"\n[LOG] {v.name} has arrived at the destination!")
                continue

            # Default: fast movement
            if v.pause_frames > 0:
                v.pause_frames -= 1
                step_size = base_step * 0.5
            else:
                step_size = base_step

            target = np.array(v.dense_path[v.segment_index + 1])
            direction = target - v.pos
            dist = np.linalg.norm(direction)
            # If we somehow got a zero-length segment, skip it to avoid freezing
            if dist < 1e-6:
                v.segment_index += 1
                continue
            # Keep previous unit direction for Arduino steering decisions
            prev_dir = getattr(v, "move_dir", None)

            if dist <= step_size:
                v.pos = target
                v.segment_index += 1
            else:
                v.pos = v.pos + (direction / dist) * step_size

            # ARDUINO SYNC: Determine motion command (L/R/F) based on current movement direction
            if dist > 1e-6:
                new_dir = direction / dist
                prev_dir = getattr(v, "move_dir", None)
                cmd = self._direction_to_arduino_cmd(prev_dir, new_dir)
                
                # Only send if the command has changed to avoid serial flooding
                last_cmd = getattr(v, "last_sent_cmd", None)
                if cmd != last_cmd and not self.obstacle_detected:
                    arduino_send(cmd)
                    v.last_sent_cmd = cmd
                v.move_dir = new_dir
            elif not self.obstacle_detected:
                arduino_send("F")
                v.last_sent_cmd = "F"
            
            # Store direction for vehicle rendering
            if dist > 1e-6:
                v.last_direction = direction / dist
            elif hasattr(v, 'last_direction'):
                pass  # Keep previous direction
            else:
                v.last_direction = np.array([1.0, 0.0])

            # Trail
            v.trail.append(v.pos.copy())
            if len(v.trail) > self.trail_length:
                v.trail.pop(0)

            # Track customer visits
            if hasattr(v, 'visited_customers'):
                for node_idx in v.route_nodes:
                    if node_idx > 0 and node_idx <= len(self.customers) and node_idx not in v.visited_customers:
                        customer_pos = self.points[node_idx]
                        dist_to_customer = np.linalg.norm(v.pos - customer_pos)
                        if dist_to_customer < 2.0:  # Close enough to customer
                            v.visited_customers.add(node_idx)
                            print(f"✓ Visited Customer {node_idx} at ({customer_pos[0]:.1f}, {customer_pos[1]:.1f})")

        # Redraw (clear then fully re-draw so the emoji stays on top)
        self.ax.clear()
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal')
        self.ax.set_title('Delivery Route Optimization - Quantum VRP', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X Coordinate', fontsize=10)
        self.ax.set_ylabel('Y Coordinate', fontsize=10)
        self.ax.grid(False)
        self.draw_static_elements()

        # ANIMATION: Draw obstacles as traffic cones (matching demo style)
        if self.obstacle_segment is not None:
            p1, p2 = self.obstacle_segment
            # Get animated alpha from obstacle animation manager
            if self.obstacle_anim:
                obs_alpha = self.obstacle_anim.get_alpha()
            else:
                obs_alpha = 1.0
            
            # Draw traffic cones along the obstacle segment
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            
            # Draw multiple cones along the blocked segment
            num_cones = 3
            for i in range(num_cones):
                t = (i + 1) / (num_cones + 1)
                cone_x = p1[0] + t * (p2[0] - p1[0])
                cone_y = p1[1] + t * (p2[1] - p1[1])
                self._draw_traffic_cone((cone_x, cone_y), alpha_val=obs_alpha, zorder=20)
            
            # Draw warning cloud/halo around obstacle area
            cloud_radius = 4.0
            cloud = Circle((mid_x, mid_y), cloud_radius, color='#FFA500', 
                         alpha=obs_alpha*0.3, zorder=19)
            self.ax.add_patch(cloud)
            
            # Add text annotation
            self.ax.text(mid_x, mid_y + 6, "OBSTACLE", color='#FF4500', fontsize=9, fontweight='bold', 
                        ha='center', zorder=21, alpha=obs_alpha,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=obs_alpha*0.9, edgecolor='#FF4500', linewidth=2))
            
            # Keep block_line for compatibility but make it invisible
            self.block_line, = self.ax.plot([], [], color='red', linewidth=0, zorder=0, label='Blocked road', alpha=0)
        else:
            self.block_line, = self.ax.plot([], [], color='red', linewidth=0, zorder=0, label='Blocked road', alpha=0)

        # Trails and vehicle rendering (high zorder)
        for v in self.vehicles:
            if len(v.trail) > 1:
                for i in range(len(v.trail)-1):
                    alpha = (i+1)/len(v.trail)*0.6
                    self.ax.plot([v.trail[i][0], v.trail[i+1][0]], [v.trail[i][1], v.trail[i+1][1]],
                                 color=v.color, linewidth=2.5, alpha=alpha, zorder=5)
        
        # Draw main delivery vehicle with direction
        if self.vehicles:
            v = self.vehicles[0]
            if hasattr(v, 'last_direction'):
                angle = math.atan2(v.last_direction[1], v.last_direction[0])
            else:
                angle = 0
            
            # Draw realistic delivery vehicle (truck icon with direction)
            self._draw_delivery_vehicle(v.pos, angle, zorder=15)
            
            # Also show emoji as backup
            self.vehicle_artist = self.ax.text(v.pos[0], v.pos[1],
                                               "🚚", fontsize=24, ha="center", va="center",
                                               zorder=16, color='#1b5e20', fontweight='bold')

        # Enhanced status with QAOA info
        status = "🚚 Delivery Vehicle Running"
        if self.enable_confidence_visuals and self.confidence_score is not None:
            status += f" | QAOA Confidence: {self.confidence_score*100:.0f}%"
        if self.obstacle_detected:
            status += " | ⛔ Obstacle Detected"
        if all(v.done for v in self.vehicles):
            status = "✅ All Deliveries Complete"
            # ARDUINO INTEGRATION START
            if not getattr(self, "arduino_stop_sent", False):
                arduino_send('X')
                self.arduino_stop_sent = True
            # ARDUINO INTEGRATION END
        
        # Show QAOA route indicator
        qaoa_info = ""
        if self.enable_confidence_visuals and self.confidence_score is not None:
            qaoa_info = f"QAOA Route Selected (Confidence: {self.confidence_score*100:.0f}%)"
        
        self.status_artist = self.ax.text(5, GRID_SIZE - 5, status, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        if qaoa_info:
            self.ax.text(5, GRID_SIZE - 12, qaoa_info, fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.9))
        
        # ANIMATION: Update status panel with current metrics
        if self.status_panel and self.vehicles:
            v = self.vehicles[0]
            current_cost = calculate_route_cost(v.route_nodes, self.dist_matrix) if hasattr(v, 'route_nodes') else 0
            metrics = {
                'current_cost': f"{current_cost:.1f}",
                'penalty': getattr(self, 'current_penalty', 'N/A'),
                'num_clusters': len(set(self.cluster_labels)) if self.cluster_labels is not None else 'N/A',
                'confidence': self.confidence_score if self.confidence_score else 0,
            }
            self.status_panel.update_panel(metrics)
        
        # ANIMATION: Update metric insets
        if self.metric_insets and self.vehicles:
            v = self.vehicles[0]
            if hasattr(v, 'route_nodes'):
                # Calculate cost variance (simplified - track route cost over time)
                route_cost = calculate_route_cost(v.route_nodes, self.dist_matrix)
                if not hasattr(self, 'cost_history'):
                    self.cost_history = []
                self.cost_history.append(route_cost)
                if len(self.cost_history) > 1:
                    cost_variance = np.var(self.cost_history[-20:]) if len(self.cost_history) >= 20 else np.var(self.cost_history)
                    self.metric_insets.update_metric('cost_variance', cost_variance)

        # If alternate routes are being presented (right after obstacle
        # detection), show a small overlay panel listing them with costs so
        # the user can see the options on the UI, not only in the terminal.
        if self.show_alt_routes_ui and self.alt_routes_ui:
            lines = ["Alternate Routes (cost):"]
            selected_i = getattr(self, "selected_alternate_index", None)
            # Show up to the first 3 alternates to avoid clutter
            for _, (i, _, route_cost) in enumerate(self.alt_routes_ui[:3]):
                prefix = "→ " if (selected_i is not None and i == selected_i) else "  "
                lines.append(f"{prefix}#{i+1}: {route_cost:.1f}")

            # Selected route note (persistent)
            chosen_route = getattr(self, "active_alternate_route", None)
            chosen_cost = getattr(self, "active_alternate_cost", None)
            if selected_i is not None:
                lines.append("")
                lines.append(f"Following: Route #{selected_i+1}")
                if chosen_cost is not None:
                    lines.append(f"Cost: {float(chosen_cost):.1f}")
                if chosen_route is not None:
                    path_preview = " -> ".join([f"{n}" for n in chosen_route[:6]])
                    if len(chosen_route) > 6:
                        path_preview += " -> ..."
                    lines.append(f"Path: {path_preview}")
            text = "\n".join(lines)
            # Bottom-left corner (persistent note)
            self.ax.text(
                5,
                5,
                text,
                fontsize=9,
                va="bottom",
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='white',
                    alpha=0.9,
                    edgecolor='#1565C0',
                    linewidth=1.2,
                ),
                zorder=30,
            )
        
        legend_elements = [
            plt.Line2D([0],[0], color='#2E7D32', linewidth=5, linestyle='-', 
                      label=f'QAOA Route (Main)' + (f' [{self.confidence_score*100:.0f}%]' if self.confidence_score else '')),
            plt.Line2D([0],[0], color='#1565C0', linewidth=3, linestyle='--', dashes=(6,3), label='Alternate Routes'),
            plt.Line2D([0],[0], marker='o', color='#FFD700', markersize=8, label='Customers'),
            plt.Line2D([0],[0], marker='^', color='#FF8C00', markersize=10, label='Obstacles'),
        ]
        if self.obstacle_detected:
            legend_elements.append(plt.Line2D([0],[0], color='red', linewidth=4, linestyle='-', label='Blocked Road'))
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # Return animated artists for FuncAnimation compatibility
        return [self.vehicle_artist, self.block_line, self.status_artist]

    def _find_current_edge(self, v):
        """Find the edge the vehicle is currently traversing."""
        if not v.route_nodes or v.pos is None:
            return None
            
        best_edge = None
        min_dist = float('inf')
        
        for i in range(len(v.route_nodes) - 1):
            n1 = v.route_nodes[i]
            n2 = v.route_nodes[i + 1]
            p1 = self.points[n1]
            p2 = self.points[n2]
            
            # Point-to-segment distance
            line_vec = p2 - p1
            pt_vec = v.pos - p1
            line_len = np.dot(line_vec, line_vec)
            if line_len < 1e-6:
                dist = np.linalg.norm(v.pos - p1)
            else:
                t = max(0, min(1, np.dot(pt_vec, line_vec) / line_len))
                projection = p1 + t * line_vec
                dist = np.linalg.norm(v.pos - projection)
            
            if dist < min_dist:
                min_dist = dist
                best_edge = (n1, n2)
        return best_edge

    def _segment_matches_obstacle(self, a, b, obs):
        o1, o2 = obs
        return (np.allclose(a, o1, atol=0.5) and np.allclose(b, o2, atol=0.5)) or \
               (np.allclose(a, o2, atol=0.5) and np.allclose(b, o1, atol=0.5))

    def _check_time_window(self, vehicle):
        """Log on-time vs delayed arrivals when close to a customer point."""
        for node in vehicle.route_nodes:
            if node == 0 or node in vehicle.visited_nodes or node >= len(self.points):
                continue
            pt = self.points[node]
            if np.linalg.norm(vehicle.pos - pt) < 1.2:
                vehicle.visited_nodes.add(node)
                arrival_hour = 9 + (getattr(self, "current_frame", 0) / 30.0)  # synthetic clock
                tw = self.time_windows[node - 1] if node - 1 < len(self.time_windows) else (9, 17)
                status = "on-time" if tw[0] <= arrival_hour <= tw[1] else "delayed"
                print(f"[LOG] {vehicle.name} reached Customer {node} at {arrival_hour:.1f}h ({status}) window {tw[0]}-{tw[1]}")

        # Source/Destination legend (top-left, below status)
        self.ax.text(5, GRID_SIZE - 15, f"Source (Depot): {self._fmt_point(self.depot)}", fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.85))
        dest_label = "Destination: Depot (round trip)" if self.end_point is None else f"Destination: {self._fmt_point(self.end_point)}"
        self.ax.text(5, GRID_SIZE - 25, dest_label, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='#fdecea', alpha=0.85))
        
        # Cluster information
        if self.cluster_labels is not None:
            num_clusters = len(set(self.cluster_labels))
            cluster_info = f"Clusters: {num_clusters}"
            self.ax.text(5, GRID_SIZE - 35, cluster_info, fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    def _fmt_point(self, point_arr):
        """Format a point array as (x, y) with one decimal"""
        try:
            return f"({point_arr[0]:.1f}, {point_arr[1]:.1f})"
        except Exception:
            return "(?, ?)"
    
    def draw_vehicle_icon(self, position, color='#4169E1', label='Vehicle'):
        """Draw a realistic vehicle icon (delivery truck)"""
        x, y = position
        
        # Calculate vehicle direction (angle)
        if self.current_route_idx < len(self.current_route) - 1:
            current_node = self.current_route[self.current_route_idx]
            next_node = self.current_route[self.current_route_idx + 1]
            if current_node < len(self.points) and next_node < len(self.points):
                target = self.points[next_node]
                dx = target[0] - x
                dy = target[1] - y
                angle = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0
            else:
                angle = 0
        else:
            angle = 0
        
        # Draw truck body (rectangle)
        truck_length = VEHICLE_SIZE * 2
        truck_width = VEHICLE_SIZE * 1.2
        
        # Rotate coordinates
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Truck body corners
        corners = [
            [-truck_length/2, -truck_width/2],
            [truck_length/2, -truck_width/2],
            [truck_length/2, truck_width/2],
            [-truck_length/2, truck_width/2]
        ]
        
        # Rotate and translate corners
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + x
            ry = cx * sin_a + cy * cos_a + y
            rotated_corners.append([rx, ry])
        
        # Draw truck body - realistic brown/orange delivery truck
        truck_body = Polygon(rotated_corners, color='#ffffff', edgecolor=color, 
                            linewidth=2, zorder=7, alpha=0.95)
        self.ax.add_patch(truck_body)
        
        # Draw truck cabin (smaller rectangle)
        cabin_length = truck_length * 0.4
        cabin_corners = [
            [-truck_length/2, -truck_width/2],
            [-truck_length/2 + cabin_length, -truck_width/2],
            [-truck_length/2 + cabin_length, truck_width/2],
            [-truck_length/2, truck_width/2]
        ]
        
        rotated_cabin = []
        for cx, cy in cabin_corners:
            rx = cx * cos_a - cy * sin_a + x
            ry = cx * sin_a + cy * cos_a + y
            rotated_cabin.append([rx, ry])
        
        cabin = Polygon(rotated_cabin, color=color, edgecolor='#0d1b2a', 
                       linewidth=2, zorder=8, alpha=0.95)
        self.ax.add_patch(cabin)
        
        # Draw wheels - reduced size for realism
        wheel_radius = VEHICLE_SIZE * 0.25  # Smaller wheels
        wheel_positions = [
            [-truck_length/4, -truck_width/2 - wheel_radius/2],
            [-truck_length/4, truck_width/2 + wheel_radius/2],
            [truck_length/4, -truck_width/2 - wheel_radius/2],
            [truck_length/4, truck_width/2 + wheel_radius/2]
        ]
        
        for wx, wy in wheel_positions:
            rx = wx * cos_a - wy * sin_a + x
            ry = wx * sin_a + wy * cos_a + y
            wheel = Circle((rx, ry), wheel_radius, color='#1C1C1C', zorder=9, 
                          edgecolor='#000000', linewidth=1)
            self.ax.add_patch(wheel)
            # Wheel rim - silver
            rim = Circle((rx, ry), wheel_radius * 0.6, color='#C0C0C0', zorder=10,
                        edgecolor='#808080', linewidth=0.8)
            self.ax.add_patch(rim)
        
        # Label
        label_x = x + (truck_length/2 + 2) * cos_a
        label_y = y + (truck_length/2 + 2) * sin_a
        self.ax.text(label_x, label_y, '🚚', ha='center', va='center', 
                    fontsize=14, zorder=11, color=color, fontweight='bold')
        self.ax.text(x, y - 3.5, label, ha='center', fontsize=9, color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1))
    
    def run(self):
        """Run the animation"""
        # ARDUINO INTEGRATION START
        arduino_send('S')  # Signal START to Arduino execution layer
        arduino_send('F')  # IMMEDIATELY initiate motion
        # ARDUINO INTEGRATION END

        # Restored to original loop behavior (repeat=True) for continuous simulation
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=1000, 
                                      interval=20, blit=False, repeat=True) 
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # Utility helpers for clarity and route stability
    # ------------------------------------------------------------------ #
    def _normalize_route(self, route_raw, dest_idx):
        """
        Simplify the planned route to avoid repeated depot/destination crossings.
        - Ensures start at depot (0).
        - Removes consecutive duplicates.
        - Stops after first full return to depot (round trip) or after reaching destination.
        """
        if not route_raw:
            return [0]

        cleaned = []
        last = None
        for node in route_raw:
            if node != last:
                cleaned.append(node)
                last = node

        # Ensure start at depot
        if cleaned[0] != 0:
            cleaned.insert(0, 0)

        visited = set([0])
        playable = [0]
        for node in cleaned[1:]:
            playable.append(node)
            if node != 0:
                visited.add(node)
            # Stop once destination is reached
            if dest_idx is not None and node == dest_idx and self.end_point is not None:
                break
            
            # For round trip: stop at first full return to depot after visiting others
            if self.end_point is None and node == 0 and len(visited) >= len(self.points):
                break

        # Ensure destination/depot is last
        if self.end_point is None:
            if playable[-1] != 0:
                playable.append(0)
        else:
            if playable[-1] != dest_idx:
                playable.append(dest_idx)

        return playable

    def _ensure_clusters_before_destination(self, route_nodes):
        """
        Ensure that the active route visits at least one customer from every
        cluster **before** reaching the destination point.

        This only applies when a distinct end_point (destination) is defined;
        it leaves pure round-trip (depot-return) routes unchanged.
        """
        # Only enforce when there is a separate destination and clustering info
        if self.end_point is None or self.cluster_labels is None or len(self.cluster_labels) == 0:
            return route_nodes

        if not route_nodes:
            return route_nodes

        dest_idx = self.destination_idx

        # Find position of destination within the route (if present)
        if dest_idx in route_nodes:
            dest_pos = route_nodes.index(dest_idx)
        else:
            dest_pos = len(route_nodes)

        # Build map: cluster_id -> list of customer node indices (1-based)
        cluster_customer_map = defaultdict(list)
        for i, cluster_id in enumerate(self.cluster_labels):
            cust_idx = i + 1  # customer nodes are 1..len(customers)
            cluster_customer_map[cluster_id].append(cust_idx)

        # Customers already visited before destination
        visited_pre_dest = set(
            node
            for node in route_nodes[:dest_pos]
            if 1 <= node <= len(self.customers)
        )

        # For each cluster, ensure at least one representative customer
        reps_to_insert = []
        for cluster_id, cust_indices in cluster_customer_map.items():
            if not any(c in visited_pre_dest for c in cust_indices):
                # Choose the first customer in that cluster as representative
                rep = cust_indices[0]
                reps_to_insert.append(rep)
                visited_pre_dest.add(rep)

        if not reps_to_insert:
            return route_nodes

        # Insert missing representatives just BEFORE the destination node,
        # while avoiding duplicates and keeping destination as the final point.
        route = list(route_nodes)
        for rep in reps_to_insert:
            # Skip if already visited before destination
            if rep in route[:dest_pos]:
                continue

            # If rep exists after destination, remove it so we can move it earlier
            if rep in route[dest_pos:]:
                route = [n for n in route if n != rep]
                if dest_idx in route:
                    dest_pos = route.index(dest_idx)
                else:
                    dest_pos = len(route)

            route.insert(dest_pos, rep)
            dest_pos += 1

        return route

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def get_cluster_size_selection(num_customers):
    """Get cluster size from user"""
    print("\n" + "="*70)
    print("CLUSTER SIZE SELECTION")
    print("="*70)
    print("Choose number of clusters:")
    print(f"  [1] 1 cluster (all customers together) - Large problem")
    print(f"  [2] 2 clusters - Medium-Large problem")
    print(f"  [3] 3 clusters (recommended) - Balanced")
    print(f"  [4] 4 clusters - Medium problem")
    print(f"  [5] 5+ clusters - Small problems per cluster")
    print(f"  [C] Custom number")
    print("="*70)
    
    while True:
        choice = input(f"\nSelect cluster size (1-5/C) [default: 3]: ").strip().upper()
        
        if not choice:
            choice = "3"
        
        if choice == "1":
            n_clusters = 1
            break
        elif choice == "2":
            n_clusters = 2
            break
        elif choice == "3":
            n_clusters = 3
            break
        elif choice == "4":
            n_clusters = 4
            break
        elif choice == "5":
            n_clusters = 5
            break
        elif choice == "C":
            try:
                n_clusters = int(input(f"Enter number of clusters (1-{num_customers}): "))
                if 1 <= n_clusters <= num_customers:
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {num_customers}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
        else:
            print("❌ Invalid choice. Please enter 1-5 or C")
    
    explain_clustering_effects(n_clusters, num_customers)
    return n_clusters

def get_user_depot_selection(customers, grid_size):
    """Get depot location from user input - using customer locations only"""
    print("\n" + "="*70)
    print("DEPOT SELECTION (Starting Point)")
    print("="*70)
    print("Choose starting point (depot) from customer locations:")
    print("\nAvailable locations:")
    print("-" * 70)
    for i, customer in enumerate(customers):
        print(f"  [{i+1:2d}] Customer {i+1:2d} at coordinates: ({customer[0]:6.2f}, {customer[1]:6.2f})")
    print("-" * 70)
    print(f"  [ 0] Center of grid (default)")
    print("="*70)
    
    while True:
        try:
            choice = input(f"\nSelect starting point (0-{len(customers)}) [default: 0]: ").strip()
            if not choice:
                choice = "0"
            
            choice_num = int(choice)
            
            if choice_num == 0:
                depot = np.array([grid_size / 2, grid_size / 2])
                print(f"\n✓ Selected: Center of grid at ({depot[0]:.2f}, {depot[1]:.2f})")
                break
            elif 1 <= choice_num <= len(customers):
                cust_idx = choice_num - 1
                depot = customers[cust_idx].copy()
                print(f"\n✓ Selected: Customer {choice_num} location at ({depot[0]:.2f}, {depot[1]:.2f})")
                break
            else:
                print(f"❌ Invalid choice. Please enter a number between 0 and {len(customers)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    return depot

def get_user_end_point(customers, grid_size):
    """Get end point (destination) from user input"""
    print("\n" + "="*70)
    print("END POINT SELECTION (Destination)")
    print("="*70)
    print("Choose destination point:")
    print("\nAvailable locations:")
    print("-" * 70)
    for i, customer in enumerate(customers):
        print(f"  [{i+1:2d}] Customer {i+1:2d} at coordinates: ({customer[0]:6.2f}, {customer[1]:6.2f})")
    print("-" * 70)
    print(f"  [ 0] Return to depot (default)")
    print("="*70)
    
    while True:
        try:
            choice = input(f"\nSelect destination (0-{len(customers)}) [default: 0]: ").strip()
            if not choice:
                choice = "0"
            
            choice_num = int(choice)
            
            if choice_num == 0:
                end_point = None  # Return to depot
                print(f"\n✓ Selected: Return to depot")
                break
            elif 1 <= choice_num <= len(customers):
                cust_idx = choice_num - 1
                end_point = customers[cust_idx].copy()
                print(f"\n✓ Selected: Customer {choice_num} at ({end_point[0]:.2f}, {end_point[1]:.2f})")
                break
            else:
                print(f"❌ Invalid choice. Please enter a number between 0 and {len(customers)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    return end_point

def plot_analytical_outputs(cluster_sizes, feasibility_rates, penalties, cost_stabilities):
    """
    VISUAL & ANALYTICAL OUTPUT (REQUIRED)
    Plot cluster size vs feasibility, penalty, and cost stability
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Cluster size vs Feasibility rate
    axes[0].plot(cluster_sizes, feasibility_rates, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    axes[0].set_xlabel('Cluster Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Feasibility Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('GAP 1: Cluster Size vs Feasibility Rate', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim([0, 1.1])
    
    # Plot 2: Cluster size vs Penalty value
    axes[1].plot(cluster_sizes, penalties, 's-', linewidth=2, markersize=8, color='#A23B72')
    axes[1].set_xlabel('Cluster Size', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Penalty Value', fontsize=12, fontweight='bold')
    axes[1].set_title('GAP 4: Cluster Size vs Penalty Value', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Cost stability across multi-start runs
    if cost_stabilities:
        axes[2].boxplot(cost_stabilities, labels=[f'C{i+1}' for i in range(len(cost_stabilities))])
        axes[2].set_xlabel('Run Number', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Route Cost', fontsize=12, fontweight='bold')
        axes[2].set_title('GAP 2: Cost Stability (Multi-Start)', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_vrp_analytics.png', dpi=150, bbox_inches='tight')
    print("\n📊 Analytical plots saved to 'quantum_vrp_analytics.png'")
    plt.show()

def main():
    # ARDUINO INTEGRATION START
    # Initialize the hardware execution layer
    init_arduino_serial()
    # ARDUINO INTEGRATION END

    print("="*70)
    print("QUANTUM-ASSISTED DELIVERY ROUTE OPTIMIZATION")
    print("With Alternate Route Switching")
    print("="*70)
    
    # Step 1: Generate data
    print("\n[STEP 1] Generating customers...")
    _, customers = generate_depot_and_customers(NUM_CUSTOMERS, GRID_SIZE)
    
    # Get depot (start point) from user
    depot = get_user_depot_selection(customers, GRID_SIZE)
    print(f"\n✓ Final starting point (depot): ({depot[0]:.2f}, {depot[1]:.2f})")
    
    # Get end point from user
    end_point = get_user_end_point(customers, GRID_SIZE)
    
    # Step 2: Get cluster size from user
    print("\n[STEP 2] Clustering configuration...")
    selected_clusters = get_cluster_size_selection(NUM_CUSTOMERS)
    
    # Step 3: Calculate distance matrix (include end_point if specified)
    print("\n[STEP 3] Calculating distance matrix...")
    if end_point is not None:
        # Include end_point in distance matrix
        all_points = np.vstack([depot.reshape(1, -1), customers, end_point.reshape(1, -1)])
    else:
        all_points = np.vstack([depot.reshape(1, -1), customers])
    
    dist_matrix, points = calculate_distance_matrix(depot, customers)
    # Update points to include end_point if specified
    if end_point is not None:
        points = np.vstack([points, end_point.reshape(1, -1)])
        # Recalculate distance matrix with end_point
        n = len(points)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
    
    print(f"Distance matrix shape: {dist_matrix.shape}")
    
    # Step 4: Use Hybrid Orchestrator (GAP 6)
    print("\n[STEP 4] Running Hybrid Orchestrator...")
    final_routes, orchestrator_log, gap_metrics, clusters, cluster_labels, cluster_centers = hybrid_orchestrator(
        customers, depot, selected_clusters, dist_matrix, points
    )
    
    # Step 5: Quantum route selection on final routes
    print("\n[STEP 5] Quantum route selection on feasible routes...")
    selected_routes, comp_metrics, penalty_info = select_routes_quantum(final_routes, dist_matrix, selected_clusters)
    
    # Collect data for plotting
    cluster_sizes = [selected_clusters]
    feasibility_rates = [gap_metrics['feasibility']['feasibility_rate']]
    penalties = [penalty_info['final_penalty']]
    cost_stabilities = [gap_metrics['multi_start']['costs']] if 'multi_start' in gap_metrics else []
    
    # Step 6: Display results
    print("\n" + "="*70)
    print("FINAL ROUTES & METRICS - ALL GAPS ADDRESSED")
    print("="*70)
    
    print(f"\n📊 GAP SUMMARY:")
    print(f"  GAP 1 (Scaling): Feasibility rate = {feasibility_rates[0]*100:.1f}%")
    print(f"  GAP 2 (Sensitivity): Cost variance = {gap_metrics['multi_start'].get('variance', 0):.2f}")
    print(f"  GAP 3 (Constraints): Feasibility ratio = {gap_metrics['feasibility_ratio']*100:.1f}%")
    print(f"  GAP 4 (Penalty): Selected alpha = {penalty_info['selected_alpha']}")
    print(f"  GAP 5 (Repair): Routes repaired = {gap_metrics['repair']['routes_repaired']}")
    print(f"  GAP 6 (Orchestration): Classical tasks = {len(orchestrator_log['classical_tasks'])}, Quantum = {len(orchestrator_log['quantum_tasks'])}")
    
    if penalty_info:
        print(f"\n📊 Penalty Information (GAP 4):")
        print(f"  Base Penalty: {penalty_info['base_penalty']:.2f}")
        print(f"  Penalty Multiplier (alpha): {penalty_info['penalty_multiplier']}")
        print(f"  Final Penalty Value: {penalty_info['final_penalty']:.2f}")
        print(f"  Cost Range: {penalty_info['cost_range']:.2f}")
    
    if comp_metrics:
        print(f"\n⚡ Computation Metrics (Cluster Size: {selected_clusters}):")
        print(f"  Problem Size: {comp_metrics['problem_size']} binary variables")
        print(f"  Total Time: {comp_metrics.get('total_time', 0):.3f} seconds")
        print(f"  Average Time: {comp_metrics.get('avg_time', 0):.3f} seconds per start")
        print(f"  Success Rate: {(comp_metrics['successes']/comp_metrics['num_starts']*100):.1f}%")
    
    print(f"\n🗺️  Generated Routes:")
    for route_info in selected_routes:
        route_type, route, cost, is_valid, msg = route_info
        print(f"\n  {route_type.upper()}:")
        route_str = " -> ".join([f'Node {r}' for r in route])
        print(f"    Route: {route_str}")
        print(f"    Cost: {cost:.2f}")
        print(f"    Valid: {is_valid} ({msg})")
    
    # Generate analytical plots
    print("\n[STEP 7] Generating analytical plots...")
    plot_analytical_outputs(cluster_sizes, feasibility_rates, penalties, cost_stabilities)
    
    # Step 8: Run simulation
    print("\n" + "="*70)
    print("STARTING ANIMATION...")
    print("="*70)
    print("Visual Guide:")
    print("  - 🟢 DEPOT (SOURCE): Large green circle - starting point")
    print("  - 🟠 C1, C2, C3...: Customer locations with clear labels")
    print("  - 🟡 MAIN ROUTE: Bright GOLD/YELLOW line with arrows")
    print("  - 🔵 ALTERNATE ROUTES: Bright CYAN/MAGENTA dashed lines")
    print("  - 🔷 MOVEMENT TRAIL: Bright CYAN path showing vehicle movement")
    print("  - 🚚 VEHICLE: Blue delivery truck icon")
    print("  - 🔴 OBSTACLE: Red block (appears after ~3 seconds)")
    print("="*70)
    
    # Optional GAP-5 visual enhancements:
    # - enable_confidence_visuals: thickness of main route encodes QAOA confidence
    confidence_score = comp_metrics.get('confidence_score') if comp_metrics else None
    
    simulation = RouteSimulation(
        depot,
        customers,
        selected_routes,
        dist_matrix,
        points,
        cluster_labels,
        cluster_centers,
        end_point,
        enable_confidence_visuals=True,
        confidence_score=confidence_score,
    )
    
    print("\n" + "="*70)
    print("READY TO START ANIMATION")
    print("="*70)
    
    # ARDUINO INTEGRATION START
    # Hardware must authorize the start of the vehicle.
    if ARDUINO_SER:
        print("\n" + "="*70)
        print("[ARDUINO] WAITING FOR MISSION START")
        print("="*70)
        print("1. Place the robot on the track.")
        print("2. Press ANY KEY to start manually OR wait for Arduino 'START' signal.")
        print("-" * 70)
        
        while True:
            if msvcrt.kbhit():
                msvcrt.getch()
                print("\n✓ Manual override: Starting mission...")
                break
            
            feedback = read_arduino_feedback()
            if feedback == "READY":
                print(f"\n✓ Arduino is Ready: Starting mission...")
                break
            time.sleep(0.1)
        
        # Once authorized, trigger the hardware execution
        arduino_send('S')
        arduino_send('F')
    else:
        input("\n>>> [NO HARDWARE] Press ENTER to start the simulation... ")
    # ARDUINO INTEGRATION END
    
    simulation.run()

if __name__ == "__main__":
    main()

