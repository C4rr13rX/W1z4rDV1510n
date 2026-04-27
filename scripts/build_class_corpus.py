#!/usr/bin/env python3
"""
build_class_corpus.py — emit a (plain English description -> canonical class
code) corpus for the multi-pool fabric to learn, then evaluate.

Produces data/foundation/class_corpus.jsonl with one record per
(description, class_id) pair:

  {"class_id": "kalman_1d",
   "variation": 0,
   "description": "<plain english>",
   "code": "<canonical python class>"}

For each class we:
  1. Provide a canonical, minimal, well-tested implementation.
  2. Write 5 plain English descriptions (variations 0..4): a one-line summary,
     a method-by-method narration, a use-case framing, an analogy, and a
     formula/equation grounded one.

Methodology for the GA:
  - Train pairs (description_v, code) for v in 0..3 (variations 0..3).
  - Hold out variation 4 as the query — the multi-pool fabric should recall
    the same code from a paraphrased description. Fitness = lev_sim between
    predicted and expected code.

Each class is short enough (≤80 lines) that training cost is bounded.
"""
from __future__ import annotations
import json
import pathlib

OUT = pathlib.Path("data/foundation/class_corpus.jsonl")


# Canonical class implementations.  Kept short (≤80 lines), each one a
# minimal idiomatic Python representation of the algorithm/data structure.
# Tested style: each class has clear method signatures and the math is
# grounded in the standard reference implementation.

CLASSES: dict[str, str] = {
    # ── Data structures ──────────────────────────────────────────────────
    "stack": '''
class Stack:
    """LIFO stack with push, pop, peek, len."""
    def __init__(self):
        self._items = []
    def push(self, x):
        self._items.append(x)
    def pop(self):
        return self._items.pop()
    def peek(self):
        return self._items[-1]
    def __len__(self):
        return len(self._items)
'''.strip(),

    "queue": '''
class Queue:
    """FIFO queue using a deque for O(1) ends."""
    from collections import deque
    def __init__(self):
        self._d = self.deque()
    def enqueue(self, x):
        self._d.append(x)
    def dequeue(self):
        return self._d.popleft()
    def __len__(self):
        return len(self._d)
'''.strip(),

    "priority_queue": '''
class PriorityQueue:
    """Min-heap priority queue using heapq."""
    import heapq
    def __init__(self):
        self._h = []
    def push(self, priority, item):
        self.heapq.heappush(self._h, (priority, item))
    def pop(self):
        return self.heapq.heappop(self._h)[1]
    def peek(self):
        return self._h[0][1]
    def __len__(self):
        return len(self._h)
'''.strip(),

    "linked_list": '''
class LinkedList:
    """Singly linked list with prepend, append, find, len."""
    class Node:
        def __init__(self, v): self.v, self.nxt = v, None
    def __init__(self):
        self.head = None
    def prepend(self, v):
        n = self.Node(v); n.nxt = self.head; self.head = n
    def append(self, v):
        n = self.Node(v)
        if self.head is None: self.head = n; return
        c = self.head
        while c.nxt: c = c.nxt
        c.nxt = n
    def find(self, v):
        c = self.head
        while c:
            if c.v == v: return c
            c = c.nxt
        return None
    def __len__(self):
        n, c = 0, self.head
        while c: n += 1; c = c.nxt
        return n
'''.strip(),

    "binary_search_tree": '''
class BinarySearchTree:
    """Unbalanced BST with insert, find, in_order traversal."""
    class Node:
        def __init__(self, k): self.k, self.l, self.r = k, None, None
    def __init__(self):
        self.root = None
    def insert(self, k):
        def _ins(n, k):
            if n is None: return self.Node(k)
            if k < n.k: n.l = _ins(n.l, k)
            elif k > n.k: n.r = _ins(n.r, k)
            return n
        self.root = _ins(self.root, k)
    def find(self, k):
        n = self.root
        while n:
            if k == n.k: return True
            n = n.l if k < n.k else n.r
        return False
    def in_order(self):
        out = []
        def _walk(n):
            if not n: return
            _walk(n.l); out.append(n.k); _walk(n.r)
        _walk(self.root); return out
'''.strip(),

    "trie": '''
class Trie:
    """Prefix tree for ascii strings."""
    def __init__(self):
        self.root = {}
    def insert(self, s):
        n = self.root
        for c in s:
            n = n.setdefault(c, {})
        n["$"] = True
    def search(self, s):
        n = self.root
        for c in s:
            if c not in n: return False
            n = n[c]
        return n.get("$", False)
    def starts_with(self, prefix):
        n = self.root
        for c in prefix:
            if c not in n: return False
            n = n[c]
        return True
'''.strip(),

    "lru_cache": '''
class LRUCache:
    """Least-Recently-Used cache with OrderedDict."""
    from collections import OrderedDict
    def __init__(self, capacity: int):
        self.cap = capacity
        self.d = self.OrderedDict()
    def get(self, key):
        if key not in self.d: return None
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key, value):
        if key in self.d: self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap: self.d.popitem(last=False)
'''.strip(),

    "union_find": '''
class UnionFind:
    """Disjoint-set with path compression and union by rank."""
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]: ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]: self.r[ra] += 1
        return True
'''.strip(),

    "bloom_filter": '''
class BloomFilter:
    """Bloom filter with k hash slots over a bitarray-like list."""
    def __init__(self, size: int, k: int):
        self.size, self.k = size, k
        self.bits = [0] * size
    def _hashes(self, x):
        h = hash(x) & 0xFFFFFFFFFFFFFFFF
        for i in range(self.k):
            yield (h + i * 0x9E3779B97F4A7C15) % self.size
    def add(self, x):
        for h in self._hashes(x): self.bits[h] = 1
    def contains(self, x):
        return all(self.bits[h] for h in self._hashes(x))
'''.strip(),

    # ── Algorithms ───────────────────────────────────────────────────────
    "binary_search": '''
class BinarySearch:
    """Iterative binary search on a sorted list."""
    def find(self, arr, target):
        lo, hi = 0, len(arr) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == target: return mid
            if arr[mid] < target: lo = mid + 1
            else: hi = mid - 1
        return -1
'''.strip(),

    "dijkstra": '''
class Dijkstra:
    """Shortest path on a weighted graph with non-negative edges."""
    import heapq
    def shortest(self, graph, source):
        dist = {n: float("inf") for n in graph}; dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = self.heapq.heappop(pq)
            if d > dist[u]: continue
            for v, w in graph[u].items():
                nd = d + w
                if nd < dist[v]: dist[v] = nd; self.heapq.heappush(pq, (nd, v))
        return dist
'''.strip(),

    # ── Numerics ─────────────────────────────────────────────────────────
    "newton_raphson": '''
class NewtonRaphson:
    """Find roots of f using Newton's method with derivative df."""
    def solve(self, f, df, x0, tol=1e-9, max_iter=100):
        x = x0
        for _ in range(max_iter):
            fx, dfx = f(x), df(x)
            if dfx == 0: return x
            step = fx / dfx
            x -= step
            if abs(step) < tol: return x
        return x
'''.strip(),

    "runge_kutta_4": '''
class RungeKutta4:
    """Classical 4th-order Runge-Kutta integrator for dy/dt = f(t, y)."""
    def step(self, f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h * k1 / 2)
        k3 = f(t + h/2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
'''.strip(),

    "gradient_descent": '''
class GradientDescent:
    """Vanilla gradient descent on parameter vector x for loss with grad."""
    def __init__(self, lr: float = 0.01):
        self.lr = lr
    def step(self, x, grad):
        return [xi - self.lr * gi for xi, gi in zip(x, grad)]
    def fit(self, x0, grad_fn, steps=100):
        x = list(x0)
        for _ in range(steps):
            x = self.step(x, grad_fn(x))
        return x
'''.strip(),

    "kmeans": '''
class KMeans:
    """K-means clustering with Lloyd iterations."""
    def __init__(self, k: int, max_iter: int = 100):
        self.k, self.max_iter = k, max_iter
        self.centers = None
    def fit(self, points):
        import random
        self.centers = random.sample(list(points), self.k)
        for _ in range(self.max_iter):
            groups = [[] for _ in range(self.k)]
            for p in points:
                i = min(range(self.k), key=lambda j: sum((a-b)**2 for a,b in zip(p, self.centers[j])))
                groups[i].append(p)
            new_centers = []
            for g in groups:
                if not g: new_centers.append(self.centers[0]); continue
                new_centers.append([sum(c)/len(g) for c in zip(*g)])
            if new_centers == self.centers: break
            self.centers = new_centers
        return self.centers
'''.strip(),

    "linear_regression": '''
class LinearRegression:
    """Ordinary least squares for y = a*x + b."""
    def fit(self, xs, ys):
        n = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs) or 1.0
        self.a = num / den
        self.b = my - self.a * mx
        return self
    def predict(self, x):
        return self.a * x + self.b
'''.strip(),

    "kalman_1d": '''
class Kalman1D:
    """Scalar Kalman filter for position+velocity tracking on a noisy sensor."""
    def __init__(self, q: float, r: float):
        self.x = 0.0
        self.p = 1.0
        self.q = q  # process noise
        self.r = r  # measurement noise
    def step(self, z: float) -> float:
        # predict
        self.p += self.q
        # update
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1.0 - k)
        return self.x
'''.strip(),

    "fft_radix2": '''
class FFTRadix2:
    """Iterative radix-2 Cooley-Tukey FFT for power-of-two length input."""
    import cmath
    def transform(self, x):
        n = len(x)
        if n & (n - 1): raise ValueError("length must be a power of 2")
        x = list(x)
        # bit-reverse permutation
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit; bit >>= 1
            j ^= bit
            if i < j: x[i], x[j] = x[j], x[i]
        # butterflies
        size = 2
        while size <= n:
            half = size // 2
            angle = -2j * self.cmath.pi / size
            w = self.cmath.exp(angle)
            for k in range(0, n, size):
                wn = 1
                for m in range(half):
                    t = wn * x[k + m + half]
                    x[k + m + half] = x[k + m] - t
                    x[k + m] += t
                    wn *= w
            size <<= 1
        return x
'''.strip(),

    # ── Physics / engineering simulations ────────────────────────────────
    "pid_controller": '''
class PIDController:
    """Classic PID controller in discrete time."""
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral = 0.0
        self._prev_error = 0.0
    def step(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp*error + self.ki*self._integral + self.kd*derivative
'''.strip(),

    "rc_circuit": '''
class RCCircuit:
    """First-order RC low-pass circuit: V_out responds to V_in with tau=RC."""
    def __init__(self, R: float, C: float):
        self.tau = R * C
        self.v_out = 0.0
    def step(self, v_in: float, dt: float) -> float:
        # Forward Euler integration of dv/dt = (v_in - v_out) / tau
        self.v_out += dt * (v_in - self.v_out) / self.tau
        return self.v_out
'''.strip(),

    "spring_mass": '''
class SpringMass:
    """Damped harmonic oscillator: m x'' + c x' + k x = F(t)."""
    def __init__(self, m: float, k: float, c: float):
        self.m, self.k, self.c = m, k, c
        self.x = 0.0; self.v = 0.0
    def step(self, F: float, dt: float):
        a = (F - self.c * self.v - self.k * self.x) / self.m
        self.v += a * dt
        self.x += self.v * dt
        return self.x, self.v
'''.strip(),

    "pendulum": '''
class Pendulum:
    """Damped simple pendulum: theta'' + (g/L)*sin(theta) + (b/m)*theta' = 0."""
    import math
    def __init__(self, length: float, mass: float, damping: float = 0.0, g: float = 9.81):
        self.L, self.m, self.b, self.g = length, mass, damping, g
        self.theta = 0.0; self.omega = 0.0
    def step(self, dt: float):
        alpha = -(self.g / self.L) * self.math.sin(self.theta) - (self.b / self.m) * self.omega
        self.omega += alpha * dt
        self.theta += self.omega * dt
        return self.theta, self.omega
'''.strip(),

    # ── Biology / dynamics ───────────────────────────────────────────────
    "sir_model": '''
class SIRModel:
    """Compartmental SIR epidemic model: dS/dt = -beta*S*I/N etc."""
    def __init__(self, beta: float, gamma: float, N: int):
        self.beta, self.gamma, self.N = beta, gamma, N
    def step(self, S, I, R, dt: float):
        dS = -self.beta * S * I / self.N
        dI =  self.beta * S * I / self.N - self.gamma * I
        dR =  self.gamma * I
        return S + dS*dt, I + dI*dt, R + dR*dt
    def r_naught(self):
        return self.beta / self.gamma
'''.strip(),

    "lotka_volterra": '''
class LotkaVolterra:
    """Predator-prey ODEs: dx/dt = a*x - b*x*y; dy/dt = -c*y + d*x*y."""
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a, self.b, self.c, self.d = a, b, c, d
    def step(self, prey, predator, dt: float):
        dprey = self.a * prey - self.b * prey * predator
        dpred = -self.c * predator + self.d * prey * predator
        return prey + dprey * dt, predator + dpred * dt
'''.strip(),

    "michaelis_menten": '''
class MichaelisMenten:
    """Enzyme kinetics: v = Vmax * [S] / (Km + [S])."""
    def __init__(self, vmax: float, km: float):
        self.vmax, self.km = vmax, km
    def rate(self, substrate: float) -> float:
        return self.vmax * substrate / (self.km + substrate)
'''.strip(),

    "kuramoto": '''
class KuramotoOscillators:
    """N coupled phase oscillators: dphi_i/dt = omega_i + (K/N) * sum_j sin(phi_j - phi_i)."""
    import math
    def __init__(self, omegas, coupling: float):
        self.omegas = list(omegas); self.K = coupling
        self.phases = [0.0] * len(self.omegas)
    def step(self, dt: float):
        N = len(self.phases)
        new = []
        for i, phi_i in enumerate(self.phases):
            interaction = sum(self.math.sin(phi_j - phi_i) for phi_j in self.phases)
            new.append(phi_i + dt * (self.omegas[i] + self.K / N * interaction))
        self.phases = new
        return list(self.phases)
'''.strip(),

    "hodgkin_huxley_simple": '''
class HodgkinHuxleyNeuron:
    """Simplified leaky integrate-and-fire spiking neuron with refractory period."""
    def __init__(self, v_rest: float = -65.0, v_thresh: float = -50.0,
                 v_reset: float = -70.0, tau: float = 10.0,
                 r_input: float = 1.0, refractory: float = 2.0):
        self.v = v_rest; self.v_rest = v_rest
        self.v_thresh = v_thresh; self.v_reset = v_reset
        self.tau = tau; self.r = r_input
        self.refractory_remaining = 0.0
        self.refractory = refractory
    def step(self, I: float, dt: float) -> bool:
        if self.refractory_remaining > 0:
            self.refractory_remaining -= dt
            self.v = self.v_reset
            return False
        self.v += dt * (-(self.v - self.v_rest) + self.r * I) / self.tau
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.refractory_remaining = self.refractory
            return True
        return False
'''.strip(),

    # ── Statistics / ML ──────────────────────────────────────────────────
    "perceptron": '''
class Perceptron:
    """Single-layer perceptron with sign activation, online learning."""
    def __init__(self, n_features: int, lr: float = 0.1):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr
    def predict(self, x):
        s = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return 1 if s >= 0 else -1
    def update(self, x, y_true):
        y_pred = self.predict(x)
        if y_pred != y_true:
            for i, xi in enumerate(x):
                self.w[i] += self.lr * y_true * xi
            self.b += self.lr * y_true
'''.strip(),

    "softmax_classifier": '''
class SoftmaxClassifier:
    """Multinomial logistic regression with softmax + cross-entropy."""
    import math
    def __init__(self, n_features: int, n_classes: int, lr: float = 0.1):
        self.W = [[0.0] * n_features for _ in range(n_classes)]
        self.b = [0.0] * n_classes
        self.lr = lr
        self.n_classes = n_classes
    def softmax(self, z):
        m = max(z); e = [self.math.exp(zi - m) for zi in z]; s = sum(e)
        return [ei / s for ei in e]
    def predict_proba(self, x):
        z = [sum(wij * xj for wij, xj in zip(self.W[i], x)) + self.b[i] for i in range(self.n_classes)]
        return self.softmax(z)
    def update(self, x, y_true: int):
        p = self.predict_proba(x)
        for i in range(self.n_classes):
            err = (1 if i == y_true else 0) - p[i]
            for j, xj in enumerate(x):
                self.W[i][j] += self.lr * err * xj
            self.b[i] += self.lr * err
'''.strip(),

    "naive_bayes": '''
class NaiveBayes:
    """Gaussian Naive Bayes for continuous features."""
    import math
    def fit(self, X, y):
        classes = set(y)
        self.priors = {c: y.count(c) / len(y) for c in classes}
        self.means, self.vars = {}, {}
        for c in classes:
            xs = [X[i] for i in range(len(X)) if y[i] == c]
            self.means[c] = [sum(col)/len(xs) for col in zip(*xs)]
            self.vars[c]  = [sum((v - m)**2 for v in col)/len(xs) + 1e-9
                              for col, m in zip(zip(*xs), self.means[c])]
        return self
    def predict(self, x):
        scores = {}
        for c in self.priors:
            log_p = self.math.log(self.priors[c])
            for xi, mi, vi in zip(x, self.means[c], self.vars[c]):
                log_p -= 0.5 * (self.math.log(2 * self.math.pi * vi) + (xi - mi)**2 / vi)
            scores[c] = log_p
        return max(scores, key=scores.get)
'''.strip(),

    "pca_2d": '''
class PCA2D:
    """Principal component analysis for 2D points; returns the top axis."""
    def fit(self, points):
        n = len(points)
        mx = sum(p[0] for p in points) / n
        my = sum(p[1] for p in points) / n
        sxx = sum((p[0]-mx)**2 for p in points) / n
        syy = sum((p[1]-my)**2 for p in points) / n
        sxy = sum((p[0]-mx)*(p[1]-my) for p in points) / n
        # eigenvalues of 2x2 covariance
        tr = sxx + syy
        det = sxx*syy - sxy*sxy
        disc = max(0.0, tr*tr/4 - det)
        lam1 = tr/2 + disc**0.5
        if abs(sxy) > 1e-12:
            self.axis = (lam1 - syy, sxy)
        else:
            self.axis = (1.0, 0.0)
        self.mean = (mx, my)
        return self
'''.strip(),
}


# Plain-English description variations.  Each class gets exactly 5 strings
# (variation 0..4): one-liner, method-by-method, use-case framing, analogy,
# formula-grounded.  Held-out variation for the GA is variation 4.

DESCRIPTIONS: dict[str, list[str]] = {
    "stack": [
        "a stack data structure with push pop peek and length",
        "the class supports push to add an item pop to remove the last item peek to look at the top and len for size",
        "use this when you need last-in first-out ordering for parsing or undo histories",
        "like a stack of plates the last one placed on top is the first one removed",
        "lifo container with operations push x pop returning x and peek returning the top",
    ],
    "queue": [
        "a fifo queue with enqueue dequeue and length using a deque internally",
        "enqueue adds to the back dequeue removes from the front and len returns size",
        "use this for breadth first search task queues or buffering producer consumer streams",
        "like a line at a checkout the first person in is the first person served",
        "first in first out container backed by collections deque for constant time operations",
    ],
    "priority_queue": [
        "a min heap priority queue using heapq with push priority pop and peek",
        "push takes a priority and an item pop returns the item with the lowest priority and peek looks at the smallest",
        "use this for dijkstras shortest path scheduler queues or A star search frontiers",
        "like an emergency room triage the most urgent patient is treated first regardless of arrival",
        "min heap where the smallest priority value sits at the root and is removed by pop",
    ],
    "linked_list": [
        "a singly linked list with prepend append find and length",
        "prepend inserts at the head append walks to the tail find scans for a value and len counts nodes",
        "use this when contiguous arrays are too expensive to grow or when frequent insertions are needed",
        "like a treasure hunt where each clue points only to the next location not back",
        "nodes hold a value and a next pointer the head field is the entry point",
    ],
    "binary_search_tree": [
        "an unbalanced binary search tree with insert find and in order traversal",
        "insert places a key on the left if smaller right if larger find walks down the tree and in order yields sorted keys",
        "use this when you need ordered keys with logarithmic average insertion and lookup",
        "like a yes no decision tree where every node asks is your key smaller than mine",
        "each node has left and right children with the BST invariant left key less than node key less than right key",
    ],
    "trie": [
        "a prefix tree for ascii strings with insert search and starts with",
        "insert walks character by character creating dict nodes search returns whether the full word is present and starts with checks any prefix",
        "use this for autocompletion spell check or routing tables that share prefixes",
        "like a phone book index that branches on each letter typed",
        "a tree where each edge is one character and a terminal flag marks the end of an inserted word",
    ],
    "lru_cache": [
        "a least recently used cache with a fixed capacity using OrderedDict",
        "get moves the key to the most recent end put inserts and evicts the oldest when over capacity",
        "use this when memory is bounded and stale items can be discarded for fresher ones",
        "like a shelf where the item you grab last is at the front and the back falls off when full",
        "ordered map where access reorders keys and the head is evicted at capacity",
    ],
    "union_find": [
        "a disjoint set union find with path compression and union by rank",
        "find walks up to the root halving the path on the way and union merges two sets keeping the deeper as root",
        "use this for kruskals minimum spanning tree connected components or equivalence classes",
        "like joining two friend groups by introducing one persons leader to anothers",
        "forest of trees where two elements share a root if and only if they are in the same set",
    ],
    "bloom_filter": [
        "a bloom filter with a fixed bit array and k hash positions per element",
        "add sets k bits derived from the element contains returns true only if all k bits are set",
        "use this to test set membership probabilistically with no false negatives only false positives",
        "like a fast guess about whether a name appears on a guest list before checking the actual list",
        "probabilistic data structure where the probability of a false positive depends on size capacity and hash count",
    ],
    "binary_search": [
        "iterative binary search on a sorted list returning the index or minus one",
        "the find method maintains lo and hi pointers and halves the range each step",
        "use this when an array is already sorted and you need logarithmic lookup",
        "like guessing a number with high low feedback always halving your remaining range",
        "compares the middle element to the target then descends into one half repeatedly",
    ],
    "dijkstra": [
        "dijkstras shortest path on a weighted graph with non negative edges",
        "shortest takes a graph dict and source node and returns a dict of shortest distances using a min heap",
        "use this for road network routing internet packet paths or game map navigation",
        "like flooding water from the source where the wave reaches every node by the cheapest route",
        "maintains a priority queue keyed by tentative distance and relaxes outgoing edges as nodes are popped",
    ],
    "newton_raphson": [
        "newton raphson root finder taking f its derivative df an initial guess and tolerance",
        "the solve method iterates x minus f over df until the step is below tolerance or max iter is reached",
        "use this to find zeros of nonlinear equations smooth root finding or implicit equation solving",
        "like climbing down a hill by following the steepest local gradient toward the bottom",
        "uses a tangent line approximation x next equals x minus f x over f prime x",
    ],
    "runge_kutta_4": [
        "classical fourth order runge kutta integrator for an ODE dy over dt equals f t y",
        "step computes four slopes k1 k2 k3 k4 and combines them into a weighted average update",
        "use this for accurate numerical integration of smooth ordinary differential equations",
        "like sampling a hillside at four points and taking a weighted average to step downhill",
        "weights one two two one over six on the four slope estimates for fourth order accuracy",
    ],
    "gradient_descent": [
        "vanilla gradient descent with a learning rate and step and fit methods",
        "step subtracts learning rate times gradient from each parameter and fit calls step in a loop",
        "use this for training simple machine learning models or convex function minimization",
        "like rolling a ball downhill where you take steps proportional to the local slope",
        "x next equals x minus alpha times grad f at x for a chosen learning rate alpha",
    ],
    "kmeans": [
        "k means clustering using lloyds iterations with random initial centers",
        "fit picks k random points as centers then assigns points to the nearest center and recomputes centers until stable",
        "use this for unsupervised grouping of data points into k clusters",
        "like sorting marbles into k bowls then moving each bowl to the average of the marbles inside",
        "minimizes the within cluster sum of squared distances by iterating expectation and maximization",
    ],
    "linear_regression": [
        "ordinary least squares for the line y equals a x plus b",
        "fit computes the slope a and intercept b from sample means and the predict method returns a x plus b",
        "use this when you suspect a linear relationship between two variables and need a baseline model",
        "like fitting a ruler through a cloud of points to minimize total perpendicular error",
        "the slope is covariance over variance and the intercept aligns the line through the means",
    ],
    "kalman_1d": [
        "a scalar one dimensional kalman filter for tracking a noisy sensor reading",
        "the step method updates an estimate x and a variance p combining a process noise q and measurement noise r",
        "use this for smoothing GPS readings IMU data or any one dimensional signal with sensor noise",
        "like a wise judge weighing a new witness against an internal belief proportional to confidence in each",
        "predict adds q to variance update mixes prior and measurement weighted by k equals p over p plus r",
    ],
    "fft_radix2": [
        "iterative radix two cooley tukey fast fourier transform for power of two length input",
        "transform performs a bit reverse permutation then nested butterfly stages doubling the size each time",
        "use this for spectrum analysis convolution acceleration or signal frequency extraction",
        "like decomposing a chord into its constituent notes by recursive splitting and recombination",
        "computes the discrete fourier transform in n log n via divide and conquer over the unit circle",
    ],
    "pid_controller": [
        "a classic discrete time proportional integral derivative controller",
        "step takes a setpoint a measurement and dt and returns kp times error plus ki times integral plus kd times derivative",
        "use this for thermostats robot motor speed pumps or any plant where you steer toward a target",
        "like steering a car correcting for current error past drift and how fast the error is changing",
        "u of t equals kp e of t plus ki integral of e plus kd derivative of e for tunable gains",
    ],
    "rc_circuit": [
        "first order RC low pass circuit with time constant tau equals R times C",
        "step performs forward euler integration of dv over dt equals v in minus v out over tau",
        "use this for filtering high frequency noise from a sensor input or smoothing a setpoint command",
        "like water filling a bucket through a small hole the higher the inflow the slower the rise",
        "v out approaches v in exponentially with characteristic time tau equal to R times C",
    ],
    "spring_mass": [
        "damped harmonic oscillator m x double dot plus c x dot plus k x equals F of t",
        "step computes acceleration from force minus damping times velocity minus stiffness times position then integrates",
        "use this for mechanical vibration analysis structural dynamics or simple physics simulations",
        "like a car suspension where mass damper and spring decide how the wheel returns after a bump",
        "second order ODE with mass m damping c stiffness k integrated by forward euler in step",
    ],
    "pendulum": [
        "damped simple pendulum with length L mass m damping b and gravity g",
        "step computes angular acceleration alpha from minus g over L sine theta minus b over m theta dot then integrates",
        "use this for clock physics swinging chandelier dynamics or chaotic forced oscillation studies",
        "like a swinging weight on a string slowed by air drag and pulled by gravity",
        "theta double dot plus g over L sine theta plus b over m theta dot equals zero",
    ],
    "sir_model": [
        "compartmental SIR epidemic model with susceptible infected and recovered compartments",
        "step integrates dS over dt equals minus beta S I over N dI over dt equals beta S I over N minus gamma I and dR over dt equals gamma I",
        "use this for outbreak forecasting vaccination policy planning or qualitative epidemic shape estimation",
        "like watching dye diffuse in a tank where infected mixes with susceptible and slowly turns into recovered",
        "basic reproduction number R naught equals beta over gamma controls whether the epidemic grows or dies",
    ],
    "lotka_volterra": [
        "lotka volterra predator prey ODEs with growth a death c capture b and conversion d",
        "step integrates dx over dt equals a x minus b x y and dy over dt equals minus c y plus d x y",
        "use this for ecology models fishery dynamics or any predator prey style mutual dependence",
        "like rabbits and foxes oscillating where many rabbits feed many foxes which then eat the rabbits down",
        "two coupled first order nonlinear ODEs that produce closed orbits in the phase plane",
    ],
    "michaelis_menten": [
        "michaelis menten enzyme kinetics with maximum rate vmax and michaelis constant km",
        "rate returns vmax times substrate concentration over km plus substrate",
        "use this for enzyme reaction modeling drug receptor saturation or any binding limited rate",
        "like a vending machine that processes one customer at a time saturating when the queue is long",
        "v equals V max times S over K m plus S where K m is the substrate concentration at half maximum rate",
    ],
    "kuramoto": [
        "N coupled phase oscillators with natural frequencies omegas and coupling K",
        "step advances each phase by dt times omega plus K over N times sum of sine of phase difference",
        "use this for synchronization studies firefly flashing patterns or neural oscillator networks",
        "like a crowd clapping where each person adjusts their tempo toward the average of their neighbors",
        "the kuramoto model exhibits a phase transition to coherence at a critical coupling strength",
    ],
    "hodgkin_huxley_simple": [
        "simplified leaky integrate and fire spiking neuron with refractory period",
        "step integrates membrane potential and emits a spike when v crosses threshold then enters refractory",
        "use this for spiking neural network simulations or biophysical neuron modelling at low cost",
        "like a leaky bucket of charge that empties faster when full and resets after overflow",
        "tau dv over dt equals minus v minus v rest plus r I plus reset on threshold crossing",
    ],
    "perceptron": [
        "single layer perceptron with sign activation and online learning rule",
        "predict returns plus or minus one based on weighted sum and update adjusts weights only on errors",
        "use this for linearly separable binary classification or as a teaching baseline for ML",
        "like a stubborn judge who only adjusts opinion when caught wrong",
        "rosenblatts learning rule w next equals w plus lr times y true times x on misclassification",
    ],
    "softmax_classifier": [
        "multinomial logistic regression with softmax outputs and cross entropy gradient",
        "predict proba returns class probabilities and update applies the softmax cross entropy gradient",
        "use this for multiclass classification of feature vectors when classes are mutually exclusive",
        "like a panel of judges each scoring the input and converting scores to probabilities",
        "softmax of z i equals exp z i over sum of exp z j and gradient is one hot label minus prediction",
    ],
    "naive_bayes": [
        "gaussian naive bayes assuming feature independence given the class",
        "fit estimates per class means variances and priors and predict picks the class with highest log posterior",
        "use this as a fast baseline classifier when features are roughly independent and gaussian per class",
        "like a doctor multiplying independent symptom likelihoods to weigh competing diagnoses",
        "log p class given x proportional to log prior plus sum of log gaussian likelihood per feature",
    ],
    "pca_2d": [
        "principal component analysis for two dimensional points returning the top axis and mean",
        "fit computes the two by two covariance matrix and returns the eigenvector of the larger eigenvalue",
        "use this to find the dominant direction of spread in a planar dataset or compress to one dimension",
        "like rotating a scatter plot until most of the variance lines up with the horizontal axis",
        "axis is the eigenvector of the covariance matrix with the largest eigenvalue",
    ],
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for cid in CLASSES:
            if cid not in DESCRIPTIONS:
                print(f"WARN: no descriptions for {cid}")
                continue
            descs = DESCRIPTIONS[cid]
            if len(descs) != 5:
                print(f"WARN: {cid} has {len(descs)} descriptions, expected 5")
            code = CLASSES[cid]
            for v, desc in enumerate(descs):
                rec = {
                    "class_id":    cid,
                    "variation":   v,
                    "description": desc,
                    "code":        code,
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
    print(f"Wrote {n} (description, code) pairs across {len(CLASSES)} classes "
          f"to {OUT}")


if __name__ == "__main__":
    main()
