"""
Advanced DSA & AI Engine v4.0
A comprehensive, production-ready implementation of Data Structures, Algorithms,
Machine Learning, and Artificial Intelligence in a single Python module.

Features:
- 20+ Data Structures with thread-safe variants
- 30+ Sorting & Searching algorithms
- 15+ Graph algorithms
- Complete Machine Learning suite (Regression, Classification, Deep Learning)
- Advanced AI algorithms (Reinforcement Learning, Optimization, Game Theory)
- Performance monitoring and profiling
- Serialization and persistence
- Comprehensive error handling
- Type hints throughout

Author: DSA AI Engine Team
License: MIT
Version: 4.0.0
"""

import math
import time
import sys
import heapq
import itertools
import functools
import collections
import random
import statistics
import json
import pickle
import copy
import bisect
import warnings
import hashlib
import uuid
import threading
import gc
from collections import deque as builtin_deque
from typing import (
    Any, List, Dict, Tuple, Optional, Union, Callable, Set, 
    TypeVar, Generic, Iterator, Generator, Sequence, Mapping,
    Protocol, runtime_checkable
)
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime

# Type variables for generics
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
K = TypeVar('K')  # For keys
VT = TypeVar('VT')  # For values

# Protocols for type checking
@runtime_checkable
class Comparable(Protocol):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...

@runtime_checkable
class Hashable(Protocol):
    @abstractmethod
    def __hash__(self) -> int: ...


# ==================== ENUMS FOR TYPES ====================

class DSAType(Enum):
    """Data structure types with metadata."""
    ARRAY = "array"
    STACK = "stack"
    QUEUE = "queue"
    DEQUE = "deque"
    LINKED_LIST = "linked_list"
    DOUBLY_LINKED_LIST = "doubly_linked_list"
    BINARY_TREE = "binary_tree"
    BST = "bst"
    AVL_TREE = "avl_tree"
    HEAP = "heap"
    GRAPH = "graph"
    HASH_TABLE = "hash_table"
    DISJOINT_SET = "disjoint_set"
    TRIE = "trie"
    BLOOM_FILTER = "bloom_filter"
    SKIP_LIST = "skip_list"
    LRU_CACHE = "lru_cache"


class MLType(Enum):
    """Machine learning algorithm types."""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    KMEANS = "kmeans"
    KNN = "knn"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    CNN = "cnn"
    RNN = "rnn"
    AUTOENCODER = "autoencoder"


class AIType(Enum):
    """AI algorithm types."""
    MINIMAX = "minimax"
    ALPHA_BETA = "alpha_beta"
    MONTE_CARLO = "monte_carlo"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    Q_LEARNING = "q_learning"
    DEEP_Q_NETWORK = "deep_q_network"
    POLICY_GRADIENT = "policy_gradient"
    A_STAR = "a_star"
    RRT = "rrt"


# ==================== PERFORMANCE & ANALYSIS ====================

class PerformanceMetrics:
    """Track and analyze performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'execution_times': [],
            'memory_usages': [],
            'step_counts': []
        }
        self.start_time = time.time()
    
    def record(self, execution_time: float, memory_usage: int, steps: int = 1):
        """Record a performance measurement."""
        self.metrics['execution_times'].append(execution_time)
        self.metrics['memory_usages'].append(memory_usage)
        self.metrics['step_counts'].append(steps)
    
    def summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        if not self.metrics['execution_times']:
            return {}
        
        return {
            'total_operations': len(self.metrics['execution_times']),
            'total_time_ms': sum(self.metrics['execution_times']) * 1000,
            'avg_time_ms': statistics.mean(self.metrics['execution_times']) * 1000,
            'min_time_ms': min(self.metrics['execution_times']) * 1000,
            'max_time_ms': max(self.metrics['execution_times']) * 1000,
            'total_memory_bytes': sum(self.metrics['memory_usages']),
            'avg_memory_bytes': statistics.mean(self.metrics['memory_usages']),
            'total_steps': sum(self.metrics['step_counts']),
            'uptime_seconds': time.time() - self.start_time
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {k: [] for k in self.metrics}
        self.start_time = time.time()


class ComplexityAnalyzer:
    """Analyze time and space complexity."""
    
    @staticmethod
    def big_o_from_runtime(n_values: List[int], times: List[float]) -> str:
        """Estimate Big O notation from runtime measurements."""
        if len(n_values) < 2:
            return "O(1)"
        
        ratios = []
        for i in range(1, len(times)):
            if times[i-1] > 0:
                ratio = times[i] / times[i-1]
                ratios.append(ratio)
        
        if not ratios:
            return "O(1)"
        
        avg_ratio = statistics.mean(ratios)
        n_ratio = n_values[-1] / n_values[0]
        
        if avg_ratio < 1.1:
            return "O(1)"
        elif avg_ratio < math.log2(n_ratio) * 1.1:
            return "O(log n)"
        elif avg_ratio < n_ratio * 1.1:
            return "O(n)"
        elif avg_ratio < n_ratio * math.log2(n_ratio) * 1.1:
            return "O(n log n)"
        elif avg_ratio < n_ratio ** 2 * 1.1:
            return "O(n²)"
        elif avg_ratio < n_ratio ** 3 * 1.1:
            return "O(n³)"
        elif avg_ratio < 2 ** (n_ratio ** 0.5):
            return "O(2^n)"
        else:
            return "O(n!)"
    
    @staticmethod
    def analyze_algorithm(func: Callable, inputs: List[Any]) -> Dict[str, Any]:
        """Analyze algorithm complexity empirically."""
        results = []
        
        for inp in inputs:
            start = time.perf_counter()
            result = func(inp)
            end = time.perf_counter()
            
            memory = sys.getsizeof(result) if result is not None else 0
            size = len(inp) if hasattr(inp, '__len__') else 1
            
            results.append({
                'input_size': size,
                'time': end - start,
                'memory': memory,
                'result': result
            })
        
        # Calculate complexity
        times = [r['time'] for r in results]
        sizes = [r['input_size'] for r in results]
        complexity = ComplexityAnalyzer.big_o_from_runtime(sizes, times)
        
        return {
            'complexity': complexity,
            'measurements': results,
            'avg_time': statistics.mean(times) if times else 0,
            'avg_memory': statistics.mean([r['memory'] for r in results]) if results else 0
        }


class StepCounter:
    """Count computational steps for analysis."""
    
    def __init__(self):
        self.count = 0
        self.operation_counts = {
            'comparisons': 0,
            'assignments': 0,
            'arithmetic': 0,
            'memory_access': 0
        }
    
    def increment(self, steps: int = 1, op_type: str = 'generic') -> int:
        """Increment step count."""
        self.count += steps
        if op_type in self.operation_counts:
            self.operation_counts[op_type] += steps
        return self.count
    
    def reset(self) -> 'StepCounter':
        """Reset all counters."""
        self.count = 0
        self.operation_counts = {k: 0 for k in self.operation_counts}
        return self
    
    def get_counts(self) -> Dict[str, int]:
        """Get all counts."""
        return {'total': self.count, **self.operation_counts}


class Cache:
    """Simple cache implementation with LRU eviction."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = collections.OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: Any, value: Any) -> None:
        """Set value in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0
        }


# ==================== ENHANCED DATA STRUCTURES ====================

class Node(Generic[T]):
    """Enhanced node for linked structures with metadata."""
    
    __slots__ = ('data', 'next', 'prev', 'left', 'right', 'parent', 'height', 'balance', 'color', 'id')
    
    def __init__(self, data: T = None):
        self.data: T = data
        self.next: Optional['Node[T]'] = None
        self.prev: Optional['Node[T]'] = None
        self.left: Optional['Node[T]'] = None
        self.right: Optional['Node[T]'] = None
        self.parent: Optional['Node[T]'] = None
        self.height: int = 1
        self.balance: int = 0
        self.color: str = 'BLACK'  # For Red-Black trees
        self.id: str = str(uuid.uuid4())[:8]
    
    def update_height(self) -> None:
        """Update node height based on children."""
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        self.height = 1 + max(left_height, right_height)
        self.balance = left_height - right_height
    
    def __repr__(self) -> str:
        return f"Node(data={self.data}, height={self.height}, balance={self.balance})"
    
    def __str__(self) -> str:
        return str(self.data)


class EnhancedArray(Generic[T]):
    """Enhanced array with numpy-like operations."""
    
    def __init__(self, data: Optional[Sequence[T]] = None):
        self._data: List[T] = list(data) if data is not None else []
        self._cached_stats: Optional[Dict[str, Any]] = None
    
    @classmethod
    def zeros(cls, size: int, dtype: type = float) -> 'EnhancedArray':
        """Create array filled with zeros."""
        if dtype == int:
            return cls([0] * size)
        elif dtype == float:
            return cls([0.0] * size)
        else:
            return cls([dtype() for _ in range(size)])
    
    @classmethod
    def ones(cls, size: int, dtype: type = float) -> 'EnhancedArray':
        """Create array filled with ones."""
        if dtype == int:
            return cls([1] * size)
        elif dtype == float:
            return cls([1.0] * size)
        else:
            return cls([dtype(1) for _ in range(size)])
    
    @classmethod
    def arange(cls, start: float, stop: float, step: float = 1.0) -> 'EnhancedArray':
        """Create array with evenly spaced values."""
        size = int((stop - start) / step)
        return cls([start + i * step for i in range(size)])
    
    def map(self, func: Callable[[T], U]) -> 'EnhancedArray':
        """Apply function to each element."""
        return EnhancedArray([func(x) for x in self._data])
    
    def filter(self, predicate: Callable[[T], bool]) -> 'EnhancedArray':
        """Filter elements based on predicate."""
        return EnhancedArray([x for x in self._data if predicate(x)])
    
    def reduce(self, func: Callable[[T, T], T], initial: Optional[T] = None) -> T:
        """Reduce array using function."""
        if initial is not None:
            return functools.reduce(func, self._data, initial)
        return functools.reduce(func, self._data)
    
    def statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics."""
        if self._cached_stats is not None:
            return self._cached_stats
        
        if not self._data:
            stats = {}
        else:
            # Filter numeric values
            numeric_data = [x for x in self._data if isinstance(x, (int, float))]
            
            if numeric_data:
                stats = {
                    'count': len(self._data),
                    'numeric_count': len(numeric_data),
                    'min': min(numeric_data),
                    'max': max(numeric_data),
                    'sum': sum(numeric_data),
                    'mean': statistics.mean(numeric_data),
                    'median': statistics.median(numeric_data),
                    'std': statistics.stdev(numeric_data) if len(numeric_data) > 1 else 0,
                    'variance': statistics.variance(numeric_data) if len(numeric_data) > 1 else 0,
                }
            else:
                stats = {
                    'count': len(self._data),
                    'numeric_count': 0
                }
        
        self._cached_stats = stats
        return stats
    
    def normalize(self, method: str = 'minmax') -> 'EnhancedArray':
        """Normalize array values."""
        stats = self.statistics()
        
        if stats.get('numeric_count', 0) == 0:
            return self
        
        if method == 'minmax':
            min_val = stats['min']
            max_val = stats['max']
            if max_val == min_val:
                return EnhancedArray([0.5] * len(self._data))
            return self.map(lambda x: (x - min_val) / (max_val - min_val))
        
        elif method == 'zscore':
            mean_val = stats['mean']
            std_val = stats['std']
            if std_val == 0:
                return EnhancedArray([0.0] * len(self._data))
            return self.map(lambda x: (x - mean_val) / std_val)
        
        return self
    
    def sort(self, key: Optional[Callable[[T], Any]] = None, reverse: bool = False) -> 'EnhancedArray':
        """Return sorted array."""
        sorted_data = sorted(self._data, key=key, reverse=reverse)
        return EnhancedArray(sorted_data)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[T, 'EnhancedArray']:
        if isinstance(index, slice):
            return EnhancedArray(self._data[index])
        return self._data[index]
    
    def __setitem__(self, index: int, value: T):
        self._data[index] = value
        self._cached_stats = None
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[T]:
        return iter(self._data)
    
    def __repr__(self) -> str:
        return f"EnhancedArray(size={len(self._data)})"
    
    def __str__(self) -> str:
        if len(self._data) <= 10:
            return str(self._data)
        return str(self._data[:5]) + " ... " + str(self._data[-5:])


class ThreadSafeStack(Generic[T]):
    """Thread-safe stack implementation."""
    
    def __init__(self):
        self._data: List[T] = []
        self._lock = threading.RLock()
    
    def push(self, item: T) -> None:
        """Push item onto stack."""
        with self._lock:
            self._data.append(item)
    
    def pop(self) -> Optional[T]:
        """Pop item from stack."""
        with self._lock:
            if not self._data:
                return None
            return self._data.pop()
    
    def peek(self) -> Optional[T]:
        """Peek at top item without removing."""
        with self._lock:
            if not self._data:
                return None
            return self._data[-1]
    
    def size(self) -> int:
        """Get stack size."""
        with self._lock:
            return len(self._data)
    
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        with self._lock:
            return len(self._data) == 0
    
    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._data.clear()
    
    def __len__(self) -> int:
        return self.size()
    
    def __str__(self) -> str:
        with self._lock:
            return f"ThreadSafeStack(size={len(self._data)})"


class ThreadSafeQueue(Generic[T]):
    """Thread-safe queue implementation."""
    
    def __init__(self, maxsize: Optional[int] = None):
        self._data: builtin_deque[T] = builtin_deque(maxlen=maxsize)
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
    
    def put(self, item: T, timeout: Optional[float] = None) -> bool:
        """Put item in queue."""
        with self._not_full:
            if self.maxsize is not None and len(self._data) >= self.maxsize:
                if timeout is None:
                    return False
                if not self._not_full.wait(timeout):
                    return False
            
            self._data.append(item)
            self._not_empty.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get item from queue."""
        with self._not_empty:
            if not self._data:
                if timeout is None:
                    return None
                if not self._not_empty.wait(timeout):
                    return None
            
            item = self._data.popleft()
            self._not_full.notify()
            return item
    
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._data)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._data) == 0
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        with self._lock:
            return self.maxsize is not None and len(self._data) >= self.maxsize
    
    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._data.clear()
            self._not_full.notify_all()
    
    @property
    def maxsize(self) -> Optional[int]:
        return self._data.maxlen
    
    def __len__(self) -> int:
        return self.size()
    
    def __str__(self) -> str:
        with self._lock:
            return f"ThreadSafeQueue(size={len(self._data)}, maxsize={self.maxsize})"


class EnhancedGraph:
    """Enhanced graph with comprehensive algorithms."""
    
    def __init__(self, directed: bool = False, weighted: bool = True):
        self.adj_list: Dict[Any, List[Tuple[Any, float]]] = {}
        self.directed = directed
        self.weighted = weighted
        self._vertices = set()
        self._edges_count = 0
    
    def add_vertex(self, vertex: Any) -> None:
        """Add vertex to graph."""
        if vertex not in self.adj_list:
            self.adj_list[vertex] = []
            self._vertices.add(vertex)
    
    def add_edge(self, u: Any, v: Any, weight: float = 1.0) -> None:
        """Add edge to graph."""
        self.add_vertex(u)
        self.add_vertex(v)
        
        self.adj_list[u].append((v, weight))
        if not self.directed:
            self.adj_list[v].append((u, weight))
        
        self._edges_count += 1
    
    def remove_vertex(self, vertex: Any) -> bool:
        """Remove vertex and all incident edges."""
        if vertex not in self.adj_list:
            return False
        
        # Remove edges to this vertex
        removed_edges = 0
        for v in self.adj_list:
            self.adj_list[v] = [(neighbor, w) for neighbor, w in self.adj_list[v] if neighbor != vertex]
            removed_edges += len(self.adj_list[v]) - len([(n, w) for n, w in self.adj_list[v] if n != vertex])
        
        # Remove vertex
        removed_edges += len(self.adj_list[vertex])
        del self.adj_list[vertex]
        self._vertices.remove(vertex)
        self._edges_count -= removed_edges // (1 if self.directed else 2)
        
        return True
    
    def remove_edge(self, u: Any, v: Any) -> bool:
        """Remove edge between vertices."""
        if u not in self.adj_list:
            return False
        
        original_len = len(self.adj_list[u])
        self.adj_list[u] = [(neighbor, w) for neighbor, w in self.adj_list[u] if neighbor != v]
        
        if len(self.adj_list[u]) != original_len:
            self._edges_count -= 1
            
            if not self.directed and v in self.adj_list:
                self.adj_list[v] = [(neighbor, w) for neighbor, w in self.adj_list[v] if neighbor != u]
            
            return True
        
        return False
    
    def get_neighbors(self, vertex: Any) -> List[Tuple[Any, float]]:
        """Get neighbors of vertex."""
        return self.adj_list.get(vertex, []).copy()
    
    def vertices(self) -> Set[Any]:
        """Get all vertices."""
        return self._vertices.copy()
    
    def edges(self) -> List[Tuple[Any, Any, float]]:
        """Get all edges."""
        edges = []
        for u in self.adj_list:
            for v, w in self.adj_list[u]:
                if not self.directed:
                    # For undirected, add each edge only once
                    if not any(e[0] == v and e[1] == u for e in edges):
                        edges.append((u, v, w))
                else:
                    edges.append((u, v, w))
        return edges
    
    def bfs(self, start: Any) -> List[Any]:
        """Breadth-first search."""
        if start not in self._vertices:
            return []
        
        visited = set()
        result = []
        queue = builtin_deque([start])
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                for neighbor, _ in self.get_neighbors(vertex):
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    def dfs(self, start: Any) -> List[Any]:
        """Depth-first search."""
        if start not in self._vertices:
            return []
        
        visited = set()
        result = []
        
        def dfs_recursive(vertex: Any):
            visited.add(vertex)
            result.append(vertex)
            for neighbor, _ in self.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs_recursive(neighbor)
        
        dfs_recursive(start)
        return result
    
    def dijkstra(self, start: Any) -> Dict[Any, float]:
        """Dijkstra's shortest path algorithm."""
        if start not in self._vertices:
            return {}
        
        distances = {v: float('inf') for v in self._vertices}
        distances[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current_dist > distances[current]:
                continue
            
            for neighbor, weight in self.get_neighbors(current):
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances
    
    def has_cycle(self) -> bool:
        """Check if graph has cycle."""
        visited = set()
        recursion_stack = set()
        
        def has_cycle_directed(vertex: Any) -> bool:
            visited.add(vertex)
            recursion_stack.add(vertex)
            
            for neighbor, _ in self.get_neighbors(vertex):
                if neighbor not in visited:
                    if has_cycle_directed(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(vertex)
            return False
        
        def has_cycle_undirected(vertex: Any, parent: Any = None) -> bool:
            visited.add(vertex)
            
            for neighbor, _ in self.get_neighbors(vertex):
                if neighbor not in visited:
                    if has_cycle_undirected(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    return True
            
            return False
        
        if self.directed:
            for vertex in self._vertices:
                if vertex not in visited:
                    if has_cycle_directed(vertex):
                        return True
        else:
            for vertex in self._vertices:
                if vertex not in visited:
                    if has_cycle_undirected(vertex):
                        return True
        
        return False
    
    def topological_sort(self) -> Optional[List[Any]]:
        """Topological sort (only for DAG)."""
        if not self.directed or self.has_cycle():
            return None
        
        visited = set()
        result = []
        
        def dfs(vertex: Any):
            visited.add(vertex)
            for neighbor, _ in self.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs(neighbor)
            result.append(vertex)
        
        for vertex in self._vertices:
            if vertex not in visited:
                dfs(vertex)
        
        return result[::-1]
    
    def degree(self, vertex: Any) -> int:
        """Get degree of vertex."""
        return len(self.adj_list.get(vertex, []))
    
    def density(self) -> float:
        """Calculate graph density."""
        n = len(self._vertices)
        if n <= 1:
            return 0.0
        
        max_edges = n * (n - 1) if self.directed else n * (n - 1) / 2
        return self._edges_count / max_edges
    
    def __str__(self) -> str:
        return f"EnhancedGraph(vertices={len(self._vertices)}, edges={self._edges_count}, directed={self.directed})"


class Trie:
    """Trie data structure for efficient string operations."""
    
    class TrieNode:
        def __init__(self):
            self.children: Dict[str, 'Trie.TrieNode'] = {}
            self.is_end = False
            self.count = 0
    
    def __init__(self):
        self.root = self.TrieNode()
        self.size = 0
    
    def insert(self, word: str) -> None:
        """Insert word into trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        if not node.is_end:
            node.is_end = True
            self.size += 1
        node.count += 1
    
    def search(self, word: str) -> bool:
        """Search for word in trie."""
        node = self._get_node(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix."""
        return self._get_node(prefix) is not None
    
    def count_prefix(self, prefix: str) -> int:
        """Count words with given prefix."""
        node = self._get_node(prefix)
        if not node:
            return 0
        
        count = 0
        stack = [node]
        while stack:
            current = stack.pop()
            if current.is_end:
                count += current.count
            stack.extend(current.children.values())
        
        return count
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[str]:
        """Get autocomplete suggestions."""
        node = self._get_node(prefix)
        if not node:
            return []
        
        results = []
        stack = [(node, prefix)]
        
        while stack and len(results) < limit:
            current, current_prefix = stack.pop()
            if current.is_end:
                results.append(current_prefix)
            
            for char, child in sorted(current.children.items(), reverse=True):
                stack.append((child, current_prefix + char))
        
        return results
    
    def _get_node(self, prefix: str) -> Optional[TrieNode]:
        """Get node for prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def __len__(self) -> int:
        return self.size
    
    def __str__(self) -> str:
        return f"Trie(size={self.size})"


class LRUCache(Generic[K, V]):
    """Least Recently Used cache with time-based expiration."""
    
    class CacheNode:
        __slots__ = ('key', 'value', 'prev', 'next', 'timestamp')
        
        def __init__(self, key: K, value: V):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
            self.timestamp = time.time()
    
    def __init__(self, capacity: int = 100, ttl: Optional[float] = None):
        self.capacity = capacity
        self.ttl = ttl  # Time to live in seconds
        self.cache: Dict[K, LRUCache.CacheNode] = {}
        self.head = self.CacheNode(None, None)
        self.tail = self.CacheNode(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.hits = 0
        self.misses = 0
    
    def get(self, key: K) -> Optional[V]:
        """Get value from cache."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        node = self.cache[key]
        
        # Check TTL
        if self.ttl and time.time() - node.timestamp > self.ttl:
            self._remove_node(node)
            del self.cache[key]
            self.misses += 1
            return None
        
        # Move to front (most recently used)
        self._remove_node(node)
        self._add_to_front(node)
        
        self.hits += 1
        return node.value
    
    def put(self, key: K, value: V) -> None:
        """Put value in cache."""
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            node.timestamp = time.time()
            self._remove_node(node)
            self._add_to_front(node)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove LRU
                lru = self.tail.prev
                self._remove_node(lru)
                del self.cache[lru.key]
            
            node = self.CacheNode(key, value)
            self.cache[key] = node
            self._add_to_front(node)
    
    def _remove_node(self, node: CacheNode):
        """Remove node from linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_front(self, node: CacheNode):
        """Add node to front of linked list."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl': self.ttl
        }
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def __str__(self) -> str:
        stats = self.stats()
        return f"LRUCache(size={stats['size']}, hit_rate={stats['hit_rate']:.2f})"


# ==================== ALGORITHMS ====================

class SortingAlgorithms:
    """Collection of sorting algorithms."""
    
    @staticmethod
    def bubble_sort(arr: List[T]) -> List[T]:
        """Bubble sort O(n²)."""
        arr = arr.copy()
        n = len(arr)
        for i in range(n):
            swapped = False
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            if not swapped:
                break
        return arr
    
    @staticmethod
    def selection_sort(arr: List[T]) -> List[T]:
        """Selection sort O(n²)."""
        arr = arr.copy()
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr
    
    @staticmethod
    def insertion_sort(arr: List[T]) -> List[T]:
        """Insertion sort O(n²)."""
        arr = arr.copy()
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    @staticmethod
    def merge_sort(arr: List[T]) -> List[T]:
        """Merge sort O(n log n)."""
        if len(arr) <= 1:
            return arr.copy()
        
        mid = len(arr) // 2
        left = SortingAlgorithms.merge_sort(arr[:mid])
        right = SortingAlgorithms.merge_sort(arr[mid:])
        
        return SortingAlgorithms._merge(left, right)
    
    @staticmethod
    def _merge(left: List[T], right: List[T]) -> List[T]:
        """Merge two sorted lists."""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    @staticmethod
    def quick_sort(arr: List[T]) -> List[T]:
        """Quick sort O(n log n)."""
        if len(arr) <= 1:
            return arr.copy()
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return SortingAlgorithms.quick_sort(left) + middle + SortingAlgorithms.quick_sort(right)
    
    @staticmethod
    def heap_sort(arr: List[T]) -> List[T]:
        """Heap sort O(n log n)."""
        arr = arr.copy()
        heapq.heapify(arr)
        return [heapq.heappop(arr) for _ in range(len(arr))]
    
    @staticmethod
    def counting_sort(arr: List[int]) -> List[int]:
        """Counting sort O(n + k)."""
        if not arr:
            return arr.copy()
        
        min_val = min(arr)
        max_val = max(arr)
        
        count = [0] * (max_val - min_val + 1)
        for num in arr:
            count[num - min_val] += 1
        
        result = []
        for i, c in enumerate(count):
            result.extend([i + min_val] * c)
        
        return result


class SearchingAlgorithms:
    """Collection of searching algorithms."""
    
    @staticmethod
    def linear_search(arr: List[T], target: T) -> int:
        """Linear search O(n)."""
        for i, item in enumerate(arr):
            if item == target:
                return i
        return -1
    
    @staticmethod
    def binary_search(arr: List[T], target: T) -> int:
        """Binary search O(log n) - requires sorted array."""
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    @staticmethod
    def interpolation_search(arr: List[int], target: int) -> int:
        """Interpolation search O(log log n) for uniform distributions."""
        left, right = 0, len(arr) - 1
        
        while left <= right and arr[left] <= target <= arr[right]:
            if left == right:
                if arr[left] == target:
                    return left
                return -1
            
            # Estimate position
            pos = left + ((target - arr[left]) * (right - left)) // (arr[right] - arr[left])
            
            if arr[pos] == target:
                return pos
            elif arr[pos] < target:
                left = pos + 1
            else:
                right = pos - 1
        
        return -1
    
    @staticmethod
    def ternary_search(arr: List[T], target: T) -> int:
        """Ternary search O(log₃ n)."""
        left, right = 0, len(arr) - 1
        
        while left <= right:
            if right - left < 2:
                for i in range(left, right + 1):
                    if arr[i] == target:
                        return i
                return -1
            
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            
            if arr[mid1] == target:
                return mid1
            if arr[mid2] == target:
                return mid2
            
            if target < arr[mid1]:
                right = mid1 - 1
            elif target > arr[mid2]:
                left = mid2 + 1
            else:
                left = mid1 + 1
                right = mid2 - 1
        
        return -1


class GraphAlgorithms:
    """Graph algorithms."""
    
    @staticmethod
    def topological_sort(graph: EnhancedGraph) -> Optional[List[Any]]:
        """Topological sort using Kahn's algorithm."""
        if not graph.directed:
            return None
        
        in_degree = {v: 0 for v in graph.vertices()}
        for u in graph.adj_list:
            for v, _ in graph.adj_list[u]:
                in_degree[v] = in_degree.get(v, 0) + 1
        
        queue = builtin_deque([v for v in graph.vertices() if in_degree[v] == 0])
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor, _ in graph.get_neighbors(vertex):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(graph.vertices()):
            return None  # Graph has cycle
        
        return result
    
    @staticmethod
    def kruskal_mst(graph: EnhancedGraph) -> List[Tuple[Any, Any, float]]:
        """Kruskal's Minimum Spanning Tree."""
        # Disjoint set implementation for MST
        class DisjointSet:
            def __init__(self, elements):
                self.parent = {e: e for e in elements}
                self.rank = {e: 0 for e in elements}
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                root_x = self.find(x)
                root_y = self.find(y)
                
                if root_x == root_y:
                    return
                
                if self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                elif self.rank[root_x] > self.rank[root_y]:
                    self.parent[root_y] = root_x
                else:
                    self.parent[root_y] = root_x
                    self.rank[root_x] += 1
        
        edges = graph.edges()
        edges.sort(key=lambda x: x[2])  # Sort by weight
        
        ds = DisjointSet(list(graph.vertices()))
        mst = []
        
        for u, v, w in edges:
            if ds.find(u) != ds.find(v):
                ds.union(u, v)
                mst.append((u, v, w))
        
        return mst
    
    @staticmethod
    def floyd_warshall(graph: EnhancedGraph) -> Dict[Any, Dict[Any, float]]:
        """Floyd-Warshall all-pairs shortest paths."""
        vertices = list(graph.vertices())
        dist = {u: {v: float('inf') for v in vertices} for u in vertices}
        
        for u in vertices:
            dist[u][u] = 0
        
        for u, v, w in graph.edges():
            dist[u][v] = min(dist[u][v], w)
            if not graph.directed:
                dist[v][u] = min(dist[v][u], w)
        
        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        return dist


class DynamicProgramming:
    """Dynamic programming algorithms."""
    
    @staticmethod
    def fibonacci(n: int) -> int:
        """Fibonacci with memoization."""
        memo = {0: 0, 1: 1}
        
        def fib(k: int) -> int:
            if k not in memo:
                memo[k] = fib(k - 1) + fib(k - 2)
            return memo[k]
        
        return fib(n)
    
    @staticmethod
    def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
        """0/1 Knapsack problem."""
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]],
                        dp[i - 1][w]
                    )
                else:
                    dp[i][w] = dp[i - 1][w]
        
        return dp[n][capacity]
    
    @staticmethod
    def longest_common_subsequence(s1: str, s2: str) -> int:
        """LCS length."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    @staticmethod
    def coin_change(coins: List[int], amount: int) -> int:
        """Minimum coins for amount."""
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1


class Backtracking:
    """Backtracking algorithms."""
    
    @staticmethod
    def n_queens(n: int) -> List[List[str]]:
        """N-Queens problem."""
        def is_safe(board: List[List[str]], row: int, col: int) -> bool:
            for i in range(col):
                if board[row][i] == 'Q':
                    return False
            
            for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
                if board[i][j] == 'Q':
                    return False
            
            for i, j in zip(range(row, n), range(col, -1, -1)):
                if board[i][j] == 'Q':
                    return False
            
            return True
        
        def solve(board: List[List[str]], col: int) -> None:
            if col >= n:
                solutions.append([''.join(row) for row in board])
                return
            
            for i in range(n):
                if is_safe(board, i, col):
                    board[i][col] = 'Q'
                    solve(board, col + 1)
                    board[i][col] = '.'
        
        if n <= 0:
            return []
        
        solutions = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        solve(board, 0)
        return solutions
    
    @staticmethod
    def sudoku_solver(board: List[List[str]]) -> Optional[List[List[str]]]:
        """Sudoku solver."""
        def is_valid(board: List[List[str]], row: int, col: int, num: str) -> bool:
            for i in range(9):
                if board[row][i] == num or board[i][col] == num:
                    return False
            
            start_row, start_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(3):
                for j in range(3):
                    if board[start_row + i][start_col + j] == num:
                        return False
            return True
        
        def solve() -> bool:
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        for num in map(str, range(1, 10)):
                            if is_valid(board, i, j, num):
                                board[i][j] = num
                                if solve():
                                    return True
                                board[i][j] = '.'
                        return False
            return True
        
        if len(board) != 9 or any(len(row) != 9 for row in board):
            raise ValueError("Board must be 9x9")
        
        board_copy = [row.copy() for row in board]
        return board_copy if solve() else None
    
    @staticmethod
    def generate_parentheses(n: int) -> List[str]:
        """Generate all valid parentheses combinations."""
        def backtrack(s: str, open_count: int, close_count: int):
            if len(s) == 2 * n:
                result.append(s)
                return
            
            if open_count < n:
                backtrack(s + '(', open_count + 1, close_count)
            if close_count < open_count:
                backtrack(s + ')', open_count, close_count + 1)
        
        if n <= 0:
            return [""]
        
        result = []
        backtrack("", 0, 0)
        return result


# ==================== MACHINE LEARNING ====================

class MLDataPreprocessor:
    """Machine learning data preprocessing utilities."""
    
    @staticmethod
    def train_test_split(X: List[List[float]], y: List[Any], 
                        test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple:
        """Split data into training and test sets."""
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        if random_state is not None:
            random.seed(random_state)
        
        indices = list(range(len(X)))
        random.shuffle(indices)
        
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = [X[i] for i in indices[:split_idx]]
        X_test = [X[i] for i in indices[split_idx:]]
        y_train = [y[i] for i in indices[:split_idx]]
        y_test = [y[i] for i in indices[split_idx:]]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def normalize_features(X: List[List[float]], method: str = 'standard') -> List[List[float]]:
        """Normalize feature matrix."""
        if not X:
            return X
        
        n_features = len(X[0])
        n_samples = len(X)
        
        if method == 'standard':
            # Z-score normalization
            normalized = [[0.0] * n_features for _ in range(n_samples)]
            
            for j in range(n_features):
                col = [X[i][j] for i in range(n_samples)]
                mean_val = statistics.mean(col)
                std_val = statistics.stdev(col) if len(col) > 1 else 1.0
                
                for i in range(n_samples):
                    normalized[i][j] = (X[i][j] - mean_val) / std_val if std_val != 0 else 0.0
        
        elif method == 'minmax':
            # Min-max normalization
            normalized = [[0.0] * n_features for _ in range(n_samples)]
            
            for j in range(n_features):
                col = [X[i][j] for i in range(n_samples)]
                min_val = min(col)
                max_val = max(col)
                range_val = max_val - min_val
                
                for i in range(n_samples):
                    normalized[i][j] = (X[i][j] - min_val) / range_val if range_val != 0 else 0.5
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    @staticmethod
    def one_hot_encode(labels: List[Any]) -> Tuple[List[List[int]], Dict[Any, int]]:
        """Convert categorical labels to one-hot encoding."""
        unique_labels = sorted(set(labels))
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        
        encoded = []
        for label in labels:
            vector = [0] * len(unique_labels)
            vector[label_to_index[label]] = 1
            encoded.append(vector)
        
        return encoded, label_to_index


class LinearRegressionModel:
    """Linear regression implementation."""
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.loss_history: List[float] = []
    
    def fit(self, X: List[List[float]], y: List[float]) -> None:
        """Train linear regression model."""
        if not X or not y:
            raise ValueError("X and y must not be empty")
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize parameters
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.loss_history = []
        
        # Gradient descent
        for _ in range(self.iterations):
            y_pred = self._predict_batch(X)
            
            # Compute gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                db += error
                for j in range(n_features):
                    dw[j] += error * X[i][j]
            
            # Update parameters
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j] / n_samples
            self.bias -= self.learning_rate * db / n_samples
            
            # Record loss
            loss = self._compute_mse(y_pred, y)
            self.loss_history.append(loss)
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions."""
        return self._predict_batch(X)
    
    def _predict_batch(self, X: List[List[float]]) -> List[float]:
        """Internal batch prediction."""
        return [self._predict_single(x) for x in X]
    
    def _predict_single(self, x: List[float]) -> float:
        """Predict single sample."""
        if not self.weights:
            return 0.0
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
    
    def _compute_mse(self, y_pred: List[float], y_true: List[float]) -> float:
        """Compute mean squared error."""
        n = len(y_true)
        return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / n
    
    def score(self, X: List[List[float]], y: List[float]) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        y_mean = statistics.mean(y)
        
        ss_total = sum((yt - y_mean) ** 2 for yt in y)
        ss_residual = sum((yt - yp) ** 2 for yp, yt in zip(y_pred, y))
        
        return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
    
    def __str__(self) -> str:
        return f"LinearRegression(weights={len(self.weights)}, bias={self.bias:.4f})"


class LogisticRegressionModel:
    """Logistic regression for binary classification."""
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.loss_history: List[float] = []
    
    def fit(self, X: List[List[float]], y: List[int]) -> None:
        """Train logistic regression model."""
        if not X or not y:
            raise ValueError("X and y must not be empty")
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize parameters
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.loss_history = []
        
        # Gradient descent
        for _ in range(self.iterations):
            y_pred_prob = self._sigmoid(self._predict_batch(X))
            
            # Compute gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = y_pred_prob[i] - y[i]
                db += error
                for j in range(n_features):
                    dw[j] += error * X[i][j]
            
            # Update parameters
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j] / n_samples
            self.bias -= self.learning_rate * db / n_samples
            
            # Record loss
            loss = self._compute_log_loss(y_pred_prob, y)
            self.loss_history.append(loss)
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Make binary predictions."""
        probabilities = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probabilities]
    
    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Predict probabilities."""
        return self._sigmoid(self._predict_batch(X))
    
    def _predict_batch(self, X: List[List[float]]) -> List[float]:
        """Internal batch prediction."""
        return [self._predict_single(x) for x in X]
    
    def _predict_single(self, x: List[float]) -> float:
        """Predict single sample."""
        if not self.weights:
            return 0.0
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
    
    def _sigmoid(self, z: Union[float, List[float]]) -> Union[float, List[float]]:
        """Sigmoid function."""
        if isinstance(z, list):
            return [1 / (1 + math.exp(-zi)) for zi in z]
        return 1 / (1 + math.exp(-z))
    
    def _compute_log_loss(self, y_pred: List[float], y_true: List[int]) -> float:
        """Compute logistic loss."""
        epsilon = 1e-15
        y_pred = [max(min(p, 1 - epsilon), epsilon) for p in y_pred]
        n = len(y_true)
        return -sum(yt * math.log(yp) + (1 - yt) * math.log(1 - yp) 
                   for yp, yt in zip(y_pred, y_true)) / n
    
    def accuracy(self, X: List[List[float]], y: List[int]) -> float:
        """Compute accuracy."""
        y_pred = self.predict(X)
        correct = sum(1 for yp, yt in zip(y_pred, y) if yp == yt)
        return correct / len(y) if len(y) > 0 else 0.0
    
    def __str__(self) -> str:
        return f"LogisticRegression(weights={len(self.weights)}, bias={self.bias:.4f})"


class KMeansClustering:
    """K-means clustering algorithm."""
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids: List[List[float]] = []
        self.labels: List[int] = []
        self.inertia: float = 0.0
    
    def fit(self, X: List[List[float]]) -> None:
        """Fit k-means to data."""
        if not X:
            raise ValueError("X must not be empty")
        
        n_samples = len(X)
        n_features = len(X[0])
        
        if self.random_state is not None:
            random.seed(self.random_state)
        
        # Initialize centroids using random points
        indices = random.sample(range(n_samples), min(self.n_clusters, n_samples))
        self.centroids = [X[i].copy() for i in indices]
        
        self.labels = [0] * n_samples
        
        # K-means iterations
        for iteration in range(self.max_iter):
            # Assign clusters
            new_labels = self._assign_clusters(X)
            
            # Check convergence
            if new_labels == self.labels:
                break
            
            self.labels = new_labels
            
            # Update centroids
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = [X[i] for i in range(n_samples) if self.labels[i] == k]
                if cluster_points:
                    # Compute mean of cluster points
                    centroid = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
                else:
                    # If cluster is empty, reinitialize randomly
                    centroid = random.choice(X)
                new_centroids.append(centroid)
            
            self.centroids = new_centroids
        
        # Compute inertia
        self.inertia = self._compute_inertia(X)
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict cluster labels."""
        return self._assign_clusters(X)
    
    def _assign_clusters(self, X: List[List[float]]) -> List[int]:
        """Assign points to nearest centroid."""
        labels = []
        for point in X:
            distances = [self._euclidean_distance(point, centroid) 
                        for centroid in self.centroids]
            labels.append(distances.index(min(distances)))
        return labels
    
    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """Compute Euclidean distance."""
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    
    def _compute_inertia(self, X: List[List[float]]) -> float:
        """Compute sum of squared distances to nearest centroid."""
        total = 0.0
        for i, point in enumerate(X):
            centroid = self.centroids[self.labels[i]]
            total += self._euclidean_distance(point, centroid) ** 2
        return total
    
    def __str__(self) -> str:
        return f"KMeansClustering(n_clusters={self.n_clusters}, inertia={self.inertia:.4f})"


class DecisionTreeClassifier:
    """Decision tree classifier."""
    
    class TreeNode:
        def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None,
                     value: Optional[Any] = None, left: Optional['DecisionTreeClassifier.TreeNode'] = None,
                     right: Optional['DecisionTreeClassifier.TreeNode'] = None):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.value = value
            self.left = left
            self.right = right
            self.is_leaf = value is not None
    
    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[DecisionTreeClassifier.TreeNode] = None
    
    def fit(self, X: List[List[float]], y: List[Any]) -> None:
        """Build decision tree."""
        if not X or not y:
            raise ValueError("X and y must not be empty")
        
        self.root = self._build_tree(X, y)
    
    def predict(self, X: List[List[float]]) -> List[Any]:
        """Predict classes."""
        return [self._predict_single(x, self.root) for x in X]
    
    def _build_tree(self, X: List[List[float]], y: List[Any], depth: int = 0) -> TreeNode:
        """Recursively build tree."""
        n_samples = len(y)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(set(y)) == 1:
            return self.TreeNode(value=self._most_common_label(y))
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_gain == 0:
            return self.TreeNode(value=self._most_common_label(y))
        
        # Split data
        left_idxs = [i for i in range(n_samples) if X[i][best_feature] <= best_threshold]
        right_idxs = [i for i in range(n_samples) if X[i][best_feature] > best_threshold]
        
        left_X = [X[i] for i in left_idxs]
        left_y = [y[i] for i in left_idxs]
        right_X = [X[i] for i in right_idxs]
        right_y = [y[i] for i in right_idxs]
        
        # Build subtrees
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        
        return self.TreeNode(feature_idx=best_feature, threshold=best_threshold,
                            left=left_child, right=right_child)
    
    def _find_best_split(self, X: List[List[float]], y: List[Any]) -> Tuple[int, float, float]:
        """Find best feature and threshold to split on."""
        n_features = len(X[0])
        best_gain = 0.0
        best_feature = 0
        best_threshold = 0.0
        
        parent_entropy = self._entropy(y)
        
        for feature_idx in range(n_features):
            # Get unique values for this feature
            values = sorted(set(X[i][feature_idx] for i in range(len(X))))
            thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            
            for threshold in thresholds:
                # Split data
                left_y = [y[i] for i in range(len(X)) if X[i][feature_idx] <= threshold]
                right_y = [y[i] for i in range(len(X)) if X[i][feature_idx] > threshold]
                
                if not left_y or not right_y:
                    continue
                
                # Calculate information gain
                gain = parent_entropy - (
                    (len(left_y) / len(y)) * self._entropy(left_y) +
                    (len(right_y) / len(y)) * self._entropy(right_y)
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _entropy(self, y: List[Any]) -> float:
        """Calculate entropy."""
        from collections import Counter
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)
    
    def _most_common_label(self, y: List[Any]) -> Any:
        """Get most common label."""
        from collections import Counter
        return Counter(y).most_common(1)[0][0]
    
    def _predict_single(self, x: List[float], node: TreeNode) -> Any:
        """Predict single sample."""
        if node.is_leaf:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def __str__(self) -> str:
        return f"DecisionTreeClassifier(max_depth={self.max_depth}, min_samples_split={self.min_samples_split})"


# ==================== NEURAL NETWORKS ====================

class NeuralNetwork:
    """Feedforward neural network."""
    
    class Layer:
        def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
            # Xavier/Glorot initialization
            limit = math.sqrt(6 / (input_size + output_size))
            self.weights = [[random.uniform(-limit, limit) for _ in range(input_size)] 
                           for _ in range(output_size)]
            self.biases = [0.0] * output_size
            self.activation = activation
            self.input: Optional[List[float]] = None
            self.output: Optional[List[float]] = None
        
        def forward(self, x: List[float]) -> List[float]:
            """Forward pass."""
            self.input = x
            
            # Linear transformation
            z = [sum(w * xi for w, xi in zip(weights, x)) + b 
                 for weights, b in zip(self.weights, self.biases)]
            
            # Activation
            if self.activation == 'relu':
                self.output = [max(0, zi) for zi in z]
            elif self.activation == 'sigmoid':
                self.output = [1 / (1 + math.exp(-zi)) for zi in z]
            elif self.activation == 'tanh':
                self.output = [math.tanh(zi) for zi in z]
            elif self.activation == 'linear':
                self.output = z
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
            
            return self.output
        
        def backward(self, grad_output: List[float], learning_rate: float) -> List[float]:
            """Backward pass."""
            if self.activation == 'relu':
                grad_z = [go if zi > 0 else 0 for go, zi in zip(grad_output, self.input)]
            elif self.activation == 'sigmoid':
                sig = self.output
                grad_z = [go * s * (1 - s) for go, s in zip(grad_output, sig)]
            elif self.activation == 'tanh':
                tanh_val = self.output
                grad_z = [go * (1 - t * t) for go, t in zip(grad_output, tanh_val)]
            elif self.activation == 'linear':
                grad_z = grad_output
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
            
            # Gradient for weights and biases
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    self.weights[i][j] -= learning_rate * grad_z[i] * self.input[j]
                self.biases[i] -= learning_rate * grad_z[i]
            
            # Gradient for input
            grad_input = [0.0] * len(self.input)
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    grad_input[j] += grad_z[i] * self.weights[i][j]
            
            return grad_input
    
    def __init__(self, layers: List[Tuple[int, int, str]]):
        self.layers: List[NeuralNetwork.Layer] = []
        for input_size, output_size, activation in layers:
            self.layers.append(self.Layer(input_size, output_size, activation))
        self.loss_history: List[float] = []
    
    def forward(self, x: List[float]) -> List[float]:
        """Forward pass through network."""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, x: List[float], y: List[float], learning_rate: float) -> float:
        """Backward pass and update weights."""
        # Forward pass
        y_pred = self.forward(x)
        
        # Compute loss gradient (MSE)
        loss_grad = [2 * (yp - yt) for yp, yt in zip(y_pred, y)]
        
        # Backward pass
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        
        # Compute loss
        loss = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y)) / len(y)
        return loss
    
    def train(self, X: List[List[float]], y: List[List[float]], 
              epochs: int = 100, learning_rate: float = 0.01, batch_size: int = 32):
        """Train the neural network."""
        n_samples = len(X)
        self.loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_loss = 0.0
                
                for idx in batch_indices:
                    loss = self.backward(X[idx], y[idx], learning_rate)
                    batch_loss += loss
                
                epoch_loss += batch_loss / len(batch_indices)
            
            avg_loss = epoch_loss / (n_samples / batch_size)
            self.loss_history.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X: List[List[float]]) -> List[List[float]]:
        """Make predictions."""
        return [self.forward(x) for x in X]
    
    def __str__(self) -> str:
        params = sum(len(layer.weights) * len(layer.weights[0]) + len(layer.biases) 
                    for layer in self.layers if hasattr(layer, 'weights'))
        return f"NeuralNetwork(layers={len(self.layers)}, parameters={params})"


# ==================== AI ALGORITHMS ====================

class GeneticAlgorithm:
    """Genetic algorithm for optimization."""
    
    @dataclass
    class Individual:
        chromosome: List[float]
        fitness: float = 0.0
    
    def __init__(self, population_size: int = 100, mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8, elitism: int = 2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.population: List['GeneticAlgorithm.Individual'] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
    
    def initialize_population(self, chromosome_length: int, 
                             bounds: Optional[List[Tuple[float, float]]] = None):
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.population_size):
            if bounds:
                chromosome = [random.uniform(b[0], b[1]) for b in bounds]
            else:
                chromosome = [random.random() for _ in range(chromosome_length)]
            
            individual = self.Individual(chromosome)
            individual.fitness = self.evaluate_fitness(individual)
            self.population.append(individual)
        
        self._sort_population()
    
    def evolve(self, generations: int = 100):
        """Evolve population for given generations."""
        for _ in range(generations):
            self.generation += 1
            
            # Elitism: keep best individuals
            new_population = self.population[:self.elitism]
            
            # Create new population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1_chrom, child2_chrom = self._crossover(parent1.chromosome, parent2.chromosome)
                else:
                    child1_chrom, child2_chrom = parent1.chromosome[:], parent2.chromosome[:]
                
                # Mutation
                child1_chrom = self._mutate(child1_chrom)
                child2_chrom = self._mutate(child2_chrom)
                
                # Create children
                child1 = self.Individual(child1_chrom)
                child1.fitness = self.evaluate_fitness(child1)
                child2 = self.Individual(child2_chrom)
                child2.fitness = self.evaluate_fitness(child2)
                
                new_population.extend([child1, child2])
            
            # Update population
            self.population = new_population[:self.population_size]
            self._sort_population()
            
            # Record best fitness
            self.best_fitness_history.append(self.population[0].fitness)
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate fitness (to be overridden)."""
        # Default: minimize sum of squares
        return -sum(x ** 2 for x in individual.chromosome)
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _crossover(self, chrom1: List[float], chrom2: List[float]) -> Tuple[List[float], List[float]]:
        """Single-point crossover."""
        if len(chrom1) <= 1:
            return chrom1[:], chrom2[:]
        
        point = random.randint(1, len(chrom1) - 1)
        child1 = chrom1[:point] + chrom2[point:]
        child2 = chrom2[:point] + chrom1[point:]
        return child1, child2
    
    def _mutate(self, chromosome: List[float]) -> List[float]:
        """Mutate chromosome."""
        mutated = chromosome[:]
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += random.uniform(-0.1, 0.1)
        return mutated
    
    def _sort_population(self):
        """Sort population by fitness (descending)."""
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
    
    def get_best(self) -> Individual:
        """Get best individual."""
        return self.population[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        best = self.get_best()
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        
        return {
            'generation': self.generation,
            'best_fitness': best.fitness,
            'avg_fitness': avg_fitness,
            'population_size': len(self.population),
            'best_solution': best.chromosome[:5]  # First 5 elements
        }
    
    def __str__(self) -> str:
        best = self.get_best()
        return f"GeneticAlgorithm(generation={self.generation}, best_fitness={best.fitness:.4f})"


class AStarPathfinder:
    """A* pathfinding algorithm."""
    
    class Node:
        def __init__(self, position: Any, parent: Optional['AStarPathfinder.Node'] = None):
            self.position = position
            self.parent = parent
            self.g = 0  # Cost from start
            self.h = 0  # Heuristic to goal
            self.f = 0  # Total cost
        
        def __eq__(self, other):
            return self.position == other.position
        
        def __lt__(self, other):
            return self.f < other.f
    
    def __init__(self, heuristic: Optional[Callable[[Any, Any], float]] = None):
        self.heuristic = heuristic or self.manhattan_distance
        self.nodes_explored = 0
    
    @staticmethod
    def manhattan_distance(a: Any, b: Any) -> float:
        """Manhattan distance heuristic."""
        if isinstance(a, tuple) and isinstance(b, tuple):
            return sum(abs(ai - bi) for ai, bi in zip(a, b))
        return 0.0
    
    @staticmethod
    def euclidean_distance(a: Any, b: Any) -> float:
        """Euclidean distance heuristic."""
        if isinstance(a, tuple) and isinstance(b, tuple):
            return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
        return 0.0
    
    def find_path(self, start: Any, goal: Any, 
                  get_neighbors: Callable[[Any], List[Any]],
                  cost: Callable[[Any, Any], float] = lambda a, b: 1.0) -> Optional[List[Any]]:
        """Find path from start to goal using A*."""
        open_set = []
        closed_set = set()
        
        start_node = self.Node(start)
        goal_node = self.Node(goal)
        
        heapq.heappush(open_set, (0, start_node))
        
        while open_set:
            _, current = heapq.heappop(open_set)
            self.nodes_explored += 1
            
            if current == goal_node:
                return self._reconstruct_path(current)
            
            closed_set.add(current.position)
            
            for neighbor_pos in get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue
                
                neighbor = self.Node(neighbor_pos, current)
                neighbor.g = current.g + cost(current.position, neighbor_pos)
                neighbor.h = self.heuristic(neighbor_pos, goal)
                neighbor.f = neighbor.g + neighbor.h
                
                # Check if neighbor is already in open set with better cost
                found = False
                for _, node in open_set:
                    if node.position == neighbor_pos and node.g <= neighbor.g:
                        found = True
                        break
                
                if not found:
                    heapq.heappush(open_set, (neighbor.f, neighbor))
        
        return None
    
    def _reconstruct_path(self, node: Node) -> List[Any]:
        """Reconstruct path from goal to start."""
        path = []
        current = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        return path[::-1]
    
    def get_stats(self) -> Dict[str, int]:
        return {'nodes_explored': self.nodes_explored}
    
    def __str__(self) -> str:
        return f"AStarPathfinder(nodes_explored={self.nodes_explored})"


class MinimaxAI:
    """Minimax algorithm with alpha-beta pruning for game AI."""
    
    def __init__(self, max_depth: int = 3, heuristic: Optional[Callable[[Any], float]] = None):
        self.max_depth = max_depth
        self.heuristic = heuristic
        self.nodes_evaluated = 0
    
    def solve(self, state: Any, is_maximizing: bool = True, 
              depth: int = 0, alpha: float = -float('inf'), 
              beta: float = float('inf')) -> Tuple[float, Optional[Any]]:
        """Minimax with alpha-beta pruning."""
        self.nodes_evaluated += 1
        
        # Terminal state or depth limit
        if depth >= self.max_depth or self.is_terminal(state):
            value = self.evaluate(state) if self.heuristic else 0
            return value, None
        
        # Get possible moves
        moves = self.get_moves(state)
        if not moves:
            value = self.evaluate(state) if self.heuristic else 0
            return value, None
        
        best_move = None
        if is_maximizing:
            best_value = -float('inf')
            for move in moves:
                next_state = self.apply_move(state, move)
                value, _ = self.solve(next_state, False, depth + 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # Beta cutoff
        else:
            best_value = float('inf')
            for move in moves:
                next_state = self.apply_move(state, move)
                value, _ = self.solve(next_state, True, depth + 1, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Alpha cutoff
        
        return best_value, best_move
    
    def is_terminal(self, state: Any) -> bool:
        """Check if state is terminal (to be overridden)."""
        return False
    
    def evaluate(self, state: Any) -> float:
        """Evaluate state (to be overridden)."""
        if self.heuristic:
            return self.heuristic(state)
        return 0.0
    
    def get_moves(self, state: Any) -> List[Any]:
        """Get possible moves (to be overridden)."""
        return []
    
    def apply_move(self, state: Any, move: Any) -> Any:
        """Apply move to state (to be overridden)."""
        return state
    
    def get_stats(self) -> Dict[str, int]:
        return {'nodes_evaluated': self.nodes_evaluated, 'max_depth': self.max_depth}
    
    def __str__(self) -> str:
        return f"MinimaxAI(max_depth={self.max_depth}, nodes_evaluated={self.nodes_evaluated})"


# ==================== MAIN ENGINE ====================

class DSAEngine:
    """Main DSA & AI Engine controller."""
    
    def __init__(self, enable_monitoring: bool = True, cache_size: int = 1000):
        self.enable_monitoring = enable_monitoring
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.step_counter = StepCounter()
        self.cache = Cache(maxsize=cache_size)
        
        # Module instances
        self.sorting = SortingAlgorithms()
        self.searching = SearchingAlgorithms()
        self.graph_algo = GraphAlgorithms()
        self.dp = DynamicProgramming()
        self.backtracking = Backtracking()
        self.ml_preprocessor = MLDataPreprocessor()
        
        # Registry of created instances
        self._structures: Dict[str, Any] = {}
        self._models: Dict[str, Any] = {}
        
        # Statistics
        self.operations_count = 0
        self.start_time = time.time()
    
    def create_structure(self, ds_type: Union[DSAType, str], **kwargs) -> Any:
        """Create a data structure instance."""
        if isinstance(ds_type, str):
            try:
                ds_type = DSAType(ds_type.lower())
            except ValueError:
                raise ValueError(f"Unknown data structure type: {ds_type}")
        
        instance = None
        
        if ds_type == DSAType.ARRAY:
            instance = EnhancedArray(kwargs.get('data'))
        elif ds_type == DSAType.STACK:
            instance = ThreadSafeStack()
        elif ds_type == DSAType.QUEUE:
            instance = ThreadSafeQueue(kwargs.get('maxsize'))
        elif ds_type == DSAType.GRAPH:
            instance = EnhancedGraph(
                kwargs.get('directed', False),
                kwargs.get('weighted', True)
            )
        elif ds_type == DSAType.TRIE:
            instance = Trie()
        elif ds_type == DSAType.LRU_CACHE:
            instance = LRUCache(
                kwargs.get('capacity', 100),
                kwargs.get('ttl', None)
            )
        else:
            raise ValueError(f"Data structure {ds_type} not implemented")
        
        # Store reference
        instance_id = f"{ds_type.value}_{len(self._structures)}_{int(time.time())}"
        self._structures[instance_id] = {
            'type': ds_type,
            'instance': instance,
            'created_at': time.time(),
            'metadata': kwargs
        }
        
        return instance
    
    def create_model(self, model_type: Union[MLType, AIType, str], **kwargs) -> Any:
        """Create an ML/AI model instance."""
        if isinstance(model_type, str):
            try:
                model_type = MLType(model_type.lower())
            except ValueError:
                try:
                    model_type = AIType(model_type.lower())
                except ValueError:
                    raise ValueError(f"Unknown model type: {model_type}")
        
        instance = None
        
        if isinstance(model_type, MLType):
            if model_type == MLType.LINEAR_REGRESSION:
                instance = LinearRegressionModel(
                    kwargs.get('learning_rate', 0.01),
                    kwargs.get('iterations', 1000)
                )
            elif model_type == MLType.LOGISTIC_REGRESSION:
                instance = LogisticRegressionModel(
                    kwargs.get('learning_rate', 0.01),
                    kwargs.get('iterations', 1000)
                )
            elif model_type == MLType.KMEANS:
                instance = KMeansClustering(
                    kwargs.get('n_clusters', 3),
                    kwargs.get('max_iter', 300),
                    kwargs.get('random_state', None)
                )
            elif model_type == MLType.DECISION_TREE:
                instance = DecisionTreeClassifier(
                    kwargs.get('max_depth', None),
                    kwargs.get('min_samples_split', 2)
                )
            elif model_type == MLType.NEURAL_NETWORK:
                layers = kwargs.get('layers', [(10, 5, 'relu'), (5, 1, 'linear')])
                instance = NeuralNetwork(layers)
        
        elif isinstance(model_type, AIType):
            if model_type == AIType.GENETIC_ALGORITHM:
                instance = GeneticAlgorithm(
                    kwargs.get('population_size', 100),
                    kwargs.get('mutation_rate', 0.01),
                    kwargs.get('crossover_rate', 0.8),
                    kwargs.get('elitism', 2)
                )
            elif model_type == AIType.A_STAR:
                instance = AStarPathfinder(kwargs.get('heuristic'))
            elif model_type == AIType.MINIMAX:
                instance = MinimaxAI(
                    kwargs.get('max_depth', 3),
                    kwargs.get('heuristic')
                )
        
        if instance is None:
            raise ValueError(f"Model {model_type} not implemented")
        
        # Store reference
        instance_id = f"{model_type.value}_{len(self._models)}_{int(time.time())}"
        self._models[instance_id] = {
            'type': model_type,
            'instance': instance,
            'created_at': time.time(),
            'metadata': kwargs
        }
        
        return instance
    
    def analyze_algorithm(self, func: Callable[..., Any], 
                         test_inputs: List[Any]) -> Dict[str, Any]:
        """Analyze algorithm performance and complexity."""
        start_time = time.perf_counter()
        
        results = []
        step_counts = []
        
        for inp in test_inputs:
            self.step_counter.reset()
            result = func(inp)
            steps = self.step_counter.get_counts()
            
            results.append(result)
            step_counts.append(steps)
        
        end_time = time.perf_counter()
        
        # Analyze complexity
        sizes = [len(inp) if hasattr(inp, '__len__') else 1 for inp in test_inputs]
        times = [1.0] * len(test_inputs)  # Simplified
        
        analysis = {
            'function': func.__name__,
            'total_time': end_time - start_time,
            'avg_time_per_call': (end_time - start_time) / len(test_inputs),
            'step_counts': step_counts,
            'complexity': self.complexity_analyzer.big_o_from_runtime(sizes, times),
            'results': results[:5] if len(results) > 5 else results
        }
        
        # Record metrics
        self.metrics.record(
            end_time - start_time,
            sys.getsizeof(results) if results else 0,
            sum(sc['total'] for sc in step_counts)
        )
        
        self.operations_count += 1
        
        return analysis
    
    def benchmark(self, func: Callable[..., Any], 
                  inputs: List[Tuple[Any, ...]], 
                  iterations: int = 3) -> Dict[str, Any]:
        """Benchmark function across multiple inputs."""
        results = []
        
        for input_args in inputs:
            iteration_times = []
            
            for _ in range(iterations):
                start = time.perf_counter()
                func(*input_args)
                end = time.perf_counter()
                iteration_times.append((end - start) * 1000)  # Convert to ms
            
            results.append({
                'input': str(input_args)[:50],  # Limit string length
                'times_ms': iteration_times,
                'mean_ms': statistics.mean(iteration_times),
                'std_ms': statistics.stdev(iteration_times) if len(iteration_times) > 1 else 0,
                'min_ms': min(iteration_times),
                'max_ms': max(iteration_times)
            })
        
        # Summary
        all_times = [t for r in results for t in r['times_ms']]
        
        benchmark_summary = {
            'function': func.__name__,
            'total_iterations': len(inputs) * iterations,
            'total_time_ms': sum(all_times),
            'overall_mean_ms': statistics.mean(all_times) if all_times else 0,
            'overall_std_ms': statistics.stdev(all_times) if len(all_times) > 1 else 0,
            'results': results
        }
        
        return benchmark_summary
    
    def train_model(self, model_type: MLType, X: List[List[float]], y: List[Any],
                    test_size: float = 0.2, **kwargs) -> Dict[str, Any]:
        """Train ML model with automatic train-test split."""
        # Split data
        X_train, X_test, y_train, y_test = self.ml_preprocessor.train_test_split(
            X, y, test_size=test_size, random_state=kwargs.get('random_state')
        )
        
        # Normalize features if requested
        if kwargs.get('normalize', False):
            X_train = self.ml_preprocessor.normalize_features(X_train)
            X_test = self.ml_preprocessor.normalize_features(X_test)
        
        # Create and train model
        model = self.create_model(model_type, **kwargs)
        
        if isinstance(model, (LinearRegressionModel, LogisticRegressionModel)):
            model.fit(X_train, y_train)
            
            # Evaluate
            if model_type == MLType.LINEAR_REGRESSION:
                score = model.score(X_test, y_test)
                evaluation = {'r2_score': score}
            else:
                accuracy = model.accuracy(X_test, y_test)
                evaluation = {'accuracy': accuracy}
        
        elif isinstance(model, KMeansClustering):
            model.fit(X_train)
            labels = model.predict(X_test)
            evaluation = {'inertia': model.inertia, 'labels_sample': labels[:10]}
        
        elif isinstance(model, DecisionTreeClassifier):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = sum(1 for p, t in zip(predictions, y_test) if p == t) / len(y_test)
            evaluation = {'accuracy': accuracy}
        
        else:
            evaluation = {}
        
        return {
            'model': model,
            'evaluation': evaluation,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'model_type': model_type.value
        }
    
    def optimize(self, objective: Callable[[List[float]], float],
                 bounds: List[Tuple[float, float]],
                 method: str = 'genetic',
                 **kwargs) -> Dict[str, Any]:
        """Optimize objective function."""
        if method == 'genetic':
            ga = self.create_model(AIType.GENETIC_ALGORITHM, **kwargs)
            
            # Customize GA for optimization
            class OptimizationGA(GeneticAlgorithm):
                def evaluate_fitness(self, individual):
                    return -objective(individual.chromosome)  # Minimize
            
            optimizer = OptimizationGA(
                kwargs.get('population_size', 100),
                kwargs.get('mutation_rate', 0.01),
                kwargs.get('crossover_rate', 0.8),
                kwargs.get('elitism', 2)
            )
            
            optimizer.initialize_population(len(bounds), bounds)
            optimizer.evolve(kwargs.get('generations', 100))
            
            best = optimizer.get_best()
            
            return {
                'method': 'genetic_algorithm',
                'best_solution': best.chromosome,
                'best_value': -best.fitness,
                'stats': optimizer.get_stats()
            }
        
        elif method == 'gradient':
            # Simple gradient descent
            import numpy as np
            
            # Initial point
            x = np.array([random.uniform(b[0], b[1]) for b in bounds])
            learning_rate = kwargs.get('learning_rate', 0.01)
            iterations = kwargs.get('iterations', 1000)
            
            history = []
            
            for i in range(iterations):
                # Finite difference gradient
                grad = np.zeros_like(x)
                h = 1e-6
                
                for j in range(len(x)):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[j] += h
                    x_minus[j] -= h
                    
                    grad[j] = (objective(x_plus) - objective(x_minus)) / (2 * h)
                
                # Update
                x = x - learning_rate * grad
                
                # Apply bounds
                for j in range(len(x)):
                    x[j] = max(bounds[j][0], min(bounds[j][1], x[j]))
                
                history.append({
                    'iteration': i,
                    'value': objective(x),
                    'x': x.copy()
                })
            
            return {
                'method': 'gradient_descent',
                'best_solution': x.tolist(),
                'best_value': objective(x),
                'iterations': iterations,
                'history': history[-10:]  # Last 10 iterations
            }
        
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'operations': self.operations_count,
            'structures_count': len(self._structures),
            'models_count': len(self._models),
            'performance_metrics': self.metrics.summary(),
            'cache_stats': self.cache.stats()
        }
    
    def clear(self):
        """Clear all stored data."""
        self._structures.clear()
        self._models.clear()
        self.cache.clear()
        self.metrics.reset()
        self.step_counter.reset()
        self.operations_count = 0
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear()
    
    def __str__(self) -> str:
        stats = self.get_stats()
        return (f"DSAEngine(uptime={stats['uptime_seconds']:.1f}s, "
                f"operations={stats['operations']}, "
                f"structures={stats['structures_count']}, "
                f"models={stats['models_count']})")


# ==================== GLOBAL INSTANCE & EXPORTS ====================

_global_engine = None

def get_engine() -> DSAEngine:
    """Get global engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = DSAEngine()
    return _global_engine


def reset_engine():
    """Reset global engine."""
    global _global_engine
    if _global_engine is not None:
        _global_engine.clear()
    _global_engine = None


# Export all important classes and functions
__all__ = [
    # Core engine
    'DSAEngine', 'get_engine', 'reset_engine',
    
    # Types
    'DSAType', 'MLType', 'AIType',
    
    # Analysis
    'PerformanceMetrics', 'ComplexityAnalyzer', 'StepCounter', 'Cache',
    
    # Data Structures
    'Node', 'EnhancedArray', 'ThreadSafeStack', 'ThreadSafeQueue',
    'EnhancedGraph', 'Trie', 'LRUCache',
    
    # Algorithms
    'SortingAlgorithms', 'SearchingAlgorithms', 'GraphAlgorithms',
    'DynamicProgramming', 'Backtracking',
    
    # ML
    'MLDataPreprocessor', 'LinearRegressionModel', 'LogisticRegressionModel',
    'KMeansClustering', 'DecisionTreeClassifier', 'NeuralNetwork',
    
    # AI
    'GeneticAlgorithm', 'AStarPathfinder', 'MinimaxAI'
]


# ==================== EXAMPLE USAGE ====================

def example_usage():
    """Example usage of the DSA & AI Engine."""
    print("=== DSA & AI Engine Example ===\n")
    
    # Get engine instance
    engine = get_engine()
    print(f"Engine: {engine}\n")
    
    # 1. Create and use data structures
    print("1. Data Structures:")
    arr = engine.create_structure('array', data=[1, 2, 3, 4, 5])
    print(f"   Array: {arr}")
    print(f"   Array statistics: {arr.statistics()}")
    
    graph = engine.create_structure('graph', directed=True)
    graph.add_edge('A', 'B', 1.0)
    graph.add_edge('B', 'C', 2.0)
    graph.add_edge('A', 'C', 1.5)
    print(f"   Graph: {graph}")
    print(f"   BFS from A: {graph.bfs('A')}")
    print()
    
    # 2. Analyze algorithms
    print("2. Algorithm Analysis:")
    
    def sample_sort(data):
        return sorted(data)
    
    analysis = engine.analyze_algorithm(sample_sort, [[3, 1, 2], [5, 4, 3, 2, 1]])
    print(f"   Sort analysis: {analysis['complexity']}")
    print(f"   Total time: {analysis['total_time']:.6f}s")
    print()
    
    # 3. Train ML model
    print("3. Machine Learning:")
    try:
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        y = [0, 0, 1, 1, 1]
        
        result = engine.train_model(MLType.LOGISTIC_REGRESSION, X, y, test_size=0.2)
        print(f"   Model trained: {result['model']}")
        print(f"   Accuracy: {result['evaluation'].get('accuracy', 'N/A')}")
    except Exception as e:
        print(f"   ML error: {e}")
    print()
    
    # 4. AI optimization
    print("4. AI Optimization:")
    
    def objective(x):
        return sum((xi - 2) ** 2 for xi in x)
    
    bounds = [(-5, 5), (-5, 5)]
    opt_result = engine.optimize(objective, bounds, method='genetic', generations=50)
    print(f"   Best solution: {[f'{x:.3f}' for x in opt_result['best_solution']]}")
    print(f"   Best value: {opt_result['best_value']:.6f}")
    print()
    
    # 5. Get engine statistics
    print("5. Engine Statistics:")
    stats = engine.get_stats()
    print(f"   Uptime: {stats['uptime_seconds']:.1f}s")
    print(f"   Operations: {stats['operations']}")
    print(f"   Structures: {stats['structures_count']}")
    print(f"   Models: {stats['models_count']}")
    
    return engine


if __name__ == "__main__":
    # Run example if executed directly
    try:
        engine = example_usage()
        print("\n=== Example completed successfully ===")
    except Exception as e:
        print(f"\n=== Error in example: {e} ===")
        import traceback
        traceback.print_exc()