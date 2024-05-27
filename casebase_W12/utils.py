import math
import heapq
from collections import Counter
from graphviz import Digraph

class HuffmanNode:
    def __init__(self, probability, symbol=None, left=None, right=None):
        self.probability = probability
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.probability < other.probability

def compute_average_bits(probabilities, codes):
    if isinstance(probabilities, dict):
        return sum(probabilities[symbol] * len(codes[symbol]) for symbol in probabilities)
    elif isinstance(probabilities, list):
        return sum(probabilities[i] * len(codes[i]) for i in range(len(probabilities)))
    else:
        raise ValueError("Unsupported data structure for probabilities.")


def compute_entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def build_huffman_tree(probabilities):
    heap = [HuffmanNode(p, symbol) for symbol, p in probabilities.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(left.probability + right.probability, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)
    
    return codebook

def arithmetic_encode(symbols, probabilities):
    low, high = 0.0, 1.0
    for symbol in symbols:
        range_ = high - low
        high = low + range_ * probabilities[symbol][1]
        low = low + range_ * probabilities[symbol][0]
    return (low + high) / 2

def arithmetic_decode(encoded_value, probabilities, length):
    decoded_symbols = []
    low, high = 0.0, 1.0
    for _ in range(length):
        range_ = high - low
        symbol = next(s for s in probabilities if probabilities[s][0] <= (encoded_value - low) / range_ < probabilities[s][1])
        decoded_symbols.append(symbol)
        high = low + range_ * probabilities[symbol][1]
        low = low + range_ * probabilities[symbol][0]
    return decoded_symbols

def calculate_probabilities(data):
    total = len(data)
    counter = Counter(data)
    probabilities = {symbol: count / total for symbol, count in counter.items()}
    return probabilities

def calculate_cumulative_probabilities(probabilities):
    cumulative_probabilities = {}
    cumulative = 0.0
    for symbol, probability in probabilities.items():
        cumulative_probabilities[symbol] = (cumulative, cumulative + probability)
        cumulative += probability
    return cumulative_probabilities


def draw_huffman_tree(node):
    def add_nodes_edges(node, dot=None):
        if dot is None:
            dot = Digraph()
            dot.node(name=str(id(node)), label=str(node.symbol) if node.symbol is not None else '')
        
        if node.left:
            dot.node(name=str(id(node.left)), label=str(node.left.symbol) if node.left.symbol is not None else '')
            dot.edge(str(id(node)), str(id(node.left)), label='0')
            dot = add_nodes_edges(node.left, dot=dot)
        
        if node.right:
            dot.node(name=str(id(node.right)), label=str(node.right.symbol) if node.right.symbol is not None else '')
            dot.edge(str(id(node)), str(id(node.right)), label='1')
            dot = add_nodes_edges(node.right, dot=dot)
        
        return dot

    dot = add_nodes_edges(node)
    return dot