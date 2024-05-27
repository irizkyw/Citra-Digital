from utils import (
    compute_average_bits, compute_entropy, build_huffman_tree, generate_huffman_codes,
    arithmetic_encode, arithmetic_decode, calculate_probabilities, calculate_cumulative_probabilities
)

def main():
    levels = list(range(8))
    probabilities = [0.19, 0.25, 0.21, 0.16, 0.08, 0.06, 0.03, 0.02]
    fixed_length_codes = ["000", "001", "010", "011", "100", "101", "110", "111"]
    variable_length_codes = ["00", "11", "01", "101", "1001", "10001", "100001", "100000"]

    average_bits_fixed = len(fixed_length_codes[0])
    average_bits_variable = compute_average_bits(probabilities, variable_length_codes)

    compression_ratio = average_bits_fixed / average_bits_variable
    data_redundancy = 1 - (1 / compression_ratio)

    entropy = compute_entropy(probabilities)

    print(f"Exercise 1 Results:")
    print(f"Average bits per pixel (fixed-length): {average_bits_fixed}")
    print(f"Average bits per pixel (variable-length): {average_bits_variable:.2f}")
    print(f"Compression ratio: {compression_ratio:.2f}")
    print(f"Data redundancy: {data_redundancy:.2%}")
    print(f"Entropy: {entropy:.2f} bits per pixel\n")


if __name__ == "__main__":
    main()
