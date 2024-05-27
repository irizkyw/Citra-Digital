from utils import (
    compute_average_bits, compute_entropy, build_huffman_tree, generate_huffman_codes,
    arithmetic_encode, arithmetic_decode, calculate_probabilities, calculate_cumulative_probabilities,draw_huffman_tree
)

def main():
    image_data = [
        21, 21, 21, 95, 169, 243, 243, 243,
        21, 21, 21, 95, 169, 243, 243, 243,
        21, 21, 21, 95, 169, 243, 243, 243,
        21, 21, 21, 95, 169, 243, 243, 243
    ]

    pixel_probabilities = calculate_probabilities(image_data)

    image_entropy = compute_entropy(list(pixel_probabilities.values()))

    huffman_tree = build_huffman_tree(pixel_probabilities)
    
    dot = draw_huffman_tree(pixel_probabilities)
    dot.render('huffman_tree', format='png', cleanup=True)

    huffman_codes = generate_huffman_codes(huffman_tree)
    
    average_bits_huffman = compute_average_bits(pixel_probabilities, huffman_codes)

    compression_ratio_huffman = 8 / average_bits_huffman

    print(f"Exercise 2 Results:")
    print(f"Pixel probabilities: {pixel_probabilities}")
    print(f"Huffman codes: {huffman_codes}")
    print(f"Entropy of the image: {image_entropy:.2f} bits per pixel")
    print(f"Average bits per pixel (Huffman coding): {average_bits_huffman:.2f}")
    print(f"Compression ratio (Huffman coding {8}/{average_bits_huffman}): {compression_ratio_huffman:.2f}\n")

    sequence = [21, 95, 21, 169, 243]
    cumulative_probabilities = calculate_cumulative_probabilities(pixel_probabilities)

    encoded_value = arithmetic_encode(sequence, cumulative_probabilities)

    decoded_sequence = arithmetic_decode(encoded_value, cumulative_probabilities, len(sequence))

    print(f"Exercise 3 Results:")
    print(f"Encoded value: {encoded_value}")
    print(f"Decoded sequence: {decoded_sequence}")

if __name__ == "__main__":
    main()
