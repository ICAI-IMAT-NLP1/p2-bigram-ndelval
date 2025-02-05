from data_processing import (
    load_and_preprocess_data,
    char_to_index,
    index_to_char,
    count_bigrams,
)

from bigram_model import (
    bigrams_count_to_probabilities,
    generate_name,
    calculate_log_likelihood,
    calculate_neg_mean_log_likelihood,
)

if __name__ == "__main__":
    # Define file path and tokens
    file_path = "data/nombres_raw.txt"
    start_token = "-"
    end_token = "."
    alphabet = "abcdefghijklmnopqrstuvwxyz "

    # Load and preprocess data
    bigrams = load_and_preprocess_data(file_path, start_token, end_token)

    # Create character indices
    char_to_idx = char_to_index(alphabet, start_token, end_token)
    idx_to_char = index_to_char(char_to_idx)

    # Count bigrams and convert to probabilities
    bigram_counts = count_bigrams(bigrams, char_to_idx)
    bigram_probabilities = bigrams_count_to_probabilities(bigram_counts)

    num_names_to_generate = 10
    print("Generated Names:")
    names = []
    for _ in range(num_names_to_generate):
        new_name = generate_name(
            start_token, end_token, char_to_idx, idx_to_char, bigram_probabilities
        )
        neg_log_likelihood = -calculate_log_likelihood(
            new_name, bigram_probabilities, char_to_idx, start_token, end_token
        )
        names.append(new_name.capitalize())
        print(f"{new_name.capitalize()}: NLL = {neg_log_likelihood}")

    # Print mean negative log likelihood
    mean_neg_log_likelihood = calculate_neg_mean_log_likelihood(
        names, bigram_probabilities, char_to_idx, start_token, end_token
    )
    print(f"\nMean NLL: {mean_neg_log_likelihood}")
