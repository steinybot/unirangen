from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare


def perform_chi_square_test(all_results: List[List[int]], min_num: int, max_num: int):
    # Flatten the list of lists into a single list
    all_numbers = [number for result in all_results for number in result]

    # Calculate observed frequencies
    observed_freq = [all_numbers.count(i) for i in range(min_num, max_num + 1)]

    # Expected frequency (assuming uniform distribution)
    expected_freq = [len(all_numbers) / (max_num - min_num + 1)] * (max_num - min_num + 1)

    # Perform chi-square test
    chi2, p_value = chisquare(f_obs=observed_freq, f_exp=expected_freq)

    return chi2, p_value, observed_freq, expected_freq


def plot_frequencies(observed_freq: List[int], expected_freq: List[float], min_num: int, max_num: int):
    numbers = list(range(min_num, max_num + 1))

    plt.bar(numbers, observed_freq, color='blue', alpha=0.7, label='Observed')
    plt.plot(numbers, expected_freq, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8,
             label='Expected')

    plt.title('Observed vs Expected Frequencies of Random Numbers')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def calculate_probabilities(min_num: int, max_num: int, max_batch_size: int) -> pd.DataFrame:
    data = []
    total_numbers = max_num - min_num + 1

    for k in range(min_num, max_num + 1):
        row = {}
        numbers_greater_than_k = max_num - k
        for batch_size in range(1, max_batch_size + 1):
            if numbers_greater_than_k >= batch_size:
                # Calculate the probability of not picking any number <= k in batch_size draws
                probability = 1.0
                for i in range(batch_size):
                    probability *= (numbers_greater_than_k - i) / (total_numbers - i)
            else:
                # If the batch size is larger than the numbers available, it's impossible to avoid all <= k
                probability = 0.0
            row[f'Batch Size {batch_size}'] = probability
        data.append(row)

    # Create a DataFrame
    df = pd.DataFrame(data, index=[f'k = {k}' for k in range(min_num, max_num + 1)])
    return df
