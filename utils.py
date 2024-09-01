from typing import Callable, Generator, List

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare


def run_and_collect_data(run: Callable[[], Generator[int, None, None]], iterations: int):
    all_results = []

    for _ in range(iterations):
        result = list(run())
        all_results.append(result)

    return all_results


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


def run_and_plot(run: Callable[[int, int, int], Generator[int, None, None]], min_num: int, max_num: int,
                 batch_size: int,
                 iterations: int):
    # Generate the data
    all_results = run_and_collect_data(lambda: run(min_num, max_num, batch_size), iterations)

    # Perform chi-square test
    chi2, p_value, observed_freq, expected_freq = perform_chi_square_test(all_results, min_num, max_num)

    # Print the chi-square test results
    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-Value: {p_value}")

    # Plot observed vs expected frequencies
    plot_frequencies(observed_freq, expected_freq, min_num, max_num)


def run_and_ignore(run: Callable[[], Generator[int, None, None]]):
    for _ in run():
        pass


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


# Parameters
min_num = 1
max_num = 10
max_batch_size = 5

# Generate the table
probability_table = calculate_probabilities_without_replacement(min_num, max_num, max_batch_size)

# Display the table
import ace_tools as tools;

tools.display_dataframe_to_user(name="Probability Table Without Replacement", dataframe=probability_table)
