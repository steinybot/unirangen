from typing import Callable, Generator

from plot import plot_frequencies, perform_chi_square_test


def collect(run: Callable[[], Generator[int, None, None]], iterations: int):
    all_results = []

    for _ in range(iterations):
        result = list(run())
        all_results.append(result)

    return all_results


def plot(run: Callable[[int, int, int], Generator[int, None, None]], min_num: int, max_num: int,
         batch_size: int,
         iterations: int):
    # Generate the data
    all_results = collect(lambda: run(min_num, max_num, batch_size), iterations)

    # Perform chi-square test
    chi2, p_value, observed_freq, expected_freq = perform_chi_square_test(all_results, min_num, max_num)

    # Print the chi-square test results
    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-Value: {p_value}")

    # Plot observed vs expected frequencies
    plot_frequencies(observed_freq, expected_freq, min_num, max_num)


def ignore(run: Callable[[], Generator[int, None, None]]):
    for _ in run():
        pass
