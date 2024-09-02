import timeit
from typing import Callable, Generator

import pandas

from plot import plot_frequencies, perform_chi_square_test


def collect(run: Callable[[], Generator[int, None, None]], iterations: int) -> list[list[int]]:
    all_results = []

    for _ in range(iterations):
        result = list(run())
        all_results.append(result)

    return all_results


def plot(run: Callable[[int, int, int], Generator[int, None, None]], min_num: int, max_num: int,
         batch_size: int,
         iterations: int):
    all_results = collect(lambda: run(min_num, max_num, batch_size), iterations)

    chi2, p_value, observed_freq, expected_freq = perform_chi_square_test(all_results, min_num, max_num)

    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-Value: {p_value}")

    plot_frequencies(observed_freq, expected_freq, min_num, max_num)


def ignore(run: Callable[[], Generator[int, None, None]]):
    for _ in run():
        pass


def measure_time(generate: Callable[[int, int], Generator[int, None, None]],
                 size_range: range = range(1000, 10001, 1000),
                 iterations: int = 10) -> pandas.DataFrame:
    data = []

    for range_size in size_range:
        for batch_percentage_int in range(0, 101, 10):
            batch_percentage = batch_percentage_int / 100
            batch_size = range_size * batch_percentage

            execution_time = timeit.timeit(lambda: ignore(lambda: generate(range_size, int(batch_size))),
                                           number=iterations)

            data.append({
                'range_size': range_size,
                'batch_percentage': batch_percentage,
                'batch_size': batch_size,
                'execution_time': execution_time
            })

    return pandas.DataFrame(data)
