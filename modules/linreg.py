from typing import Dict, Tuple

import numpy as np

# ? maybe I should add * and / in function parameters


def LinearRegression(
    xValues: list,
    yValues: list,
    xErrors: list,
    yErrors: list,
    k_test: float
) -> Dict[str, float]:
    # __check_data(xValues, yValues, xErrors, yErrors)

    xValues = np.array(xValues)
    yValues = np.array(yValues)
    xErrors = np.array(xErrors)
    yErrors = np.array(yErrors)

    # ! this formulas are wrong, I cannot calculate them in a loop

    sigma_yi = np.sqrt(np.square(yErrors) + (k_test**2) * np.square(xErrors))
    sigma_y = np.sum(sigma_yi)

    len_data = len(xValues)
    squared_x = np.square(xValues)

    weights = np.power(sigma_yi, -2)
    sum_weights = np.sum(weights)
    sum_weight_dot_x_vals = np.sum(xValues @ weights)
    sum_weight_dot_y_vals = np.sum(yValues @ weights)
    sum_weight_dot_squared_x = np.sum(weights @ squared_x)
    sum_dot_wxy = np.sum(weights @ xValues @ yValues)

    Delta = len_data * sum_weights * sum_weight_dot_squared_x - sum_weight_dot_x_vals**2

    intercept = (
        sum_weight_dot_squared_x * sum_weight_dot_y_vals
        - sum_weight_dot_x_vals * sum_dot_wxy
    ) / Delta
    coeff = (
        sum_weights * sum_dot_wxy - sum_weight_dot_x_vals * sum_weight_dot_y_vals
    ) / Delta

    coeffErr = np.sqrt(sum_weight_dot_squared_x / Delta)
    interceptErr = np.sqrt(sum_weights / Delta)

    return {
        "Coefficient": coeff,
        "Intercept": intercept,
        "CoefficientError": coeffErr,
        "InterceptError": interceptErr,
        "Sigmay": sigma_y,
        # "Sigma_yi": sigma_yi,
    }


def LinearRegressionStdErr(
    xValues: Tuple[float],
    yValues: Tuple[float],
) -> Dict[str, float]:
    __check_data(xValues, yValues)

    xValues = np.array(xValues)
    yValues = np.array(yValues)

    sum_x = np.sum(xValues)
    squared_x = np.square(xValues)
    sum_y = np.sum(yValues)
    sum_dot_prod = np.sum(xValues @ yValues)

    len_data = len(xValues)

    Delta = len_data * np.sum(squared_x) - (sum_x) ** 2

    intercept = (np.sum(squared_x) * sum_y - sum_x * sum_dot_prod) / Delta
    coeff = (len_data * sum_dot_prod - sum_x * sum_y) / Delta

    sigma_y = np.sqrt(
        np.sum(np.square(yValues - coeff * xValues - intercept)) / (len_data - 2)
    )

    coeffErr = sigma_y * np.sqrt(np.sum(squared_x) / Delta)
    interceptErr = sigma_y * np.sqrt(len_data / Delta)

    return {
        "Coefficient": coeff,
        "Intercept": intercept,
        "CoefficientError": coeffErr,
        "InterceptError": interceptErr,
        "Sigmay": sigma_y,
    }


def expected(xValues: Tuple[float], coeff: float, intercept: float):
    if not isinstance(xValues, np.ndarray):
        try:
            xValues = np.array(xValues)
        except Exception:
            raise Exception("Something went ka-boom")
    return np.array(coeff * xValues + intercept)


def __check_data(*data: Tuple[Tuple[float]]) -> None:
    for index, list in enumerate(data):
        for elem in list:
            if not isinstance(elem, int | float):
                raise ValueError("All elements should be integers of floats")

        if index == 0:
            previous = len(list)
            continue
        if len(list) != previous:
            raise ValueError("Lists should have the same number of elements")
        previous = len(list)


def main():
    print(expected([1, 2, 3, 4], 2.01, 0))
    print(LinearRegressionStdErr([1, 2, 3, 4], [2, 4, 6.1, 8]))


if __name__ == "__main__":
    main()
