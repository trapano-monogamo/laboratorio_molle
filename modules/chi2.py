from typing import Dict, Tuple

import numpy as np

# ! should add a check function


def calculate_chi2(
    observed: Tuple[float],
    expected: Tuple[float],
    sigma_y: Tuple[float],
    deg_of_freedom: int,
) -> Dict[str, float | Tuple[float]]:

    values = {
        "observed": observed,
        "expected": expected,
        "sigma_y": sigma_y,
    }

    for key, value in values.items():
        if not isinstance(value, np.ndarray):
            try:
                values[key] = np.array(value)
            except Exception:
                raise Exception("Something went ka-boom")

    chi2i = np.divide(
        np.square(values["observed"] - values["expected"]), np.square(values["sigma_y"])
    )
    chi2 = np.sum(chi2i)
    chi2r = chi2 / deg_of_freedom

    return {"chi2i": chi2i, "chi2": chi2, "chi2r": chi2r}


def main():
    exp = [2.01, 4.02, 6.03, 8.04]
    obs = [2, 4, 6.1, 8]
    print(calculate_chi2(exp, obs, [0.2836, 1.2225, 0.5016, 0.4477], len(exp) - 2))


if __name__ == "__main__":
    main()
