from typing import Any


def flatten(data: list[dict]):
    results = {}
    for elem in data:
        for key, value in elem.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
    return results


def flatten_data(data: list[dict], n: int) -> Any:
    if n == 0: return data
    flattened = flatten(data)
    results = {}
    for key, value in flattened.items():
        results[key] = flatten_data(value, n - 1)
    return results


def swap_groups(values: dict[str, dict]):
    results = {}
    for metric_name, metric in values.items():
        for group_name, group in metric.items():
            if group_name not in results:
                results[group_name] = {}
            results[group_name][metric_name] = group
    return results


def swap_layers(values: dict, n: int):
    if n == 0: return swap_groups(values)
    results = {}
    for name, value in values.items():
        results[name] = swap_layers(value, n - 1)
    return results