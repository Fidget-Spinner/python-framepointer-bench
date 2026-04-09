# Credits to Mike Droettboom for the code here.
# This is adapted from https://github.com/faster-cpython/bench_runner
from typing import Any

from os import PathLike

from pathlib import Path
import json
import sys
from operator import itemgetter

import pyperf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

EXCLUDED = {
}

BASELINE_LINUX = "y4-cpy314/json/2025-12-06_20-29-paper_3.14.2_noopt-00bf18d590b2.json"

COMPARISONS_ALL = [
    ("macOS M3Pro", "savannah/aarch64_macos/baseline-darwin-jones.json", "savannah/aarch64_macos/fp-darwin-jones.json"),
    ("Debian RPi5", "savannah/aarch64_linux/baseline-linux-blueberry.json", "savannah/aarch64_linux/fp-linux-blueberry.json"),
    ("Ubuntu i712700H", "kenjin/x86_64/2026-04-05_19-10-frame-pointers-baseline.json", "kenjin/x86_64/2026-04-05_19-27-frame-pointers-fp.json"),
]


# Source: https://stackoverflow.com/questions/43099542/python-easy-way-to-do-geometric-mean-in-python
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def get_combined_data(
    ref_data: dict[str, np.ndarray], head_data: dict[str, np.ndarray]
) -> list:

    def calculate_diffs(ref_values, head_values) -> tuple[np.ndarray | None, float]:
        values = np.outer(ref_values, 1.0 / head_values).flatten()
        values.sort()
        return values, float(np.median(values))

    combined_data = []
    for name, ref in ref_data.items():
        if len(ref) != 0 and name in head_data and name not in EXCLUDED:
            head = head_data[name]
            if len(ref) == len(head):
                combined_data.append((name, *calculate_diffs(ref, head)))
    combined_data.sort(key=itemgetter(2))
    return combined_data

class BenchmarkData:
    def __init__(self, filename: Path):
        with filename.open("rb") as fd:
            self.contents = json.load(fd)

    def get_timing_data(self) -> dict[str, np.ndarray]:
        data = {}

        for benchmark in self.contents["benchmarks"]:
            name = benchmark.get("metadata", self.contents["metadata"])["name"]
            if name not in EXCLUDED:
                row = []
                for run in benchmark["runs"]:
                    row.extend(run.get("values", []))
                data[name] = np.array(row, dtype=np.float64)

        return data

    
    def get_diff(self, other):
        me = self.get_timing_data()
        compare_to = other.get_timing_data()
        combined = get_combined_data(me, compare_to)

def plot_diff_pair(ax, data):
    if not len(data):
        return []

    all_data = []

    medians_all = []
    for config in data:
        medians = []
        for i, (name, values, median) in enumerate(config):
            if values is not None:
                medians.append(median)
            else:
                medians.append(1.0)
        print(f"{np.median(medians):.2f}")
        medians_all.append(medians)

    bplot = ax.boxplot(
        medians_all,
        vert=True,
        # showmeans=True,
        showfliers=False,
        patch_artist=True,
        tick_labels=[x[0] for x in COMPARISONS_ALL]
    )

    len_plots = len(bplot['boxes'])
    for i in range(len_plots):
        ax.plot(i+1, geo_mean(medians_all[i]),marker='o', color='green', markersize=4, label='Geometric Mean')
        if i % 2:
            bplot['boxes'][i].set_facecolor('lightgrey')
        else:
            bplot['boxes'][i].set_facecolor('white')

    return all_data
    


def formatter(val, pos):
    return f"{val:.02f}×"


def plot_diff(
    combined_data: list,
    output_filename: PathLike,
) -> None:
    _, axs = plt.subplots(layout="constrained")
    plot_diff_pair(axs, combined_data)
    axs.yaxis.set_major_formatter(formatter)
    axs.grid()
    # title = axs.set_title("\% Speedup normalized to cpy_3.14.2_noopt baseline")

    output_filename = Path(output_filename)
    plt.xticks(rotation=45, ha="right")
    # plt.gcf().subplots_adjust(bottom=0.4)
    axs.set_ylabel(r"Effect on pyperformance")
    axs.set_xlabel(f"System configurations")
    plt.axhline(1.0)
    plt.savefig(output_filename, dpi=500)

    plt.close("all")

if __name__ == "__main__":
    # base = BenchmarkData(Path(sys.argv[1]))
    # changed = BenchmarkData(Path(sys.argv[2]))
    everything = []
    for name, base, changed in COMPARISONS_ALL:
        combined = get_combined_data(BenchmarkData(Path(base)).get_timing_data(), BenchmarkData(Path(changed)).get_timing_data())
        everything.append(combined)
    plot_diff(everything, "fp_perf_over_baseline.pdf")