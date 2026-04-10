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

COMPARISONS_ALL = [
    ("macOS M2", "diego/macmini-base.json", "diego/macmini-head.json"),    
    ("macOS M3Pro", "savannah/aarch64_macos/baseline-darwin-jones.json", "savannah/aarch64_macos/fp-darwin-jones.json"),
    ("Debian RPi5", "savannah/aarch64_linux/baseline-linux-blueberry.json", "savannah/aarch64_linux/fp-linux-blueberry.json"),    
    ("Ubuntu Ampere Altra Max", "diego/altramax-base.json", "diego/altramax-head.json"),
    ("Ubuntu AWS Graviton c7g.16xlarge", "diego/graviton3-base.json", "diego/graviton3-head.json"),
    ("Ubuntu Intel i7-12700H", "kenjin/x86_64/2026-04-05_19-10-frame-pointers-baseline.json", "kenjin/x86_64/2026-04-05_19-27-frame-pointers-fp.json"),    
    ("Ubuntu AMD EPYC 9654", "diego/epyc-base.json", "diego/epyc-head.json"),
    ("Ubuntu Intel Xeon Platinum 8480", "diego/platinum-base.json", "diego/platinum-head.json"),
]

AARCH64_COUNT = 5

# Misisng on some of the results
EXCLUDED = {
    'async_tree_eager',
    'async_tree_eager_cpu_io_mixed',
    'async_tree_eager_cpu_io_mixed_tg',
    'async_tree_eager_io',
    'async_tree_eager_io_tg',
    'async_tree_eager_memoization',
    'async_tree_eager_memoization_tg',
    'async_tree_eager_tg',
    'asyncio_tcp',
    'asyncio_tcp_ssl',
    'bench_mp_pool',
    'bench_thread_pool',
    'dask',
    'django_template',
    'pickle',
    'pickle_dict',
    'pickle_list',
    'sympy_expand',
    'sympy_integrate',
    'sympy_str',
    'sympy_sum',
    'unpickle',
    'unpickle_list',
}

BENCH_NAMES = [
    '2to3', 
    'async_generators',
    'async_tree_cpu_io_mixed', 'async_tree_cpu_io_mixed_tg', 'async_tree_eager', 'async_tree_eager_cpu_io_mixed', 'async_tree_eager_cpu_io_mixed_tg', 'async_tree_eager_io', 'async_tree_eager_io_tg', 'async_tree_eager_memoization', 'async_tree_eager_memoization_tg', 'async_tree_eager_tg', 'async_tree_io', 'async_tree_io_tg', 'async_tree_memoization', 'async_tree_memoization_tg', 'async_tree_none', 'async_tree_none_tg', 'asyncio_tcp', 'asyncio_tcp_ssl', 'asyncio_websockets', 'bench_mp_pool', 'bench_thread_pool', 'bpe_tokeniser', 'chameleon', 'chaos', 'comprehensions', 'coroutines', 'create_gc_cycles', 'crypto_pyaes', 'dask', 'deepcopy', 'deepcopy_memo', 'deepcopy_reduce', 'deltablue', 'django_template', 'docutils', 'dulwich_log', 'fannkuch', 'float', 'generators', 'genshi_text', 'genshi_xml', 'go', 'hexiom', 'html5lib', 'json_dumps', 'json_loads', 'logging_format', 'logging_silent', 'logging_simple', 'mako', 'many_optionals', 'mdp', 'meteor_contest', 'nbody', 'nqueens', 'pathlib', 'pickle', 'pickle_dict', 'pickle_list', 'pickle_pure_python', 'pidigits', 'pprint_pformat', 'pprint_safe_repr', 'pyflate', 'python_startup', 'python_startup_no_site', 'raytrace', 'regex_compile', 'regex_dna', 'regex_effbot', 'regex_v8', 'richards', 'richards_super', 'scimark_fft', 'scimark_lu', 'scimark_monte_carlo', 'scimark_sor', 'scimark_sparse_mat_mult', 'spectral_norm', 'sphinx', 'sqlglot_v2_normalize', 'sqlglot_v2_optimize', 'sqlglot_v2_parse', 'sqlglot_v2_transpile', 'subparsers', 'sympy_expand', 'sympy_integrate', 'sympy_str', 'sympy_sum', 'telco', 'tomli_loads', 'tornado_http', 'typing_runtime_protocols', 'unpickle', 'unpickle_list', 'unpickle_pure_python', 'xdsl_constant_fold', 'xml_etree_generate', 'xml_etree_iterparse', 'xml_etree_parse', 'xml_etree_process'
]

BENCH_NAMES = [x for x in BENCH_NAMES if x not in EXCLUDED]

def get_combined_data(
    ref_data: dict[str, np.ndarray], head_data: dict[str, np.ndarray]
) -> dict:

    def calculate_diffs(ref_values, head_values) -> tuple[np.ndarray | None, float]:
        values = np.outer(ref_values, 1.0 / head_values).flatten()
        values.sort()
        return values

    combined_data = []
    for name, ref in ref_data.items():
        if len(ref) != 0 and name in head_data and name in BENCH_NAMES:
            head = head_data[name]
            if len(ref) == len(head):
                combined_data.append((name, calculate_diffs(ref, head)))
    combined_data.sort(key=lambda a: np.median(a[1]))
    return dict(combined_data)

class BenchmarkData:
    def __init__(self, filename: Path):
        with filename.open("rb") as fd:
            self.contents = json.load(fd)

    def get_timing_data(self) -> dict[str, np.ndarray]:
        data = {}

        for benchmark in self.contents["benchmarks"]:
            name = benchmark.get("metadata", self.contents["metadata"])["name"]
            if name in BENCH_NAMES:
                row = []
                for run in benchmark["runs"]:
                    row.extend(run.get("values", []))
                data[name] = np.array(row, dtype=np.float64)

        return data


def plot_diff_pair(ax, bench_name, data):
    if not len(data):
        return []

    all_data = []

    values_all = []
    for config in data:   
        values_all.append(config[bench_name])

    bplot = ax.boxplot(
        values_all,
        vert=True,
        #showmeans=True,
        showfliers=False,
        patch_artist=True,
    )

    # Set color
    len_plots = len(bplot['boxes'])
    for i in range(len_plots):
        if i >= AARCH64_COUNT:
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
    fig, axs = plt.subplots(len(BENCH_NAMES)//6 + 1, 6, sharex=False, sharey=False, layout="constrained")
    og_axes = axs
    axs = axs.flatten()
    for i, bench_name in enumerate(BENCH_NAMES):
        plot_diff_pair(axs[i], bench_name, combined_data)
        axs[i].yaxis.set_major_formatter(formatter)
        axs[i].grid()
        axs[i].set_xlabel(bench_name)
        axs[i].tick_params(labelbottom=False)
    all_labels = [x[0] for x in COMPARISONS_ALL]
    plt.xticks(np.arange(1, len(all_labels) + 1), all_labels, rotation=90)

    for x in range(i+1, len(axs)-1):
        fig.delaxes(axs[x])
    fig.set_size_inches(8.5 * 2, 11 * 2)


    fig.supylabel("Relative performance")
    fig.supxlabel("Individual benchmarks in pyperformance for CPython 3.15a configurations versus their respective baselines.")
    output_filename = Path(output_filename)

    # plt.gcf().subplots_adjust(bottom=0.4)

    # plt.axhline(1.0)
    plt.savefig(output_filename, dpi=1000)

    plt.close("all")

if __name__ == "__main__":
    everything = []
    for name, base, changed in COMPARISONS_ALL:
        print(name)
        combined = get_combined_data(BenchmarkData(Path(base)).get_timing_data(), BenchmarkData(Path(changed)).get_timing_data())
        everything.append(combined)       
    plot_diff(everything, "pep-0831_perf_over_baseline_indiv.svg") 