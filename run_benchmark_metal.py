"""Run benchmarks using float32 initialization (for Metal GPU backend)."""
import argparse

from npbench.infrastructure import (generate_framework, LineCount,
                                    Test, utilities as util)
from npbench.infrastructure.metal_benchmark import MetalBenchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", type=str, nargs="?", required=True)
    parser.add_argument("-f", "--framework", type=str, nargs="?", default="appy_metal")
    parser.add_argument("-p", "--preset", choices=['S', 'M', 'L', 'paper'],
                        nargs="?", default='S')
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v", "--validate", type=util.str2bool, nargs="?", default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-t", "--timeout", type=float, nargs="?", default=200.0)
    args = vars(parser.parse_args())

    bench = MetalBenchmark(args["benchmark"])
    frmwrk = generate_framework(args["framework"])
    numpy = generate_framework("numpy")
    lcount = LineCount(bench, frmwrk, numpy)
    lcount.count()
    test = Test(bench, frmwrk, numpy)
    test.run(args["preset"], args["validate"], args["repeat"], args["timeout"])
