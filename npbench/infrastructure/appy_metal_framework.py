# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict


class APPyMetalFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        super().__init__(fname)

    def version(self) -> str:
        import appy
        return getattr(appy, '__version__', '0.1')

    def copy_func(self) -> Callable:
        import appy.np_shared as nps
        return nps.copy

    def imports(self) -> Dict[str, Any]:
        import appy
        import appy.np_shared as nps
        import numpy as np
        return {'appy': appy, 'nps': nps, 'np': np}

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        arg_str = self.arg_str(bench, impl)
        return "__npb_result = __npb_impl({a})".format(a=arg_str)
