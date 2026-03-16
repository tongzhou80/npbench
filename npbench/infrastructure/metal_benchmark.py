# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
from npbench.infrastructure.benchmark import Benchmark


class MetalBenchmark(Benchmark):
    """A Benchmark subclass that forces float32 initialization for Metal."""

    def get_data(self, preset: str = 'L'):
        if preset in self.bdata.keys():
            return self.bdata[preset]

        import copy
        # Get a mutable copy of info so we can inject datatype
        info = self.info

        data = dict()
        if preset not in info["parameters"].keys():
            raise NotImplementedError(
                "{b} doesn't have a {p} preset.".format(b=self.bname, p=preset))
        for k, v in info["parameters"][preset].items():
            data[k] = v

        if "init" in info.keys() and info["init"]:
            module_pypath = "npbench.benchmarks.{r}.{m}".format(
                r=info["relative_path"].replace('/', '.'),
                m=info["module_name"])
            exec_str = "from {m} import {i}".format(
                m=module_pypath, i=info["init"]["func_name"])
            exec(exec_str, data)

            # Inject np and datatype=np.float32 into the init call
            data['np'] = np
            iargs = ", ".join(info["init"]["input_args"]) + ", datatype=np.float32"
            init_str = "{oargs} = {i}({iargs})".format(
                oargs=",".join(info["init"]["output_args"]),
                i=info["init"]["func_name"],
                iargs=iargs)
            exec(init_str, data)
            del data[info["init"]["func_name"]]

        self.bdata[preset] = data
        return self.bdata[preset]
