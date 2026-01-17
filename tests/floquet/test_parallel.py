import textwrap
from pathlib import Path

import pytest

from ..order import TEST_SHORT
from ..parallel_utils import run_parallel_script


@pytest.mark.run(order=TEST_SHORT)
def test_floquet_parallel_cpu_devices():
    repo_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp

        import dynamiqs as dq
        from dynamiqs.distributed import DataParallel, make_mesh


        def main():
            if jax.device_count() != 2:
                raise RuntimeError(
                    f"Expected 2 CPU devices, got {jax.device_count()}."
                )
            mesh = make_mesh(jax.devices(), axis_names="d")
            parallel = DataParallel(mesh=mesh, axis_names="d", batch_axis=0)
            options = dq.Options(parallel=parallel)

            H = dq.stack([dq.sigmaz(), dq.sigmaz()])
            tsave = jnp.linspace(0.0, 0.2, 3)

            result = dq.floquet(H, 0.2, tsave, options=options)
            result.modes.block_until_ready()
            devices = result.modes.devices()
            if len(devices) != 2:
                raise RuntimeError(f"Expected sharded output, got {devices}.")

            print("OK")


        if __name__ == "__main__":
            main()
        """
    )
    run_parallel_script(script, repo_root=repo_root)
