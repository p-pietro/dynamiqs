import textwrap
from pathlib import Path

import jax.numpy as jnp
import pytest

import dynamiqs as dq
from dynamiqs.distributed import DataParallel, make_mesh

from ..order import TEST_SHORT
from ..parallel_utils import run_parallel_script


@pytest.mark.run(order=TEST_SHORT)
def test_sesolve_parallel_cpu_devices():
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
            mesh = make_mesh(jax.devices(), axis_name="d")
            parallel = DataParallel(mesh=mesh, axis_name="d", batch_axis=0)
            options = dq.Options(parallel=parallel)

            H = dq.stack([dq.sigmaz(), dq.sigmaz()])
            psi0 = dq.stack([dq.ground(), dq.ground()])
            tsave = jnp.linspace(0.0, 0.1, 3)

            result = dq.sesolve(H, psi0, tsave, options=options)
            result.states.block_until_ready()
            devices = result.states.devices()
            if len(devices) != 2:
                raise RuntimeError(f"Expected sharded output, got {devices}.")

            print("OK")


        if __name__ == "__main__":
            main()
        """
    )
    run_parallel_script(script, repo_root=repo_root)


@pytest.mark.run(order=TEST_SHORT)
def test_sesolve_parallel_flat_batching_unsupported():
    mesh = make_mesh()
    parallel = DataParallel(mesh=mesh, axis_name='d', batch_axis=0)
    options = dq.Options(cartesian_batching=False, parallel=parallel)

    H = dq.sigmaz().broadcast_to(2, 3, 2, 2)
    psi0 = dq.stack([dq.ground()] * 3)
    tsave = jnp.linspace(0.0, 0.1, 3)

    with pytest.raises(ValueError, match='flat batching'):
        dq.sesolve(H, psi0, tsave, options=options)
