import textwrap
from pathlib import Path

import pytest

from ..order import TEST_SHORT
from ..parallel_utils import run_parallel_script


@pytest.mark.run(order=TEST_SHORT)
def test_dsmesolve_parallel_cpu_devices():
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
            Ls = [dq.sigmax()]
            rho0 = dq.stack([dq.ground_dm(), dq.ground_dm()])
            tsave = jnp.linspace(0.0, 0.2, 3)
            keys = jax.random.split(jax.random.key(0), 2)
            method = dq.method.EulerMaruyama(dt=0.1)

            result = dq.dsmesolve(
                H,
                Ls,
                jnp.array([0.5]),
                rho0,
                tsave,
                keys,
                method=method,
                options=options,
            )
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
