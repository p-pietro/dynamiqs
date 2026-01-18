import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_device_batching_shard_map_cpu():
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import dynamiqs as dq
        from dynamiqs.method import Tsit5
        from jax.sharding import PartitionSpec as P

        assert jax.device_count() == 2

        n = 2
        key = jax.random.PRNGKey(0)
        (
            key_h,
            key_psi,
            key_h_cart,
            key_psi_cart,
            key_h_me,
            key_l_me,
            key_rho,
        ) = jax.random.split(key, 7)
        tsave = jnp.linspace(0.0, 0.01, 3)
        ntsave = tsave.shape[0]

        # flat batching (diffrax)
        H = dq.random.operator(key_h, n, batch=(2, 4))
        psi0 = dq.random.ket(key_psi, n, batch=(2, 4))
        options = dq.Options(
            cartesian_batching=False,
            device_batching=dq.DeviceBatching(mesh_shape=(2,), batch_axes=(1,)),
            progress_meter=False,
        )
        result = dq.sesolve(H, psi0, tsave, method=Tsit5(), options=options)
        states = result.states.to_jax()
        assert states.shape == (2, 4, ntsave, n, 1)
        assert states.sharding.spec == P(None, "d0")

        # cartesian batching (diffrax)
        H_cart = dq.random.operator(key_h_cart, n, batch=(2, 4))
        psi0_cart = dq.random.ket(key_psi_cart, n, batch=(3,))
        options_cart = dq.Options(
            cartesian_batching=True,
            device_batching=dq.DeviceBatching(mesh_shape=(2,), batch_axes=(1,)),
            progress_meter=False,
        )
        result_cart = dq.sesolve(
            H_cart, psi0_cart, tsave, method=Tsit5(), options=options_cart
        )
        states_cart = result_cart.states.to_jax()
        assert states_cart.shape == (2, 4, 3, ntsave, n, 1)
        assert states_cart.sharding.spec == P(None, "d0")

        # list-of-operators batching (diffrax)
        H_me = dq.random.herm(key_h_me, (2, 4, n, n))
        Ls = [dq.random.operator(key_l_me, n, batch=(2, 4))]
        rho0 = dq.random.dm(key_rho, n, batch=(2, 4))
        options_me = dq.Options(
            cartesian_batching=False,
            device_batching=dq.DeviceBatching(mesh_shape=(2,), batch_axes=(1,)),
            progress_meter=False,
        )
        result_me = dq.mesolve(
            H_me, Ls, rho0, tsave, method=Tsit5(), options=options_me
        )
        states_me = result_me.states.to_jax()
        assert states_me.shape == (2, 4, ntsave, n, n)
        assert states_me.sharding.spec == P(None, "d0")
        """
    )
    env = os.environ.copy()
    flags = env.get('XLA_FLAGS', '')
    extra = '--xla_force_host_platform_device_count=2'
    env['XLA_FLAGS'] = f'{flags} {extra}'.strip()
    env['PYTHONPATH'] = str(repo_root)
    subprocess.run([sys.executable, '-c', script], env=env, check=True)
