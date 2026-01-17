import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_SHORT


@pytest.mark.run(order=TEST_SHORT)
def test_mesolve_offload_states_cpu_noop():
    H = dq.sigmaz()
    Ls = [dq.sigmax()]
    rho0 = dq.ground_dm()
    tsave = jnp.linspace(0.0, 0.1, 3)

    options = dq.Options(offload_states=True)
    res = dq.mesolve(H, Ls, rho0, tsave, options=options)
    ref = dq.mesolve(H, Ls, rho0, tsave)

    assert res.states.shape == ref.states.shape
    assert jnp.allclose(res.states.to_jax(), ref.states.to_jax()).item()
