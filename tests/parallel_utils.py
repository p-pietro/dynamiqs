from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _build_xla_flags(*, device_count: int) -> str:
    existing_flags = os.environ.get('XLA_FLAGS', '')
    force_flag = f'--xla_force_host_platform_device_count={device_count}'
    filtered_flags = ' '.join(
        flag
        for flag in existing_flags.split()
        if not flag.startswith('--xla_force_host_platform_device_count')
    ).strip()
    return f'{filtered_flags} {force_flag}'.strip()


def run_parallel_script(script: str, *, repo_root: Path, device_count: int = 2) -> None:
    env = os.environ.copy()
    env['XLA_FLAGS'] = _build_xla_flags(device_count=device_count)
    env['JAX_PLATFORM_NAME'] = 'cpu'

    python = Path(sys.executable)
    result = subprocess.run(
        [str(python), '-c', script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            'Parallel test script failed.\n'
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        )
