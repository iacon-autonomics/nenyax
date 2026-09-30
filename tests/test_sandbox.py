import shutil
import subprocess
import sys

import pytest

from nenyax import UnsupportedCapability, sandbox
from nenyax.sandbox import BackendNotConfigured, Isolation, SandboxSpec, WarmPool
from nenyax.sandbox.process import ProcessBackend
from nenyax.testing import assert_sandbox_conforms


class KeyedBackend(ProcessBackend):
    """Stands in for a bring-your-own-key hosted provider."""

    name = "keyed-test"
    credentials = ("NENYAX_TEST_PROVIDER_KEY",)


def test_isolation_levels_are_ordered():
    assert Isolation.MICROVM.satisfies(Isolation.CONTAINER)
    assert not Isolation.PROCESS.satisfies("container")


def test_process_backend_conforms():
    assert_sandbox_conforms(ProcessBackend())


def test_process_sandbox_blocks_path_escape():
    with ProcessBackend().create() as sb, pytest.raises(PermissionError):
        sb.read_file("/workspace/../../etc/passwd")


def test_process_fork_is_independent():
    with ProcessBackend().create() as parent:
        parent.write_file("x.txt", "1")
        with parent.fork() as child:
            child.write_file("x.txt", "2")
            assert child.exec("cat x.txt").stdout == "2"
        assert parent.read_file("x.txt") == b"1"


def test_select_returns_none_when_no_isolation_needed():
    assert sandbox.select("none") is None


def test_select_prefers_lightest_backend():
    assert sandbox.select("process").name == "process"


def test_keyed_backend_reports_missing_key(monkeypatch):
    monkeypatch.delenv("NENYAX_TEST_PROVIDER_KEY", raising=False)
    sandbox.register(KeyedBackend)
    with pytest.raises(BackendNotConfigured, match="set NENYAX_TEST_PROVIDER_KEY"):
        sandbox.get("keyed-test")
    assert sandbox.get("keyed-test", nenyax_test_provider_key="k").available()


def test_select_honours_preference_once_key_is_present(monkeypatch):
    sandbox.register(KeyedBackend)
    monkeypatch.delenv("NENYAX_TEST_PROVIDER_KEY", raising=False)
    assert sandbox.select("process", prefer=["keyed-test"]).name == "process"  # no key: skipped
    monkeypatch.setenv("NENYAX_TEST_PROVIDER_KEY", "k")
    assert sandbox.select("process", prefer=["keyed-test"]).name == "keyed-test"


def test_unmeetable_requirements_are_explicit():
    with pytest.raises(BackendNotConfigured, match="fork>=memory"):
        sandbox.select("process", fork="memory")


def test_warm_pool_serves_identical_fresh_state():
    with WarmPool(ProcessBackend(), setup=["echo seeded > seed.txt"], size=2) as pool:
        first = pool.acquire()
        first.write_file("seed.txt", "dirty")
        first.destroy()
        with pool.acquire() as second:
            assert second.read_file("seed.txt") == b"seeded\n"


def _docker_ready() -> bool:
    return (
        bool(shutil.which("docker"))
        and subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.skipif(not _docker_ready(), reason="Docker is not running")
def test_docker_backend_conforms_and_exposes_ports():
    import os

    backend = sandbox.get("docker", advertise_host=os.environ.get("NENYAX_DOCKER_HOST"))
    assert_sandbox_conforms(backend)
    with pytest.raises(UnsupportedCapability):
        backend.create(SandboxSpec(ports=(8000,), network=False))
    with backend.create(SandboxSpec(ports=(8000,), network=True)) as sb:
        sb.exec("nohup python3 -m http.server 8000 >/dev/null 2>&1 &")
        import time
        import urllib.request

        body, last_error = b"", None
        for _ in range(40):
            try:
                body = urllib.request.urlopen(sb.url(8000), timeout=1).read()
                break
            except OSError as e:
                last_error = e
                time.sleep(0.25)
        assert b"Directory listing" in body, f"{sb.url(8000)} unreachable: {last_error}"


# -- regressions from the adversarial campaign ---------------------------------------------------


def test_output_flood_is_capped_not_buffered():
    spec = SandboxSpec(output_cap=4096)
    with ProcessBackend().create(spec) as sb:
        r = sb.exec("yes AAAA | head -c 50000000")
    assert r.ok and r.truncated and len(r.stdout) == 4096


def test_background_process_holding_pipes_does_not_hang_exec():
    with ProcessBackend().create() as sb:
        r = sb.exec("sleep 30 & echo started")
    assert r.ok and r.stdout.strip() == "started" and r.duration_s < 10


def test_signal_kill_is_reported_as_killed_not_timeout():
    with ProcessBackend().create() as sb:
        r = sb.exec("kill -9 $$")
    assert r.killed and not r.timed_out and not r.ok


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.skipif(not _docker_ready(), reason="Docker is not running")
def test_docker_memory_bomb_is_killed_and_host_survives():
    with sandbox.get("docker").create(SandboxSpec(memory_mb=256)) as sb:
        r = sb.exec('python3 -c "x = bytearray(2 * 1024**3)"', timeout_s=30)
        # Either outcome means the limit held: the kernel refuses the allocation (MemoryError,
        # typical on Linux hosts) or the OOM killer stops the process (typical on Docker Desktop).
        assert not r.ok and not r.timed_out
        assert r.killed or "MemoryError" in r.stderr, r
        assert sb.exec("echo alive").ok


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="memory rlimits are Linux-only")
def test_process_backend_enforces_memory_on_linux():
    backend = ProcessBackend()
    assert "memory" in backend.info.enforces
    with backend.create(SandboxSpec(memory_mb=256)) as sb:
        r = sb.exec('python3 -c "x = bytearray(1024**3); print(len(x))"', timeout_s=30)
    assert not r.ok and "MemoryError" in r.stderr
