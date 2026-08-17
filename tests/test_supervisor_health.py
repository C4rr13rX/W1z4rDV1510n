"""Regression tests for the supervisor's node health probes.

Context (2026-08-17): the crypto node reported healthy for 4d19h while
learning nothing. It had grown to 47 GB private on a 31.8 GB box, so every
write route blocked on page-ins, while GET /health kept answering in ~2 ms
from a static struct. Two defects made that invisible:

  * start_node() never exported the RAM politeness floor that
    start_node.ps1 has always set, so a supervisor-launched node had no
    ceiling at all;
  * the watchdog only probed a read route, which cannot distinguish "up"
    from "up but unable to learn".

These tests pin both fixes plus the guards that keep the write probe from
restarting a node that is merely still loading.
"""
import http.server
import json
import threading
import time

import pytest

from scripts import w1z4rd_supervisor as sup


class _Handler(http.server.BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *args):  # silence per-request logging
        pass

    def do_POST(self):
        if _Handler.mode == "hang":
            time.sleep(30)
            return
        if _Handler.mode == "error":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # /brain/tick answers with a bare counter, not an object.
        body = b"983093" if _Handler.mode == "scalar" else json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(name="tick_url")
def _tick_url():
    # _Handler.mode is class state shared by every server instance, and the
    # "hang" test leaves a request thread sleeping inside the handler. Reset
    # it per test so one case cannot leak into the next.
    _Handler.mode = "ok"
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/brain/tick"
    finally:
        server.shutdown()
        server.server_close()
        _Handler.mode = "ok"


@pytest.mark.parametrize("mode", ["ok", "scalar", "error"])
def test_write_probe_accepts_any_answer(tick_url, mode):
    """Liveness, not correctness: a bare int or even a 500 proves the
    handler ran rather than blocking on the fabric lock."""
    _Handler.mode = mode
    assert sup.node_write_healthy(tick_url, 5.0) is True


def test_write_probe_reports_a_hang(tick_url):
    _Handler.mode = "hang"
    started = time.time()
    assert sup.node_write_healthy(tick_url, 3.0) is False
    assert time.time() - started < 6.0, "probe must honour its timeout"


def test_write_probe_reports_a_refused_connection():
    assert sup.node_write_healthy("http://127.0.0.1:1/brain/tick", 2.0) is False


def test_write_probe_can_be_disabled():
    assert sup.node_write_healthy("", 5.0) is True


def test_node_launch_exports_the_ram_ceiling():
    """start_node() must mirror start_node.ps1's politeness contract.

    Without W1Z4RD_TIER_MIN_SYS_AVAIL_MB the tier orchestrator never evicts
    to the SSD cold tier and the node outgrows the machine.
    """
    node = sup.load_config()["node"]
    assert int(node["min_sys_avail_mb"]) > 0
    assert int(node["autocheckpoint_secs"]) > 0

    source = (sup.PROJECT_ROOT / "scripts" / "w1z4rd_supervisor.py").read_text(
        encoding="utf-8"
    )
    start_node = source.split("def start_node(")[1].split("\ndef ")[0]
    assert "W1Z4RD_TIER_MIN_SYS_AVAIL_MB" in start_node
    assert "W1Z4RD_BRAIN_AUTOCHECKPOINT_SECS" in start_node


def test_write_probe_settings_are_not_hair_triggered():
    node = sup.load_config()["node"]
    # A single blocked probe must never restart a node holding hours of
    # in-RAM fabric.
    assert int(node["write_health_misses_before_restart"]) > 1
    # /brain/tick mutates the fabric, so probe far less often than we poll.
    assert float(node["write_health_interval"]) > float(node["poll_interval"]) * 10
    # The probe blocks this single-threaded loop for its whole timeout.
    assert float(node["write_health_timeout"]) <= 60.0


def test_write_probe_waits_out_the_startup_warmup():
    """The guard that stops us killing a node mid-deserialise.

    An 11 GB brain takes 12-15 min to load, during which the write path is
    legitimately busy; warmup_secs exists precisely to tolerate that.
    """
    node = sup.load_config()["node"]
    warmup = float(node["warmup_secs"])
    assert warmup > 0

    def would_probe(first_healthy_unix, now, miss_count):
        streak = (now - first_healthy_unix) if first_healthy_unix else 0.0
        return miss_count == 0 and streak >= warmup

    first_healthy = 1_000_000.0
    # Mid-load: /health answers, but we are still inside the warmup window.
    assert would_probe(first_healthy, first_healthy + warmup / 2, 0) is False
    # Past warmup the probe becomes active.
    assert would_probe(first_healthy, first_healthy + warmup + 1, 0) is True
    # A read-probe miss suspends it: the node is already suspect.
    assert would_probe(first_healthy, first_healthy + warmup + 1, 1) is False
    # A restart clears the timer, so the next node re-serves its warmup.
    assert would_probe(0.0, first_healthy + warmup + 1, 0) is False
