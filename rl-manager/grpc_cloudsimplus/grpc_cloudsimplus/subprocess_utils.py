"""
Subprocess utilities for launching CloudSim gRPC servers.

Each Python worker process (created by SubprocVecEnv) calls start_java_server()
to spawn a dedicated Java JVM running the gRPC server, then connects to it.

This avoids socket collisions and allows true parallelism when using SubprocVecEnv
with the fork/start_method.
"""
import subprocess
import os
import time
import grpc
import sys
from typing import Optional, Tuple

# Path to the built CloudSim JAR (built via: ./gradlew bootJar)
# Mounted from cloudsimplus-gateway/build/libs/ into the container
DEFAULT_JAR_PATH = os.environ.get(
    "CLOUDSIM_GATEWAY_JAR",
    "/app/cloudsimplus-gateway/build/libs/cloudsimplus-gateway-0.1.0-boot.jar",
)
DEFAULT_PORT_RANGE_START = 50051
GRPC_STARTUP_TIMEOUT = 60  # seconds to wait for Java server to be ready


def _find_free_port(start: int = 50051) -> int:
    """Find a free port starting from `start`."""
    import socket
    for port in range(start, start + 1000):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find a free port")


def start_java_server(
    port: int = None,
    jar_path: str = DEFAULT_JAR_PATH,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> Tuple[subprocess.Popen, int]:
    """
    Spawn a Java CloudSim gRPC server subprocess and wait for it to be ready.

    Args:
        port:          TCP port for the gRPC server. If None, a free port is auto-selected.
        jar_path:      Path to the Spring Boot JAR containing the CloudSim gRPC server.
        log_level:     Java log level (INFO, DEBUG, WARN, ERROR).
        log_file:      Optional path to write Java stdout/stderr. If None, inherits parent.

    Returns:
        (subprocess_handle, assigned_port) - the Popen handle and the port the server is on.
        The caller should call proc.terminate() when done.

    Note:
        With SubprocVecEnv fork start_method, this must be called AFTER fork
        (i.e., inside the _init function passed to SubprocVecEnv), so each
        worker gets its own JVM.
    """
    if port is None:
        port = _find_free_port(DEFAULT_PORT_RANGE_START)

    java_cmd = [
        "java",
        "-jar",
        jar_path,
        "--server.port=" + str(port),
        f"--logging.level.root={log_level}",
    ]

    java_env = os.environ.copy()
    # Disable JVM AOT/paging to reduce startup time in containers
    java_env.setdefault("JAVA_TOOL_OPTIONS", "-XX:+UseSerialGC")

    dev_null = open(os.devnull, "w") if sys.platform == "win32" else subprocess.DEVNULL

    proc = subprocess.Popen(
        java_cmd,
        env=java_env,
        stdout=open(log_file, "w") if log_file else dev_null,
        stderr=subprocess.STDOUT,
    )

    # Wait for the gRPC server to be ready by polling
    deadline = time.time() + GRPC_STARTUP_TIMEOUT
    channel = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Java gRPC server process exited prematurely with code {proc.returncode}"
            )
        try:
            channel = grpc.insecure_channel(f"localhost:{port}")
            grpc.channel_ready_future(channel).result(timeout=2)
            channel.close()
            break
        except Exception:
            time.sleep(0.5)
            continue

    if channel is not None:
        channel.close()

    return proc, port


def stop_java_server(proc: subprocess.Popen, timeout: float = 10.0):
    """Gracefully terminate a Java gRPC server subprocess."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
