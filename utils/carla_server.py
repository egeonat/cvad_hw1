import os
import subprocess
from time import sleep

import psutil


def find_carla_procs():
    carla_procs = []
    for process in psutil.process_iter():
        if process.name().lower().startswith("CarlaUE4".lower()):
            carla_procs.append(process)
    return carla_procs


def kill_carla_server():
    """Find and terminate/kill existing carla processes."""
    still_alive = find_carla_procs()
    # Ask to terminate politely
    for process in still_alive:
        process.terminate()
    sleep(1)
    still_alive = find_carla_procs()
    # Kill process if still alive
    if still_alive:
        for process in still_alive:
            process.kill()
    psutil.wait_procs(still_alive)


def start_carla_server(port):
    """Start carla server and wait for it to initialize."""
    kill_carla_server()
    carla_path = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
    cmd = [carla_path,
           f"-carla-rpc-port={port}",
           "-quality-level=Epic"]
    subprocess.Popen(cmd)
