#!/usr/bin/env python3
"""
run_all.py

This launcher does the following:
 1. Runs `initiate.py` in the current (main) terminal.
 2. After 30 seconds, opens a new GNOME Terminal window to run `RML_yolo.py`.
 3. After 5 more seconds, opens another GNOME Terminal window to run `missioncorrect.py`.
 4. If you hit Ctrl+C in the main terminal, it will send SIGINT to all three child processes (including the two GNOME‐Terminal windows), so they all shut down together.
"""

import subprocess
import time
import signal
import sys
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────────
# Make sure these paths point to your scripts. If run_all.py lives in the same folder
# as initiate.py, RML_yolo.py, and missioncorrect.py, you can leave them like this:
BASE_DIR = Path(__file__).parent.resolve()
INITIATE_SCRIPT       = BASE_DIR / "Scripts/Initiate.py"
RML_YOLO_SCRIPT       = BASE_DIR / "Scripts/RML_yolo.py"
MISSIONCORRECT_SCRIPT = BASE_DIR / "Scripts/Missioncorrrect.py"

# Command to launch a new GNOME Terminal running a given script:
def launch_in_new_gnome_terminal(python_script: Path) -> subprocess.Popen:
    """
    Opens a new gnome-terminal window, runs `python3 <python_script>`, then keeps the shell open.
    The `exec bash` ensures you can see output/logs until you manually close the terminal window.
    """
    return subprocess.Popen([
        "gnome-terminal",
        "--",                      # anything after this goes to the shell inside GNOME Terminal
        "bash", "-c",
        f"python3 \"{python_script}\"; exec bash"
    ])

# ── GLOBALS ──────────────────────────────────────────────────────────────────────
child_processes = []  # will hold Popen objects for initiate.py, gnome-terminal#1, gnome-terminal#2

def on_sigint(signum, frame):
    """
    This handler will be called when you press Ctrl+C in the main terminal.
    We forward SIGINT (Ctrl+C) to every child process (including the two gnome-terminal windows),
    then exit.
    """
    print("\n[!] Caught Ctrl+C in main terminal. Terminating all children…")
    for p in child_processes:
        try:
            # Try sending a SIGINT first
            p.send_signal(signal.SIGINT)
            time.sleep(0.1)
            # If it’s still alive, force‐kill it
            if p.poll() is None:
                p.kill()
        except Exception:
            pass

    sys.exit(0)


def main():
    # Register our SIGINT handler so that Ctrl+C in main goes to on_sigint()
    signal.signal(signal.SIGINT, on_sigint)

    # 1) Launch initiate.py in the current (main) terminal
    if not INITIATE_SCRIPT.exists():
        print(f"[ERROR] Cannot find {INITIATE_SCRIPT}")
        sys.exit(1)

    print("[+] Launching initiate.py in this terminal…")
    p1 = subprocess.Popen(
        ["python3", str(INITIATE_SCRIPT)],
        # stdout/stderr inherit from this script, so you see initiate.py output here
    )
    child_processes.append(p1)
    print(f"    → PID(initiate.py) = {p1.pid}")

    # 2) Wait 30 seconds
    print("[…] Waiting 30 seconds before starting RML_yolo.py …")
    time.sleep(30)

    # 3) Launch RML_yolo.py in a new GNOME Terminal
    if not RML_YOLO_SCRIPT.exists():
        print(f"[ERROR] Cannot find {RML_YOLO_SCRIPT}")
        # If you want to keep going even if this fails, comment out sys.exit(1)
        sys.exit(1)

    print("[+] Opening a new GNOME Terminal for RML_yolo.py …")
    p2 = launch_in_new_gnome_terminal(RML_YOLO_SCRIPT)
    child_processes.append(p2)
    print(f"    → PID(gnome-terminal for RML_yolo.py) = {p2.pid}")

    # 4) Wait 5 more seconds
    print("[…] Waiting 5 seconds before starting missioncorrect.py …")
    time.sleep(5)

    # 5) Launch missioncorrect.py in another new GNOME Terminal
    if not MISSIONCORRECT_SCRIPT.exists():
        print(f"[ERROR] Cannot find {MISSIONCORRECT_SCRIPT}")
        sys.exit(1)

    print("[+] Opening another GNOME Terminal for missioncorrect.py …")
    p3 = launch_in_new_gnome_terminal(MISSIONCORRECT_SCRIPT)
    child_processes.append(p3)
    print(f"    → PID(gnome-terminal for missioncorrect.py) = {p3.pid}")

    # 6) Now keep the launcher alive until all children have exited
    print("[…] Launcher is now waiting for all scripts to finish. Press Ctrl+C to kill them all.")
    while True:
        # Check if every process has terminated
        if all(p.poll() is not None for p in child_processes):
            print("[+] All child processes have exited. Exiting launcher.")
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
