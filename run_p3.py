"""
Crash-resilient launcher for train_p3.py.

Restarts training automatically whenever the subprocess exits with a non-zero
code (e.g. CUDA illegal memory access from GPU contention with the display
driver).  Resumes from the latest checkpoint each time, so at most
P3_CHECKPOINT_EVERY steps are lost per crash.

Usage:
    python run_p3.py
"""

import subprocess
import sys
import time

MAX_RESTARTS   = 50       # give up after this many consecutive crashes
RESTART_DELAY  = 5        # seconds to wait before restarting

def main():
    restarts = 0
    while restarts < MAX_RESTARTS:
        print(f"\n{'='*60}")
        if restarts > 0:
            print(f"[run_p3] Restart #{restarts} — resuming from checkpoint...")
        else:
            print("[run_p3] Starting training...")
        print(f"{'='*60}\n")

        result = subprocess.run(
            [sys.executable, "train_p3.py"],
            # Inherit stdout/stderr so tqdm and loss logs appear normally.
        )

        if result.returncode == 0:
            print("[run_p3] Training finished successfully.")
            break

        restarts += 1
        print(f"\n[run_p3] Process exited with code {result.returncode}. "
              f"Restarting in {RESTART_DELAY}s... ({restarts}/{MAX_RESTARTS})")
        time.sleep(RESTART_DELAY)
    else:
        print(f"[run_p3] Gave up after {MAX_RESTARTS} restarts.")
        sys.exit(1)


if __name__ == "__main__":
    main()
