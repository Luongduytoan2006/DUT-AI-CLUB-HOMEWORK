# -*- coding: utf-8 -*-
"""Wrapper auto-resume: chay train, neu crash thi tu chay lai (resume tu last.pt)."""
import time
import traceback

import config as C
from train import main, log

MAX_RETRY = 20

if __name__ == "__main__":
    attempt = 0
    while attempt < MAX_RETRY:
        try:
            main()
            log("=== HOAN TAT (thoat wrapper) ===")
            break
        except Exception:
            attempt += 1
            log(f"!!! CRASH lan {attempt}: {traceback.format_exc()}")
            log(f"Thu lai sau 10s (resume tu {C.CKPT_LAST}) ...")
            time.sleep(10)

