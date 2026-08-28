#!/usr/bin/env python3
"""Patch pbb/utils.py runexp so 'fashion-mnist' builds the FCN posterior net (same as mnist).
Idempotent. Only the FCN branch needed fixing — the CNN branches use `else` for non-cifar."""
from pathlib import Path
f = Path("/root/autodl-tmp/pacbayes-cert/third_party/PBB/pbb/utils.py")
s = f.read_text()
old = "        elif name_data == 'mnist':"
new = "        elif name_data in ('mnist', 'fashion-mnist'):"
if new in s:
    print("ALREADY_PATCHED")
elif old in s:
    n = s.count(old)
    f.write_text(s.replace(old, new))
    print(f"PATCHED ({n} occurrence)")
else:
    raise SystemExit("anchor 'elif name_data == mnist' not found")
