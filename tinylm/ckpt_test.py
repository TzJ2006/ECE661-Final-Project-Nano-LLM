"""Week 1 check that checkpoints survive a TPU session ending (same save code as train.py).

Run once, stop the session, start a new one, run again:
python -m tinylm.ckpt_test --dir ~/project/ckpt_test
First run saves; second run restores and checks the values match.
"""
import argparse, os
import jax, jax.numpy as jnp
from flax import serialization

from tinylm.train import save

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/project/ckpt_test"))
    a = ap.parse_args()
    print("devices:", jax.devices())

    expected = {"w": jax.random.normal(jax.random.PRNGKey(0), (768, 768)), "step": 1234}
    path = os.path.join(a.dir, "ckpt.msgpack")
    if not os.path.exists(path):
        os.makedirs(a.dir, exist_ok=True)
        save(path, expected)
        print(f"saved to {path}. Now stop the session, start a new one and rerun this command.")
    else:
        got = serialization.from_bytes(expected, open(path, "rb").read())
        ok = bool(jnp.array_equal(got["w"], expected["w"])) and got["step"] == expected["step"]
        print("restore OK: checkpoints survive the session" if ok else "MISMATCH: restored values differ")
