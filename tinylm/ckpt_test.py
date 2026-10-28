"""Week 1 check that checkpoints survive a TPU session ending.

Run once, stop the session, start a new one, run again:
python -m tinylm.ckpt_test --dir /mnt/group/ckpt_test
First run saves; second run restores and checks the values match.
"""
import argparse, os
import jax, jax.numpy as jnp
import orbax.checkpoint as ocp


def make_state():
    k = jax.random.PRNGKey(0)
    return {"w": jax.random.normal(k, (768, 768)), "step": jnp.array(1234)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/group/ckpt_test")
    a = ap.parse_args()
    print("devices:", jax.devices())

    mgr = ocp.CheckpointManager(os.path.abspath(a.dir), options=ocp.CheckpointManagerOptions(max_to_keep=1))
    expected = make_state()
    step = mgr.latest_step()
    if step is None:
        mgr.save(1, args=ocp.args.StandardSave(expected))
        mgr.wait_until_finished()
        print(f"saved to {a.dir}. Now stop the session, start a new one and rerun this command.")
    else:
        got = mgr.restore(step, args=ocp.args.StandardRestore(expected))
        ok = all(bool(jnp.array_equal(got[k], expected[k])) for k in expected)
        print("restore OK: checkpoints survive the session" if ok else "MISMATCH: restored values differ")
