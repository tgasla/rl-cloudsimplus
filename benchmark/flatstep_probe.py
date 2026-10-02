"""Does the agent lose to the rules because it decides a whole step at once?

The RL policy emits one action per job slot from ONE observation: `JobPlacementEnv.action_masks`
is built from `_last_infr_obs`, the snapshot at the start of the step, and by design a full DC
stays legal. So 32 slots are chosen without any of them seeing what the others just did.

The rules are not scored under that constraint. R1/R2/R3 walk their slots in priority order and
update as they go -- R1 decrements the chosen host's free PEs and adds the job's core-timesteps to
that DC's backlog before the next slot is decided (reference_policies.R1.act; the gateway does the
same, WrappedSimulation line 304: "backlog_core_ts plus the core-timesteps placed there this
step"). They get within-step contention accounting; the agent does not.

This probe isolates exactly that one variable, with no training. Each rule is replayed in a FLAT
variant that decides every slot from the pristine start-of-step view -- the agent's information --
and is otherwise byte-identical. The sequential/flat difference then prices the handicap:

    gap_handicap = G(rule, sequential) - G(rule, flat)

against the measured agent-rule gap on the same member and split. If the flat rules fall most of
the way to the agent, the gap is the action space, not the policy class, and an autoregressive
head is the fix. If they barely move, the architecture is not being handicapped and the agent's
deficit is its own.

    python3 benchmark/flatstep_probe.py [--members S] [--split val] [--num-cpu 8]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference_policies as rp  # noqa: E402
import run_references as rr  # noqa: E402


def flatten(factory, base_name):
    """Wrap a rule so it cannot see its own placements inside a step.

    POLICIES holds classes (R1, R2) and functools.partial objects (the R3 family), so this wraps
    the FACTORY rather than subclassing. The rules keep their within-step state in locals of
    act() -- R1 in `free` and `placed_work` -- so rather than reach into each one, act() is run
    once per slot on a view exposing only that slot. Same decision rule, same tie-breaks, same
    DC state every time: no slot can observe what another just did.
    """

    def make(topo, **kwargs):
        return _FlatWrapper(factory(topo, **kwargs), base_name + "-flat")

    return make


class _FlatWrapper:
    def __init__(self, inner, name):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "name", name)

    def act(self, view, ctx):
        out = np.zeros(self._inner.topo.n_slots, dtype=np.int64)
        for i in view.slots:
            out[i] = self._inner.act(_SingleSlotView(view, i), ctx)[i]
        return out

    def __getattr__(self, k):
        return getattr(self._inner, k)


class _SingleSlotView:
    """The step's view restricted to one slot; everything else delegated unchanged, so the rule
    sees exactly the DC state, reach and job features it would have seen first in the step."""

    def __init__(self, view, slot):
        object.__setattr__(self, "_v", view)
        object.__setattr__(self, "_slot", slot)

    @property
    def slots(self):
        return [self._slot]

    def __getattr__(self, k):
        return getattr(self._v, k)


FLAT = {}
for _n in ("R1", "R2") + tuple(rp.R3_FAMILY):
    FLAT[_n + "-flat"] = flatten(rp.POLICIES[_n], _n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="S")
    ap.add_argument("--split", default="val", choices=["val", "test", "lockbox"])
    ap.add_argument("--num-cpu", type=int, default=8)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results_flatstep"))
    a = ap.parse_args()

    rp.POLICIES.update(FLAT)                       # register before the runner resolves names
    rp.ONLINE_POLICIES = tuple(rp.ONLINE_POLICIES) + tuple(FLAT)
    names = [n for n in ("R1", "R2") + tuple(rp.R3_FAMILY)]
    both = names + [n + "-flat" for n in names]
    os.makedirs(a.out_dir, exist_ok=True)

    argv = ["run_references.py", "--members", a.members, "--split", a.split,
            "--policies", ",".join(both), "--rollouts", "1",
            "--num-cpu", str(a.num_cpu), "--plan-workers", str(a.num_cpu),
            "--out-dir", a.out_dir]
    sys.argv = argv
    print("playing:", ", ".join(both), flush=True)
    rr.main()

    d = pd.read_csv(os.path.join(a.out_dir, f"references_{a.split}.csv"))
    d = d[d.policy.isin(both)]
    piv = d.pivot_table(index=["member", "level_id"], columns="policy", values="unshaped_return")
    rows = []
    for n in names:
        if n in piv and n + "-flat" in piv:
            seq, flat = piv[n], piv[n + "-flat"]
            rows.append(dict(rule=n, sequential=seq.mean(), flat=flat.mean(),
                             handicap=(seq - flat).mean(),
                             handicap_pct=100 * (seq - flat).mean() / abs(seq.mean())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out_dir, "flatstep_summary.csv"), index=False, float_format="%.6g")
    pd.set_option("display.width", 160)
    print("\n=== within-step contention handicap (raw unshaped return) ===")
    print(out.round(4).to_string(index=False))
    print("\nIf 'flat' lands near the agent, the action space is the problem, not the policy class.")


if __name__ == "__main__":
    main()
