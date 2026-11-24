"""Frozen-selection evaluation on new coordinates. No optimizer or training."""
import argparse
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.special import roots_hermite
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import paths, require, source_is_local, model_directory
CONFIG = paths()
ROOT = CONFIG.archive
OUT = CONFIG.test
import verify_models as audit
source_is_local(audit)

DATA = audit.DATA
REF = OUT / "test_references"
PRED = OUT / "predictions"
RECORDS = OUT / "run_records"
PROTOCOL = OUT / "frozen_test_plan.json"
FAMILIES = audit.FAMILIES
PAIRS = audit.PAIRS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def task_id(eq, mode, reg):
    return eq + "_" + ("inverse" if mode == "inverse" else "forward") + "_" + reg


def ref_id(eq, reg):
    return eq + "_" + (reg if eq in audit.extended.CASES else "baseline")


def mesh(axes):
    return np.stack([v.ravel() for v in np.meshgrid(*axes, indexing="ij")], axis=1)


def midpoints(n, low=0., high=1.):
    return low + (np.arange(n) + .5) * ((high-low)/n)


def grids():
    result = {}
    for eq in ("Burgers", "KdV", "AllenCahn", "Wave1D"):
        for reg in (("base", "stress") if eq in audit.extended.CASES else ("baseline",)):
            result[ref_id(eq, reg)] = dict(case=eq, regime=reg, shape=[256, 100],
                axes=[midpoints(256, -1, 1).tolist(), midpoints(100).tolist()])
    for reg in ("base", "stress"):
        result[ref_id("Heat2D", reg)] = dict(case="Heat2D", regime=reg, shape=[32, 32, 25],
            axes=[midpoints(32).tolist(), midpoints(32).tolist(), midpoints(25).tolist()])
        result[ref_id("Helmholtz2D", reg)] = dict(case="Helmholtz2D", regime=reg, shape=[128, 128],
            axes=[midpoints(128).tolist(), midpoints(128).tolist()])
    result[ref_id("NS_periodic", "baseline")] = dict(case="NS_periodic", regime="baseline",
        shape=[32, 32, 40], axes=[midpoints(32, 0, 2*math.pi).tolist(),
        midpoints(32, 0, 2*math.pi).tolist(), midpoints(40).tolist()])
    return result


def freeze():
    assert not PROTOCOL.exists(), "Protocol already frozen; do not select again"
    prior = CONFIG.runs
    rows = read(prior)
    group = collections.defaultdict(list)
    for r in rows:
        if r["eq"] == "NS_wake" or r["reg"] not in ("baseline", "base", "stress"):
            continue
        if r["model"] not in PAIRS and r["model"] != "MLP":
            continue
        group[r["eq"], r["mode"], r["reg"]].append(r)
    assert len(group) == 21 and sum(map(len, group.values())) == 1071
    tasks = []
    for key, members in group.items():
        scores = {name: statistics.mean(r["Eu"] for r in members if r["model"] == name)
                  for name in (*PAIRS, "MLP")}
        best = min(PAIRS, key=scores.get)
        same = min((name for name, (a, b) in PAIRS.items() if a == b), key=scores.get)
        runs = []
        for r in members:
            source = model_directory(r["run"])
            meta = read(source / "result.json")
            assert sha(source / "final.pt") == meta["weights_sha256"]
            runs.append(dict(r, id=task_id(*key)+"_"+r["model"]+"_seed"+str(r["seed"]),
                             weights_sha256=meta["weights_sha256"]))
        tasks.append(dict(id=task_id(*key), eq=key[0], mode=key[1], reg=key[2],
                          selected=best, same_family=same, validation_means=scores,
                          reference=ref_id(key[0], key[2]), runs=runs))
    protocol = dict(version=1, source_validation_records_sha256=sha(prior),
        selection="Original reference evaluation is validation; mean field error across seeds 42,43,44",
        selection_locked_before_new_predictions=True, new_test_used_for_selection=False,
        scope="21 main task/condition matrices, 16 pairs plus MLP, three seeds; 1071 models",
        excluded=["NS_wake: existing window has already been evaluated in full",
                  "Auxiliary noise/sampling/initialization/legacy architecture experiments are not part of this batch"],
        interpretation="Prospective new-coordinate test of frozen models on the same PDE and condition; not new-task or new-parameter generalization",
        precision="FP64 inference and metrics; final archived weights, no optimization",
        statistic="Eu=100*sqrt(sum((prediction-reference)^2)/sum(reference^2)); joint u,v for NS; unweighted interior midpoint grids",
        aggregation="Mean and sample standard deviation across three seeds; gains are ratios of means",
        reference_plan={
            "Burgers":"Cole-Hopf Gauss-Hermite 512; compare 1024 at identical new coordinates",
            "KdV":"Original ETDRK4, n=512, dt=1.25e-5; compare n=512 dt=2.5e-5 and n=1024 dt=1.25e-5; midpoint output times",
            "NS_periodic":"Original spectral vorticity solver, n=128, rtol=2e-11, atol=2e-13; compare n=64 at new coordinates/times",
            "AllenCahn":"BDF central differences 8192 intervals, rtol=1e-10, atol=1e-12; compare 4096 same tolerances",
            "Heat2D/Helmholtz2D/Wave1D":"Direct analytic formulas in FP64 at predefined new coordinates"},
        grids=grids(), tasks=tasks,
        train_code_sha256={p.name: sha(p) for p in audit.CODE.glob("*.py")})
    write_new(PROTOCOL, protocol)
    write_new(OUT / "protocol_digest.json", {"sha256":sha(PROTOCOL)})
    print("Frozen before new evaluation:", len(tasks), "tasks, 1071 models", flush=True)


def checked_protocol():
    require(PROTOCOL, OUT / "protocol_digest.json")
    assert sha(PROTOCOL) == read(OUT / "protocol_digest.json")["sha256"]
    p = read(PROTOCOL)
    from source_integrity import archive_sources, matches_source, training_hashes
    current, archived = training_hashes(audit.CODE), archive_sources()
    for name, digest in p["train_code_sha256"].items():
        assert matches_source(name, digest, current, archived), name
    return p


def old_coordinates(eq, reg):
    if eq in ("Burgers", "KdV"):
        path = next(DATA.glob(eq+"reference/*"+(".mat" if eq == "KdV" else ".npz")))
        a = audit.loadmat(path) if eq == "KdV" else audit.loadz(path)
        return mesh([a["x"].ravel(), a["t"].ravel()])
    if eq == "NS_periodic":
        return audit.loadz(next(DATA.glob("NSreference/independent*diagnostic_data.npz")))["coordinates"]
    cn = "base" if reg == "base" else "hard"
    return audit.loadz(next(DATA.glob(eq+"*/forward_"+cn+"_case_reference.npz")))["coordinates"]


def overlap_count(a, b):
    dtype = np.dtype([(str(i), "<f8") for i in range(a.shape[1])])
    av = np.ascontiguousarray(np.round(a, 10)).view(dtype).ravel()
    bv = np.ascontiguousarray(np.round(b, 10)).view(dtype).ravel()
    return len(np.intersect1d(av, bv))


def burgers_reference(z, order):
    nodes, w = roots_hermite(order)
    keep = w > 0
    nodes, w = nodes[keep], w[keep]
    values = []
    nu = .01 / np.pi
    for start in range(0, len(z), 256):
        q = z[start:start+256]
        y = q[:, :1] - 2*np.sqrt(nu*q[:, 1:2])*nodes
        lw = np.log(w)[None, :] - np.cos(np.pi*y)/(2*np.pi*nu)
        weights = np.exp(lw-lw.max(axis=1, keepdims=True))
        values.append((-np.sum(weights*np.sin(np.pi*y), axis=1)/weights.sum(axis=1))[:, None])
    return np.concatenate(values)


def kdv_reference(grid):
    source = CONFIG.reference / "kdv_reference.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve")
    class MidpointOutput(ast.NodeTransformer):
        changes = 0
        def visit_Constant(self, n):
            if isinstance(n.value, float) and n.value == .01:
                self.changes += 1
                return ast.copy_location(ast.Constant(.005), n)
            if isinstance(n.value, int) and n.value == 101:
                self.changes += 1
                return ast.copy_location(ast.Constant(201), n)
            return n
    modifier = MidpointOutput()
    node = modifier.visit(node)
    assert modifier.changes == 2
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = {"np":np}
    exec(compile(tree, str(source), "exec"), scope)
    solve = scope["solve"]
    arrays = []
    for n, dt in ((512, 2.5e-5), (512, 1.25e-5), (1024, 1.25e-5)):
        x, t, u = solve(n, dt)
        xids = np.arange(n//512, n, n//256)
        tids = np.arange(1, 200, 2)
        assert np.allclose(x[xids], grid["axes"][0], atol=1e-15, rtol=0)
        assert np.allclose(t[tids], grid["axes"][1], atol=1e-15, rtol=0)
        arrays.append(u[np.ix_(xids, tids)].reshape(-1, 1))
    rel = lambda a,b: float(100*np.linalg.norm(a-b)/np.linalg.norm(b))
    check = dict(generator_sha256=sha(source), time_refinement_percent=rel(arrays[0], arrays[1]),
                 space_refinement_percent=rel(arrays[1], arrays[2]))
    assert max(check["time_refinement_percent"], check["space_refinement_percent"]) < 1e-7
    return arrays[1], check


def allencahn_reference(grid):
    p = audit.extended.physics("AllenCahn", grid["regime"])
    times = np.r_[0., grid["axes"][1]]
    fields = []
    for intervals in (4096, 8192):
        x = np.linspace(-1, 1, intervals+1)
        dx, nu, reaction = x[1]-x[0], p["epsilon"], p["reaction"]
        op = diags([np.full(intervals-2, nu/dx**2), np.full(intervals-1, -2*nu/dx**2),
                    np.full(intervals-2, nu/dx**2)], [-1,0,1], format="csc")
        boundary = np.zeros(intervals-1)
        boundary[[0,-1]] = -nu/dx**2
        def rhs(t,u):return op@u+boundary-reaction*(u**3-u)
        def jac(t,u):return op+diags(-reaction*(3*u*u-1), 0, format="csc")
        solution = solve_ivp(rhs, (0,1), x[1:-1]**2*np.cos(np.pi*x[1:-1]),
            method="BDF", t_eval=times, jac=jac, rtol=1e-10, atol=1e-12)
        assert solution.success
        full = np.vstack([-np.ones((1,len(times))), solution.y, -np.ones((1,len(times)))])
        ids = np.arange(intervals//512, intervals, intervals//256)
        assert np.allclose(x[ids], grid["axes"][0], atol=1e-15, rtol=0)
        fields.append(full[ids, 1:].reshape(-1,1))
    delta = float(100*np.linalg.norm(fields[0]-fields[1])/np.linalg.norm(fields[1]))
    assert delta < .001
    return fields[1], dict(refinement_4096_8192_percent=delta)


def ns_reference(grid):
    times = np.r_[0., grid["axes"][2]]
    fields = []
    for n, rtol, atol in ((64, 2e-10, 2e-12), (128, 2e-11, 2e-13)):
        a = audit.study.base.ns.spectral_solution(n, times, rtol, atol)[0]
        ids = np.arange(n//64, n, n//32)
        assert np.allclose(np.arange(n)[ids]*2*np.pi/n, grid["axes"][0], atol=1e-14, rtol=0)
        selected = a[1:][:, ids][:, :, ids]
        fields.append(selected.transpose(1,2,0,3).reshape(-1,2))
    delta = float(100*np.linalg.norm(fields[0]-fields[1])/np.linalg.norm(fields[1]))
    assert delta < .0001
    return fields[1], dict(refinement_64_128_percent=delta)


def references():
    p = checked_protocol()
    REF.mkdir(exist_ok=True)
    for rid, grid in p["grids"].items():
        destination = REF / (rid+".npz")
        meta = REF / (rid+".json")
        if destination.exists() or meta.exists():
            assert destination.exists() and meta.exists()
            assert sha(destination) == read(meta)["file_sha256"]
            continue
        z = mesh([np.array(axis) for axis in grid["axes"]])
        eq, reg = grid["case"], grid["regime"]
        assert len(z) == math.prod(grid["shape"])
        overlap = overlap_count(z, old_coordinates(eq, reg))
        assert overlap == 0, (rid, overlap)
        if eq == "Burgers":
            target = burgers_reference(z, 512)
            check = dict(order_512_1024_max_difference=float(np.max(np.abs(target-burgers_reference(z, 1024)))))
            assert check["order_512_1024_max_difference"] < 1e-10
        elif eq == "KdV":target, check = kdv_reference(grid)
        elif eq == "AllenCahn":target, check = allencahn_reference(grid)
        elif eq == "NS_periodic":target, check = ns_reference(grid)
        else:
            target = audit.extended.exact(torch.tensor(z), eq, audit.extended.physics(eq, reg)).numpy()
            check = dict(method="analytic FP64")
        assert np.isfinite(target).all() and np.linalg.norm(target)>0
        np.savez_compressed(destination, coordinates=z, reference=target,
                            test_index=np.arange(len(z), dtype=np.int64))
        write_new(meta, dict(case=eq, regime=reg, points=len(z), channels=target.shape[1],
            coordinates_sha256=array_sha(z), file_sha256=sha(destination),
            overlap_with_original_grid=overlap, check=check, protocol_sha256=sha(PROTOCOL)))
        print("Reference", rid, len(z), check, flush=True)


def prediction(model, z, eq):
    if eq in ("NS_periodic", "NS_wake"):
        u,v,_ = audit.study.base.ns.velocity_fields(model, z.requires_grad_())
        return torch.cat((u,v), 1)
    with torch.no_grad():return model(z)


def evaluate(device):
    p = checked_protocol()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4 if device == "cpu" else 1)
    PRED.mkdir(exist_ok=True)
    RECORDS.mkdir(exist_ok=True)
    all_runs = [(task, r) for task in p["tasks"] for r in task["runs"]]
    cache = {}
    started = time.perf_counter()
    for number, (task, r) in enumerate(all_runs, 1):
        dest = RECORDS / (r["id"]+".json")
        predpath = PRED / (r["id"]+".npz")
        if dest.exists():
            prev = read(dest)
            assert prev["protocol_sha256"] == sha(PROTOCOL)
            assert predpath.exists() and sha(predpath) == prev["prediction_file_sha256"]
            continue
        assert not predpath.exists(), "Unrecorded partial prediction must be investigated, not silently replaced"
        rid = task["reference"]
        if rid not in cache:
            refpath = REF / (rid+".npz")
            meta = read(REF / (rid+".json"))
            assert sha(refpath) == meta["file_sha256"]
            a = audit.loadz(refpath)
            cache[rid] = (a["coordinates"], a["reference"], meta)
        z, target, meta = cache[rid]
        model, _ = audit.make_model(r)
        weight = model_directory(r["run"]) / "final.pt"
        assert sha(weight) == r["weights_sha256"]
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
        model = model.double().to(device).eval()
        oldz, oldpred = audit.load_reload_probe(r)
        probe = prediction(model, torch.tensor(oldz, dtype=torch.float64, device=device), r["eq"])
        reload_error = float(np.max(np.abs(probe.detach().cpu().numpy()-oldpred)))
        assert reload_error < 1e-8, (r["id"], reload_error)
        values = []
        for start in range(0,len(z),512):
            batch = torch.tensor(z[start:start+512], dtype=torch.float64, device=device)
            value = prediction(model, batch, r["eq"])
            values.append(value.detach().cpu().numpy())
            del value, batch
        predicted = np.concatenate(values)
        assert predicted.shape == target.shape and np.isfinite(predicted).all()
        error = float(100*np.linalg.norm(predicted-target)/np.linalg.norm(target))
        np.savez_compressed(predpath, prediction=predicted, test_index=np.arange(len(z)),
                            reference_id=np.array(rid), coordinates_sha256=np.array(meta["coordinates_sha256"]))
        result = dict(id=r["id"], task=task["id"], eq=r["eq"], mode=r["mode"], reg=r["reg"],
            model=r["model"], seed=r["seed"], parameters=r["n"], old_validation_error=r["Eu"],
            new_test_error=error, max_absolute_error=float(np.max(np.abs(predicted-target))),
            points=len(z), reference_id=rid, reference_file_sha256=meta["file_sha256"],
            weights_sha256=r["weights_sha256"], prediction_file_sha256=sha(predpath),
            protocol_sha256=sha(PROTOCOL), reload_max_difference=reload_error,
            device=device, torch_version=torch.__version__, dtype="float64", run=r["run"],
            parameter_errors=r.get("param", {}))
        if r["mode"] == "inverse":
            coeff = model.log_coefficients.detach().cpu().double().exp().numpy()
            names = list(r["true"])
            errs = 100*np.abs(coeff/np.array(list(r["true"].values()))-1)
            assert np.allclose(errs, list(r["param"].values()), atol=1e-9, rtol=1e-10)
            result["parameter_errors"] = dict(zip(names, errs.tolist()))
        write_new(dest, result)
        del model, probe, values, predicted
        if number % 25 == 0 or number == len(all_runs):
            print("New test", number, "/", len(all_runs), "elapsed_s", round(time.perf_counter()-started,1), flush=True)


def summarize():
    p = checked_protocol()
    records = []
    for task in p["tasks"]:
        for r in task["runs"]:
            records.append(read(RECORDS / (r["id"]+".json")))
    assert len(records) == 1071
    summary = []
    configurations = []
    for task in p["tasks"]:
        lookup = {}
        for model in (*PAIRS, "MLP"):
            rows = sorted((r for r in records if r["task"]==task["id"] and r["model"]==model), key=lambda r:r["seed"])
            assert [r["seed"] for r in rows] == [42,43,44]
            eu = [r["new_test_error"] for r in rows]
            v = dict(task=task["id"], eq=task["eq"], mode=task["mode"], reg=task["reg"], model=model,
                old_mean=task["validation_means"][model], new_mean=statistics.mean(eu),
                new_std=statistics.stdev(eu), seed_errors=eu, points=rows[0]["points"],
                selected=(model==task["selected"]), best_same_by_validation=(model==task["same_family"]))
            configurations.append(v)
            lookup[model] = v
        best, same, mlp = [lookup[model] for model in (task["selected"], task["same_family"], "MLP")]
        summary.append(dict(task=task["id"], eq=task["eq"], mode=task["mode"], reg=task["reg"],
            selected=best["model"], same=same["model"], points=best["points"],
            old_selected=best["old_mean"], new_selected=best["new_mean"], new_selected_std=best["new_std"],
            old_same=same["old_mean"], new_same=same["new_mean"], new_same_std=same["new_std"],
            old_mlp=mlp["old_mean"], new_mlp=mlp["new_mean"], new_mlp_std=mlp["new_std"],
            old_gain_mlp=mlp["old_mean"]/best["old_mean"], new_gain_mlp=mlp["new_mean"]/best["new_mean"],
            old_reduction_same=100*(1-best["old_mean"]/same["old_mean"]),
            new_reduction_same=100*(1-best["new_mean"]/same["new_mean"])))
    write_new(OUT/"summary_data.json", dict(summary=summary, configurations=configurations, runs=records,
        protocol_sha256=sha(PROTOCOL), selected_using_new_test=False))
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "references", "evaluate", "summarize"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = parser.parse_args()
    if a.action == "evaluate" and a.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu.")
    if a.action == "freeze":
        require(CONFIG.runs)
        OUT.mkdir(parents=True, exist_ok=True)
    else:
        require(PROTOCOL, OUT / "protocol_digest.json")
    if a.action == "references":
        require(DATA)
    with threadpool_limits(limits=1):
        if a.action == "freeze":freeze()
        elif a.action == "references":references()
        elif a.action == "evaluate":evaluate(a.device)
        else:summarize()
