"""Read-only audit of archived experiments; writes only to this new folder."""
import argparse
import ast
import collections
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_paths import paths, require, source_is_local, model_directory
CONFIG = paths()
ROOT = CONFIG.archive
OUT = CONFIG.output / "method_and_reference_checks"
CODE = CONFIG.code
DATA = CONFIG.data
import adaptive_basis as study
import burgers_training as burgers
import inverse_training as inverse
import pde_benchmarks as extended
import inverse_benchmarks as ext_inverse
import cylinder_wake as wake
from architecture_controls import hidden_layer, hidden_values

for module in (study, burgers, inverse, extended, ext_inverse, wake):
    source_is_local(module)

ORIGINAL_FACTORY = study.AdaptiveModel
FAMILIES = ("Chebyshev", "Fourier", "RBF", "Wavelet")
PAIRS = {a if b == "Chebyshev" else a + "_H" + b: (a, b)
         for a in FAMILIES for b in FAMILIES}
torch.set_num_threads(1)
torch.serialization.add_safe_globals([
    np._core.multiarray._reconstruct, np._core.multiarray.scalar,
    np.ndarray, np.dtype, np.dtypes.UInt32DType, np.dtypes.Float64DType,
])


def readj(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")


def loadz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def reload_samples():
    path = CONFIG.archive.parent / "03_verification_inputs/model_reload_checks.json"
    return readj(path)["runs"]


def load_reload_probe(row):
    sample = reload_samples()[row["run"]]
    return (np.asarray(sample["coordinates"], dtype=np.float64),
            np.asarray(sample["prediction"], dtype=np.float64))


def independent_values(q, family, terms, phase_kind):
    q = np.asarray(q)
    if family == "Chebyshev":
        return np.polynomial.chebyshev.chebvander(q, terms - 1)
    if family == "Fourier":
        phase = np.pi * q.copy()
        if phase_kind == "physical":
            phase[:, -1] = np.pi * (q[:, -1] + 1) / 2
        elif phase_kind == "unit":
            phase = np.pi * (q + 1) / 2
        phase = phase[..., None] * np.arange(1, terms // 2 + 1)
        return np.concatenate((np.cos(phase), np.sin(phase)), axis=-1)
    r = (q[..., None] - np.linspace(-1, 1, terms)) / (2 / np.sqrt(terms))
    g = np.exp(-r * r / 2)
    return g if family == "RBF" else (1 - r * r) * g


def formulas():
    output = []
    rng = np.random.default_rng(20261107)
    for dim in (2, 3):
        q = rng.uniform(-1, 1, (61, dim))
        for family in FAMILIES:
            for phase, fn in (("physical", study.basis_values),
                              ("unit", extended.values),
                              ("hidden", hidden_values)):
                terms = 8 if phase == "hidden" and dim == 3 else 16
                actual = fn(torch.tensor(q), family, terms).detach().numpy()
                expected = independent_values(q, family, terms, phase)
                error = float(np.max(np.abs(actual - expected)))
                assert error < 1e-12, (dim, family, phase, error)
                output.append(dict(dim=dim, family=family, phase=phase,
                                   max_absolute_difference=error))
    return output


def make_model(row):
    eq, name = row["eq"], row["model"]
    inv = row["mode"] == "inverse"
    first, hidden = PAIRS.get(name, ("MLP", None))
    torch.manual_seed(row["seed"])
    if eq in extended.CASES:
        spec = dict(case=eq, regime=row["reg"], variant=first)
        model = ext_inverse.Model(spec) if inv else extended.Model(spec)
        net = model.network if inv else model
        if hidden:
            net.kan[2].family = hidden
    else:
        factory = burgers.BurgersModel if eq == "Burgers" else (
            wake.Network if eq == "NS_wake" else ORIGINAL_FACTORY)
        if inv:
            if eq == "NS_wake":
                inverse.INITIAL[eq] = [1.5, .02]
            study.AdaptiveModel = factory
            try:
                model = inverse.InverseModel(eq, first)
            finally:
                study.AdaptiveModel = ORIGINAL_FACTORY
            net = model.network
        else:
            model = net = factory(eq, first)
        if hidden and hidden != "Chebyshev":
            net.kan[2] = hidden_layer(net.kan[2], hidden)
    return model, net


def normalized_basis_error(net, row):
    layer = net.kan[0]
    first, hidden = PAIRS[row["model"]]
    kind = "unit" if row["eq"] in extended.CASES else "physical"
    dim = layer.coeff.shape[0]
    grid = np.repeat(np.linspace(-1, 1, 2049)[:, None], dim, axis=1)
    values = independent_values(grid, first, 16, kind)
    mu = values.mean(axis=0)
    sd = np.sqrt(np.mean((values - mu) ** 2, axis=0))
    constant = sd < 1e-10
    mu[constant] = 0
    sd[constant] = 1
    return max(float(np.max(np.abs(layer.means.detach().numpy() - mu.astype("float32")))),
               float(np.max(np.abs(layer.scales.detach().numpy() - sd.astype("float32")))))


def audit_models():
    wake.DATA = DATA / "wake_observations"
    rows = readj(CONFIG.runs)
    from source_integrity import archive_sources, matches_source, training_hashes
    current_hashes = training_hashes(CODE)
    archived = archive_sources()
    source_failures = []
    for row in rows:
        for name, digest in row["sources"].items():
            if not matches_source(name, digest, current_hashes, archived):
                source_failures.append([row["run"], name])
    core = [r for r in rows if r["reg"] in ("baseline", "base", "stress", "clean")
            and (r["model"] in PAIRS or r["model"] == "MLP")]
    groups = collections.defaultdict(list)
    for r in core:
        groups[r["eq"], r["mode"], r["reg"]].append(r)
    assert len(groups) == 22 and len(core) == 1122
    for group in groups.values():
        assert len(group) == 51
    tests, failures = [], []
    for index, row in enumerate(core):
        try:
            model, net = make_model(row)
            n = sum(p.numel() for p in model.parameters())
            assert n == row["n"], (n, row["n"])
            folder = model_directory(row["run"])
            initial = sorted(folder.rglob("*initial*.pt"))
            initial_equal = None
            initial_diff = None
            if initial:
                assert len(initial) == 1
                payload = torch.load(initial[0], map_location="cpu", weights_only=True)
                state = payload.get("model", payload)
                current = model.state_dict()
                initial_equal = current.keys() == state.keys() and all(
                    torch.equal(current[k], state[k]) for k in current)
                assert current.keys() == state.keys(), "Initial checkpoint keys differ"
                initial_diff = max(float((current[k]-state[k]).abs().max()) for k in current)
                assert all(torch.allclose(current[k], state[k], rtol=2e-6, atol=2e-7)
                           for k in current), ("Initial tensors differ", initial_diff)
            basis_diff = normalized_basis_error(net, row) if row["model"] in PAIRS else None
            assert basis_diff is None or basis_diff < 2e-7, basis_diff
            model.load_state_dict(torch.load(folder / "final.pt", map_location="cpu",
                                            weights_only=True), strict=True)
            model.double().eval()
            coordinates, expected = load_reload_probe(row)
            z = torch.tensor(coordinates, dtype=torch.float64)
            if row["eq"].startswith("NS_"):
                u, v, _ = study.base.ns.velocity_fields(model, z.requires_grad_())
                pred = torch.cat((u, v), 1)
            else:
                with torch.no_grad():
                    pred = model(z)
            delta = float(np.max(np.abs(pred.detach().numpy() - expected)))
            assert delta < 1e-9, delta
            tests.append(dict(run=row["run"], parameters=n, initial_checkpoint=bool(initial),
                              initial_tensors_equal=initial_equal, initial_max_difference=initial_diff,
                              basis_buffer_difference=basis_diff,
                              final_prediction_max_difference=delta))
        except Exception as exc:
            failures.append(dict(run=row["run"], error=repr(exc)))
        if (index + 1) % 150 == 0:
            print("Models", index + 1, "/", len(core), "failures", len(failures), flush=True)
    count_table = []
    for key, group in groups.items():
        mlp = next(r["n"] for r in group if r["model"] == "MLP")
        nums = sorted({r["n"] for r in group if r["model"] in PAIRS})
        count_table.append(dict(task=list(key), mlp=mlp, hybrid_counts=nums,
                                difference_percent_of_mlp=100 * (mlp - nums[0]) / mlp))
    result = dict(source_runs_checked=len(rows), source_hash_failures=source_failures,
                  core_runs=len(core), tasks=len(groups), formula_tests=formulas(),
                  models=tests, failures=failures, parameter_counts=count_table)
    save("model_checks.json", result)
    print("MODEL SUMMARY", json.dumps(dict(checked=len(core), passed=len(tests),
          source_hash_failures=len(source_failures), failures=failures[:3]), ensure_ascii=False), flush=True)


def audit_splits():
    output = []
    paths = sorted(DATA.rglob("*split.npz")) + sorted(DATA.rglob("*data_partitions.npz"))
    paths = sorted(set(paths))
    for path in paths:
        a = loadz(path)
        train, test = a["train"], a["test"]
        output.append(dict(path=str(path.relative_to(DATA)), train=len(train), test=len(test),
                           intersection=len(np.intersect1d(train, test)), keys=list(a)))
    exposure = []
    for eq, refglob, splitpath in (
        ("NS_periodic", "independent*diagnostic_data.npz", DATA / "KdVand periodicNSobservation_split/NS_periodic_split.npz"),
        ("NS_wake", "cylinder_wake_diagnostic_data.npz", DATA / "NSreference/cylinder_wake_fixed_split.npz")):
        ref = loadz(next((DATA / "NSreference").glob(refglob)))
        s = loadz(splitpath)
        positive = np.flatnonzero(ref["coordinates"][:, -1] > 0)
        leftover = np.setdiff1d(positive, np.union1d(s["train"], s["test"]))
        exposure.append(dict(case=eq, positive_time_points=len(positive),
                             outside_train_and_named_test=len(leftover),
                             all_positive_points_used_for_full_field_metric=True))
    result = dict(splits=output, full_grid_exposure=exposure,
                  independent_validation_split_present=False)
    save("partition_audit.json", result)
    print("SPLITS", json.dumps(result, ensure_ascii=False), flush=True)


def audit_analytic():
    output = []
    for case in ("Heat2D", "Helmholtz2D", "Wave1D"):
        for regime, cn in (("base", "base"), ("stress", "hard")):
            path = next(DATA.glob(case + "*/forward_" + cn + "_case_reference.npz"))
            ref = loadz(path)
            z = torch.tensor(ref["coordinates"], dtype=torch.float64)
            p = extended.physics(case, regime)
            pred = extended.exact(z, case, p).numpy()
            delta = float(np.max(np.abs(pred - ref["target"])))
            assert delta < 1e-12
            class Exact(torch.nn.Module):
                def forward(self, zz):
                    return extended.exact(zz, case, p)
            exact = Exact()
            exact.case, exact.p = case, p
            probe = torch.tensor(np.random.default_rng(20261107).uniform(
                .01, .99, (79, extended.dimension(case))), requires_grad=True)
            residual = extended.residual(exact, probe)
            rms = float(residual.detach().abs().max())
            assert rms < 1e-9
            output.append(dict(case=case, regime=regime, file_sha256=sha(path),
                               max_saved_reference_difference=delta, max_analytic_residual=rms))
    return output


def audit_burgers():
    from scipy.special import roots_hermite
    path = next(DATA.glob("Burgersreference/*.npz"))
    ref = loadz(path)
    x, t = np.meshgrid(ref["x"].ravel(), ref["t"].ravel()[1:], indexing="ij")
    xs, ts = x.ravel(), t.ravel()
    output = []
    for order in (256, 512):
        nodes, weights = roots_hermite(order)
        keep = weights > 0
        nodes, weights = nodes[keep], weights[keep]
        prediction = []
        for start in range(0, len(xs), 256):
            y = xs[start:start+256, None] - 2 * np.sqrt(
                burgers.NU * ts[start:start+256, None]) * nodes
            lw = np.log(weights)[None, :] - np.cos(np.pi*y)/(2*np.pi*burgers.NU)
            w = np.exp(lw - lw.max(axis=1, keepdims=True))
            prediction.append(-np.sum(w*np.sin(np.pi*y), axis=1)/w.sum(axis=1))
        output.append(np.concatenate(prediction).reshape(x.shape))
    return dict(file_sha256=sha(path), points=x.size,
                maximum_saved_difference=float(np.max(np.abs(output[-1]-ref["usol"][:, 1:]))),
                maximum_order_256_512_difference=float(np.max(np.abs(output[0]-output[1]))))


def audit_allencahn():
    from scipy.integrate import solve_ivp
    from scipy.sparse import diags
    reports = []
    for regime, cn in (("base", "base"), ("stress", "hard")):
        p = extended.physics("AllenCahn", regime)
        times = np.linspace(0, 1, 101)
        solutions = []
        for intervals in (1024, 2048, 4096):
            x = np.linspace(-1, 1, intervals+1)
            dx, nu, reaction = x[1]-x[0], p["epsilon"], p["reaction"]
            operator = diags([np.full(intervals-2, nu/dx**2),
                              np.full(intervals-1, -2*nu/dx**2),
                              np.full(intervals-2, nu/dx**2)], [-1, 0, 1], format="csc")
            boundary = np.zeros(intervals-1)
            boundary[[0, -1]] = -nu/dx**2
            def rhs(t, u):
                return operator@u + boundary - reaction*(u**3-u)
            def jac(t, u):
                return operator + diags(-reaction*(3*u*u-1), 0, format="csc")
            sol = solve_ivp(rhs, (0., 1.), x[1:-1]**2*np.cos(np.pi*x[1:-1]),
                            method="BDF", jac=jac, t_eval=times, rtol=1e-9, atol=1e-11)
            assert sol.success
            field = np.vstack([-np.ones((1, len(times))), sol.y, -np.ones((1, len(times)))])
            solutions.append(field[::intervals//256])
        path = next(DATA.glob("AllenCahn*/forward_" + cn + "_case_reference.npz"))
        ref = loadz(path)
        saved_delta = float(np.max(np.abs(solutions[-1][:, 1:].reshape(-1, 1)-ref["target"])))
        coarse = float(np.linalg.norm(solutions[0]-solutions[1])/np.linalg.norm(solutions[1]))
        fine = float(np.linalg.norm(solutions[1]-solutions[2])/np.linalg.norm(solutions[2]))
        reports.append(dict(regime=regime, maximum_saved_difference=saved_delta,
                            relative_difference_1024_2048=coarse, relative_difference_2048_4096=fine))
        print("AllenCahn", regime, reports[-1], flush=True)
    return reports


def audit_ns():
    ns = study.base.ns
    times = np.linspace(0, 1, 41)
    coarse = ns.spectral_solution(32, times, 2e-8, 2e-10)
    print("NS 32 grid complete", flush=True)
    fine = ns.spectral_solution(64, times, 2e-10, 2e-12)
    ref = loadz(next(DATA.glob("NSreference/independent*diagnostic_data.npz")))
    comparable = fine[0].transpose(1, 2, 0, 3).reshape(-1, 2)
    return dict(maximum_saved_difference=float(np.max(np.abs(comparable-ref["velocity"]))),
                relative_difference_32_64=float(np.linalg.norm(coarse[0]-fine[0][:, ::2, ::2]) /
                                               np.linalg.norm(fine[0][:, ::2, ::2])),
                coarse=coarse[3], fine=fine[3])


def audit_references():
    result = dict(analytic=audit_analytic(), burgers=audit_burgers())
    save("reference_checks.json", result)
    print("Analytic and Burgers references complete", flush=True)
    result["AllenCahn"] = audit_allencahn()
    save("reference_checks.json", result)
    result["NS_periodic"] = audit_ns()
    save("reference_checks.json", result)
    result["KdV"] = audit_kdv()
    save("reference_checks.json", result)
    print("REFERENCES COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)


def audit_kdv():
    folder = DATA / "KdVreference"
    source = CONFIG.reference / "kdv_reference.py"
    old = readj(folder / "KdVreference_crosscheck.json")
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve")
    scope = {"np": np}
    # Run the original solver without its historical-path writes or main block.
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
    solve = scope["solve"]
    _, _, a = solve(256, 5e-5)
    x, t, b = solve(256, 2.5e-5)
    _, _, c = solve(512, 2.5e-5)
    _, _, d = solve(256, 1e-5, "rk4")
    path = next(folder.glob("*.mat"))
    saved = loadmat(path)
    assert np.array_equal(saved["x"].ravel(), x) and np.array_equal(saved["t"].ravel(), t)
    maximum = float(np.max(np.abs(b - saved["usol"])))
    assert maximum < 1e-10, maximum
    rel = lambda left, right: float(100*np.linalg.norm(left-right)/np.linalg.norm(right))
    values = dict(dt_halving_difference_percent=rel(a, b),
                  space_doubling_difference_percent=rel(b, c[::2]),
                  independent_RK4_difference_percent=rel(d, b))
    assert all(v < 1e-7 for v in values.values()), values
    report = dict(file_sha256=sha(path), generator_sha256=sha(source),
                  original_generator=source.relative_to(CONFIG.package).as_posix(),
                  convergence_recomputed=True, maximum_saved_difference=maximum,
                  recomputed=values, original_record={k: old[k] for k in values},
                  comparison_absolute_difference={k: abs(values[k]-old[k]) for k in values})
    print("KdV", json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("models", "references", "kdv", "all"), default="all")
    args = parser.parse_args()
    require(DATA)
    if args.part in ("models", "all"):
        require(CONFIG.runs)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        if args.part in ("models", "all"):
            audit_splits()
            audit_models()
        if args.part in ("references", "all"):
            audit_references()
        if args.part == "kdv":
            report = readj(OUT / "reference_checks.json") if (OUT / "reference_checks.json").exists() else {}
            report["KdV"] = audit_kdv()
            save("reference_checks.json", report)
    print("Elapsed seconds", time.perf_counter()-start, flush=True)
