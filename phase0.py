"""Data description, time split with leakage and shortcut checks, and a drift check per scenario.

Stages (run in order):
  describe  data only: columns, timestamp coverage, attack schedule, class presence per window, duplicates;
  screen    builds the time split from the config, runs leakage checks, removes source duplicates and screens
            single features for shortcuts on the source train block;
  drift     trains the detector on the source period and reports macro-F1 per deployment window; optional
            cross-network mode with --deploy-config.

Usage:
  python phase0.py describe --csv <path to csv> --name unsw
  python phase0.py screen   --config configs/phase0_unsw.json
  python phase0.py drift    --config configs/phase0_unsw.json [--deploy-config configs/phase0_ton_s1c.json]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from driftids import schema as S
from driftids.data import (Flows, apply_class_map, check_split, dedupe_source, make_time_split, read_header,
                           read_netflow, row_hashes)
from driftids.leakage import assert_before, assert_no_excluded_features, assert_no_raw_column_copies
from driftids.metrics import binary_metrics, macro_f1, macro_f1_block_bootstrap
from driftids.shortcut import single_feature_auc

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "phase0"
CANDIDATE_WINDOWS = (25_000, 50_000, 100_000)
M_PROVISIONAL = 20


def _iso(ms) -> str:
    return pd.to_datetime(int(ms), unit="ms", utc=True).isoformat()


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _load(csv: str, nrows=None) -> Flows:
    t0 = time.time()
    f = read_netflow(csv, nrows=nrows)
    print(f"read {len(f):,} flows, {len(f.features)} features in {time.time() - t0:.0f}s", flush=True)
    return f


# ---------------------------------------------------------------- describe
def describe(args) -> None:
    out = RESULTS / args.name
    cols = read_header(args.csv)
    f = _load(args.csv, args.nrows)
    feats = f.features
    assert_no_excluded_features(feats)
    dur = f.end - f.start
    cls = pd.Series(f.attack)                         # categorical: 2 bytes per flow instead of an object array
    codes = f.attack.codes.astype(np.int64)
    K = len(f.attack.categories)
    vc_all = cls.value_counts()
    summary = {
        "csv": str(args.csv), "n_flows": len(f), "n_columns": len(cols), "columns": cols,
        "n_features": len(feats), "features": feats,
        "excluded_columns_present": [c for c in cols if S.is_excluded(c)],
        "missing_expected_v3_temporal": [c for c in S.EXPECTED_V3_TEMPORAL if c not in cols],
        "time_first_start": _iso(f.start[0]), "time_last_start": _iso(f.start[-1]),
        "span_hours": float((f.start[-1] - f.start[0]) / 3.6e6),
        "n_distinct_start_ms": int(np.unique(f.start).size),
        "duration_ms_quantiles": {q: float(np.quantile(dur, q)) for q in (0, .5, .9, .99, .999, 1)},
        "negative_durations": int((dur < 0).sum()),
        "class_counts": {str(k): int(v) for k, v in vc_all[vc_all > 0].items()},
        "label_vs_attack_mismatch": int(((f.label == 1) != (cls != S.BENIGN).to_numpy()).sum()),
        "nan_per_feature": {c: v for c in feats if (v := int(np.isnan(f.X[c].to_numpy()).sum()))},
    }
    if "FLOW_DURATION_MILLISECONDS" in feats:
        d2 = f.X["FLOW_DURATION_MILLISECONDS"].to_numpy(np.float64)
        summary["duration_vs_FLOW_DURATION_max_abs_diff"] = float(np.nanmax(np.abs(d2 - dur)))
    # duplicates over the whole file (descriptive only)
    h_xy, h_x = row_hashes(f.X, f.attack), row_hashes(f.X)
    summary["exact_duplicate_share"] = float(pd.Series(h_xy).duplicated().mean())
    del h_xy
    g = pd.DataFrame({"h": h_x, "y": codes.astype(np.int16)}).groupby("h")["y"].nunique()
    summary["conflicting_feature_vectors"] = int((g > 1).sum())
    summary["rows_with_conflicting_vectors"] = int(pd.Series(h_x).isin(g.index[g > 1]).sum())
    del g, h_x
    _save_json(summary, out / "summary.json")

    # attack schedule over time (hours since first flow); counted with bincount, not crosstab (memory)
    hour = ((f.start - f.start[0]) // 3_600_000).astype(np.int64)
    H = int(hour.max()) + 1
    tab = np.bincount(hour * K + codes, minlength=H * K).reshape(H, K)
    keep = tab.sum(1) > 0
    pd.DataFrame(tab[keep], index=pd.Index(np.arange(H)[keep], name="hour"),
                 columns=[str(c) for c in f.attack.categories]).to_csv(out / "flows_per_hour_by_class.csv")
    gaps = np.diff(f.start)
    top = np.argsort(gaps)[::-1][:20]
    pd.DataFrame({"after_position": top, "gap_minutes": gaps[top] / 6e4,
                  "at": [_iso(f.start[i]) for i in top]}).to_csv(out / "largest_time_gaps.csv", index=False)
    # class presence per candidate window size (data only, no model)
    for w in CANDIDATE_WINDOWS:
        rows = []
        for k, a in enumerate(range(0, len(f) - w + 1, w)):
            yy = cls.iloc[a:a + w]
            vc = yy.value_counts()
            rows.append({"window": k, "t_start": _iso(f.start[a]), "t_end": _iso(f.start[a + w - 1]),
                         "hours": (f.start[a + w - 1] - f.start[a]) / 3.6e6,
                         "attack_share": float((yy != S.BENIGN).mean()),
                         f"n_classes_ge_{M_PROVISIONAL}": int((vc >= M_PROVISIONAL).sum()), **vc.to_dict()})
        pd.DataFrame(rows).fillna(0).to_csv(out / f"windows_{w}.csv", index=False)
    _plot_schedule(f, cls, out)
    print(json.dumps({k: summary[k] for k in ("n_flows", "n_features", "time_first_start", "time_last_start",
                                              "span_hours", "missing_expected_v3_temporal",
                                              "exact_duplicate_share", "class_counts")}, indent=2, default=str))
    print(f"written to {out}")


def _plot_schedule(f: Flows, cls: pd.Series, out: Path, w: int = 50_000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_win = len(f) // w
    if n_win < 2:
        return
    K = len(f.attack.categories)
    codes = f.attack.codes[:n_win * w].astype(np.int64)
    counts = np.bincount((np.arange(n_win * w) // w) * K + codes, minlength=n_win * K).reshape(n_win, K)
    tab = pd.DataFrame(counts, columns=[str(c) for c in f.attack.categories])
    tab = tab.loc[:, tab.sum() > 0].drop(columns=[S.BENIGN], errors="ignore")
    tab = tab[sorted(tab.columns)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(np.log10(tab.T.to_numpy() + 1), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(tab.shape[1]), tab.columns, fontsize=7)
    ax.set_xlabel(f"window of {w:,} flows (time order)")
    ax.set_title("Attack flows per window (log10 count + 1)")
    fig.tight_layout()
    fig.savefig(out / "attack_schedule.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- config and split
def _config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("name", "csv", "window_size", "n_source", "gap_flows"):
        if k not in cfg:
            raise SystemExit(f"{path}: missing '{k}'")
    cfg.setdefault("inner_gap", 0)
    cfg.setdefault("fracs", [0.6, 0.1, 0.1, 0.2])
    cfg.setdefault("m", M_PROVISIONAL)
    cfg.setdefault("known_class_min_train", 100)
    cfg.setdefault("noise_rule", {"block": 1000, "B": 200})
    cfg.setdefault("delta", 0.05)
    cfg.setdefault("split_mode", "time")
    cfg.setdefault("chunk_size", None)
    cfg.setdefault("max_windows", None)
    cfg.setdefault("class_map", None)
    return cfg


def _split(cfg: dict, f: Flows):
    sp = make_time_split(f.start, f.end, n_source=cfg["n_source"], window_size=cfg["window_size"],
                         gap_flows=cfg["gap_flows"], inner_gap=cfg["inner_gap"], fracs=cfg["fracs"],
                         mode=cfg["split_mode"], chunk_size=cfg["chunk_size"], max_windows=cfg["max_windows"])
    dedupe_source(sp, row_hashes(f.X, f.attack))
    check_split(sp, f.start, f.end)
    return sp


_CACHE: dict = {}


def _read_cfg_flows(cfg_path: str):
    cfg = _config(cfg_path)
    cfg["_path"] = cfg_path
    csv = Path(cfg["csv"])
    if not csv.is_absolute():
        csv = (Path(cfg_path).resolve().parent / csv).resolve()
    key = (str(csv), json.dumps(cfg["class_map"], sort_keys=True))
    if key not in _CACHE:                      # one dataset read per process (several configs share a file)
        _CACHE.clear()
        _CACHE[key] = apply_class_map(_load(str(csv)), cfg["class_map"])
    return cfg, _CACHE[key]


def _shortcut_exclusions(cfg: dict, features: list[str]) -> tuple[list[str], dict]:
    """Main-variant exclusions: an explicit 'shortcut_rule', or the mechanical
    'shortcut_policy': exclude every listed fingerprint feature
    whose single-feature AUC on the source train block is >= the threshold; other flagged features stay (reported)."""
    if "shortcut_rule" in cfg:
        rule = cfg["shortcut_rule"] if isinstance(cfg["shortcut_rule"], dict) else {}
        return list(rule.get("exclude", [])), rule
    pol = cfg["shortcut_policy"]
    scr_path = RESULTS / cfg["name"] / "shortcut_screen.csv"
    if not scr_path.exists():
        raise SystemExit(f"{scr_path} missing: run 'screen' for {cfg['name']} first")
    scr = pd.read_csv(scr_path)
    flagged = list(scr.loc[scr["auc"] >= pol["auc_threshold"], "feature"])
    excl = [c for c in flagged if c in pol["fingerprint_features"] and c in features]
    rule = {"exclude": excl, "flagged_kept": [c for c in flagged if c not in excl],
            "all_features_variant": bool(pol.get("all_features_variant", True)), "derived_from_policy": True}
    return excl, rule


def screen(args) -> None:
    cfg, f = _read_cfg_flows(args.config)
    out = RESULTS / cfg["name"]
    sp = _split(cfg, f)
    assert_no_excluded_features(f.features)
    tr = sp.blocks["train"]
    raw = {S.TIME_START: f.start[tr], S.TIME_END: f.end[tr]}
    assert_no_raw_column_copies(f.X.iloc[tr].to_numpy(), raw)
    info = {"blocks": {}, "windows": len(sp.windows), "dropped": sp.dropped, "split": sp.info}
    cls = np.asarray(f.attack, dtype=object)
    for b, pos in sp.blocks.items():
        info["blocks"][b] = {"n": len(pos), "t_first": _iso(f.start[pos].min()), "t_last_end": _iso(f.end[pos].max()),
                             "classes": pd.Series(cls[pos]).value_counts().to_dict()}
    src = sp.source
    info["max_source_duration_ms"] = int((f.end[src] - f.start[src]).max())
    w0 = sp.windows[0]
    info["embargo_time_ms"] = int(f.start[w0[0]] - f.end[src].max())
    info["first_window_start"] = _iso(f.start[w0[0]])
    info["last_window_end"] = _iso(f.start[sp.windows[-1][-1]])
    # deployment rows that exactly copy a source-train row (reported, kept)
    htr = np.unique(row_hashes(f.X.iloc[tr]))
    hall = row_hashes(f.X)
    info["deploy_copy_of_train_share_per_window"] = [float(np.isin(hall[w], htr).mean()) for w in sp.windows]
    _save_json(info, out / "split_summary.json")
    scr = single_feature_auc(f.X.iloc[tr], f.label[tr])
    scr["flag_auc_ge_0.99"] = scr["auc"] >= 0.99
    scr.to_csv(out / "shortcut_screen.csv", index=False)
    print(json.dumps({k: info[k] for k in ("windows", "dropped", "max_source_duration_ms", "embargo_time_ms")},
                     indent=2))
    print(scr.head(12).to_string(index=False))
    if "shortcut_policy" in cfg:
        print(f"written to {out}. Shortcut rule is derived mechanically from the fixed policy in 'drift'.")
    else:
        print(f"written to {out}. Next: fix 'shortcut_rule' and sets 'fixed': true in {args.config}")


# ---------------------------------------------------------------- drift (go/no-go)
def _detector(kind: str = "auto", seed: int = 0):
    """LightGBM with default settings; 'hgb' forces the HistGradientBoosting model."""
    if kind in ("auto", "lightgbm"):
        try:
            import lightgbm as lgb
            n_jobs = int(os.environ.get("DRIFTIDS_THREADS", "-1"))   # set per job by tools/run_batch.py
            return lgb.LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1), "lightgbm"
        except ImportError:
            if kind == "lightgbm":
                raise SystemExit("lightgbm is not installed")
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=seed), "hgb_preview"


def _require_fixed(cfg: dict, path: str) -> None:
    if not cfg.get("fixed") or ("shortcut_rule" not in cfg and "shortcut_policy" not in cfg):
        raise SystemExit(f"{path}: run 'screen' first, then fix 'shortcut_rule' (or 'shortcut_policy') and set \"fixed\": true "
                         "")


def _boot(y, p, classes, nr: dict, seed: int) -> np.ndarray:
    return macro_f1_block_bootstrap(y, p, classes, block=nr["block"], B=nr["B"], seed=seed)


def drift(args) -> None:
    cfg, f = _read_cfg_flows(args.config)
    _require_fixed(cfg, args.config)
    sp = _split(cfg, f)
    excl, rule = _shortcut_exclusions(cfg, f.features)
    unknown = [c for c in excl + sum(rule.get("sensitivity", {}).values(), []) if c not in f.features]
    if unknown:
        raise SystemExit(f"shortcut_rule names unknown features: {unknown}")
    feats = [c for c in f.features if c not in excl]
    assert_no_excluded_features(feats)
    flagged = []
    scr_path = RESULTS / cfg["name"] / "shortcut_screen.csv"
    if scr_path.exists():
        scr = pd.read_csv(scr_path)
        flagged = list(scr.loc[scr["auc"] >= 0.99, "feature"])
    variants = {"main": feats}
    for vname, extra in rule.get("sensitivity", {}).items():       # main minus further features
        variants[vname] = [c for c in feats if c not in extra]
    if rule.get("all_features_variant"):
        variants["all_features"] = list(f.features)
    if not rule and any(c in feats for c in flagged):                 # no rule given: flagged-removed variant
        variants["no_flagged"] = [c for c in feats if c not in flagged]

    deploy_cfg, g, cross = cfg, f, bool(args.deploy_config)
    if cross:
        deploy_cfg, g = _read_cfg_flows(args.deploy_config)
        if g.features != f.features:
            raise SystemExit("feature columns differ between the two datasets")
        w = deploy_cfg["window_size"]
        windows = [np.arange(a, a + w) for a in range(0, len(g) - w + 1, w)]   # S2: whole target stream
    else:
        windows = sp.windows
    cls_f = np.asarray(f.attack, dtype=object).astype(str)
    cls_g = np.asarray(g.attack, dtype=object).astype(str)
    tr, te = sp.blocks["train"], sp.blocks["test"]
    if not cross:
        for wpos in windows:
            assert_before(tr, wpos, f.start, f.end, "train -> window")
    shared = None
    if cross:
        shared = sorted(set(map(str.lower, np.unique(cls_f[tr]))) & set(map(str.lower, np.unique(cls_g))))
    m, delta, nr = cfg["m"], cfg["delta"], cfg["noise_rule"]
    z = 1.6448536269514722                                            # one-sided 95%
    # known classes: at least known_class_min_train unique flows in the source train block (fixed before evaluation)
    tr_counts = pd.Series(cls_f[tr]).value_counts()
    known = sorted(tr_counts.index[tr_counts >= cfg["known_class_min_train"]])
    seen_lower = {c.lower() for c in tr_counts.index}           # classes with at least one training flow
    if shared is not None:
        known = [c for c in known if c.lower() in set(shared)]
    tag = f"{cfg['name']}_to_{deploy_cfg['name']}" if cross else cfg["name"]
    out = RESULTS / tag
    out.mkdir(parents=True, exist_ok=True)
    summary = {"config": cfg, "deploy_config": deploy_cfg if cross else None, "shared_classes": shared,
               "flagged_features": flagged, "excluded_by_rule": excl, "shortcut_rule_used": rule,
               "split": sp.info, "python": sys.version,
               "machine": platform.node(), "variants": {}}
    kind = None
    for vname, cols in variants.items():
        model, kind = _detector(args.detector, 0)
        t0 = time.time()
        model.fit(f.X.iloc[tr][cols].to_numpy(), cls_f[tr])   # fitted on the source train block only
        fit_s = time.time() - t0
        p_te = model.predict(f.X.iloc[te][cols].to_numpy())
        src_f1, src_cls = macro_f1(cls_f[te], p_te, m, restrict_to=shared)
        src_known, src_kcls = macro_f1(cls_f[te], p_te, m, restrict_to=known)
        se_src = float(np.std(_boot(cls_f[te], p_te, src_cls, nr, 1), ddof=1))
        se_src_k = float(np.std(_boot(cls_f[te], p_te, src_kcls, nr, 2), ddof=1))
        src_bin = binary_metrics(cls_f[te], p_te)
        rows = []
        for k, wpos in enumerate(windows):
            p = model.predict(g.X.iloc[wpos][cols].to_numpy())
            mf, used = macro_f1(cls_g[wpos], p, m, restrict_to=shared)
            mk, kused = macro_f1(cls_g[wpos], p, m, restrict_to=known)
            se_w = float(np.std(_boot(cls_g[wpos], p, used, nr, 1000 + k), ddof=1))
            se_wk = float(np.std(_boot(cls_g[wpos], p, kused, nr, 2000 + k), ddof=1))
            drop = src_f1 - mf if mf == mf else np.nan
            drop_k = src_known - mk if mk == mk else np.nan
            # new (unseen) class: present in the window (>= m flows) but with no flow at all in the source train block
            new_cls = [c for c in used if c.lower() not in seen_lower]
            rows.append({"window": k, "t_start": _iso(g.start[wpos[0]]), "new_classes": "|".join(new_cls),
                         "has_new_class": bool(new_cls), "macro_f1": mf, "drop": drop,
                         "drop_lb": drop - z * np.hypot(se_src, se_w), "se_window": se_w,
                         "classes": "|".join(used), "macro_f1_known": mk, "drop_known": drop_k,
                         "drop_known_lb": drop_k - z * np.hypot(se_src_k, se_wk),
                         **binary_metrics(cls_g[wpos], p)})
        df = pd.DataFrame(rows)
        df.to_csv(out / f"drift_windows_{vname}_{kind}.csv", index=False)
        n_drop = int((df["drop"] >= delta).sum())
        n_sig = int(((df["drop"] >= delta) & (df["drop_lb"] > 0)).sum())
        sig_k = (df["drop_known"] >= delta) & (df["drop_known_lb"] > 0)
        n_sig_k = int(sig_k.sum())
        known_only = ~df["has_new_class"]
        bin_drop = src_bin["bin_f1"] - df["bin_f1"]
        far_rise = df["false_alarm_rate"] - src_bin["false_alarm_rate"]
        summary["variants"][vname] = {
            "detector": kind, "n_features": len(cols), "fit_seconds": round(fit_s, 1),
            "source_test_macro_f1": src_f1, "source_test_se": se_src, "source_test_classes": src_cls,
            "source_test_binary": src_bin, "known_classes": known, "source_test_macro_f1_known": src_known,
            "source_test_se_known": se_src_k,
            "n_windows": len(df), "windows_with_drop_ge_delta": n_drop,
            "windows_with_drop_ge_delta_beyond_noise": n_sig,
            "windows_with_known_drop_ge_delta": int((df["drop_known"] >= delta).sum()),
            "windows_with_known_drop_ge_delta_beyond_noise": n_sig_k,
            "n_windows_without_new_class": int(known_only.sum()),
            "known_only_windows_with_known_drop_beyond_noise": int((sig_k & known_only).sum()),
            "windows_with_binary_f1_drop_ge_delta": int((bin_drop >= delta).sum()),
            "windows_with_false_alarm_rate_rise_ge_0.01": int((far_rise >= 0.01).sum()),
            "median_window_binary_f1": float(df["bin_f1"].median()),
            "median_window_false_alarm_rate": float(df["false_alarm_rate"].median()),
            "median_window_macro_f1": float(df["macro_f1"].median()),
            "go_criterion_met": n_drop >= 3, "go_criterion_beyond_noise_met": n_sig >= 3}
        print(f"[{vname}] {kind}: source macro-F1 {src_f1:.3f} (SE {se_src:.3f}); windows with drop >= {delta}: "
              f"{n_drop}/{len(df)}, beyond noise: {n_sig}/{len(df)}; known-class drop beyond noise: {n_sig_k} "
              f"(in windows without new classes: {int((sig_k & known_only).sum())}/{int(known_only.sum())}); "
              f"binary F1 {src_bin['bin_f1']:.3f} -> median {df['bin_f1'].median():.3f}, "
              f"windows with binary drop >= {delta}: {int((bin_drop >= delta).sum())}")
    if kind != "lightgbm":
        summary["note"] = "Preview with HistGradientBoosting."
    _save_json(summary, out / f"drift_summary_{kind}.json")
    print(f"written to {out}")


def _report_peak_ram() -> None:
    """Print this process's peak RAM at exit (Windows), so tools/run_batch.py records the real interpreter's peak
    (a venv's python.exe can be a small launcher that starts the real interpreter as a child process)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        gpmi = ctypes.windll.psapi.GetProcessMemoryInfo
        gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]   # without this the handle is cut
        gpmi.restype = wintypes.BOOL
        if gpmi(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            print(f"PEAK_RAM_GB {pmc.PeakWorkingSetSize / 1e9:.2f}", flush=True)
    except Exception:
        pass


def main(argv=None) -> None:
    import atexit
    atexit.register(_report_peak_ram)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe")
    d.add_argument("--csv", required=True)
    d.add_argument("--name", required=True)
    d.add_argument("--nrows", type=int, default=None, help="read only the first rows (smoke test)")
    s = sub.add_parser("screen")
    s.add_argument("--config", required=True)
    r = sub.add_parser("drift")
    r.add_argument("--config", required=True)
    r.add_argument("--deploy-config", default=None)
    r.add_argument("--detector", choices=["auto", "lightgbm", "hgb"], default="auto",
                   help="auto: LightGBM if installed, else the HistGradientBoosting preview")
    args = ap.parse_args(argv)
    {"describe": describe, "screen": screen, "drift": drift}[args.cmd](args)


if __name__ == "__main__":
    main()
