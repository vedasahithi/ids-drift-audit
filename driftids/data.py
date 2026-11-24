"""NetFlow v3 loading, time ordering, time-ordered splits with embargo, and fixed-size windows.

Leakage rules implemented here : A1 stable time order, A2 source blocks (strict time order,
or interleaved chunks, v3), A3 embargo gaps and removal of flows that cross a boundary, E13 duplicate removal within
the source period, and class grouping (apply_class_map).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S
from .leakage import assert_before, assert_disjoint

SOURCE_BLOCKS = ("train", "val", "calib", "test")


@dataclass
class Flows:
    """Time-ordered flows. Row i is the i-th flow by start time; `row_id` is the row in the original file."""
    X: pd.DataFrame          # features only (float32)
    start: np.ndarray        # int64 ms
    end: np.ndarray          # int64 ms
    attack: pd.Categorical   # class name per flow
    label: np.ndarray        # int8, 0 benign / 1 attack
    row_id: np.ndarray       # int64

    def __len__(self) -> int:
        return len(self.start)

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)


def read_header(path: str | Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _engine(engine: str) -> str:
    """'arrow' (pyarrow streaming CSV reader, multi-threaded) if available, else 'pandas' (C parser)."""
    import os
    engine = os.environ.get("DRIFTIDS_CSV_ENGINE", engine)
    if engine == "auto":
        try:
            import pyarrow.csv  # noqa: F401
            return "arrow"
        except ImportError:
            return "pandas"
    return engine


def _limit(chunks, nrows):
    """Stop a chunk stream after nrows rows (each chunk is a dict of equally long arrays)."""
    done = 0
    for ch in chunks:
        k = len(ch["start"])
        if nrows is not None and done + k > nrows:
            ch = {key: v[:nrows - done] for key, v in ch.items()}
            k = nrows - done
        if k <= 0:
            return
        yield ch
        done += k
        if nrows is not None and done >= nrows:
            return


def _chunks_pandas(path, feats, chunksize, only_start=False):
    if only_start:
        for c in pd.read_csv(path, usecols=[S.TIME_START], dtype={S.TIME_START: "int64"}, chunksize=4 * chunksize):
            yield {"start": c[S.TIME_START].to_numpy()}
        return
    dtype = {c: "float32" for c in feats}
    dtype.update({S.TIME_START: "int64", S.TIME_END: "int64", S.LABEL_BIN: "int8", S.LABEL_CLASS: "category"})
    for c in pd.read_csv(path, usecols=feats + list(S.REQUIRED_COLS), dtype=dtype, chunksize=chunksize):
        cat = c[S.LABEL_CLASS].cat
        yield {"start": c[S.TIME_START].to_numpy(), "end": c[S.TIME_END].to_numpy(),
               "label": c[S.LABEL_BIN].to_numpy(), "X": c[feats].to_numpy(dtype=np.float32),
               "cls_codes": cat.codes.to_numpy(), "cls_names": np.asarray([str(x) for x in cat.categories])}


def _chunks_arrow(path, feats, chunksize, only_start=False):
    """pyarrow streaming reader: parses blocks with several threads and yields them one by one (low memory).
    Features are parsed as float64 and then cast to float32, the same two steps as the pandas C parser."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    ropt = pacsv.ReadOptions(block_size=64 << 20, use_threads=True)
    if only_start:
        copt = pacsv.ConvertOptions(include_columns=[S.TIME_START], column_types={S.TIME_START: pa.int64()})
        for b in pacsv.open_csv(path, read_options=ropt, convert_options=copt):
            yield {"start": b.column(0).to_numpy()}
        return
    types = {c: pa.float64() for c in feats}
    types.update({S.TIME_START: pa.int64(), S.TIME_END: pa.int64(), S.LABEL_BIN: pa.int8(),
                  S.LABEL_CLASS: pa.string()})
    copt = pacsv.ConvertOptions(include_columns=feats + list(S.REQUIRED_COLS), column_types=types)
    for b in pacsv.open_csv(path, read_options=ropt, convert_options=copt):
        names = b.schema.names
        X = np.empty((b.num_rows, len(feats)), dtype=np.float32)
        for j, c in enumerate(feats):
            X[:, j] = b.column(names.index(c)).to_numpy(zero_copy_only=False)
        d = b.column(names.index(S.LABEL_CLASS)).dictionary_encode()
        if d.null_count:
            raise ValueError(f"{path}: missing class labels")
        yield {"start": b.column(names.index(S.TIME_START)).to_numpy(),
               "end": b.column(names.index(S.TIME_END)).to_numpy(),
               "label": b.column(names.index(S.LABEL_BIN)).to_numpy(),
               "X": X, "cls_codes": d.indices.to_numpy(zero_copy_only=False),
               "cls_names": np.asarray(d.dictionary.to_pylist(), dtype=object).astype(str)}


def read_netflow(path: str | Path, chunksize: int = 1_000_000, nrows: int | None = None,
                 engine: str = "auto") -> Flows:
    """Read a NetFlow v3 CSV in chunks, skipping identifier columns, and time-order it.

    Two passes keep peak memory at about one copy of the feature matrix (needed for the 20-28 M-row datasets):
    pass 1 reads only the start times and fixes the time order (A1: stable, ties by original row order); pass 2
    reads the full rows chunk by chunk and writes each row straight to its sorted position in a pre-allocated
    float32 array. engine: 'arrow' (pyarrow, multi-threaded; used when installed), 'pandas', or 'auto'.
    """
    cols = read_header(path)
    missing = [c for c in S.REQUIRED_COLS if c not in cols]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    feats = S.feature_columns(cols)
    eng = _engine(engine)
    chunks = _chunks_arrow if eng == "arrow" else _chunks_pandas
    # pass 1: start times -> order and target position (rank) of every file row
    start_raw = np.concatenate([c["start"] for c in _limit(chunks(path, feats, chunksize, only_start=True), nrows)])
    n = len(start_raw)
    order = np.argsort(start_raw, kind="stable")       # stable = ties keep the original row order
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    start = start_raw[order]
    del start_raw
    # pass 2: scatter rows to their sorted positions
    X = np.empty((n, len(feats)), dtype=np.float32, order="F")
    end = np.empty(n, dtype=np.int64)
    label = np.empty(n, dtype=np.int8)
    codes = np.empty(n, dtype=np.int16)
    cat_index: dict[str, int] = {}
    off = 0
    for ch in _limit(chunks(path, feats, chunksize), nrows):
        k = len(ch["start"])
        dest = rank[off:off + k]
        if not np.array_equal(start[dest], ch["start"]):
            raise ValueError(f"{path}: file changed between the two reading passes")
        block = ch["X"]
        block[np.isinf(block)] = np.nan
        X[dest] = block
        end[dest] = ch["end"]
        label[dest] = ch["label"]
        if (ch["cls_codes"] < 0).any():
            raise ValueError(f"{path}: missing class labels")
        for name in ch["cls_names"]:
            cat_index.setdefault(str(name), len(cat_index))
        lut = np.array([cat_index[str(c)] for c in ch["cls_names"]], dtype=np.int16)
        codes[dest] = lut[ch["cls_codes"]]
        off += k
    if off != n:
        raise ValueError(f"{path}: pass 2 read {off} rows, pass 1 read {n}")
    attack = pd.Categorical.from_codes(codes, categories=list(cat_index))
    Xdf = pd.DataFrame(X, columns=feats, copy=False)
    return Flows(X=Xdf, start=start, end=end, attack=attack, label=label, row_id=order.astype(np.int64))


def row_hashes(X: pd.DataFrame, attack=None) -> np.ndarray:
    """Hash of the feature vector (plus the class, if given) per row, for exact-duplicate checks.

    The class hash is combined arithmetically, so the feature matrix is never copied (matters for 27 M rows).
    """
    h = pd.util.hash_pandas_object(X, index=False).to_numpy()
    if attack is None:
        return h
    hy = pd.util.hash_array(np.asarray(attack, dtype=object))
    with np.errstate(over="ignore"):
        return h ^ (hy * np.uint64(0x9E3779B97F4A7C15) + np.uint64(0x632BE59BD9B4E019))


@dataclass
class TimeSplit:
    blocks: dict[str, np.ndarray]                      # source blocks: positions into Flows
    windows: list[np.ndarray]                          # deployment windows: positions into Flows
    dropped: dict[str, int] = field(default_factory=dict)  # counts of removed rows, by reason
    info: dict = field(default_factory=dict)
    segments: list = field(default_factory=list)      # time-ordered source segments: (block name, positions)

    @property
    def source(self) -> np.ndarray:
        return np.concatenate([self.blocks[b] for b in SOURCE_BLOCKS])


# Interleaved mode: chunks of the source period are dealt to the blocks in this repeating order,
# which gives train 6 : val 1 : calib 1 : test 2 and spreads every block over the whole source period.
INTERLEAVE_PATTERN = ("train", "test", "train", "val", "train", "train", "calib", "train", "test", "train")


def _trim_crossing(a: np.ndarray, b_first: int, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, int]:
    """A3: drop flows of segment a that end at or after the first flow of the next segment starts."""
    t = start[b_first]
    keep = (end[a] < t) & (start[a] < t)
    return a[keep], int((~keep).sum())


def make_time_split(start: np.ndarray, end: np.ndarray, *, n_source: int, window_size: int, gap_flows: int,
                    inner_gap: int = 0, fracs=(0.6, 0.1, 0.1, 0.2), max_windows: int | None = None,
                    mode: str = "time", chunk_size: int | None = None) -> TimeSplit:
    """Split of a stream already sorted by start time into source blocks and later deployment windows.

    mode "time":        train | gap | val | gap | calib | gap | test | embargo | window 1 | window 2 | ...
    mode "interleaved": the source period is cut into chunks of `chunk_size` flows separated by `inner_gap` flows;
                        chunks are dealt to train/val/calib/test in INTERLEAVE_PATTERN order (6:1:1:2), so attack
                        types that occur anywhere in the source period reach every block; then embargo and windows.
    In both modes flows that cross into the next segment are removed, every segment ends before the next one starts,
    and the whole source period ends before the first deployment window. The last partial window is dropped.
    """
    n = len(start)
    if np.any(np.diff(start) < 0):
        raise ValueError("start times must be sorted")
    if n_source + gap_flows + window_size > n:
        raise ValueError("stream too short for this split")
    segs: list[tuple[str, np.ndarray]] = []
    dropped = {"inner_gap": 0, "embargo": 0, "crossing": 0}
    if mode == "time":
        usable = n_source - inner_gap * (len(SOURCE_BLOCKS) - 1)
        sizes = [int(usable * f) for f in fracs]
        sizes[0] += usable - sum(sizes)
        pos = 0
        for i, (name, size) in enumerate(zip(SOURCE_BLOCKS, sizes)):
            segs.append((name, np.arange(pos, pos + size)))
            pos += size
            if i < len(SOURCE_BLOCKS) - 1:
                pos += inner_gap
                dropped["inner_gap"] += inner_gap
    elif mode == "interleaved":
        if tuple(fracs) != (0.6, 0.1, 0.1, 0.2):
            raise ValueError("interleaved mode uses the fixed 6:1:1:2 pattern")
        if not chunk_size:
            raise ValueError("interleaved mode needs chunk_size")
        pos, k = 0, 0
        while pos < n_source:
            stop = min(pos + chunk_size, n_source)
            segs.append((INTERLEAVE_PATTERN[k % len(INTERLEAVE_PATTERN)], np.arange(pos, stop)))
            k += 1
            pos = stop
            if pos < n_source:
                g = min(inner_gap, n_source - pos)
                pos += g
                dropped["inner_gap"] += g
    else:
        raise ValueError(f"unknown split mode {mode}")
    pos = n_source + gap_flows
    dropped["embargo"] = gap_flows
    windows = []
    while pos + window_size <= n and (max_windows is None or len(windows) < max_windows):
        windows.append(np.arange(pos, pos + window_size))
        pos += window_size
    # A3: remove flows that cross into the next segment (the last source segment against the first window)
    nxt = [s[1][0] for s in segs[1:]] + [windows[0][0]]
    trimmed = []
    for (name, a), b_first in zip(segs, nxt):
        a, k = _trim_crossing(a, b_first, start, end)
        dropped["crossing"] += k
        trimmed.append((name, a))
    blocks = {b: np.concatenate([a for name, a in trimmed if name == b] or [np.array([], dtype=np.int64)])
              for b in SOURCE_BLOCKS}
    split = TimeSplit(blocks=blocks, windows=windows, dropped=dropped, segments=trimmed,
                      info={"mode": mode, "n_segments": len(trimmed)})
    check_split(split, start, end)
    return split


def dedupe_source(split: TimeSplit, hashes: np.ndarray) -> int:
    """E13: drop exact duplicates (features + class) within the source period, keeping the earliest copy in time."""
    allpos = np.sort(split.source)
    dup = pd.Series(hashes[allpos]).duplicated(keep="first").to_numpy()
    drop = allpos[dup]
    for name in SOURCE_BLOCKS:
        pos = split.blocks[name]
        split.blocks[name] = pos[~np.isin(pos, drop)]
    split.segments = [(name, a[~np.isin(a, drop)]) for name, a in split.segments]
    split.dropped["source_duplicates"] = int(len(drop))
    return int(len(drop))


def check_split(split: TimeSplit, start: np.ndarray, end: np.ndarray) -> None:
    """A4: blocks are disjoint; each source segment ends before the next starts; the source ends before the
    first deployment window starts."""
    order = list(SOURCE_BLOCKS)
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            assert_disjoint(split.blocks[a], split.blocks[b], f"{a}/{b}")
    segs = [(n, a) for n, a in split.segments if len(a)]
    for (na, a), (nb, b) in zip(segs, segs[1:]):
        assert_before(a, b, start, end, f"segment {na} -> {nb}")
    src = split.source
    for k, w in enumerate(split.windows):
        assert_disjoint(src, w, f"source/window {k}")
    if split.windows:
        assert_before(src, split.windows[0], start, end, "source -> first window")


def apply_class_map(flows: Flows, mapping: dict[str, str] | None) -> Flows:
    """Group fine class labels into families (e.g. CSE-CIC-IDS2018 tools -> UQ families). Every non-benign label
    must be mapped explicitly, so a label missing from the map stops the run instead of passing silently."""
    if not mapping:
        return flows
    labels = np.asarray(flows.attack, dtype=object).astype(str)
    present = set(np.unique(labels))
    missing = sorted(present - set(mapping) - {S.BENIGN})
    if missing:
        raise ValueError(f"class_map does not cover labels: {missing}")
    mapped = np.array([mapping.get(v, v) for v in sorted(present)], dtype=object)
    lut = dict(zip(sorted(present), mapped))
    flows.attack = pd.Categorical(pd.Series(labels).map(lut).to_numpy())
    return flows
