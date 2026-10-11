"""Fetch and curate isolated short peptides from the PDB.

"Isolated" means the whole deposited entry is 8-50 residues, so the peptide's
conformation is not being dictated by a large partner chain that the graph has
no representation of. 84% of these are solution NMR, which is a feature: each
entry carries 10-20 models, an experimental conformational ensemble, across
~2,250 distinct sequence clusters instead of this project's 21.

The user's designed peptides are deliberately **not** in here. They are the
held-out set, together with a held-out slice of PDB clusters, so that model
quality and the predicted-structure domain gap can be told apart.

Downloads land in their own directory and are parsed as untrusted input.
Non-canonical residues are kept when their backbone is complete and map to the
one-hot's "other" slot, because a single modified residue would otherwise
discard an entire peptide.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pepgraph.residues import THREE_TO_ONE, is_canonical  # noqa: E402
from pepgraph.structure import read_cif  # noqa: E402

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
FILES = "https://files.rcsb.org/download/{}.cif.gz"

QUERY = {
    "query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_sample_sequence_length",
            "operator": "range", "value": {"from": 8, "to": 50}}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_entity_polymer_type",
            "operator": "exact_match", "value": "Protein"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
            "operator": "less_or_equal", "value": 50}},
    ]},
    "return_type": "entry",
}


def post(payload, retries=3):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        SEARCH, data=data, headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as fh:
                return json.load(fh)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise
            print(f"    retry after {exc}")
            time.sleep(2 * (attempt + 1))


def entry_ids(page=100):
    total = post({**QUERY, "request_options": {"return_counts": True}})["total_count"]
    print(f"{total} isolated peptide entries")
    ids = []
    for start in range(0, total, page):
        res = post({**QUERY, "request_options": {
            "paginate": {"start": start, "rows": page}}})
        ids.extend(r["identifier"] for r in res.get("result_set", []))
        if start % 1000 == 0:
            print(f"    {len(ids)}/{total}")
    return sorted(set(ids))


def _intact(path: Path) -> bool:
    """A gzip file that can be read to the end. Guards against a download cut
    short -- by a sleeping machine, say -- leaving a truncated file that a mere
    existence check would happily accept on resume."""
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except Exception:                                   # noqa: BLE001
        return False


def download(ids, dest: Path, pause=0.05, verify=True):
    dest.mkdir(parents=True, exist_ok=True)
    got, failed, repaired = 0, [], 0
    for i, pdb_id in enumerate(ids, 1):
        out = dest / f"{pdb_id}.cif.gz"
        if out.exists() and out.stat().st_size > 0:
            if not verify or _intact(out):
                got += 1
                continue
            # truncated from an interrupted run: discard and fetch again
            out.unlink(missing_ok=True)
            repaired += 1
        # write to a .part file and rename, so a partial download can never
        # look like a finished one
        tmp = out.with_suffix(".gz.part")
        try:
            with urllib.request.urlopen(FILES.format(pdb_id), timeout=60) as r, \
                    open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh)
            tmp.replace(out)
            got += 1
        except Exception as exc:                        # noqa: BLE001
            failed.append((pdb_id, repr(exc)))
            tmp.unlink(missing_ok=True)
            out.unlink(missing_ok=True)
        if i % 250 == 0:
            print(f"    {i}/{len(ids)} downloaded, {len(failed)} failed"
                  + (f", {repaired} truncated re-fetched" if repaired else ""))
        time.sleep(pause)
    return got, failed, repaired


def n_models(path):
    """How many models the mmCIF holds (NMR ensembles carry 10-20)."""
    seen = set()
    with gzip.open(path, "rt", errors="replace") as fh:
        cols, inloop = [], False
        for line in fh:
            if line.startswith("_atom_site."):
                cols.append(line.split(".", 1)[1].strip())
                inloop = True
                continue
            if inloop:
                if line.startswith(("ATOM", "HETATM")):
                    parts = line.split()
                    if "pdbx_PDB_model_num" in cols:
                        idx = cols.index("pdbx_PDB_model_num")
                        if idx < len(parts):
                            seen.add(parts[idx])
                elif seen or line.startswith("#"):
                    break
    return sorted(seen, key=lambda s: int(s) if s.isdigit() else 0) or ["1"]


def curate(paths, min_length, max_length, max_models, drops: Counter):
    """Yield one record per (entry, model) that survives the filters."""
    for path in paths:
        pdb_id = path.name.split(".")[0]
        try:
            models = n_models(path)
        except Exception:                               # noqa: BLE001
            drops["unreadable file"] += 1
            continue
        if not models:
            drops["no models"] += 1
            continue
        kept_any = False
        for model in models[:max_models]:
            try:
                res_list = read_cif(str(path), min_length=min_length,
                                    model=int(model), allow_noncanonical=True)
            except Exception:                           # noqa: BLE001
                drops["parse error"] += 1
                continue
            if not res_list:
                drops["no complete-backbone chain"] += 1
                continue
            if not (min_length <= len(res_list) <= max_length):
                drops["length out of range"] += 1
                continue
            seq = "".join(THREE_TO_ONE.get(r.name, "X") for r in res_list)
            frac_x = seq.count("X") / len(seq)
            if frac_x > 0.34:
                drops["too many non-canonical"] += 1
                continue
            coords = np.array([[r.coord("N"), r.coord("CA"), r.coord("C")]
                               for r in res_list], dtype=np.float32)
            if not np.isfinite(coords).all():
                drops["non-finite coordinates"] += 1
                continue
            yield pdb_id, int(model), seq, coords
            kept_any = True
        if not kept_any:
            drops["entry contributed nothing"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True,
                    help="directory for downloaded mmCIF (its own, empty)")
    ap.add_argument("--out", type=Path, default=Path("outputs/pdb_peptides"))
    ap.add_argument("--min-length", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=50)
    ap.add_argument("--max-models", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N entries, for a trial run")
    ap.add_argument("--ids-cache", type=Path,
                    default=Path("outputs/pdb_peptides/entry_ids.json"))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.ids_cache.parent.mkdir(parents=True, exist_ok=True)
    if args.ids_cache.exists():
        ids = json.loads(args.ids_cache.read_text())
        print(f"{len(ids)} entry IDs from cache {args.ids_cache}")
    else:
        ids = entry_ids()
        args.ids_cache.write_text(json.dumps(ids))
        print(f"cached {len(ids)} entry IDs to {args.ids_cache}")
    if args.limit:
        ids = ids[:args.limit]
        print(f"limited to {len(ids)} entries")

    if not args.skip_download:
        got, failed, repaired = download(ids, args.raw)
        print(f"downloaded {got}/{len(ids)}, {len(failed)} failed"
              + (f", {repaired} truncated files re-fetched" if repaired else ""))
        for pdb_id, why in failed[:5]:
            print(f"    {pdb_id}: {why}")

    paths = sorted(args.raw.glob("*.cif.gz"))
    print(f"\ncurating {len(paths)} files...")
    drops = Counter()
    records = list(curate(paths, args.min_length, args.max_length,
                          args.max_models, drops))

    entries = {r[0] for r in records}
    seqs = {r[2] for r in records}
    print(f"\nkept {len(records)} (entry, model) conformers "
          f"from {len(entries)} entries, {len(seqs)} distinct sequences")
    print("dropped:")
    for k, v in drops.most_common():
        print(f"    {k:<32} {v}")

    if not records:
        print("\nnothing survived curation -- stopping before the write step")
        return 1

    lengths = np.array([len(r[2]) for r in records])
    frac_x = np.array([r[2].count("X") / len(r[2]) for r in records])
    print(f"\nlength: {lengths.min()}-{lengths.max()} "
          f"(median {int(np.median(lengths))})")
    print(f"non-canonical residues: {100 * (frac_x > 0).mean():.1f}% of "
          f"conformers contain at least one, "
          f"{100 * frac_x.mean():.1f}% of residues overall")

    # group by sequence so splits can be made without leakage
    by_seq = {}
    for pdb_id, model, seq, coords in records:
        by_seq.setdefault(seq, []).append((pdb_id, model, coords))
    np.savez_compressed(
        args.out / "peptides.npz",
        **{f"c{i}": np.stack([c for _, _, c in v])
           for i, (s, v) in enumerate(sorted(by_seq.items()))},
        meta=json.dumps([
            {"index": i, "sequence": s, "n_conformers": len(v),
             "entries": sorted({p for p, _, _ in v})}
            for i, (s, v) in enumerate(sorted(by_seq.items()))]))
    size = (args.out / "peptides.npz").stat().st_size / 1e6
    print(f"\nwrote {args.out / 'peptides.npz'} ({size:.1f} MB), "
          f"{len(by_seq)} sequence groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
