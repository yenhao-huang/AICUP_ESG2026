"""Loop001 train_grid normalization.

Resolves dev OPEN ISSUES (a)+(c): builds canonical ST1 training mix files for
every source A1..A4 against the ONE canonical real-train file
data/raw_data/vpesg_4k_train_1000.json (n=1000). Each mix file =
real(1000) + N synthetic rows, with N = {B1:500, B2:1000, B3:2000} computed on
the 1000 base for EVERY source (A4 included -> base 1000, not 800). Every row is
projected to id/data/promise_status only (uniform projection). Synthetic rows
are sampled from each source's offline pool; with replacement when N exceeds the
unique pool (ids suffixed _kK). seed=42, deterministic.

Runtime/eval ST1 path is unaffected; this only authors offline training files.
"""
import json, random, os

SEED = 42
REAL = "data/raw_data/vpesg_4k_train_1000.json"
OUT = "data/generated/loop001_canonical_mix"
B = {"b1": 500, "b2": 1000, "b3": 2000}

POOLS = {
    "a1": "data/generated/loop001_synth_st1/pool_a1_data_only.json",
    "a3": "data/generated/loop001_synth_st1/pool_a3_data_plus_promise.json",
    "a2": "data/generated/synth_st1_a2/synth_st1_a2_promise_pool.json",
    "a4": "data/generated/synth_st1_a4/synth_st1_a4_pdf_pool.json",
}

def proj(row, i, src):
    rid = str(row.get("id", f"{src}_{i}"))
    return {"id": rid, "data": row["data"], "promise_status": row["promise_status"]}

def sample(pool, n, seed):
    rng = random.Random(seed)
    if n <= len(pool):
        idx = rng.sample(range(len(pool)), n)
        return [(pool[j], 0) for j in idx]
    # with replacement: each unique used floor(n/len) times then remainder
    out = []
    base = list(range(len(pool)))
    k = 0
    while len(out) < n:
        rng.shuffle(base)
        for j in base:
            out.append((pool[j], k))
            if len(out) >= n:
                break
        k += 1
    return out

def main():
    os.makedirs(OUT, exist_ok=True)
    real = json.load(open(REAL))
    real_proj = [proj(r, i, "real") for i, r in enumerate(real)]
    manifest = {"real_file": REAL, "n_real": len(real_proj), "seed": SEED, "B": B, "arms": {}}
    for src, pool_path in POOLS.items():
        pool = json.load(open(pool_path))
        for bname, n in B.items():
            chosen = sample(pool, n, SEED)
            syn = []
            for (row, rep) in chosen:
                p = proj(row, len(syn), src)
                if rep > 0:
                    p["id"] = f"{p['id']}_k{rep}"
                syn.append(p)
            mix = real_proj + syn
            out_path = f"{OUT}/mix_{src}_{bname}.json"
            json.dump(mix, open(out_path, "w"), ensure_ascii=False, indent=1)
            from collections import Counter
            c = Counter(x["promise_status"] for x in mix)
            cs = Counter(x["promise_status"] for x in syn)
            manifest["arms"][f"{src}_{bname}"] = {
                "path": out_path, "n_total": len(mix), "n_real": len(real_proj),
                "n_synth": len(syn), "synth_unique": len(set(x["data"] for x in syn)),
                "label_total": dict(c), "label_synth": dict(cs),
            }
            print(f"{src}_{bname}: total={len(mix)} real={len(real_proj)} synth={len(syn)} "
                  f"synth_uniq={len(set(x['data'] for x in syn))} labels={dict(c)}")
    # A0 control = real only, projected
    a0 = f"{OUT}/mix_a0_real_only.json"
    json.dump(real_proj, open(a0, "w"), ensure_ascii=False, indent=1)
    from collections import Counter
    manifest["arms"]["a0_real_only"] = {"path": a0, "n_total": len(real_proj),
        "n_real": len(real_proj), "n_synth": 0,
        "label_total": dict(Counter(x["promise_status"] for x in real_proj))}
    print(f"a0_real_only: total={len(real_proj)} labels={dict(Counter(x['promise_status'] for x in real_proj))}")
    json.dump(manifest, open(f"{OUT}/canonical_manifest.json", "w"), ensure_ascii=False, indent=2)
    print("manifest ->", f"{OUT}/canonical_manifest.json")

if __name__ == "__main__":
    main()
