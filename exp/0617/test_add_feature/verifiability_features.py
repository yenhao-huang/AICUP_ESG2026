"""Data-only verifiability features for Stage 3 (evidence_quality / clarity).

Method A motivation (docs/plans/0617_improvement.md, §P3):
    Clear vs Not Clear is fundamentally a *verifiability* judgment, not a
    surface-language one. On data/raw_data/vpesg_4k_train_1000.json the
    discriminative signal is quantifiability, NOT hedging:

        signal              Clear    NotClear   diff
        digit               0.793    0.363      +0.431
        num+unit            0.697    0.315      +0.383
        year(20xx)          0.601    0.282      +0.319
        percent             0.223    0.065      +0.158
        target_verb         0.556    0.468      +0.088
        hedge(持續/致力..)  0.732    0.742      -0.010   <- NO signal

These features are derived ONLY from the raw ``data`` text (regex over the model
input). They use NO annotation-derived field (promise_string / evidence_string /
labels). They are therefore data-only compliant, and are consumed as **auxiliary
training supervision** (multitask heads) — not as a deterministic post-process
rule (which the harness forbids and which the stats above show would fail anyway,
since "has a number" alone splits NC at only 0.36/0.79).

The auxiliary tasks force the shared [CLS] representation to encode the
components of verifiability, aligning the Clear/NC decision boundary with
"is this promise checkable" rather than with hedging language.
"""

from __future__ import annotations

import re
from typing import Dict, List

# ── Lexicons / patterns (operate on raw `data` text only) ────────────────────────

# Half-width or full-width digits.
_NUM = r"[0-9０-９]+(?:[.,][0-9０-９]+)?"

# Units that turn a bare number into a *quantified target*.
_UNIT = (
    r"(?:%|％|成|倍|"                       # %, ％, 成, 倍
    r"噸|公噸|度|kWh|MWh|GWh|MW|GW|kW|"  # 噸 公噸 度 kWh...
    r"億|萬|千瓦|瓦|"                # 億 萬 千瓦 瓦
    r"公斤|公升|公頃|平方公尺|"  # 公斤 公升 公頃 平方公尺
    r"座|件|人次|人|小時|個|元|台|輛|"  # 座件人次人小時個元台輛
    r"名|項|家|次|份|棵|株|場|期"     # 名項家次份棵株場期
    r")"
)

# Quantified target: a number immediately bound to a unit, or any percent sign.
_RE_NUM_UNIT = re.compile(_NUM + r"\s*" + _UNIT)
_RE_PERCENT = re.compile(r"[%％]")

# Temporal anchor: an explicit calendar year (19xx/20xx) or a relative deadline
# like "3年內/前/底/後".
_RE_YEAR = re.compile(r"(?:19|20)[0-9０-９]{2}")
_RE_REL_DEADLINE = re.compile(_NUM + r"\s*年(?:內|前|底|後)")  # N年(內|前|底|後)

# Binding / concrete commitment markers — capture the "Clear without a number"
# case (e.g. 簽署不使用衝突礦產保證書, 取得 ISO 認證, 制定政策). Verifiable because
# the action is a discrete, checkable fact.
_RE_BINDING = re.compile(
    r"(?:簽署|簽訂|"                     # 簽署 簽訂
    r"制定|訂定|建立|導入|"  # 制定 訂定 建立 導入
    r"取得|通過|認證|驗證|查證|稽核|盤查|"  # 取得 通過 認證 驗證 查證 稽核 盤查
    r"揭露|發布|認養|"           # 揭露 發布 認養
    r"ISO|GRI|SBTi|RBA|TCFD|SASB|RoHS|UNEP)"            # external standards
)

# Quantified target (number+unit OR percent).
def _has_quantified_target(t: str) -> int:
    return int(bool(_RE_NUM_UNIT.search(t)) or bool(_RE_PERCENT.search(t)))


def _has_percent(t: str) -> int:
    return int(bool(_RE_PERCENT.search(t)))


def _has_temporal_anchor(t: str) -> int:
    return int(bool(_RE_YEAR.search(t)) or bool(_RE_REL_DEADLINE.search(t)))


def _has_binding_commitment(t: str) -> int:
    return int(bool(_RE_BINDING.search(t)))


# Ordered auxiliary-task registry. Order defines the aux-logit / aux-target
# column order; keep stable across train and eval.
AUX_FEATURES = [
    ("has_quantified_target", _has_quantified_target),
    ("has_temporal_anchor", _has_temporal_anchor),
    ("has_binding_commitment", _has_binding_commitment),
    ("has_percent", _has_percent),
]

AUX_NAMES: List[str] = [name for name, _ in AUX_FEATURES]
NUM_AUX = len(AUX_FEATURES)


def extract_vector(text: str) -> List[float]:
    """Return the ordered 0/1 auxiliary-target vector for one `data` text."""
    return [float(fn(text)) for _, fn in AUX_FEATURES]


def extract_dict(text: str) -> Dict[str, int]:
    """Return a named {feature: 0/1} dict (for diagnostics)."""
    return {name: int(fn(text)) for name, fn in AUX_FEATURES}


if __name__ == "__main__":
    # Quick self-check / separation report on the canonical train file.
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "data/raw_data/vpesg_4k_train_1000.json"
    )
    rows = json.load(open(path))
    clear = [r["data"] for r in rows if r.get("evidence_quality") == "Clear"]
    nc = [r["data"] for r in rows if r.get("evidence_quality") == "Not Clear"]
    print(f"Clear={len(clear)}  NotClear={len(nc)}")
    print(f"{'aux feature':24s} {'Clear':>7s} {'NotClear':>9s} {'diff':>7s}")
    for name, fn in AUX_FEATURES:
        c = sum(fn(t) for t in clear) / max(1, len(clear))
        n = sum(fn(t) for t in nc) / max(1, len(nc))
        print(f"{name:24s} {c:7.3f} {n:9.3f} {c - n:+7.3f}")
