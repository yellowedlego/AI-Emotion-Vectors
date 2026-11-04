"""층별 탐색 답변의 토큰·어절 빈도와 n-gram(이어진 n개) 빈도. 대시보드의 /words 페이지가 쓴다.

답변 글을 Kanana 토크나이저로 다시 쪼갠다. 생성 때의 토큰과 거의 같지만, 이모지처럼 한 글자가
여러 토큰으로 나뉜 곳은 한 조각으로 합친다. 측정 중에 파일이 계속 늘어나므로 새로 붙은 줄만 더 읽는다.

명령줄에서 CSV로 뽑기:
    .venv\\Scripts\\python.exe scripts\\word_freq.py joy --n 2 --unit word
"""
import argparse
import csv
import io
import json
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results" / "layer_sweep.jsonl"
CONTROL = "random0"
TOP = 1000  # CSV에 넣는 최대 줄 수
MIN_COUNT = 5  # 배수로 줄 세울 때 최소 횟수

box = {"pos": 0, "lock": threading.Lock(), "cache": {}}


def get_tok():
    if "tok" not in box:
        from transformers import AutoTokenizer

        box["tok"] = AutoTokenizer.from_pretrained("kakaocorp/kanana-2-3b-instruct")
    return box["tok"]


def junk(piece):
    """공백·문장부호·마크다운 기호로만 된 조각."""
    return not any(c.isalnum() for c in piece)


class Unit:
    """토큰이나 어절을 번호로 바꿔 한 줄로 이어 붙여 둔다."""

    def __init__(self):
        self.vocab, self.names, self.ids, self.starts = {}, [], [], []

    def add(self, pieces):
        self.starts.append(len(self.ids))
        for p in pieces:
            i = self.vocab.setdefault(p, len(self.vocab))
            if i == len(self.names):
                self.names.append(p)
            self.ids.append(i)


def token_pieces(text, offsets):
    out, end = [], -1
    for a, b in offsets:
        if a < end:  # 한 글자가 여러 토큰으로 나뉜 곳
            continue
        out.append(text[a:b])
        end = b
    return out


def word_pieces(text):
    out = []
    for w in text.split():
        w = w.strip("*#-:.,!?()[]\"'`~·…")
        if w:
            out.append(w)
    return out


def load():
    """새로 붙은 답변을 읽어 넣는다. 처음에는 13만 줄이라 30초쯤 걸린다."""
    with box["lock"]:
        if "rows" not in box:
            box.update(rows={"cond": [], "layer": [], "s": [], "broken": []}, token=Unit(), word=Unit())
        if not SWEEP.exists():
            return box
        with SWEEP.open("rb") as f:
            f.seek(box["pos"])
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1
        lines = [json.loads(x) for x in chunk[:end].splitlines()]
        if not lines:
            return box
        tok = get_tok()
        for i in range(0, len(lines), 5000):
            part = lines[i : i + 5000]
            enc = tok([x["text"] for x in part], return_offsets_mapping=True, add_special_tokens=False)
            for x, offs in zip(part, enc["offset_mapping"]):
                for k, v in (("cond", x["condition"]), ("layer", x["layer"]), ("s", x["s"]), ("broken", x["collapsed2"])):
                    box["rows"][k].append(v)
                box["token"].add(token_pieces(x["text"], offs))
                box["word"].add(word_pieces(x["text"]))
        box["pos"] += end
        box["arrays"] = arrays()
        box["cache"].clear()
        return box


def arrays():
    r = box["rows"]
    out = {"cond": np.array(r["cond"]), "layer": np.array(r["layer"]), "s": np.array(r["s"]), "broken": np.array(r["broken"], bool)}
    for name in ("token", "word"):
        u = box[name]
        ids = np.array(u.ids, np.int32)
        starts = np.array(u.starts + [len(u.ids)], np.int64)
        owner = np.repeat(np.arange(len(u.starts)), np.diff(starts))
        out[name] = {"ids": ids, "owner": owner, "junk": np.array([junk(p) for p in u.names], bool)}
    return out


def grams(unit, n, keep, skip_junk):
    """골라 둔 답변(keep)에서 이어진 n개를 [답변 번호, id1..idn] 행으로 모은다."""
    u = box["arrays"][unit]
    ids, owner = u["ids"], u["owner"]
    m = len(ids) - n + 1
    if m <= 0:
        return np.zeros((0, n + 1), np.int64)
    ok = keep[owner[:m]] & (owner[:m] == owner[n - 1 :])
    cols = [ids[k : k + m] for k in range(n)]
    if skip_junk:
        for c in cols:
            ok &= ~u["junk"][c]
    return np.column_stack([owner[:m][ok]] + [c[ok] for c in cols])


def count(rows):
    """[무리 번호, id...] 행을 무리별로 센다. (무리, 조각 묶음, 횟수)."""
    if not len(rows):
        return np.zeros((0, rows.shape[1]), np.int64), np.zeros(0, np.int64)
    return np.unique(rows, axis=0, return_counts=True)


def void(rows):
    """행 하나를 값 하나로 묶어 정렬·검색할 수 있게 한다."""
    rows = np.ascontiguousarray(rows, np.int64)
    return rows.view(np.dtype((np.void, 8 * rows.shape[1]))).ravel()


def masks(cond, layer, broken):
    a = box["arrays"]
    sel = a["cond"] != "base"
    sel &= (a["cond"] != CONTROL) if cond == "all" else (a["cond"] == cond)
    if layer != "all":
        sel &= a["layer"] == int(layer)
    if broken == "drop":
        sel &= ~a["broken"]
    elif broken == "only":
        sel &= a["broken"]
    return sel


def tally(sel, s_vals, unit, n, skip_junk):
    """세기마다, 그리고 증가·감소·전체로 묶어 센다. 무리 이름 -> (정렬된 키, 횟수, 조각 총수, 답변 수)."""
    a = box["arrays"]
    names = [*s_vals, "증가", "감소", "전체"]
    parts = [a["s"] == s for s in s_vals] + [a["s"] > 0, a["s"] < 0, np.ones(len(sel), bool)]
    out = {}
    gid = np.full(len(sel), -1)
    for i, m in enumerate(parts[: len(s_vals)]):
        gid[sel & m] = i
    rows = grams(unit, n, gid >= 0, skip_junk)
    if len(rows):
        rows[:, 0] = gid[rows[:, 0]]
    keys, counts = count(rows)
    for i, s in enumerate(s_vals):
        m = keys[:, 0] == i
        out[s] = (keys[m, 1:], counts[m], int((gid == i).sum()))
    for name, m in zip(names[len(s_vals) :], parts[len(s_vals) :]):
        k, c = count(grams(unit, n, sel & m, skip_junk)[:, 1:])
        out[name] = (k, c, int((sel & m).sum()))
    res = {}
    for name, (k, c, answers) in out.items():
        v = void(k) if len(k) else np.zeros(0, "V8")
        order = np.argsort(v)
        res[name] = (k, c, v[order], c[order], int(c.sum()), answers)
    return res


def lookup(table, keys):
    """table에서 keys 행마다의 횟수. 없으면 0."""
    _, _, sv, sc, _, _ = table
    if not len(sv) or not len(keys):
        return np.zeros(len(keys), np.int64)
    v = void(keys)
    i = np.clip(np.searchsorted(sv, v), 0, len(sv) - 1)
    return np.where(sv[i] == v, sc[i], 0)


def label(unit, key):
    names = box[unit].names
    sep = "" if unit == "token" else " "
    return sep.join(names[i] for i in key)


def query(cond="all", layer="all", unit="token", n=1, skip_junk=True, broken="drop", top=10, sort=1):
    """세기마다, 그리고 기준·증가·감소·전체 무리마다 많이 나온 조각.
    기준 대비: 주입 없는 답변(10개뿐이라 거칠다)보다 몇 배 자주 나왔는지.
    다른 감정 대비: 같은 층·세기에서 나머지 감정들보다 몇 배 자주 나왔는지. 세게 넣어 생기는 공통 버릇을 걸러 낸다."""
    load()
    a = box["arrays"]
    key = (cond, layer, unit, n, skip_junk, broken)
    if key not in box["cache"]:
        sel = masks(cond, layer, broken)
        s_vals = sorted(set(a["s"][sel].tolist()))
        mine = tally(sel, s_vals, unit, n, skip_junk)
        every = None
        if cond != "all":
            akey = ("all", layer, unit, n, skip_junk, broken, "tally")
            if akey not in box["cache"]:
                allsel = masks("all", layer, broken)
                box["cache"][akey] = tally(allsel, sorted(set(a["s"][allsel].tolist())), unit, n, skip_junk)
            every = box["cache"][akey]
        base = (a["cond"] == "base") & (~a["broken"] if broken != "keep" else True)  # 무너진 답변만 볼 때도 기준은 멀쩡한 답변
        bk, bc = count(grams(unit, n, base, skip_junk)[:, 1:])
        bv = void(bk) if len(bk) else np.zeros(0, "V8")
        o = np.argsort(bv)
        base_t = (bk, bc, bv[o], bc[o], int(bc.sum()), int(base.sum()))

        def pack(t, name):
            keys, counts, _, _, total, answers = t
            order = np.argsort(-counts, kind="stable")[:TOP]
            keys, counts = keys[order], counts[order]
            share = counts / max(1, total)
            b = lookup(base_t, keys)
            lift = share / ((b + 1) / max(1, base_t[4]))
            other = [None] * len(keys)
            if every is not None and name in every:
                e = every[name]
                rest = lookup(e, keys) - counts
                other = np.round(share / ((rest + 1) / max(1, e[4] - total)), 2).tolist()
            items = [[label(unit, k), int(c), round(int(c) / max(1, answers), 3), round(float(l), 2), x]
                     for k, c, l, x in zip(keys.tolist(), counts, lift, other)]
            return {"answers": answers, "total": total, "items": items}

        box["cache"][key] = {
            "strengths": [[s, pack(mine[s], s)] for s in s_vals],
            "summary": {"기준": pack(base_t, "기준"), **{k: pack(mine[k], k) for k in ("증가", "감소", "전체")}},
        }
    res = box["cache"][key]

    def cut(g):
        items = g["items"]
        if sort == 3 or (sort == 4 and items and items[0][4] is not None):  # 배수로 줄 세울 때는 한두 번 나온 우연을 뺀다
            items = sorted((x for x in items if x[1] >= MIN_COUNT and x[sort] is not None), key=lambda x: -x[sort])
        return {**g, "items": items[:top]}

    return {
        "cond": cond, "layer": layer, "unit": unit, "n": n,
        "emotions": sorted(set(a["cond"].tolist()) - {"base"}),
        "layers": sorted(set(a["layer"].tolist()) - {-1}),
        "answers": len(a["cond"]),
        "strengths": [[s, cut(g)] for s, g in res["strengths"]],
        "summary": {k: cut(g) for k, g in res["summary"].items()},
    }


def to_csv(cond="all", layer="all", unit="token", n=1, skip_junk=True, broken="drop"):
    """세기·무리마다 상위 1000개를 한 표로. 엑셀에서 한글이 깨지지 않게 BOM을 붙인다."""
    r = query(cond, layer, unit, n, skip_junk, broken, TOP)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["감정", "층", "단위", "n", "무리", "세기", "순위", "조각", "횟수", "답변당", "기준 대비 배수", "다른 감정 대비 배수", "무리 답변 수"])
    blocks = [(k, "", g) for k, g in r["summary"].items()] + [("세기별", s, g) for s, g in r["strengths"]]
    for name, s, g in blocks:
        for rank, (text, c, per, lift, other) in enumerate(g["items"], 1):
            w.writerow([cond, layer, unit, n, name, s, rank, text.replace("\n", "\\n"), c, per, lift, "" if other is None else other, g["answers"]])
    return ("﻿" + out.getvalue()).encode("utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", nargs="?", default="all")
    ap.add_argument("--layer", default="all")
    ap.add_argument("--unit", choices=["token", "word"], default="token")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--keep-junk", action="store_true")
    ap.add_argument("--broken", choices=["drop", "keep", "only"], default="drop")
    a = ap.parse_args()
    path = ROOT / "results" / f"word_freq_{a.cond}_{a.unit}{a.n}.csv"
    path.write_bytes(to_csv(a.cond, a.layer, a.unit, a.n, not a.keep_junk, a.broken))
    print(path)
