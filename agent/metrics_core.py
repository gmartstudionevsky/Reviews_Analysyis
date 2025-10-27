# agent/metrics_core.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd
import json

# периоды, которые мы агрегируем
PERIOD_LEVELS: List[Tuple[str, str]] = [
    ("week",    "week_key"),
    ("month",   "month_key"),
    ("quarter", "quarter_key"),
    ("year",    "year_key"),
]

def _json_list(cell: Any) -> List[Any]:
    """Чтение JSON-строки в python-list безопасно."""
    if isinstance(cell, list):
        return cell
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    try:
        return json.loads(cell)
    except Exception:
        return []


def build_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица history:
    - period_type (week/month/quarter/year)
    - period_key
    - reviews (кол-во отзывов)
    - avg10 (средняя оценка /10)
    - pos/neu/neg (кол-во отзывов по общей тональности)
    """
    rows = []
    for ptype, col in PERIOD_LEVELS:
        if col not in df_raw.columns:
            continue
        sub = df_raw.dropna(subset=[col])
        if sub.empty:
            continue

        for pkey, g in sub.groupby(col, dropna=True):
            rows.append({
                "period_type": ptype,
                "period_key":  pkey,
                "reviews":     int(g["review_id"].nunique()),
                "avg10":       round(float(g["rating10"].dropna().mean()), 2)
                               if g["rating10"].notna().any() else "",
                "pos":         int((g["sentiment_overall"] == "pos").sum()),
                "neu":         int((g["sentiment_overall"] == "neu").sum()),
                "neg":         int((g["sentiment_overall"] == "neg").sum()),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["period_type","period_key","reviews","avg10","pos","neu","neg"])

    # сортируем красиво: сначала week, потом month, потом quarter, потом year
    order = {p:i for i,(p,_) in enumerate(PERIOD_LEVELS)}
    out["__ord"] = out["period_type"].map(order).fillna(999)
    out = out.sort_values(by=["__ord", "period_key"], ascending=[True, True], ignore_index=True)
    out = out.drop(columns="__ord")
    return out


def build_sources_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица sources_history:
    - week_key
    - source
    - reviews
    - avg10
    - pos/neu/neg
    Используется в блоке C (матрица Источник × Период).
    """
    if "week_key" not in df_raw.columns or "source" not in df_raw.columns:
        return pd.DataFrame(columns=["week_key","source","reviews","avg10","pos","neu","neg"])

    sub = df_raw.dropna(subset=["week_key","source"])
    if sub.empty:
        return pd.DataFrame(columns=["week_key","source","reviews","avg10","pos","neu","neg"])

    rows = []
    for (wk, src), g in sub.groupby(["week_key","source"], dropna=True):
        rows.append({
            "week_key": wk,
            "source":   src,
            "reviews":  int(g["review_id"].nunique()),
            "avg10":    round(float(g["rating10"].dropna().mean()), 2)
                        if g["rating10"].notna().any() else "",
            "pos":      int((g["sentiment_overall"] == "pos").sum()),
            "neu":      int((g["sentiment_overall"] == "neu").sum()),
            "neg":      int((g["sentiment_overall"] == "neg").sum()),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(by=["week_key","source"], ascending=[True,True], ignore_index=True)
    return out


def build_semantic_agg_aspects_period(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица semantic_agg_aspects_period:
    - period_type / period_key
    - source_scope ("all" и отдельно каждый источник)
    - aspect_key
    - mentions_total / pos_mentions / neg_mentions / neu_mentions
    - pos_share / neg_share
    - pos_weight / neg_weight
      *pos_weight* ~= насколько этот аспект свойственен позитивным отзывам недели
      *neg_weight* ~= насколько этот аспект свойственен негативным отзывам недели

    Это прямой источник для блоков B1 (сильные стороны), B2 (риски),
    B3 (существенные отклонения).
    """
    rows = []
    # список уникальных источников для source_scope
    sources = sorted([s for s in df_raw["source"].dropna().unique().tolist() if s])

    for ptype, col in PERIOD_LEVELS:
        if col not in df_raw.columns:
            continue
        sub_per = df_raw.dropna(subset=[col])
        if sub_per.empty:
            continue

        for scope in ["all"] + sources:
            if scope == "all":
                scoped = sub_per
            else:
                scoped = sub_per[sub_per["source"] == scope]

            if scoped.empty:
                continue

            # какие отзывы считаем «позитивными» / «негативными»
            pos_reviews = set(scoped.loc[scoped["sentiment_overall"]=="pos","review_id"])
            neg_reviews = set(scoped.loc[scoped["sentiment_overall"]=="neg","review_id"])
            pos_total = len(pos_reviews)
            neg_total = len(neg_reviews)

            for pkey, g in scoped.groupby(col, dropna=True):

                # собираем статистику по аспектам
                aspects_stats: Dict[str, Dict[str,set]] = {}
                # структура: aspect_key -> { "m": all_mentions, "p":pos_mentions, "n":neg_mentions, "u":neu_mentions }
                # где значения — множества review_id

                for _, r in g.iterrows():
                    rid = r["review_id"]
                    topics_all = set(_json_list(r.get("topics_all")))
                    topics_pos = set(_json_list(r.get("topics_pos")))
                    topics_neg = set(_json_list(r.get("topics_neg")))
                    topics_neu = topics_all - topics_pos - topics_neg

                    for a in topics_all:
                        aspects_stats.setdefault(a, {"m":set(),"p":set(),"n":set(),"u":set()})["m"].add(rid)
                    for a in topics_pos:
                        aspects_stats.setdefault(a, {"m":set(),"p":set(),"n":set(),"u":set()})["p"].add(rid)
                    for a in topics_neg:
                        aspects_stats.setdefault(a, {"m":set(),"p":set(),"n":set(),"u":set()})["n"].add(rid)
                    for a in topics_neu:
                        aspects_stats.setdefault(a, {"m":set(),"p":set(),"n":set(),"u":set()})["u"].add(rid)

                for aspect_key, bucket in aspects_stats.items():
                    mentions_total = len(bucket["m"])
                    pos_mentions   = len(bucket["p"])
                    neg_mentions   = len(bucket["n"])
                    neu_mentions   = len(bucket["u"])

                    # «вес» аспекта в позитиве/негативе —
                    # доля позитивных/негативных отзывов недели,
                    # в которых этот аспект упомянут
                    pos_weight = (
                        len(bucket["p"].intersection(pos_reviews))/pos_total
                        if pos_total else 0.0
                    )
                    neg_weight = (
                        len(bucket["n"].intersection(neg_reviews))/neg_total
                        if neg_total else 0.0
                    )

                    rows.append({
                        "period_type": ptype,
                        "period_key": pkey,
                        "source_scope": scope,
                        "aspect_key": aspect_key,
                        "mentions_total": mentions_total,
                        "pos_mentions": pos_mentions,
                        "neg_mentions": neg_mentions,
                        "neu_mentions": neu_mentions,
                        "pos_share": round(pos_mentions/mentions_total,4) if mentions_total else 0.0,
                        "neg_share": round(neg_mentions/mentions_total,4) if mentions_total else 0.0,
                        "pos_weight": round(pos_weight,4),
                        "neg_weight": round(neg_weight,4),
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "period_type","period_key","source_scope","aspect_key",
            "mentions_total","pos_mentions","neg_mentions","neu_mentions",
            "pos_share","neg_share","pos_weight","neg_weight",
        ])

    # сортируем чтобы смотреть глазами:
    # 1. week -> month -> quarter -> year
    # 2. period_key по возрастанию
    # 3. сначала scope=all, потом конкретные источники
    # 4. внутри — по mentions_total по убыванию (что чаще всего всплывает)
    order = {p:i for i,(p,_) in enumerate(PERIOD_LEVELS)}
    out["__o"] = out["period_type"].map(order).fillna(999)
    out["__s"] = out["source_scope"].apply(lambda s: 0 if s=="all" else 1)
    out = out.sort_values(
        by=["__o","period_key","__s","mentions_total"],
        ascending=[True,True,True,False],
        ignore_index=True
    )
    out = out.drop(columns=["__o","__s"])
    return out


def build_semantic_agg_pairs_period(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица semantic_agg_pairs_period:
    - period_type / period_key
    - source_scope ("all" и по источникам)
    - pair_key (строка "aspectA|aspectB" в сортированном порядке)
    - category (systemic_risk / expectations_conflict / loyalty_driver)
    - distinct_reviews (сколько разных отзывов упомянули эту пару)
    - example_quote (какой-нибудь характерный фрагмент)

    Это источник блока B4.
    """
    rows = []
    sources = sorted([s for s in df_raw["source"].dropna().unique().tolist() if s])

    for ptype, col in PERIOD_LEVELS:
        if col not in df_raw.columns:
            continue
        sub_per = df_raw.dropna(subset=[col])
        if sub_per.empty:
            continue

        for scope in ["all"] + sources:
            if scope == "all":
                scoped = sub_per
            else:
                scoped = sub_per[sub_per["source"] == scope]

            if scoped.empty:
                continue

            for pkey, g in scoped.groupby(col, dropna=True):

                pair_to_reviews: Dict[Tuple[str,str], set] = {}
                pair_to_quote: Dict[Tuple[str,str], str] = {}

                for _, r in g.iterrows():
                    rid    = r["review_id"]
                    pairs  = _json_list(r.get("pair_tags"))
                    quotes = _json_list(r.get("quote_candidates"))
                    quote0 = quotes[0] if quotes else ""

                    for p in pairs:
                        a = (p.get("a") or "").strip()
                        b = (p.get("b") or "").strip()
                        cat = (p.get("cat") or "").strip()
                        if not a or not b or not cat:
                            continue

                        pair_sorted = "|".join(sorted([a,b]))
                        key = (pair_sorted, cat)

                        pair_to_reviews.setdefault(key, set()).add(rid)
                        # сохраняем первую увиденную цитату как пример
                        if key not in pair_to_quote and quote0:
                            pair_to_quote[key] = quote0

                for (pair_key, cat), revs in pair_to_reviews.items():
                    rows.append({
                        "period_type": ptype,
                        "period_key": pkey,
                        "source_scope": scope,
                        "pair_key": pair_key,
                        "category": cat,
                        "distinct_reviews": len(revs),
                        "example_quote": pair_to_quote.get((pair_key, cat), ""),
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "period_type","period_key","source_scope",
            "pair_key","category","distinct_reviews","example_quote"
        ])

    order = {p:i for i,(p,_) in enumerate(PERIOD_LEVELS)}
    out["__o"] = out["period_type"].map(order).fillna(999)
    out["__s"] = out["source_scope"].apply(lambda s: 0 if s=="all" else 1)
    out = out.sort_values(
        by=["__o","period_key","__s","distinct_reviews"],
        ascending=[True,True,True,False],
        ignore_index=True
    )
    out = out.drop(columns=["__o","__s"])
    return out
