# -*- coding: utf-8 -*-
"""
Reviews Weekly Report Agent (v1)
--------------------------------
Самодостаточный агент для недельного отчёта по отзывам.

Возможности:
- Загрузка истории из локального XLSX или Google Drive (если заданы креды).
- Гибкая привязка столбцов (дата/источник/оценка/текст) по нескольким кандидатам.
- Нормализация оценок к 10-балльной шкале (по источнику или авто-инфер).
- Извлечение аспектов по вашему lexicon_module (с использованием display_short/long_hint).
- Сводки по неделе: totals, per-source, top positive/negative aspects.
- Генерация HTML summary + CSV-приложений.
- Опциональная отправка письма через SMTP.

Ничего не меняет в metrics_core/surveys_* и не требует reviews_core,
но имеет крюк для его использования (см. блок "OPTIONAL: hook to reviews_core").
"""

from __future__ import annotations

import os
import io
import re
import sys
import math
import json
import time
import uuid
import base64
import typing as T
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ===== Optional Google Drive deps (guarded) =====
_DRIVE_AVAILABLE = True
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as gbuild
    from googleapiclient.http import MediaIoBaseDownload
except Exception:
    _DRIVE_AVAILABLE = False

# ===== Optional SMTP (guarded) =====
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# ===== Your lexicon module =====
from lexicon_module import Lexicon

# ===== OPTIONAL: hook to reviews_core (if you сделали свою классификацию) =====
_HAS_REVIEW_CORE = False
try:
    import reviews_core as rcore
    # например, если у вас есть rcore.classify_reviews_df(...)
    _HAS_REVIEW_CORE = hasattr(rcore, "classify_reviews_df")
except Exception:
    _HAS_REVIEW_CORE = False


# ------------------------------------------------------------
# Параметры и утилиты
# ------------------------------------------------------------

SOURCE_DEFAULT_SCALES = {
    # можно расширять: используем для нормализации к /10
    # 'booking': 10,  # (пример, если у вас так)
    "google": 5,
    "yandex": 5,
    "2gis": 5,
    "tripadvisor": 5,
    "booking": 10,   # часто 10-балльная (adjust if иначе)
    "ostrovok": 10,
}

DATE_CANDIDATES = ["date", "review_date", "created_at", "created", "published_at"]
SOURCE_CANDIDATES = ["source", "platform", "channel"]
RATING_CANDIDATES = ["rating", "score", "rating_value", "review_rating"]
RATING_SCALE_CANDIDATES = ["rating_scale", "scale", "max_rating"]
TEXT_CANDIDATES = ["text", "review_text", "review", "content", "body", "comment"]
ID_CANDIDATES = ["review_id", "id", "external_id", "guid"]

REQUIRED_PANDAS_VERSION = (1, 3)  # на всякий случай


def _ensure_pandas_version():
    v = tuple(int(x) for x in pd.__version__.split(".")[:2])
    if v < REQUIRED_PANDAS_VERSION:
        print(f"[WARN] pandas {pd.__version__} < {'.'.join(map(str, REQUIRED_PANDAS_VERSION))}: "
              f"рекомендуется обновить.")


def _pick_col(df: pd.DataFrame, candidates: T.Sequence[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
        # иногда в колонках бывают разные кейсы/пробелы
        for col in df.columns:
            if col.lower().strip() == c.lower():
                return col
    if required:
        raise KeyError(f"Не найден ни один из столбцов: {candidates}")
    return None


def _parse_date(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col, errors="coerce").dt.tz_localize(None)


def _infer_scale_per_source(df: pd.DataFrame, source_col: str, rating_col: str) -> pd.Series:
    """
    Возвращает pd.Series scale_per_row для нормализации к /10.

    Логика:
    - если есть явный столбец scale — используем его;
    - иначе пробуем по источнику (SOURCE_DEFAULT_SCALES);
    - иначе берём 5, если максимум в источнике <= 5.0; иначе 10.
    """
    scale = pd.Series(index=df.index, dtype=float)

    # сначала по источнику — если известна карта
    if source_col in df.columns:
        src_norm = df[source_col].astype(str).str.lower().str.strip()
        for src_val, default_scale in SOURCE_DEFAULT_SCALES.items():
            mask = src_norm == src_val
            scale.loc[mask] = default_scale

    # оставшимся — по данным
    mask_nan = scale.isna()
    if mask_nan.any():
        # группируем по источнику; если не задан источник — общий максимум
        if source_col in df.columns:
            for src, grp in df.loc[mask_nan].groupby(df[source_col]):
                mx = pd.to_numeric(grp[rating_col], errors="coerce").max()
                guessed = 5.0 if (mx is not None and mx <= 5.0) else 10.0
                scale.loc[grp.index] = guessed
        else:
            mx = pd.to_numeric(df[rating_col], errors="coerce").max()
            guessed = 5.0 if (mx is not None and mx <= 5.0) else 10.0
            scale.loc[mask_nan] = guessed

    return scale


def _normalize_to_10(rating: pd.Series, scale: pd.Series) -> pd.Series:
    r = pd.to_numeric(rating, errors="coerce")
    s = pd.to_numeric(scale, errors="coerce").replace(0, np.nan)
    return (r / s) * 10.0


def _last_complete_week(today: dt.date | None = None) -> tuple[dt.datetime, dt.datetime]:
    """Возвращает (start, end) прошлого ПОЛНОГО календарного понедельник-воскресенье в UTC-naive."""
    if today is None:
        today = dt.date.today()
    # понедельник текущей недели
    this_monday = today - dt.timedelta(days=today.weekday())
    # конец прошлой недели = вчера от этого понедельника
    last_sunday = this_monday - dt.timedelta(days=1)
    last_monday = last_sunday - dt.timedelta(days=6)
    start = dt.datetime.combine(last_monday, dt.time.min)
    end = dt.datetime.combine(last_sunday, dt.time.max)
    return start, end


def _format_daterange(start: dt.datetime, end: dt.datetime) -> str:
    return f"{start:%d.%m.%Y} — {end:%d.%m.%Y}"


# ------------------------------------------------------------
# Загрузка истории
# ------------------------------------------------------------

def _load_from_local(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_excel(path)


def _drive_service_from_env():
    if not _DRIVE_AVAILABLE:
        return None
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    sa_content = os.getenv("GOOGLE_SERVICE_ACCOUNT_CONTENT") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT")
    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")

    creds = None
    if not sa_content and not sa_path and b64:
        try:
            sa_content = base64.b64decode(b64).decode("utf-8")
        except Exception:
            pass

    if sa_content:
        info = json.loads(sa_content)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    elif sa_path and os.path.exists(sa_path):
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    else:
        return None

    return gbuild("drive", "v3", credentials=creds, cache_discovery=False)


def _drive_download_generic(file_id: str, service) -> bytes:
    meta = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    mime = meta.get("mimeType", "")
    if mime == "application/vnd.google-apps.spreadsheet":
        request = service.files().export(
            fileId=file_id,
            mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return buf.getvalue()


def _drive_find_history_file(service, folder_id: str, explicit_file_id: str | None = None) -> bytes:
    if explicit_file_id:
        return _drive_download_generic(explicit_file_id, service)

    q = f"'{folder_id}' in parents and trashed = false and (mimeType contains 'spreadsheet' or name contains '.xls')"
    resp = service.files().list(q=q, pageSize=1000,
                                fields="files(id,name,mimeType,modifiedTime)").execute()
    files = resp.get("files", [])
    if not files:
        raise FileNotFoundError("В папке на Drive не найден файл истории отзывов.")
    # приоритет: имя с 'history/истори' и табличные расширения → самая свежая
    def score(f):
        n = f["name"].lower()
        hist = ("history" in n) or ("истори" in n)
        xls = n.endswith(".xlsx") or n.endswith(".xls") or "spreadsheet" in f.get("mimeType","")
        return (1 if hist else 0, 1 if xls else 0, f["modifiedTime"])
    file = sorted(files, key=score, reverse=True)[0]
    return _drive_download_generic(file["id"], service)


def load_reviews_history() -> pd.DataFrame:
    # 1) локальный путь (опционально)
    path = os.getenv("REVIEWS_HISTORY_PATH")
    if path:
        return _load_from_local(path)

    # 2) Google Drive (секреты с твоими именами)
    folder_id = os.getenv("REVIEWS_DRIVE_FOLDER_ID") or os.getenv("DRIVE_FOLDER_ID")
    file_id = os.getenv("REVIEWS_HISTORY_FILE_ID") or os.getenv("SHEETS_HISTORY_ID")

    service = _drive_service_from_env() if (folder_id or file_id) else None
    if service is not None:
        raw = _drive_find_history_file(service, folder_id or "root", explicit_file_id=file_id)
        # читаем экспорт в pandas
        return pd.read_excel(io.BytesIO(raw))

    raise RuntimeError("Не задан ни локальный путь, ни переменные для Drive.")



# ------------------------------------------------------------
# Извлечение аспектов (через ваш lexicon_module)
# ------------------------------------------------------------

@dataclass(frozen=True)
class AspectHit:
    aspect_code: str
    display_short: str
    long_hint: str
    polarity_hint: str  # positive / negative / neutral


class AspectMatcher:
    """
    Минималистичный матчинг аспектов по регуляркам из вашего Lexicon.
    Работает без reviews_core, но если у вас есть собственный быстрый классификатор,
    можете подключить его вместо (см. use_reviews_core=True в run()).
    """
    def __init__(self, lexicon: Lexicon):
        self.lex = lexicon
        self._compiled: dict[str, list[re.Pattern]] = self._compile()

    def _compile(self) -> dict[str, list[re.Pattern]]:
        compiled: dict[str, list[re.Pattern]] = {}
        # Берём AspectRule-ы из lexicon
        rules = self.lex.get_aspect_rules()  # ожидается в вашем модуле (мы добавляли такой метод)
        for aspect_code, rule in rules.items():
            pats: list[re.Pattern] = []
            for lang, arr in rule.patterns_by_lang.items():
                for rx in arr:
                    try:
                        pats.append(re.compile(rx, re.I | re.U))
                    except re.error:
                        # пропускаем битые регексы, чтобы не падать отчёт
                        pass
            compiled[aspect_code] = pats
        return compiled

    def extract(self, text: str) -> list[AspectHit]:
        if not isinstance(text, str) or not text.strip():
            return []
        txt = text.strip()
        hits: list[AspectHit] = []
        seen: set[str] = set()

        for aspect_code, patterns in self._compiled.items():
            for pat in patterns:
                if pat.search(txt):
                    if aspect_code in seen:
                        break
                    seen.add(aspect_code)
                    display_short = self.lex.aspect_display(aspect_code) or aspect_code
                    long_hint = self.lex.aspect_hint(aspect_code) or ""
                    polarity = self.lex.get_aspect_polarity(aspect_code) or "neutral"
                    hits.append(AspectHit(
                        aspect_code=aspect_code,
                        display_short=display_short,
                        long_hint=long_hint,
                        polarity_hint=polarity
                    ))
                    break
        return hits


# ------------------------------------------------------------
# Подготовка данных и расчёт недельных метрик
# ------------------------------------------------------------

@dataclass
class WeeklySummary:
    period_start: dt.datetime
    period_end: dt.datetime
    n_reviews: int
    avg_score_10: float
    pos_share: float
    neg_share: float
    by_source: pd.DataFrame
    aspects_pos: pd.DataFrame
    aspects_neg: pd.DataFrame
    sample: pd.DataFrame


def _prepare_dataframe(df: pd.DataFrame) -> dict:
    col_date = _pick_col(df, DATE_CANDIDATES, required=True)
    col_text = _pick_col(df, TEXT_CANDIDATES, required=True)
    col_source = _pick_col(df, SOURCE_CANDIDATES, required=False) or "_source"
    col_rating = _pick_col(df, RATING_CANDIDATES, required=False) or "_rating"
    col_scale = _pick_col(df, RATING_SCALE_CANDIDATES, required=False)
    col_id = _pick_col(df, ID_CANDIDATES, required=False) or "_rid"

    df = df.copy()
    if col_source not in df.columns:
        df[col_source] = "unknown"
    if col_rating not in df.columns:
        df[col_rating] = np.nan
    if col_id not in df.columns:
        df[col_id] = [str(x) for x in range(len(df))]

    df[col_date] = _parse_date(df[col_date])
    df["_source_norm"] = df[col_source].astype(str).str.lower().str.strip()

    if col_scale and col_scale in df.columns:
        # явная шкала
        scale = pd.to_numeric(df[col_scale], errors="coerce")
    else:
        scale = _infer_scale_per_source(df, col_source, col_rating)

    df["_score_10"] = _normalize_to_10(df[col_rating], scale)

    return dict(
        df=df,
        col_date=col_date,
        col_text=col_text,
        col_source=col_source,
        col_rating=col_rating,
        col_id=col_id
    )


def _slice_period(df: pd.DataFrame, col_date: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    m = df[col_date].between(start, end, inclusive="both")
    return df.loc[m].copy()


def _classify_aspects_df(df: pd.DataFrame, col_text: str, lex: Lexicon, use_reviews_core: bool = False) -> list[list[AspectHit]]:
    """
    Возвращает список списков AspectHit на каждый отзыв.
    Если у вас реализован свой быстрый пайплайн в reviews_core, можно задействовать его.
    """
    if use_reviews_core and _HAS_REVIEW_CORE:
        # пример – подстройте под свою сигнатуру:
        # cls_df = rcore.classify_reviews_df(df[[col_text]], text_col=col_text, lexicon=lex)
        # ожидаем колонку 'aspect_codes' (list[str]); здесь преобразуем в AspectHit
        # Ниже — безопасный fallback:
        pass

    # fallback: встроенный матчинг
    matcher = AspectMatcher(lex)
    return [matcher.extract(str(t)) for t in df[col_text].tolist()]


def build_weekly_summary(df_raw: pd.DataFrame, week_start: dt.datetime, week_end: dt.datetime, sample_n: int = 25) -> WeeklySummary:
    prep = _prepare_dataframe(df_raw)
    df = _slice_period(prep["df"], prep["col_date"], week_start, week_end)
    if df.empty:
        return WeeklySummary(
            period_start=week_start, period_end=week_end,
            n_reviews=0, avg_score_10=float("nan"),
            pos_share=float("nan"), neg_share=float("nan"),
            by_source=pd.DataFrame(), aspects_pos=pd.DataFrame(),
            aspects_neg=pd.DataFrame(), sample=pd.DataFrame(),
        )

    # аспекты
    lex = Lexicon()
    aspect_hits = _classify_aspects_df(df, prep["col_text"], lex, use_reviews_core=True)
    df["_aspect_codes"] = [[h.aspect_code for h in hits] for hits in aspect_hits]
    df["_aspect_pols"] = [[h.polarity_hint for h in hits] for hits in aspect_hits]
    df["_aspect_display_short"] = [[h.display_short for h in hits] for hits in aspect_hits]
    df["_aspect_long_hint"] = [[h.long_hint for h in hits] for hits in aspect_hits]

    # простая тональность по оценке (можно заменить вашей логикой позже)
    s10 = pd.to_numeric(df["_score_10"], errors="coerce")
    df["_sent_bucket"] = pd.cut(
        s10,
        bins=[-0.01, 3.0, 5.0, 7.0, 8.5, 10.01],
        labels=["strong_neg", "soft_neg", "neutral", "soft_pos", "strong_pos"]
    )

    n_reviews = len(df)
    avg_score_10 = float(np.nanmean(s10))
    pos_share = float((df["_sent_bucket"].isin(["soft_pos", "strong_pos"])).mean())
    neg_share = float((df["_sent_bucket"].isin(["soft_neg", "strong_neg"])).mean())

    # по источникам
    by_source = (
        df.groupby("_source_norm")
          .agg(n=(" _rid" if " _rid" in df.columns else prep["col_id"], "count"),
               avg10=("_score_10", "mean"))
          .reset_index()
          .rename(columns={"_source_norm": "source"})
          .sort_values(["n", "avg10"], ascending=[False, False])
    )

    # агрегация аспектов
    # разворачиваем список -> строки
    recs: list[dict] = []
    for i, (codes, pols, dsp) in enumerate(zip(df["_aspect_codes"], df["_aspect_pols"], df["_aspect_display_short"])):
        for c, p, d in zip(codes, pols, dsp):
            recs.append({"aspect_code": c, "polarity": p, "display_short": d})

    aspects_df = pd.DataFrame(recs)
    if aspects_df.empty:
        aspects_pos = pd.DataFrame(columns=["aspect_code", "display_short", "cnt"])
        aspects_neg = aspects_pos.copy()
    else:
        agg = (aspects_df
               .groupby(["aspect_code", "display_short", "polarity"])
               .size()
               .reset_index(name="cnt"))
        aspects_pos = (agg[agg["polarity"] == "positive"]
                       .sort_values("cnt", ascending=False)
                       .head(25)
                       .drop(columns=["polarity"]))
        aspects_neg = (agg[agg["polarity"] == "negative"]
                       .sort_values("cnt", ascending=False)
                       .head(25)
                       .drop(columns=["polarity"]))

    # сэмпл отзывов
    sample_cols = [prep["col_date"], prep["col_source"], prep["col_rating"], "_score_10", prep["col_id"], prep["col_text"]]
    sample = df[sample_cols].sort_values(prep["col_date"], ascending=False).head(sample_n).copy()

    return WeeklySummary(
        period_start=week_start, period_end=week_end,
        n_reviews=n_reviews, avg_score_10=avg_score_10,
        pos_share=pos_share, neg_share=neg_share,
        by_source=by_source, aspects_pos=aspects_pos, aspects_neg=aspects_neg,
        sample=sample
    )


# ------------------------------------------------------------
# Рендер HTML и сохранение приложений
# ------------------------------------------------------------

def _html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _pct(x: float) -> str:
    if x != x or x is None:
        return "—"
    return f"{x*100:.1f}%"

def _fmt2(x: float) -> str:
    if x != x or x is None:
        return "—"
    return f"{x:.2f}"

def _render_html(summary: WeeklySummary, title_prefix: str = "") -> str:
    period = _format_daterange(summary.period_start, summary.period_end)
    title = f"{title_prefix} Отчёт по отзывам за {period}".strip()

    # таблица источников
    src_rows = []
    if not summary.by_source.empty:
        for _, r in summary.by_source.iterrows():
            src_rows.append(
                f"<tr><td>{_html_escape(str(r['source']))}</td>"
                f"<td style='text-align:right'>{int(r['n'])}</td>"
                f"<td style='text-align:right'>{_fmt2(float(r['avg10']))}</td></tr>"
            )
    src_html = "\n".join(src_rows) if src_rows else "<tr><td colspan='3' style='color:#888'>нет данных</td></tr>"

    # аспекты (+)
    pos_rows = []
    if not summary.aspects_pos.empty:
        for _, r in summary.aspects_pos.iterrows():
            title = _html_escape(Lexicon().aspect_hint(r["aspect_code"]) or "")
            pos_rows.append(
                f"<tr><td title='{title}'>{_html_escape(str(r['display_short']))}</td>"
                f"<td style='text-align:right'>{int(r['cnt'])}</td></tr>"
            )
    pos_html = "\n".join(pos_rows) if pos_rows else "<tr><td colspan='2' style='color:#888'>нет данных</td></tr>"

    # аспекты (−)
    neg_rows = []
    if not summary.aspects_neg.empty:
        for _, r in summary.aspects_neg.iterrows():
            title = _html_escape(Lexicon().aspect_hint(r["aspect_code"]) or "")
            neg_rows.append(
                f"<tr><td title='{title}'>{_html_escape(str(r['display_short']))}</td>"
                f"<td style='text-align:right'>{int(r['cnt'])}</td></tr>"
            )
    neg_html = "\n".join(neg_rows) if neg_rows else "<tr><td colspan='2' style='color:#888'>нет данных</td></tr>"

    html = f"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<title>{_html_escape(title)}</title>
<style>
body {{
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  color: #1f2937; background: #fff; line-height: 1.45; padding: 24px;
}}
h1 {{ margin: 0 0 12px; font-size: 20px; }}
.kpi {{ display: inline-block; margin: 8px 16px 16px 0; padding: 10px 14px; background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; }}
.kpi .v {{ font-size:18px; font-weight:700; }}
.kpi .t {{ font-size:12px; color:#6b7280 }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 24px; }}
th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; font-size: 13px; }}
th {{ background: #f3f4f6; text-align: left; }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
small {{ color:#6b7280 }}
</style>
</head>
<body>
  <h1>{_html_escape(title)}</h1>
  <div class="kpi"><div class="v">{summary.n_reviews}</div><div class="t">Отзывов</div></div>
  <div class="kpi"><div class="v">{_fmt2(summary.avg_score_10)}</div><div class="t">Средняя оценка (из 10)</div></div>
  <div class="kpi"><div class="v">{_pct(summary.pos_share)}</div><div class="t">Доля позитивных (по оценке)</div></div>
  <div class="kpi"><div class="v">{_pct(summary.neg_share)}</div><div class="t">Доля негативных (по оценке)</div></div>

  <h2>Разбивка по источникам</h2>
  <table>
    <thead><tr><th>Источник</th><th>Кол-во</th><th>Средн. оценка /10</th></tr></thead>
    <tbody>
      {src_html}
    </tbody>
  </table>

  <h2>Топ аспектов (+)</h2>
  <table>
    <thead><tr><th>Аспект</th><th>Упоминаний</th></tr></thead>
    <tbody>
      {pos_html}
    </tbody>
  </table>

  <h2>Топ аспектов (−)</h2>
  <table>
    <thead><tr><th>Аспект</th><th>Упоминаний</th></tr></thead>
    <tbody>
      {neg_html}
    </tbody>
  </table>

  <p><small>Примечание: аспекты извлекаются из текста отзывов по словарю (lexicon_module), где для каждого
  аспекта задано короткое название и подсказка. Подробности — в тултипах ячеек аспектов.</small></p>
</body>
</html>
""".strip()
    return html


def _out_dir() -> str:
    return os.getenv("OUT_DIR") or "/mnt/data"

def _save_csv(df: pd.DataFrame, fname: str) -> str:
    os.makedirs(_out_dir(), exist_ok=True)
    path = os.path.join(_out_dir(), fname)
    df.to_csv(path, index=False)
    return path


def _maybe_send_email(subject: str, html: str, attachments: list[tuple[str, bytes]]):
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "0") or 0)
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    to   = os.getenv("MAIL_TO")
    from_ = os.getenv("MAIL_FROM") or user

    if not host or not port:
        # авто-настройка по домену почты
        domain = (user or "").split("@")[-1].lower()
        guess = {
            "gmail.com": ("smtp.gmail.com", 587),
            "yandex.ru": ("smtp.yandex.ru", 587),
            "ya.ru": ("smtp.yandex.ru", 587),
            "yandex.com": ("smtp.yandex.com", 587),
            "outlook.com": ("smtp.office365.com", 587),
            "office365.com": ("smtp.office365.com", 587),
            "hotmail.com": ("smtp.office365.com", 587),
            "live.com": ("smtp.office365.com", 587),
            "mail.ru": ("smtp.mail.ru", 587),
            "bk.ru": ("smtp.mail.ru", 587),
            "inbox.ru": ("smtp.mail.ru", 587),
            "list.ru": ("smtp.mail.ru", 587),
        }
        if domain in guess:
            host, port = guess[domain]

    if not (host and port and user and pwd and to and from_):
        print("[info] SMTP не настроен — письмо не отправляется.")
        return

    msg = MIMEMultipart()
    msg["From"] = from_
    msg["To"] = to
    prefix = os.getenv("MAIL_SUBJECT_PREFIX", "").strip()
    msg["Subject"] = f"{prefix} {subject}".strip()

    msg.attach(MIMEText(html, "html", "utf-8"))
    for fname, data in attachments:
        part = MIMEApplication(data, Name=os.path.basename(fname))
        part["Content-Disposition"] = f'attachment; filename="{os.path.basename(fname)}"'
        msg.attach(part)

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)
        print(f"[ok] Письмо отправлено: {to}")



# ------------------------------------------------------------
# Основной сценарий
# ------------------------------------------------------------

def run(reference_date: dt.date | None = None) -> dict:
    """
    Основной сценарий: загрузка, срез недели, расчёт, HTML, приложений, (опц.) отправка письма.
    Возвращает словарь с путями к CSV и HTML (для дебага/логов).
    """
    _ensure_pandas_version()

    # 1) загрузка
    df = load_reviews_history()
    print(f"[ok] Загружено записей: {len(df)}")

    # 2) неделя
    start, end = _last_complete_week(reference_date)
    print(f"[info] Период: {start:%Y-%m-%d} .. {end:%Y-%m-%d}")

    # 3) расчёт
    summary = build_weekly_summary(df, start, end)

    # 4) html
    subject = f"Отзывы: неделя {_format_daterange(start, end)}"
    html = _render_html(summary, title_prefix=os.getenv("MAIL_SUBJECT_PREFIX", ""))

    # 5) сохранить CSV
    paths = {}
    if not summary.by_source.empty:
        p1 = _save_csv(summary.by_source, f"reviews_weekly_sources_{start:%Y%m%d}_{end:%Y%m%d}.csv")
        paths["by_source_csv"] = p1
    if not summary.aspects_pos.empty or not summary.aspects_neg.empty:
        p2 = _save_csv(summary.aspects_pos, f"reviews_weekly_aspects_pos_{start:%Y%m%d}_{end:%Y%m%d}.csv")
        p3 = _save_csv(summary.aspects_neg, f"reviews_weekly_aspects_neg_{start:%Y%m%d}_{end:%Y%m%d}.csv")
        paths["aspects_pos_csv"] = p2
        paths["aspects_neg_csv"] = p3
    if not summary.sample.empty:
        p4 = _save_csv(summary.sample, f"reviews_weekly_sample_{start:%Y%m%d}_{end:%Y%m%d}.csv")
        paths["sample_csv"] = p4

    # 6) сохранить HTML (на всякий случай)
    html_path = os.path.join(_out_dir(), f"reviews_weekly_report_{start:%Y%m%d}_{end:%Y%m%d}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    paths["html"] = html_path

    # 7) (опц.) отправка по SMTP
    atts: list[tuple[str, bytes]] = []
    for key, p in paths.items():
        if p.endswith(".csv") and os.path.exists(p):
            with open(p, "rb") as fh:
                atts.append((os.path.basename(p), fh.read()))
    _maybe_send_email(subject=subject, html=html, attachments=atts)

    print("[ok] Готово.")
    return paths


if __name__ == "__main__":
    run()
