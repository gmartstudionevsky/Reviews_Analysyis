# agent/weekly_report_agent.py
# Еженедельный агент: отчёт «Неделя vs MTD/QTD/YTD» + источники + темы + графики
import os, io, re, json, math, datetime as dt
from datetime import date
import pandas as pd
import numpy as np

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import smtplib, mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Графики (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from agent.metrics_core import (
        HISTORY_TAB, TOPICS_TAB, SOURCES_TAB, SURVEYS_TAB,
        iso_week_monday, week_range_for_monday,
        period_ranges_for_week, prev_period_ranges,
        aggregate_weeks_from_history, deltas_week_vs_period,
        role_of_week_in_period, sources_summary_for_periods,
        week_label, month_label, quarter_label, year_label
    )
except ModuleNotFoundError:
    # fallback, если файл запущен как скрипт и пакет 'agent' не найден
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from metrics_core import (
        HISTORY_TAB, TOPICS_TAB, SOURCES_TAB, SURVEYS_TAB,
        iso_week_monday, week_range_for_monday,
        period_ranges_for_week, prev_period_ranges,
        aggregate_weeks_from_history, deltas_week_vs_period,
        role_of_week_in_period, sources_summary_for_periods,
        week_label, month_label, quarter_label, year_label
    )

# --- наше ядро метрик / периодов ---
from agent.metrics_core import (
    HISTORY_TAB, TOPICS_TAB, SOURCES_TAB, SURVEYS_TAB,
    iso_week_monday, week_range_for_monday,
    period_ranges_for_week, prev_period_ranges,
    aggregate_weeks_from_history, deltas_week_vs_period,
    role_of_week_in_period, sources_summary_for_periods,
    week_label, month_label, quarter_label, year_label
)

# ======================================
# ENV / Google API clients / constants
# ======================================
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]
CREDS = Credentials.from_service_account_file(
    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"], scopes=SCOPES
)
DRIVE  = build("drive",  "v3", credentials=CREDS)
SHEETS = build("sheets", "v4", credentials=CREDS).spreadsheets()

DRIVE_FOLDER_ID   = os.environ["DRIVE_FOLDER_ID"]
HISTORY_SHEET_ID  = os.environ["SHEETS_HISTORY_ID"]
RECIPIENTS        = [e.strip() for e in os.environ["RECIPIENTS"].split(",") if e.strip()]
SMTP_USER         = os.environ["SMTP_USER"]
SMTP_PASS         = os.environ["SMTP_PASS"]
SMTP_FROM         = os.environ.get("SMTP_FROM", SMTP_USER)
SMTP_HOST         = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.environ.get("SMTP_PORT", 587))

# Ожидаемая схема файла отзывов
EXPECTED = ["Дата","Рейтинг","Источник","Автор","Код языка","Текст отзыва","Наличие ответа"]
RENAMES  = {"Дата":"Дата","Рейтинг":"Рейтинг","Источник":"Источник","Автор":"Автор",
            "Код языка":"Код языка","Текст отзыва":"Текст","Наличие ответа":"Ответ"}

# ====================
# Utils: Drive/Sheets
# ====================
def ensure_tab(spreadsheet_id: str, tab_name: str, header: list[str]):
    meta = SHEETS.get(spreadsheetId=spreadsheet_id).execute()
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    if tab_name not in tabs:
        SHEETS.batchUpdate(spreadsheetId=spreadsheet_id,
                           body={"requests":[{"addSheet":{"properties":{"title":tab_name}}}]}).execute()
        SHEETS.values().update(spreadsheetId=spreadsheet_id,
                               range=f"{tab_name}!A1:{chr(64+len(header))}1",
                               valueInputOption="RAW",
                               body={"values":[header]}).execute()

def gs_get_df(tab: str, a1: str) -> pd.DataFrame:
    try:
        res = SHEETS.values().get(spreadsheetId=HISTORY_SHEET_ID, range=f"{tab}!{a1}").execute()
        vals = res.get("values", [])
        return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals)>1 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def gs_append(tab: str, a1: str, rows: list[list]):
    if not rows: return
    SHEETS.values().append(spreadsheetId=HISTORY_SHEET_ID,
                           range=f"{tab}!{a1}",
                           valueInputOption="RAW",
                           body={"values": rows}).execute()

def latest_reviews_file_from_drive() -> tuple[str,str,date]:
    regex = re.compile(r"^Reviews_\d{2}-\d{2}-\d{4}\.xls(x)?$", re.IGNORECASE)
    res = DRIVE.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false",
        fields="files(id,name,modifiedTime)",
        pageSize=1000
    ).execute()
    items = res.get("files", [])
    def parse_date_from_name(name: str):
        m = re.search(r"^Reviews_(\d{2})-(\d{2})-(\d{4})\.xls(x)?$", name, re.IGNORECASE)
        if not m: return None
        dd, mm, yyyy = map(int, m.groups()[:3])
        return date(yyyy, mm, dd)
    items = [(i["id"], i["name"], parse_date_from_name(i["name"])) for i in items if regex.match(i["name"])]
    if not items:
        raise RuntimeError("В папке нет файлов вида Reviews_dd-mm-yyyy.xls(x).")
    items.sort(key=lambda t: t[2], reverse=True)
    return items[0]  # id, name, date

def drive_download(file_id: str) -> bytes:
    req = DRIVE.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

# ===============
# IO: Reviews XLS
# ===============
def normalize_to10(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False).str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )
    vmax, vmin = num.max(skipna=True), num.min(skipna=True)
    if pd.isna(vmax): return num
    if vmax <= 5: return num*2
    if vmax <= 10: return num
    if vmax <= 100 and vmin >= 0: return num/10
    if vmax <= 1: return num*10
    return num.clip(upper=10)

def load_reviews_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)  # .xls/.xlsx
    low = {str(c).strip().lower(): c for c in df.columns}
    ren = {}
    for k in EXPECTED:
        hit = low.get(k.lower())
        if hit: ren[hit] = RENAMES[k]
        else:
            cand = next((c for lc,c in low.items() if k.split()[0].lower() in lc), None)
            if cand: ren[cand] = RENAMES[k]
    df = df.rename(columns=ren)
    missing = [v for v in RENAMES.values() if v not in df.columns]
    if missing: raise RuntimeError(f"Нет ожидаемых колонок: {missing}")
    return df[[*RENAMES.values()]]

# ===============
# Sentiment/topics
# ===============
NEG_CUES = re.compile(
    r"\bне\b|\bнет\b|проблем|ошиб|задерж|долг|ждал|ждали|"
    r"сложн|плохо|неудоб|путан|непонят|неясн|не приш|не сработ|не открыл|не откры|скромн|бедн|мало",
    re.IGNORECASE
)
POS_LEX = re.compile(r"отличн|прекрасн|замечат|идеальн|классн|любезн|комфортн|удобн|чисто|тихо|вкусн|быстро", re.IGNORECASE)
NEG_LEX = re.compile(r"ужасн|плох|грязн|громк|холодн|жарко|слаб|плохо|долго|скучн|бедн|мало|дорог", re.IGNORECASE)

TOPIC_PATTERNS = {
    "Локация":        [r"расположен|расположение|рядом|вокзал|невск"],
    "Персонал":       [r"персонал|сотрудник|администрат|ресепшен|ресепшн|менеджер"],
    "Чистота":        [r"чисто|уборк|бель[её]|пятн|полотенц|постель"],
    "Номер/оснащение":[r"номер|кухн|посудомо|стирал|плита|чайник|фен|сейф"],
    "Сантехника/вода":[r"душ|слив|кран|смесител|бойлер|водонагрев|давлен|температур|канализац|засор"],
    "AC/шум":         [r"кондицион|вентиляц|шум|тихо|громк"],
    "Завтраки":       [r"завтрак|ланч-?бокс|ресторан"],
    "Чек-ин/вход":    [r"заселен|заезд|чек-?ин|вход|инструкц|селф|код|ключ|карта"],
    "Wi-Fi/интернет": [r"wi-?fi|wifi|вай-?фай|интернет|парол"],
    "Цена/ценность":  [r"цена|стоимост|дорог|дешев|соотношен.*цена"],
}

def sentiment_sign(text: str, rating10: float):
    t = str(text)
    pos = len(POS_LEX.findall(t))
    neg = len(NEG_LEX.findall(t)) + len(NEG_CUES.findall(t))
    if neg - pos >= 2: return "negative"
    if pos - neg >= 1 and neg == 0: return "positive"
    if rating10 is not None and not (isinstance(rating10, float) and math.isnan(rating10)):
        if rating10 < 6: return "negative"
        if rating10 >= 8: return "positive"
        return "neutral"
    return "neutral"

def sentence_split(text: str):
    parts = re.split(r"(?<=[\.\!\?\n])\s+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip())>=12]

def classify_topics(text: str):
    tl = str(text).lower()
    hits = set()
    for tp, pats in TOPIC_PATTERNS.items():
        for p in pats:
            if re.search(p, tl, re.IGNORECASE):
                hits.add(tp); break
    return hits

def summarize_topics(wk_df: pd.DataFrame):
    if wk_df.empty:
        return pd.DataFrame(columns=["topic","mentions","share","avg10","neg_text_share"]), {}
    rows=[]; quotes={}
    total = len(wk_df)
    for idx, r in wk_df.iterrows():
        t = str(r["Текст"])
        found = classify_topics(t)
        if not found: continue
        for tp in found:
            rows.append({"topic": tp, "idx": idx, "rating10": r["_rating10"], "sent": sentiment_sign(t, r["_rating10"])})
            for s in sentence_split(t):
                if any(re.search(p, s, re.IGNORECASE) for p in TOPIC_PATTERNS[tp]):
                    qm = quotes.setdefault(tp, {"pos": None, "neg": None})
                    sg = sentiment_sign(s, r["_rating10"])
                    if sg=="positive" and not qm["pos"]: qm["pos"]=s
                    if sg=="negative" and not qm["neg"]: qm["neg"]=s
    if not rows:
        return pd.DataFrame(columns=["topic","mentions","share","avg10","neg_text_share"]), quotes
    tmp = pd.DataFrame(rows)
    agg = (tmp.groupby("topic")
             .agg(mentions=("topic","size"),
                  avg10=("rating10","mean"),
                  neg_text_share=("sent", lambda s: (s=="negative").mean()))
             .reset_index())
    agg["share"]          = (agg["mentions"]/total*100).round(1)
    agg["avg10"]          = agg["avg10"].round(2)
    agg["neg_text_share"] = (agg["neg_text_share"]*100).round(1)
    return agg.sort_values("mentions", ascending=False), quotes

# ===================
# Week calc & append
# ===================
def last_week_range(today: date):
    monday = today - dt.timedelta(days=today.weekday())
    end = monday - dt.timedelta(days=1)
    start = end - dt.timedelta(days=6)
    return start, end

def analyze_week(df: pd.DataFrame, start: date, end: date):
    # даты в файле: ДД.ММ.ГГГГ
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", dayfirst=True).dt.date

    # срез прошлой недели
    wk = df[(df["Дата"] >= start) & (df["Дата"] <= end)].copy()

    # нормализуем рейтинг и считаем сентимент
    wk["_rating10"] = normalize_to10(wk["Рейтинг"])
    wk["_sent"] = wk.apply(lambda r: sentiment_sign(r["Текст"], r["_rating10"]), axis=1)

    # агрегаты недели
    agg = {
        "reviews": int(len(wk)),
        "avg10": round(float(wk["_rating10"].mean()), 2) if len(wk) else None,
        "pos": int((wk["_sent"] == "positive").sum()),
        "neu": int((wk["_sent"] == "neutral").sum()),
        "neg": int((wk["_sent"] == "negative").sum()),
    }
    return wk, agg

def ensure_storage_tabs():
    ensure_tab(HISTORY_SHEET_ID, HISTORY_TAB, ["period_type","period_key","reviews","avg10","pos","neu","neg"])
    ensure_tab(HISTORY_SHEET_ID, SOURCES_TAB, ["week_key","source","reviews","avg10","pos","neu","neg"])
    ensure_tab(HISTORY_SHEET_ID, TOPICS_TAB,   ["week_key","topic","mentions","share_pct","avg10","neg_text_share_pct","by_channel_json"])

def append_history_week(week_start: date, agg: dict):
    iso_y, iso_w, _ = week_start.isocalendar()
    week_key = f"{iso_y}-W{iso_w}"
    hist = gs_get_df(HISTORY_TAB, "A:G")
    if len(hist[(hist["period_type"]=="week") & (hist["period_key"]==week_key)])==0:
        rows = [["week", week_key, agg["reviews"], agg["avg10"], agg["pos"], agg["neu"], agg["neg"]]]
        gs_append(HISTORY_TAB, "A:G", rows)
    return week_key

def append_sources_week(week_key: str, wk_df: pd.DataFrame):
    exist = gs_get_df(SOURCES_TAB, "A:G")
    have = set(zip(exist.get("week_key",[]), exist.get("source",[]))) if not exist.empty else set()
    rows=[]
    for src, sub in wk_df.groupby("Источник"):
        if (week_key, src) in have: continue
        n = int(len(sub))
        avg10 = round(float(sub["_rating10"].mean()), 2) if n else None
        pos = int((sub["_sent"]=="positive").sum())
        neu = int((sub["_sent"]=="neutral").sum())
        neg = int((sub["_sent"]=="negative").sum())
        rows.append([week_key, src, n, avg10, pos, neu, neg])
    gs_append(SOURCES_TAB, "A:G", rows)


# =============
# formatters (RU)
# =============
def _comma_num(x, nd=2):
    if x is None or (isinstance(x, float) and (pd.isna(x) or np.isnan(x))):
        return "—"
    return f"{float(x):.{nd}f}".replace(".", ",")

def fmt_avg(x):   return _comma_num(x, 2)          # 9,05
def fmt_pct(x):   return "—" if x is None else f"{float(x):.1f}%".replace(".", ",")    # 91,7%
def fmt_pp(x):    return "—" if x is None else f"{float(x):+.2f} п.п.".replace(".", ",")
def fmt_int(x):   return "—" if x is None else str(int(x))

# --- источники, которые показываем в 5-балльной шкале (для всех периодов в таблице)
SOURCES_5PT = {
    "TL: Marketing", "Yandex", "Яндекс Путешествия", "TripAdvisor", "2GIS", "Google", "Trip.com"
}

def normalize_source_name(s: str) -> str:
    if s is None: return ""
    t = str(s).strip().lower()
    if "yandex" in t or "яндекс" in t:   # объединяем все «яндексы»
        return "Yandex"
    return str(s).strip()

def unify_sources_df(df: pd.DataFrame) -> pd.DataFrame:
    """Склеиваем 'Yandex' и 'Яндекс Путешествия' в один источник 'Yandex' с корректной взвешенной средней."""
    if df is None or df.empty:
        return df
    x = df.copy()
    x["source"] = x["source"].map(normalize_source_name)
    for c in ["reviews","avg10","pos","neu","neg"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    def ag(g):
        n = int(g["reviews"].sum())
        pos = int(g.get("pos",0).sum()); neu = int(g.get("neu",0).sum()); neg = int(g.get("neg",0).sum())
        avg = _weighted_avg(g["avg10"], g["reviews"])
        return pd.Series({
            "reviews": n,
            "avg10": avg,
            "pos": pos, "neu": neu, "neg": neg,
            "pos_share": _safe_pct(pos, n), "neg_share": _safe_pct(neg, n),
        })
    out = (x.groupby("source", as_index=False).apply(ag).reset_index(drop=True))
    # сортируем по объёму → средней
    return out.sort_values(["reviews","avg10"], ascending=[False, False])

def display_avg_for_source(avg10: float, src: str) -> str:
    """Для ряда источников отображаем в /5 (для читателя), расчёты остаются в /10."""
    if avg10 is None or (isinstance(avg10, float) and (pd.isna(avg10) or np.isnan(avg10))):
        return "—"
    if normalize_source_name(src) in SOURCES_5PT:
        return _comma_num(float(avg10)/2.0, 2)  # 10→5
    return fmt_avg(avg10)


def header_human_block(week_start: date, labels: dict, week_agg: dict, mtd_agg: dict, qtd_agg: dict, ytd_agg: dict, deltas: dict):
    s, e = week_start, week_start + dt.timedelta(days=6)

    def cmp_line(name_ru, lab, agg, d):
        return (
          f"<p><b>{name_ru} — {lab}.</b> "
          f"Средняя оценка: <b>{fmt_avg(agg['avg10'])}/10</b>; "
          f"доля позитива: <b>{fmt_pct(agg['pos_share'])}</b>; "
          f"негатива: <b>{fmt_pct(agg['neg_share'])}</b>."
          f"<br>Неделя относительно {name_ru.lower()}: "
          f"{'выше' if d['avg10_delta'] and d['avg10_delta']>0 else ('ниже' if d['avg10_delta'] and d['avg10_delta']<0 else 'на уровне')} по средней ({fmt_avg(d['avg10_delta'])}), "
          f"изменение позитива {fmt_pp(d['pos_delta_pp'])}, негатива {fmt_pp(d['neg_delta_pp'])}; "
          f"на долю недели пришлось {fmt_pct(d['week_share_pct'])} отзывов {name_ru.lower()}."
          f"</p>"
        )

    html = (
      f"<h2>ARTSTUDIO Nevsky — отзывы за неделю {week_label(s, e)}</h2>"
      f"<p><b>Итоги недели:</b> {fmt_int(week_agg['reviews'])} отзывов; "
      f"средняя <b>{fmt_avg(week_agg['avg10'])}/10</b>; "
      f"позитив {fmt_pct(100.0*week_agg['pos']/week_agg['reviews']) if week_agg['reviews'] else '—'}, "
      f"негатив {fmt_pct(100.0*week_agg['neg']/week_agg['reviews']) if week_agg['reviews'] else '—'}."
      f"</p>"
      + cmp_line("Текущий месяц",   labels['mtd'], mtd_agg, deltas['mtd'])
      + cmp_line("Текущий квартал", labels['qtd'], qtd_agg, deltas['qtd'])
      + cmp_line("Текущий год",     labels['ytd'], ytd_agg, deltas['ytd'])
    )
    return html

def topics_table_html(topics_df: pd.DataFrame, prev_topics_df: pd.DataFrame | None):
    prev = prev_topics_df.set_index("topic") if (prev_topics_df is not None and not prev_topics_df.empty) else None
    rows=[]
    if topics_df.empty:
        return "<h3>Темы недели</h3><p>Нет данных</p>"
    for _, r in topics_df.head(10).iterrows():
        d_m = d_n = ""
        if prev is not None and r["topic"] in prev.index:
            dm = int(r["mentions"] - prev.loc[r["topic"], "mentions"])
            dn = float(r["neg_text_share"] - prev.loc[r["topic"], "neg_text_share"])
            d_m = f" ({'+' if dm>=0 else ''}{dm})"
            d_n = f" ({'+' if dn>=0 else ''}{round(dn,1)} п.п.)"
        rows.append(
            f"<tr><td>{r['topic']}</td><td>{int(r['mentions'])}{d_m}</td>"
            f"<td>{r['share']}%</td><td>{fmt_avg(r['avg10'])}</td><td>{r['neg_text_share']}%{d_n}</td></tr>"
        )
    return ("<h3>Темы недели</h3>"
            "<table border='1' cellspacing='0' cellpadding='6'>"
            "<tr><th>Тема</th><th>Упоминания</th><th>Доля</th><th>Средняя /10</th><th>Негатив по тексту</th></tr>"
            + "".join(rows) + "</table>")

def quotes_block(quotes: dict):
    parts=[]
    for tp, q in quotes.items():
        line=[]
        if q.get("pos"): line.append(f"<b>+</b> «{q['pos']}»")
        if q.get("neg"): line.append(f"<b>–</b> «{q['neg']}»")
        if line: parts.append(f"<p><b>{tp}:</b> " + " / ".join(line) + "</p>")
    return ("<h3>Цитаты</h3>" + "".join(parts)) if parts else ""

def sources_table_block(summ: dict, wk_src_avg: dict):
    """Неделя/Месяц/Квартал/Год + Итог за всё время (оценка), с отображением /5 для ряда источников."""
    def to_map(df):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return {}
        out = {}
        for _, r in df.iterrows():
            src = str(r["source"])
            out[src] = {
                "reviews": int(r["reviews"]) if pd.notna(r["reviews"]) else 0,
                "avg10": float(r["avg10"]) if pd.notna(r["avg10"]) else None,
                "pos": float(r.get("pos_share")) if pd.notna(r.get("pos_share")) else None,
                "neg": float(r.get("neg_share")) if pd.notna(r.get("neg_share")) else None,
            }
        return out

    W = to_map(summ.get("week"))
    M = to_map(summ.get("mtd"))
    Q = to_map(summ.get("qtd"))
    Y = to_map(summ.get("ytd"))
    A = to_map(summ.get("alltime"))  # итог за всё время (оценка)

    # если где-то нет avg10, но есть недельная средняя — подставим
    def fill_avg_with_week(d):
        for k, v in d.items():
            if (v.get("avg10") is None or (isinstance(v["avg10"], float) and np.isnan(v["avg10"]))) and k in wk_src_avg:
                v["avg10"] = float(wk_src_avg[k])
    for D in (W, M, Q, Y):
        fill_avg_with_week(D)

    all_sources = sorted(set(W) | set(M) | set(Q) | set(Y) | set(A))

    def cell(d, src):
        if src not in d:
            return "<td>—</td><td>—</td><td>—</td><td>—</td>"
        v = d[src]
        return (
            f"<td>{display_avg_for_source(v['avg10'], src)}</td>"
            f"<td>{fmt_int(v['reviews'])}</td>"
            f"<td>{fmt_pct(v['pos'])}</td>"
            f"<td>{fmt_pct(v['neg'])}</td>"
        )

    def cell_all(src):
        if src not in A: return "<td>—</td>"
        return f"<td><b>{display_avg_for_source(A[src]['avg10'], src)}</b></td>"

    rows = [
        f"<tr><td><b>{s}</b></td>{cell(W,s)}{cell(M,s)}{cell(Q,s)}{cell(Y,s)}{cell_all(s)}</tr>"
        for s in all_sources
    ]

    return f"""
    <h3>Источники (по периодам)</h3>
    <p>Неделя: {summ['labels']['week']} • Месяц: {summ['labels']['mtd']} • Квартал: {summ['labels']['qtd']} • Год: {summ['labels']['ytd']}</p>
    <table border='1' cellspacing='0' cellpadding='6'>
      <tr>
        <th rowspan="2">Источник</th>
        <th colspan="4">Неделя</th>
        <th colspan="4">Текущий месяц</th>
        <th colspan="4">Текущий квартал</th>
        <th colspan="4">Текущий год</th>
        <th rowspan="2">Итог (всё время)</th>
      </tr>
      <tr>
        <th>Ср.</th><th>Отз.</th><th>Позитив</th><th>Негатив</th>
        <th>Ср.</th><th>Отз.</th><th>Позитив</th><th>Негатив</th>
        <th>Ср.</th><th>Отз.</th><th>Позитив</th><th>Негатив</th>
        <th>Ср.</th><th>Отз.</th><th>Позитив</th><th>Негатив</th>
      </tr>
      {''.join(rows) if rows else '<tr><td colspan="21">Нет данных</td></tr>'}
    </table>
    """

def sources_deltas_block(summ: dict, wk_src_avg: dict):
    """Дельты средних /10 по источникам: месяц/квартал/год vs прошлые периоды и vs 'всё время'."""
    prevM = summ.get("prev_month"); prevQ = summ.get("prev_quarter"); prevY = summ.get("prev_year")
    ALL   = summ.get("alltime")

    def to_map(df):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return {}
        m = {normalize_source_name(r["source"]): (float(r["avg10"]) if pd.notna(r["avg10"]) else None) for _, r in df.iterrows()}
        # фоллбек на недельную среднюю, если совсем пусто
        for k, v in m.items():
            if v is None and k in wk_src_avg:
                m[k] = float(wk_src_avg[k])
        return m

    M, PM = to_map(summ.get("mtd")),  to_map(prevM)
    Q, PQ = to_map(summ.get("qtd")),  to_map(prevQ)
    Y, PY = to_map(summ.get("ytd")),  to_map(prevY)
    A     = to_map(ALL)

    all_sources = sorted(set(M) | set(PM) | set(Q) | set(PQ) | set(Y) | set(PY) | set(A))

    def dd(a, b):
        if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
            return "—"
        return f"{(float(a) - float(b)):+.2f}".replace(".", ",")

    rows = []
    for s in all_sources:
        rows.append(
            "<tr>"
            f"<td><b>{s}</b></td>"
            f"<td>{dd(M.get(s), PM.get(s))}</td>"
            f"<td>{dd(Q.get(s), PQ.get(s))}</td>"
            f"<td>{dd(Y.get(s), PY.get(s))}</td>"
            f"<td>{dd(M.get(s), A.get(s))}</td>"
            f"<td>{dd(Q.get(s), A.get(s))}</td>"
            f"<td>{dd(Y.get(s), A.get(s))}</td>"
            "</tr>"
        )

    return f"""
    <h3>Дельты по источникам</h3>
    <p>Сравнение средних /10: текущий месяц vs предыдущий месяц; текущий квартал vs предыдущий квартал; текущий год vs предыдущий год; а также каждая величина относительно общей оценки за всё время.</p>
    <table border='1' cellspacing='0' cellpadding='6'>
      <tr>
        <th rowspan="2">Источник</th>
        <th colspan="3">Δ к прошлым периодам</th>
        <th colspan="3">Δ к общей оценке (всё время)</th>
      </tr>
      <tr>
        <th>Месяц</th><th>Квартал</th><th>Год</th>
        <th>Месяц→Общая</th><th>Квартал→Общая</th><th>Год→Общая</th>
      </tr>
      {''.join(rows) if rows else '<tr><td colspan="7">Нет данных</td></tr>'}
    </table>
    """

def trends_text_block(history_df: pd.DataFrame, week_agg: dict, w_start: date):
    wk = history_df[history_df["period_type"] == "week"].copy()
    if wk.empty:
        return ""

    # аккуратно приводим к числам (учитываем '9,05')
    wk["reviews"] = pd.to_numeric(wk["reviews"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    wk["avg10"]   = pd.to_numeric(wk["avg10"].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # последние 8 недель (включая текущую, если уже есть в history)
    def worder(k):
        try:
            y, w = str(k).split("-W")
            return int(y) * 100 + int(w)
        except:
            return 0
    wk = wk.sort_values(by="period_key", key=lambda s: s.map(worder)).tail(8)

    base_avg = wk["avg10"].mean(skipna=True)
    base_cnt = wk["reviews"].mean(skipna=True)

    delta_avg = None
    if week_agg.get("avg10") is not None and pd.notna(base_avg):
        delta_avg = round(float(week_agg["avg10"]) - float(base_avg), 2)

    delta_cnt = None
    if week_agg.get("reviews") is not None and pd.notna(base_cnt):
        delta_cnt = round(float(week_agg["reviews"]) - float(base_cnt), 1)

    trend = "на уровне" if (delta_avg is None or abs(delta_avg) < 0.01) else ("выше" if delta_avg > 0 else "ниже")

    return (
        "<h3>Тренды и динамика</h3>"
        f"<p>Средняя оценка недели {trend} средней последних 8 недель "
        f"({fmt_avg(base_avg)}) на {fmt_avg(delta_avg)}; "
        f"объём отзывов {('выше' if (delta_cnt is not None and delta_cnt>0) else ('ниже' if (delta_cnt is not None and delta_cnt<0) else 'на уровне'))} "
        f"среднего за 8 недель ({fmt_int(base_cnt)}) на {fmt_int(delta_cnt)}.</p>"
    )

# ===========
# Charts
# ===========
def plot_ratings_reviews_trend(history_df, path_png):
    wk = history_df[history_df["period_type"]=="week"].copy()
    if wk.empty: return None
    for c in ["reviews","avg10"]:
        wk[c] = pd.to_numeric(wk[c], errors="coerce")
    def worder(k):
        try:
            y,w = str(k).split("-W"); return int(y)*100+int(w)
        except: return 0
    wk = wk.sort_values(by="period_key", key=lambda s: s.map(worder)).tail(8)
    if wk.empty: return None
    x = wk["period_key"]; y1=wk["avg10"]; y2=wk["reviews"]
    fig, ax1 = plt.subplots(figsize=(9,4.5))
    ax1.plot(x, y1, marker="o")
    ax1.set_ylabel("Средняя /10")
    ax1.set_title("Динамика: средняя /10 и объём отзывов (8 недель)")
    ax2 = ax1.twinx()
    ax2.bar(x, y2, alpha=0.25)
    ax2.set_ylabel("Отзывы")
    plt.xticks(rotation=45); fig.tight_layout()
    plt.savefig(path_png); plt.close(); return path_png

def plot_topics_trend(topics_hist, path_png):
    if topics_hist.empty: return None
    def worder(k):
        try: y,w=str(k).split("-W"); return int(y)*100+int(w)
        except: return 0
    recent = sorted(topics_hist["week_key"].unique(), key=worder)[-8:]
    th = topics_hist[topics_hist["week_key"].isin(recent)].copy()
    if th.empty: return None
    top_topics = th.groupby("topic")["mentions"].sum().sort_values(ascending=False).head(5).index.tolist()
    pv = th[th["topic"].isin(top_topics)].pivot_table(index="week_key", columns="topic", values="mentions", aggfunc="sum").fillna(0)
    plt.figure(figsize=(9,5))
    for col in pv.columns:
        plt.plot(pv.index, pv[col], marker="o", label=col)
    plt.legend(); plt.title("Тренды тем (8 недель)"); plt.ylabel("Упоминания"); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(path_png); plt.close(); return path_png

def plot_sources_trends(sources_hist, path_avg_png, path_vol_png):
    if sources_hist.empty: return None, None
    def worder(k):
        try: y,w=str(k).split("-W"); return int(y)*100+int(w)
        except: return 0
    sh = sources_hist.copy()
    for c in ["reviews","avg10"]:
        sh[c] = pd.to_numeric(sh[c], errors="coerce")
    top_src = (sh.groupby("source")["reviews"].sum().sort_values(ascending=False).head(5).index.tolist())
    recent = sorted(sh["week_key"].unique(), key=worder)[-8:]
    sh = sh[(sh["source"].isin(top_src)) & (sh["week_key"].isin(recent))]

    if sh.empty: return None, None
    pv_avg = sh.pivot_table(index="week_key", columns="source", values="avg10", aggfunc="mean")
    pv_cnt = sh.pivot_table(index="week_key", columns="source", values="reviews", aggfunc="sum")
    # avg
    plt.figure(figsize=(9,5))
    for col in pv_avg.columns:
        plt.plot(pv_avg.index, pv_avg[col], marker="o", label=col)
    plt.legend(); plt.title("Средняя /10 по источникам (8 недель)"); plt.ylabel("Средняя /10"); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(path_avg_png); plt.close()
    # volumes
    plt.figure(figsize=(9,5))
    bottom = np.zeros(len(pv_cnt.index))
    for col in pv_cnt.columns:
        plt.bar(pv_cnt.index, pv_cnt[col].values, bottom=bottom, label=col)
        bottom += pv_cnt[col].fillna(0).values
    plt.legend(); plt.title("Объём отзывов по источникам (8 недель, stacked)"); plt.ylabel("Отзывы"); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(path_vol_png); plt.close()
    return path_avg_png, path_vol_png

# ===========
# Email
# ===========
def attach_file(msg, path):
    if not path or not os.path.exists(path): return
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None: ctype='application/octet-stream'
    maintype, subtype = ctype.split('/',1)
    with open(path,"rb") as fp:
        part = MIMEBase(maintype, subtype); part.set_payload(fp.read()); encoders.encode_base64(part)
        part.add_header('Content-Disposition','attachment', filename=os.path.basename(path)); msg.attach(part)

def send_email(subject, html_body, attachments=None):
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject; msg["From"] = SMTP_FROM; msg["To"] = ", ".join(RECIPIENTS)
    alt = MIMEMultipart("alternative"); alt.attach(MIMEText(html_body, "html", "utf-8")); msg.attach(alt)
    for p in (attachments or []): attach_file(msg, p)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.sendmail(msg["From"], RECIPIENTS, msg.as_string())

# ===========
# MAIN
# ===========
def main():
    # 0) подготовка
    ensure_storage_tabs()

    # 1) забираем последний файл отзывов с диска и парсим прошлую неделю
    fid, fname, fdate = latest_reviews_file_from_drive()
    blob = drive_download(fid); tmp = "/tmp/reviews.xls"; open(tmp,"wb").write(blob)

    today = dt.date.today()
    w_start, w_end = last_week_range(today)

    df = load_reviews_df(tmp)
    wk_df, week_agg = analyze_week(df, w_start, w_end)
    wk_src_avg = (
        wk_df.groupby("Источник")["_rating10"].mean().round(2).to_dict()
    )
    week_key = append_history_week(w_start, week_agg)
    append_sources_week(week_key, wk_df)  # обновляем sources_history для недели

    # 2) читаем историю и строим агрегаты периода «на конец недели»
    hist_df     = gs_get_df(HISTORY_TAB, "A:G")
    topics_hist = gs_get_df(TOPICS_TAB,   "A:G")
    sources_hist= gs_get_df(SOURCES_TAB,  "A:G")

    ranges = period_ranges_for_week(w_start)
    labels = {k: v["label"] for k,v in ranges.items()}

    # агрегаты из недель
    mtd_agg = aggregate_weeks_from_history(hist_df, ranges["mtd"]["start"], ranges["mtd"]["end"])
    qtd_agg = aggregate_weeks_from_history(hist_df, ranges["qtd"]["start"], ranges["qtd"]["end"])
    ytd_agg = aggregate_weeks_from_history(hist_df, ranges["ytd"]["start"], ranges["ytd"]["end"])

    deltas = {
        "mtd": deltas_week_vs_period(week_agg, mtd_agg),
        "qtd": deltas_week_vs_period(week_agg, qtd_agg),
        "ytd": deltas_week_vs_period(week_agg, ytd_agg),
    }

    # 3) темы недели
    topics_df, quotes = summarize_topics(wk_df)
    # prev week topics для дельт
    prev_week_key = f"{w_start.isocalendar()[0]}-W{w_start.isocalendar()[1]-1}" if w_start.isocalendar()[1]>1 else None
    prev_topics_df = pd.DataFrame()
    if prev_week_key and not topics_hist.empty:
        th = topics_hist.rename(columns={"share_pct":"share","neg_text_share_pct":"neg_text_share"})
        prev_topics_df = th[th["week_key"]==prev_week_key][["topic","mentions","share","avg10","neg_text_share"]]

    # 4) блоки по источникам
    sources_summ = sources_summary_for_periods(sources_hist, w_start)

    # склеиваем Яндекс + Яндекс.Путешествия → Yandex
for k in ["week","mtd","qtd","ytd","prev_month","prev_quarter","prev_year"]:
    sources_summ[k] = unify_sources_df(sources_summ[k])

# добавляем «всё время» как последний столбец (для таблицы по источникам)
if not sources_hist.empty:
    tmp = sources_hist.copy()
    tmp["mon"] = tmp["week_key"].map(lambda k: iso_week_monday(str(k)))
    all_start, all_end = tmp["mon"].min(), tmp["mon"].max()
    all_df = aggregate_sources_from_history(sources_hist, all_start, all_end)
    sources_summ["alltime"] = unify_sources_df(all_df)
    sources_summ["labels"]["alltime"] = "всё время"
else:
    sources_summ["alltime"] = pd.DataFrame(columns=["source","reviews","avg10","pos_share","neg_share","pos","neu","neg"])
    sources_summ["labels"]["alltime"] = "всё время"

    
    # 5) собираем HTML
    head_html   = header_human_block(w_start, labels, week_agg, mtd_agg, qtd_agg, ytd_agg, deltas)
    topics_html = topics_table_html(topics_df, prev_topics_df)
    quotes_html = quotes_block(quotes)
    src_tbl     = sources_table_block(sources_summ, wk_src_avg)
    src_deltas  = sources_deltas_block(sources_summ, wk_src_avg)

    html = head_html + topics_html + quotes_html + src_tbl + src_deltas
    html += "<p><i>История и сравнения: листы <b>history</b>, <b>sources_history</b>, <b>topics_history</b>. Графики — во вложениях.</i></p>"

    trend_html = trends_text_block(hist_df, week_agg, w_start)
    html = head_html + trend_html + topics_html + quotes_html + src_tbl + src_deltas
    
    # 6) графики
    charts=[]
    p1="/tmp/ratings_trend.png"; plot_ratings_reviews_trend(hist_df, p1); charts.append(p1)
    p2="/tmp/topics_trends.png"; plot_topics_trend(topics_hist, p2);  charts.append(p2)
    a="/tmp/sources_avg.png"; v="/tmp/sources_vol.png"; 
    a,v = plot_sources_trends(sources_hist, a, v); 
    if a: charts.append(a)
    if v: charts.append(v)
    charts = [p for p in charts if p and os.path.exists(p)]

html += """
<hr>
<p><i>* Средняя /10 — средневзвешенная оценка по всем отзывам периода. Позитив/негатив — доля отзывов с положительной/отрицательной тональностью по тексту (и/или по оценке, если текст нейтрален). Δ — разница недели к соответствующему периоду; п.п. — процентные пункты. «На долю недели пришлось …» — вклад текущей недели в объём отзывов месяца/квартала/года.</i></p>
<p><i>** В таблице по источникам оценки для TL: Marketing, Yandex (объединяет Яндекс и Яндекс Путешествия), TripAdvisor, 2GIS, Google, Trip.com показаны в 5-балльной шкале; для остальных источников — в 10-балльной. Все расчёты выполняются в 10-балльной шкале, конвертация в 5-балльную — только для отображения.</i></p>
"""


    # 7) письмо
    subject = f"ARTSTUDIO Nevsky. Анализ отзывов за неделю {w_start:%d.%m}–{w_end:%d.%m}"
    send_email(subject, html, attachments=charts)

if __name__ == "__main__":
    main()
