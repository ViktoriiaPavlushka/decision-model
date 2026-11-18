"""Streamlit playground for the decision model.   streamlit run app.py"""
import html
import re
import uuid

import streamlit as st

from decision_model import DecisionModel

st.set_page_config(page_title="Модель рішень", page_icon=":material/rule:", layout="centered")
st.markdown("""
<style>
:root { --pink:#f3c3d3; --pink-soft:#fcedf2; --pink-line:#ecc9d6; --pink-dark:#a2546f;
        --yellow:#f6e3a1; --green:#cfe5cc; --green-dark:#4f7d52; --ink:#3f3539;
        --muted:#7a6c72; --line:#ebe1e4; }
.stApp, header[data-testid="stHeader"] { background:#fcfaf8 !important; color:var(--ink); }
h1,h2,h3,p,label,li,.stMarkdown { color:var(--ink); }
textarea, input { -webkit-text-fill-color:var(--ink) !important; background:#fff !important; }
textarea::placeholder, input::placeholder { -webkit-text-fill-color:#a8999f !important; }
.block-container { max-width:860px; padding-top:3rem; padding-bottom:3rem; }
.header { padding:0 0 .8rem 1rem; border-left:4px solid var(--pink); margin-bottom:1.4rem; }
.header h1 { font-size:1.7rem; margin:0; }
.header p { color:var(--muted); margin:.2rem 0 0; }
.section { font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
           font-weight:600; margin:1.4rem 0 .5rem; }
.hint { color:var(--muted); font-size:.82rem; margin:-.3rem 0 .4rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:var(--line) !important;
           border-radius:4px !important; }
.card { background:#fff; border:1px solid var(--line); border-left:4px solid var(--yellow);
        border-radius:4px; padding:.85rem 1.05rem; margin-bottom:.8rem; }
.card.noul { border-left-color:var(--green); }
.card.choice { border-left-color:var(--pink); }
.card .top { display:flex; justify-content:space-between; align-items:baseline; gap:1rem; }
.card h4 { margin:0; font-size:1rem; }
.card .tag { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }
.card .q { color:var(--muted); font-size:.88rem; margin:.15rem 0 .55rem; }
.card .big { font-size:1.3rem; font-weight:600; }
.card .meta { color:var(--muted); font-size:.85rem; margin-bottom:.45rem; }
.row { display:flex; align-items:center; gap:.7rem; font-size:.86rem; margin:.25rem 0; }
.row .k { width:42%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.row.win .k { font-weight:600; }
.row .bar { flex:1; height:8px; background:#f4f0ee; border-radius:2px; }
.row .bar div { height:8px; background:var(--pink); border-radius:2px; }
.row.win .bar div { background:var(--pink-dark); }
.row .v { width:2.8rem; text-align:right; font-variant-numeric:tabular-nums; }
.conf { display:inline-block; font-size:.75rem; padding:.05rem .45rem; border-radius:3px; margin-left:.5rem;
        vertical-align:middle; }
.conf.hi { background:var(--green); color:var(--green-dark); }
.conf.mid { background:#fbf1cf; color:#7d6a2c; }
.conf.lo { background:var(--pink-soft); color:var(--pink-dark); }
div.stButton > button { border-radius:4px; border:1px solid var(--pink-line); color:var(--ink);
        background:#fff; }
div.stButton > button[kind="primary"] { background:var(--pink); border-color:var(--pink-line); color:var(--ink); }
</style>
<div class="header"><h1>Модель рішень</h1>
<p>Відповідає на ваші запитання про текст і показує, наскільки вона впевнена у відповіді</p></div>
""", unsafe_allow_html=True)

TYPES = {"choice": "Вибір варіанта", "score": "Оцінка за шкалою", "noul": "Так / ні"}
OPTION_HINT = {
    "choice": "Кожен варіант з нового рядка у форматі «Назва — опис». Опис можна не писати.",
    "score": "Рівні шкали з нового рядка, від найнижчого до найвищого.",
}
EXAMPLE_TEXT = ("Вітаю! З мене двічі списали оплату за місячну підписку. Поверніть, будь ласка, зайве. "
                "Акаунт працює нормально, але я вже третій день не можу до вас додзвонитися.")
EXAMPLE_QUESTIONS = [
    {"name": "Відділ", "type": "choice", "instructions": "Яка команда має обробити цей запит?",
     "options": "Оплати — Рахунки, оплати, повернення коштів\n"
                "Технічна підтримка — Баги, збої, помилки системи\n"
                "Продажі — Ціни, нові контракти\n"
                "Інше — Усе інше"},
    {"name": "Роздратування", "type": "score", "instructions": "Наскільки клієнт роздратований?",
     "options": "Спокійний, нейтральний тон\nПомірне невдоволення\nСильне роздратування або погрози піти"},
    {"name": "Повернення коштів", "type": "noul", "instructions": "Клієнт явно просить повернути кошти?",
     "options": ""},
]


@st.cache_resource(show_spinner="Завантаження моделі...")
def load():
    return DecisionModel("artifacts/heads.joblib")


# ------------------------------------------------------------------ state
def new_question(q=None):
    qid = uuid.uuid4().hex[:8]
    q = q or {"name": "", "type": "noul", "instructions": "", "options": ""}
    st.session_state[f"name_{qid}"] = q["name"]
    st.session_state[f"type_{qid}"] = q["type"]
    st.session_state[f"instr_{qid}"] = q["instructions"]
    st.session_state[f"opts_{qid}"] = q["options"]
    return qid


if "qids" not in st.session_state:
    st.session_state.text = EXAMPLE_TEXT
    st.session_state.qids = [new_question(q) for q in EXAMPLE_QUESTIONS]


def reset_example():
    st.session_state.text = EXAMPLE_TEXT
    st.session_state.qids = [new_question(q) for q in EXAMPLE_QUESTIONS]
    st.session_state.pop("result", None)


def add_question():
    st.session_state.qids.append(new_question())


def remove_question(qid):
    st.session_state.qids.remove(qid)


def parse_options(qtype, raw):
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if qtype == "score":
        return lines
    out = {}
    for l in lines:
        parts = re.split(r"\s+[—–-]\s+|:\s+", l, maxsplit=1)
        out[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""
    return out


# ------------------------------------------------------------------ input
st.markdown('<div class="section">Текст</div>', unsafe_allow_html=True)
st.text_area("Текст", key="text", height=110, label_visibility="collapsed",
             placeholder="Вставте текст, про який треба прийняти рішення")

st.markdown('<div class="section">Запитання</div>', unsafe_allow_html=True)
for n, qid in enumerate(st.session_state.qids, 1):
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2.2, 0.6], vertical_alignment="bottom")
        c1.text_input("Назва", key=f"name_{qid}", placeholder=f"Запитання {n}")
        c2.selectbox("Тип", list(TYPES), format_func=TYPES.get, key=f"type_{qid}")
        c3.button("", icon=":material/delete:", key=f"del_{qid}", help="Видалити запитання",
                  on_click=remove_question, args=(qid,))
        st.text_input("Запитання", key=f"instr_{qid}", placeholder="Наприклад: Клієнт просить повернути кошти?")
        qtype = st.session_state[f"type_{qid}"]
        if qtype != "noul":
            st.markdown(f'<div class="hint">{OPTION_HINT[qtype]}</div>', unsafe_allow_html=True)
            st.text_area("Варіанти", key=f"opts_{qid}", height=110, label_visibility="collapsed")

b1, b2, b3 = st.columns([1.4, 1.6, 3])
b1.button("Додати запитання", icon=":material/add:", on_click=add_question)
b2.button("Відновити приклад", icon=":material/restart_alt:", on_click=reset_example)
run = b3.button("Отримати відповіді", type="primary", use_container_width=True)


# ------------------------------------------------------------------ run
def build_questions():
    qs, meta, errors = {}, {}, []
    for n, qid in enumerate(st.session_state.qids, 1):
        name = st.session_state[f"name_{qid}"].strip() or f"Запитання {n}"
        qtype = st.session_state[f"type_{qid}"]
        instr = st.session_state[f"instr_{qid}"].strip()
        if not instr:
            errors.append(f"«{name}»: напишіть текст запитання.")
            continue
        q = {"type": qtype, "instructions": instr}
        if qtype != "noul":
            q["criteria"] = parse_options(qtype, st.session_state[f"opts_{qid}"])
            if len(q["criteria"]) < 2:
                errors.append(f"«{name}»: потрібно щонайменше два варіанти.")
                continue
        while name in qs:
            name += " "
        qs[name] = q
        meta[name] = q
    return qs, meta, errors


if run:
    text = st.session_state.text.strip()
    qs, meta, errors = build_questions()
    if not text:
        errors.insert(0, "Введіть текст.")
    if not qs and not errors:
        errors.append("Додайте хоча б одне запитання.")
    if errors:
        for e in errors:
            st.warning(e)
        st.session_state.pop("result", None)
    else:
        with st.spinner("Аналізую текст..."):
            st.session_state.result = (load().predict(text, qs)["answers"], meta)


# ------------------------------------------------------------------ output
def conf_badge(c):
    cls, word = ("hi", "висока") if c >= .8 else ("mid", "середня") if c >= .55 else ("lo", "низька")
    return f'<span class="conf {cls}">{word} впевненість</span>'


def bars(items, winner):
    out = []
    for label, v in items:
        w = " win" if label == winner else ""
        lab = html.escape(label)
        out.append(f'<div class="row{w}"><div class="k" title="{lab}">{lab}</div>'
                   f'<div class="bar"><div style="width:{100 * v:.1f}%"></div></div>'
                   f'<div class="v">{round(100 * v)}%</div></div>')
    return "".join(out)


if "result" in st.session_state:
    answers, meta = st.session_state.result
    st.markdown('<div class="section">Відповіді</div>', unsafe_allow_html=True)
    for name, a in answers.items():
        instr = html.escape(meta[name]["instructions"])
        if a["type"] == "noul":
            p = a["noul"]
            yes = p >= .5
            conf = p if yes else 1 - p
            body = (f'<div class="big">{"Так" if yes else "Ні"}{conf_badge(conf)}</div>'
                    f'<div class="meta">Ймовірність відповіді «так» — {round(100 * p)}%</div>'
                    + bars([("Так", p), ("Ні", 1 - p)], "Так" if yes else "Ні"))
        elif a["type"] == "choice":
            body = (f'<div class="big">{html.escape(a["choice"])}{conf_badge(a["confidence"])}</div>'
                    f'<div class="meta">Впевненість — {round(100 * a["confidence"])}%</div>'
                    + bars(list(a["probabilities"].items()), a["choice"]))
        else:
            levels = list(a["legend"].values())
            best = max(a["probabilities"], key=a["probabilities"].get)
            body = (f'<div class="big">{html.escape(a["legend"][best])}{conf_badge(a["confidence"])}</div>'
                    f'<div class="meta">Оцінка {a["score"]:.2f} за шкалою від 0 до {len(levels) - 1}</div>'
                    + bars([(a["legend"][k], v) for k, v in a["probabilities"].items()], a["legend"][best]))
        st.markdown(f'<div class="card {a["type"]}"><div class="top"><h4>{html.escape(name.strip())}</h4>'
                    f'<span class="tag">{TYPES[a["type"]]}</span></div>'
                    f'<div class="q">{instr}</div>{body}</div>', unsafe_allow_html=True)
