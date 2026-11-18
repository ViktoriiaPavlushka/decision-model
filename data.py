"""Build training / test examples for the decision model from public datasets.

Every example is a dict:
    {task, lang, type, held_out, state, q: {type, instructions, criteria?}, label}
label: 1/0 for noul, index of the correct option for choice/score.

In-domain tasks (train + test split):
    noul   boolq (en), ua-news "is it about X?" (uk), ukr-pravda summary match (uk),
           ag_news "is it about X?" (en)
    choice ag_news 4 topics (en), ua-news 4 topics (uk), yahoo answers 10 topics (en)
    score  amazon reviews 1-5 stars (en)
Held-out tasks (test only, never seen in training -> zero-shot generalisation):
    noul   ukr toxicity (uk)
    choice dbpedia 14 classes (en), emotion 6 classes (en)
    score  SST-5 sentiment (en)
"""
from __future__ import annotations

import random

from datasets import load_dataset

SEED = 42


def _cut(t: str, n: int = 900) -> str:
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "..."


def _choice_q(rng, instr_variants, options: dict, drop_desc_p=0.25):
    keys = list(options)
    rng.shuffle(keys)                                  # option order must not matter
    drop = rng.random() < drop_desc_p                  # sometimes only bare labels
    crit = {k: ("" if drop else options[k]) for k in keys}
    return {"type": "choice", "instructions": rng.choice(instr_variants), "criteria": crit}


def _sample(ds, n, rng):
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return ds.select(idx[:n])


# --------------------------------------------------------------------------- noul
def boolq(split, n, rng):
    ds = _sample(load_dataset("google/boolq", split=split), n, rng)
    for r in ds:
        q = r["question"].strip()
        yield dict(task="boolq", lang="en", type="noul", state=_cut(r["passage"]),
                   q={"type": "noul", "instructions": q[0].upper() + q[1:] + "?"},
                   label=int(r["answer"]))


UA_TOPICS = {
    "політика": "Політика, влада, вибори, міжнародні відносини",
    "спорт": "Спортивні змагання, команди, спортсмени",
    "бізнес": "Економіка, компанії, фінанси, ринки",
    "технології": "Технології, гаджети, IT, наука",
}
UA_TOPIC_NOUL = {
    "політика": ["Чи йдеться в цій новині про політику?", "Ця новина стосується політики або влади?"],
    "спорт": ["Чи йдеться в цій новині про спорт?", "Це спортивна новина?"],
    "бізнес": ["Чи йдеться в цій новині про бізнес або економіку?", "Ця новина про економіку чи фінанси?"],
    "технології": ["Чи йдеться в цій новині про технології?", "Ця новина стосується IT або гаджетів?"],
}


def _ua_news(split, n, rng):
    ds = load_dataset("FIdo-AI/ua-news", split=split).filter(lambda r: r["target"] in UA_TOPICS)
    return _sample(ds, n, rng)


def ua_news_noul(split, n, rng):
    for r in _ua_news(split, n, rng):
        pos = rng.random() < 0.5
        topic = r["target"] if pos else rng.choice([t for t in UA_TOPICS if t != r["target"]])
        yield dict(task="ua_news_noul", lang="uk", type="noul",
                   state=_cut(r["title"] + ". " + r["text"]),
                   q={"type": "noul", "instructions": rng.choice(UA_TOPIC_NOUL[topic])},
                   label=int(pos))


def ukr_summary_noul(split, n, rng):
    ds = load_dataset("d0p3/ukr-pravda-news-summary", split="train")
    half = len(ds) // 2                                   # train/test by index range
    ds = ds.select(range(0, half)) if split == "train" else ds.select(range(half, len(ds)))
    ds = _sample(ds, 2 * n, rng)
    rows = list(ds)
    for i in range(n):
        r = rows[i]
        pos = rng.random() < 0.5
        summ = r["summary"] if pos else rows[n + i]["summary"]
        tmpl = rng.choice(["Чи правдиве щодо цього тексту твердження: «{}»?",
                           "Чи відповідає тексту таке твердження: «{}»?"])
        yield dict(task="ukr_summary_noul", lang="uk", type="noul", state=_cut(r["text"]),
                   q={"type": "noul", "instructions": tmpl.format(summ.strip())}, label=int(pos))


AG = {
    "World": "World news, politics, international affairs",
    "Sports": "Sports events, teams, athletes",
    "Business": "Business, economy, companies, markets",
    "Sci/Tech": "Science and technology",
}
AG_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def ag_news_noul(split, n, rng):
    ds = _sample(load_dataset("fancyzhx/ag_news", split=split), n, rng)
    for r in ds:
        gold = AG_NAMES[r["label"]]
        pos = rng.random() < 0.5
        topic = gold if pos else rng.choice([t for t in AG_NAMES if t != gold])
        yield dict(task="ag_news_noul", lang="en", type="noul", state=_cut(r["text"]),
                   q={"type": "noul", "instructions": f"Is this article about {AG[topic].lower()}?"},
                   label=int(pos))


def ukr_toxicity(split, n, rng):
    ds = load_dataset("ukr-detect/ukr-toxicity-dataset", split="train")
    pos = [r for r in ds if r["toxic"] == 1]
    neg = [r for r in ds if r["toxic"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    for r in pos[:n // 2] + neg[:n // 2]:
        yield dict(task="ukr_toxicity", lang="uk", type="noul", state=_cut(r["text"]),
                   q={"type": "noul",
                      "instructions": "Чи містить цей текст образи, лайку або токсичні висловлювання?"},
                   label=int(r["toxic"]))


# --------------------------------------------------------------------------- choice
def _choice_example(task, lang, state, q, gold_key):
    keys = list(q["criteria"])
    return dict(task=task, lang=lang, type="choice", state=state, q=q, label=keys.index(gold_key))


def ag_news_choice(split, n, rng):
    ds = _sample(load_dataset("fancyzhx/ag_news", split=split), n, rng)
    instr = ["What is the topic of this news article?", "Which section of a newspaper does this belong to?"]
    for r in ds:
        q = _choice_q(rng, instr, AG)
        yield _choice_example("ag_news", "en", _cut(r["text"]), q, AG_NAMES[r["label"]])


def ua_news_choice(split, n, rng):
    instr = ["Яка тематика цієї новини?", "До якої рубрики належить ця новина?"]
    for r in _ua_news(split, n, rng):
        q = _choice_q(rng, instr, UA_TOPICS)
        yield _choice_example("ua_news", "uk", _cut(r["title"] + ". " + r["text"]), q, r["target"])


YAHOO = {
    "Society & Culture": "religion, traditions, culture, society",
    "Science & Mathematics": "science, math, physics, chemistry",
    "Health": "medicine, diseases, fitness, diet",
    "Education & Reference": "school, university, studying, grammar",
    "Computers & Internet": "software, hardware, internet, programming",
    "Sports": "sports, games, teams",
    "Business & Finance": "money, jobs, investing, companies",
    "Entertainment & Music": "movies, music, celebrities, TV",
    "Family & Relationships": "dating, marriage, friends, parenting",
    "Politics & Government": "politics, law, elections, government",
}


def yahoo_choice(split, n, rng):
    ds = load_dataset("community-datasets/yahoo_answers_topics", split=split)
    ds = ds.select(range(0, 20000))                    # avoid shuffling 1.4M rows
    ds = _sample(ds, n, rng)
    names = list(YAHOO)
    instr = ["Which category does this question belong to?", "What is this question about?"]
    for r in ds:
        state = _cut(f"{r['question_title']} {r['question_content']} Answer: {r['best_answer']}")
        q = _choice_q(rng, instr, YAHOO)
        yield _choice_example("yahoo", "en", state, q, names[r["topic"]])


DBPEDIA = {
    "Company": "a business or corporation", "EducationalInstitution": "a school or university",
    "Artist": "a musician, painter or writer", "Athlete": "a sports person",
    "OfficeHolder": "a politician or official", "MeanOfTransportation": "a vehicle, ship or aircraft",
    "Building": "a building or structure", "NaturalPlace": "a mountain, river or lake",
    "Village": "a village or small settlement", "Animal": "an animal species", "Plant": "a plant species",
    "Album": "a music album", "Film": "a movie", "WrittenWork": "a book or publication",
}


def dbpedia_choice(split, n, rng):
    ds = load_dataset("fancyzhx/dbpedia_14", split="test").select(range(0, 70000, 7))
    ds = _sample(ds, n, rng)
    names = list(DBPEDIA)
    for r in ds:
        q = _choice_q(rng, ["What kind of entity is described in the text?"], DBPEDIA, drop_desc_p=0)
        yield _choice_example("dbpedia", "en", _cut(r["title"] + ". " + r["content"]), q, names[r["label"]])


EMOTION = {
    "sadness": "the author feels sad or unhappy", "joy": "the author feels happy or glad",
    "love": "the author feels love or affection", "anger": "the author feels angry or irritated",
    "fear": "the author feels afraid or anxious", "surprise": "the author is surprised or amazed",
}


def emotion_choice(split, n, rng):
    ds = _sample(load_dataset("dair-ai/emotion", split="test"), n, rng)
    names = list(EMOTION)
    for r in ds:
        q = _choice_q(rng, ["Which emotion does the author express?"], EMOTION, drop_desc_p=0)
        yield _choice_example("emotion", "en", r["text"], q, names[r["label"]])


# --------------------------------------------------------------------------- score
STARS = ["Very dissatisfied, 1 star", "Dissatisfied, 2 stars", "Neutral, 3 stars",
         "Satisfied, 4 stars", "Very satisfied, 5 stars"]


def amazon_score(split, n, rng):
    sp = "train" if split == "train" else "test"
    ds = _sample(load_dataset("SetFit/amazon_reviews_multi_en", split=sp), n, rng)
    instr = ["How satisfied is the customer with the product?", "What rating would this customer give?"]
    for r in ds:
        yield dict(task="amazon", lang="en", type="score", state=_cut(r["text"]),
                   q={"type": "score", "instructions": rng.choice(instr), "criteria": STARS},
                   label=int(r["label"]))


SST = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]


def sst5_score(split, n, rng):
    ds = _sample(load_dataset("SetFit/sst5", split="test"), n, rng)
    for r in ds:
        yield dict(task="sst5", lang="en", type="score", state=r["text"],
                   q={"type": "score", "instructions": "How positive is the sentiment of this review?",
                      "criteria": SST},
                   label=int(r["label"]))


# --------------------------------------------------------------------------- recipes
TRAIN_RECIPE = [  # (builder, n_train, n_test)
    (boolq, 2000, 500), (ua_news_noul, 800, 300), (ukr_summary_noul, 800, 300), (ag_news_noul, 400, 200),
    (ag_news_choice, 500, 300), (ua_news_choice, 500, 300), (yahoo_choice, 300, 150),
    (amazon_score, 600, 300),
]
HELD_OUT = [(ukr_toxicity, 400), (dbpedia_choice, 200), (emotion_choice, 300), (sst5_score, 400)]


def build(split: str) -> list[dict]:
    rng = random.Random(SEED if split == "train" else SEED + 1)
    out = []
    for fn, ntr, nte in TRAIN_RECIPE:
        for ex in fn("train" if split == "train" else ("validation" if fn is boolq else "test"),
                     ntr if split == "train" else nte, rng):
            ex["held_out"] = False
            out.append(ex)
    if split == "test":
        for fn, n in HELD_OUT:
            for ex in fn("test", n, rng):
                ex["held_out"] = True
                out.append(ex)
    return out


# --------------------------------------------------------------------------- customer support
BITEXT = {
    "ACCOUNT": "creating, editing, deleting or recovering a user account",
    "ORDER": "placing, changing, cancelling or tracking an order",
    "REFUND": "getting money back, refund status or refund policy",
    "INVOICE": "getting or checking an invoice or bill",
    "CONTACT": "wants to talk to customer service or a human agent",
    "PAYMENT": "payment methods or problems with a payment",
    "FEEDBACK": "complaints or reviews about the service",
    "DELIVERY": "delivery options and delivery time",
    "SHIPPING": "setting up or changing a shipping address",
    "SUBSCRIPTION": "newsletter subscription",
    "CANCEL": "cancellation fees",
}
BITEXT_NOUL = {
    "get_refund": "Does the customer explicitly ask for their money back?",
    "complaint": "Is the customer complaining about the service?",
    "cancel_order": "Does the customer want to cancel an order?",
    "contact_human_agent": "Does the customer want to talk to a human?",
    "payment_issue": "Does the customer report a problem with a payment?",
    "recover_password": "Has the customer lost access to their account?",
    "track_order": "Does the customer want to know where their order is?",
    "delete_account": "Does the customer want to delete their account?",
}


def _bitext(split, n, rng):
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    idx = [i for i in range(len(ds)) if (i % 5 == 0) == (split != "train")]   # 80/20 split
    rng.shuffle(idx)
    return ds.select(idx[:n])


def bitext_choice(split, n, rng):
    instr = ["Which department should handle this request?", "What is this customer request about?"]
    for r in _bitext(split, n, rng):
        gold = r["category"]
        others = rng.sample([c for c in BITEXT if c != gold], rng.randint(2, 5))
        opts = {c: BITEXT[c] for c in [gold] + others}
        q = _choice_q(rng, instr, opts)
        yield _choice_example("bitext_choice", "en", r["instruction"], q, gold)


def bitext_noul(split, n, rng):
    ds = _bitext(split, 20 * n, rng)
    by_intent = {}
    for r in ds:
        by_intent.setdefault(r["intent"], []).append(r["instruction"])
    intents = [i for i in BITEXT_NOUL if i in by_intent]
    for _ in range(n):
        it = rng.choice(intents)
        pos = rng.random() < 0.5
        src = it if pos else rng.choice([i for i in by_intent if i != it])
        yield dict(task="bitext_noul", lang="en", type="noul", state=rng.choice(by_intent[src]),
                   q={"type": "noul", "instructions": BITEXT_NOUL[it]}, label=int(pos))


# Hand-written Ukrainian support tickets (test only) - the domain of the assignment example.
UA_SUPPORT = [  # (text, department, wants_refund)
    ("Вітаю! З мене двічі списали оплату за підписку. Поверніть, будь ласка, зайве.", "billing", 1),
    ("Додаток вилітає щоразу, коли я відкриваю налаштування. Виправте, будь ласка.", "technical", 0),
    ("Скільки коштуватиме корпоративний тариф на 50 користувачів?", "sales", 0),
    ("Не можу увійти в акаунт: пише «помилка сервера 500».", "technical", 0),
    ("Хочу скасувати підписку і отримати гроші за невикористаний місяць.", "billing", 1),
    ("Чи можна отримати знижку, якщо оплатимо одразу за рік?", "sales", 0),
    ("Підкажіть, будь ласка, де знаходиться ваш офіс у Львові?", "other", 0),
    ("Після оновлення зникли всі мої збережені файли!", "technical", 0),
    ("Мені прийшов рахунок на суму, більшу, ніж у договорі. Поясніть, чому.", "billing", 0),
    ("Ми хочемо укласти новий контракт на наступний рік, з ким можна поговорити?", "sales", 0),
    ("Я оплатив товар, але його так і не доставили. Верніть мої кошти.", "billing", 1),
    ("Сайт дуже повільно завантажується вже третій день.", "technical", 0),
    ("Дякую вашій команді за чудову підтримку минулого тижня!", "other", 0),
    ("Кнопка «Оплатити» не натискається в мобільному застосунку.", "technical", 0),
    ("Які у вас ціни на преміум-версію для навчальних закладів?", "sales", 0),
    ("Помилково оформив дві однакові підписки, прошу повернути гроші за одну.", "billing", 1),
    ("Чи працюєте ви у святкові дні?", "other", 0),
    ("Карта прив'язана, але оплата постійно відхиляється.", "billing", 0),
    ("Синхронізація між телефоном і ноутбуком не працює.", "technical", 0),
    ("Цікавить партнерська програма для реселерів, надішліть умови.", "sales", 0),
    ("Послуга не відповідає опису, вимагаю повного повернення коштів!", "billing", 1),
    ("Хочу змінити email в профілі, але не знаю як.", "other", 0),
    ("Отримав повідомлення про списання, якого я не робив. Поверніть гроші.", "billing", 1),
    ("Програма показує неправильний баланс після останнього оновлення.", "technical", 0),
]
UA_DEPTS = {"billing": "Рахунки, оплати, повернення коштів", "technical": "Баги, збої, помилки системи",
            "sales": "Ціни, нові контракти", "other": "Усе інше"}


def ua_support(split, n, rng):
    for text, dept, refund in UA_SUPPORT[:n]:
        q = {"type": "choice", "instructions": "Яка команда має обробити цей запит?", "criteria": dict(UA_DEPTS)}
        yield _choice_example("ua_support_choice", "uk", text, q, dept) | {"held_out": True}
        yield dict(task="ua_support_noul", lang="uk", type="noul", state=text, held_out=True,
                   q={"type": "noul", "instructions": "Клієнт явно просить повернути кошти?"}, label=refund)


TRAIN_RECIPE += [(bitext_choice, 700, 300), (bitext_noul, 700, 300)]
HELD_OUT += [(ua_support, len(UA_SUPPORT))]
