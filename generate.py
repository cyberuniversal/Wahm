"""
WAHM stage 4: answer generation.

Runs the model set over the benchmark and records raw answers. Every row is
identified by (qid, variety, condition, model), so MSA, the four dialects, and
the control condition are all just rows: no separate code paths.

Reads   wahm_seed_msa.csv        the MSA arm
        final_<dialect>.csv      validated dialect arms (from finalize.py)
Writes  generations.csv          append-only, resumable

Design notes
  * Zero-shot, no retrieved context: answers must come from parametric
    knowledge, which is the regime where hallucination shows up.
  * temperature 0.0 by default: we want the model's most likely answer, and
    reproducibility.
  * Raw output is stored unmodified. Cleaning and degeneration detection happen
    at scoring time, not here, so the record stays auditable.
  * Resumable: re-running skips (qid, variety, condition, model) already done.

Conditions
  direct      the question is asked as-is, model answers however it wants
  msa_answer  dialect question, but the model is told to answer in MSA.
              This is the control for the matching confound: if a dialect
              answer scores as hallucinated only because it lexically diverges
              from the MSA gold, this condition removes that artifact. If the
              drift survives here, it is real.
  dialect_aware explicitly identifies the input variety in the system prompt.
  msa_pivot   first translates the dialect question to MSA, then answers the
              translated question in a second call.
  msa_restate asks for a visible MSA restatement followed by the answer.

Usage
    python generate.py --models gpt4o allam --varieties msa gulf
    python generate.py --all --limit 20              # pilot
    python generate.py --all                         # full run
"""

import argparse, csv, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# ---------------------------------------------------------------- model registry
# Entries use either the OpenAI chat-completions or completions schema. The
# latter is needed for base checkpoints that do not define a chat template.
MODELS = {
    "gpt4o": dict(
        model_id="gpt-4o",
        base_url=None,                      # default OpenAI
        key_env="OPENAI_API_KEY",
        family="multilingual",
    ),
    "gpt55": dict(
        model_id=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5"),
        base_url_env="AZURE_OPENAI_BASE_URL",
        key_env="AZURE_OPENAI_API_KEY",
        family="multilingual",
        api_mode="responses",
        supports_temperature=False,
        max_output_tokens=2048,
        max_output_tokens_cap=8192,
    ),
    "allam": dict(
        model_id=os.getenv("ALLAM_DEPLOYMENT", "allam-2-7b"),
        base_url_env="AZURE_ALLAM_ENDPOINT",  # Azure AI Foundry endpoint
        key_env="AZURE_ALLAM_KEY",
        family="arabic_centric",
    ),
    "jais": dict(
        model_id=os.getenv("JAIS_DEPLOYMENT", "inceptionai/Jais-2-8B-Chat"),
        base_url_env="JAIS_BASE_URL",
        key_env="JAIS_API_KEY",
        family="arabic_centric",
    ),
    "fanar": dict(
        model_id=os.getenv("FANAR_MODEL", "QCRI/Fanar-1-9B"),
        base_url_env="FANAR_BASE_URL",        # from your approved QCRI API access
        key_env="FANAR_API_KEY",
        family="arabic_centric",
        api_mode=os.getenv("FANAR_API_MODE", "chat"),
    ),
    "falcon": dict(
        model_id="tiiuae/falcon-h1-34b-instruct",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        family="arabic_centric",
    ),
    "llama": dict(
        model_id="meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        family="multilingual",
    ),
    "qwen3-8b": dict(
        model_id=os.getenv("QWEN3_MODEL", "qwen/qwen3-8b"),
        base_url=os.getenv(
            "QWEN3_BASE_URL", "https://openrouter.ai/api/v1"),
        key_env="OPENROUTER_API_KEY",
        family="multilingual",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "llama-3.3-70b": dict(
        model_id=os.getenv(
            "LLAMA33_MODEL", "meta-llama/llama-3.3-70b-instruct"),
        base_url=os.getenv(
            "LLAMA33_BASE_URL", "https://openrouter.ai/api/v1"),
        key_env="OPENROUTER_API_KEY",
        family="multilingual",
    ),
    "gemma-2-9b": dict(
        model_id=os.getenv("GEMMA2_MODEL", "google/gemma-2-9b-it"),
        base_url=os.getenv(
            "GEMMA2_BASE_URL", "https://openrouter.ai/api/v1"),
        key_env="OPENROUTER_API_KEY",
        family="multilingual",
    ),
    "gemma-2-9b-base": dict(
        model_id=os.getenv("GEMMA2_BASE_MODEL", "google/gemma-2-9b"),
        base_url=os.getenv(
            "GEMMA2_BASE_URL", "http://127.0.0.1:8000/v1"),
        key_env="GEMMA2_BASE_API_KEY",
        family="multilingual",
        api_mode="completion",
    ),
}

VARIETIES = ["msa", "gulf", "egyptian", "levantine", "sudanese"]

MSA_ANSWER_SYSTEM = (
    "أجب على السؤال التالي باللغة العربية الفصحى الحديثة فقط، "
    "بغض النظر عن اللهجة المستخدمة في السؤال."
)

DIALECT_AWARE_SYSTEM = (
    "السؤال التالي مكتوب بلهجة {variety}. افهمه كما هو وأجب عنه بدقة."
)
MSA_PIVOT_SYSTEM = (
    "حوّل السؤال التالي إلى العربية الفصحى الحديثة مع الحفاظ على المعنى "
    "بالضبط. أخرج السؤال المحوّل فقط."
)
MSA_RESTATE_SYSTEM = (
    "أعد صياغة السؤال أولا بالعربية الفصحى الحديثة، ثم أجب عنه بدقة. "
    "اعرض إعادة الصياغة والإجابة بوضوح."
)

OUT = "generations.csv"
FIELDS = ["qid", "variety", "condition", "model", "family", "question",
          "gold_answer", "answer", "pivot_question", "model_id", "temperature",
          "resolved_model_id", "pivot_model_id", "timestamp", "error"]


# ---------------------------------------------------------------- clients
_clients = {}


def get_client(name):
    """One OpenAI-compatible client per model, cached."""
    if name in _clients:
        return _clients[name]
    from openai import OpenAI
    spec = MODELS[name]

    key = os.getenv(spec["key_env"])
    if not key:
        sys.exit(f"ERROR: {name} needs {spec['key_env']} in the environment")

    base = spec.get("base_url")
    if base is None and "base_url_env" in spec:
        base = os.getenv(spec["base_url_env"])
        if not base:
            sys.exit(f"ERROR: {name} needs {spec['base_url_env']} in the environment")

    _clients[name] = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
    return _clients[name]


def _chat(name, messages, temperature, max_tokens=512, retries=4):
    spec = MODELS[name]
    client = get_client(name)
    for attempt in range(retries):
        try:
            api_mode = spec.get("api_mode", "chat")
            if api_mode == "completion":
                prompt = "\n\n".join(message["content"] for message in messages)
                r = client.completions.create(
                    model=spec["model_id"], prompt=prompt,
                    temperature=temperature, max_tokens=max_tokens)
                answer = r.choices[0].text
            elif api_mode == "responses":
                base_output_tokens = spec.get("max_output_tokens", max_tokens)
                kwargs = dict(
                    model=spec["model_id"], input=messages,
                    max_output_tokens=min(
                        base_output_tokens * (2 ** attempt),
                        spec.get("max_output_tokens_cap", base_output_tokens)))
                if spec.get("supports_temperature", True):
                    kwargs["temperature"] = temperature
                r = client.responses.create(**kwargs)
                answer = r.output_text
                if not answer:
                    status = getattr(r, "status", "unknown")
                    reason = getattr(
                        getattr(r, "incomplete_details", None), "reason",
                        "empty_output")
                    raise RuntimeError(
                        f"empty response: status={status}, reason={reason}")
            else:
                kwargs = dict(
                    model=spec["model_id"], messages=messages,
                    temperature=temperature, max_tokens=max_tokens)
                if spec.get("extra_body"):
                    kwargs["extra_body"] = spec["extra_body"]
                r = client.chat.completions.create(**kwargs)
                answer = r.choices[0].message.content
            return ((answer or "").strip(), "",
                    getattr(r, "model", spec["model_id"]))
        except Exception as e:
            if attempt == retries - 1:
                return "", f"{type(e).__name__}: {e}"[:300], ""
            time.sleep(2 ** attempt)


def generate_one(name, question, variety, condition, temperature):
    pivot, pivot_model = "", ""
    if condition == "msa_answer":
        system = MSA_ANSWER_SYSTEM
    elif condition == "dialect_aware":
        system = DIALECT_AWARE_SYSTEM.format(variety=variety)
    elif condition == "msa_restate":
        system = MSA_RESTATE_SYSTEM
    elif condition == "msa_pivot":
        pivot, error, pivot_model = _chat(name, [
            {"role": "system", "content": MSA_PIVOT_SYSTEM},
            {"role": "user", "content": question},
        ], 0.0)
        if error:
            return "", error, "", "", pivot_model
        question = pivot
        system = MSA_ANSWER_SYSTEM
    else:
        system = ""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    answer, error, resolved_model = _chat(name, messages, temperature)
    return answer, error, pivot, resolved_model, pivot_model


# ---------------------------------------------------------------- benchmark loading
def load_benchmark(varieties, limit=None):
    """Returns rows of (qid, variety, question, gold)."""
    items = []

    if "msa" in varieties:
        path = "wahm_seed_msa.csv"
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found")
        with open(path, encoding="utf-8", newline="") as source:
            for r in csv.DictReader(source):
                items.append((r["qid"], "msa", r["question_msa"], r["gold_answer"]))

    for v in varieties:
        if v == "msa":
            continue
        path = f"final_{v}.csv"
        if not os.path.exists(path):
            print(f"  skipping {v}: {path} not found "
                  "(run finalize.py once the validator returns their sheet)")
            continue
        with open(path, encoding="utf-8", newline="") as source:
            for r in csv.DictReader(source):
                items.append((r["qid"], v, r["question_dialect"], r["gold_answer"]))

    if limit:
        # keep the limit per variety so arms stay parallel
        by_variety = {}
        for it in items:
            by_variety.setdefault(it[1], []).append(it)
        items = [it for v in by_variety for it in by_variety[v][:limit]]

    return items


def load_done():
    """Resume support: which (qid, variety, condition, model) already succeeded."""
    if not os.path.exists(OUT):
        return set()
    done = set()
    with open(OUT, encoding="utf-8", newline="") as source:
        for r in csv.DictReader(source):
            if r.get("answer", "").strip() and not r.get("error", "").strip():
                done.add((r["qid"], r["variety"], r["condition"], r["model"]))
    return done


# ---------------------------------------------------------------- main loop
def run(models, varieties, conditions, limit, temperature, sleep, workers=1):
    items = load_benchmark(varieties, limit)
    if not items:
        sys.exit("ERROR: no benchmark items loaded")

    done = load_done()
    todo = [(q, v, ques, gold, c, m)
            for (q, v, ques, gold) in items
            for c in conditions
            for m in models
            if (q, v, c, m) not in done
            and not (c == "msa_answer" and v == "msa")]   # control is dialect-only

    print(f"benchmark items : {len(items)}")
    print(f"models          : {', '.join(models)}")
    print(f"varieties       : {', '.join(varieties)}")
    print(f"conditions      : {', '.join(conditions)}")
    print(f"workers         : {workers}")
    print(f"already done    : {len(done)}")
    print(f"to generate     : {len(todo)}\n")
    if not todo:
        print("nothing to do")
        return

    new_file = not os.path.exists(OUT)
    if not new_file:
        with open(OUT, encoding="utf-8", newline="") as existing:
            header = next(csv.reader(existing), [])
        if header != FIELDS:
            sys.exit(f"ERROR: {OUT} has an incompatible header. Archive or "
                     "migrate it before appending this experiment schema.")
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()

        def generate_task(task):
            qid, variety, question, gold, condition, model = task
            result = generate_one(model, question, variety, condition, temperature)
            return task, result

        executor = None
        if workers > 1:
            # Initialize clients before worker threads to avoid creation races.
            for model in models:
                get_client(model)
            executor = ThreadPoolExecutor(max_workers=workers)
            generated = executor.map(generate_task, todo)
        else:
            generated = map(generate_task, todo)

        errors = 0
        try:
            for i, (task, result) in enumerate(generated, 1):
                qid, variety, question, gold, condition, model = task
                answer, err, pivot, resolved_model, pivot_model = result
                if err:
                    errors += 1
                w.writerow(dict(
                    qid=qid, variety=variety, condition=condition, model=model,
                    family=MODELS[model]["family"], question=question,
                    gold_answer=gold, answer=answer, pivot_question=pivot,
                    model_id=MODELS[model]["model_id"],
                    temperature=(temperature if MODELS[model].get(
                        "supports_temperature", True) else ""),
                    resolved_model_id=resolved_model, pivot_model_id=pivot_model,
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    error=err))
                f.flush()

                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  (errors: {errors})")
                if sleep:
                    time.sleep(sleep)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    print(f"\nwrote {OUT}")
    if errors:
        print(f"{errors} rows failed — re-run this command to retry only those")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None, choices=list(MODELS))
    p.add_argument("--varieties", nargs="+", default=None, choices=VARIETIES)
    p.add_argument("--conditions", nargs="+", default=["direct"],
                   choices=["direct", "msa_answer", "dialect_aware",
                            "msa_pivot", "msa_restate"])
    p.add_argument("--all", action="store_true", help="all models, all varieties")
    p.add_argument("--limit", type=int, default=None,
                   help="first N questions per variety (use 20 for a pilot)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sleep", type=float, default=0.0, help="pause between calls")
    p.add_argument("--workers", type=int, default=1,
                   help="concurrent requests (use only with providers that allow it)")
    a = p.parse_args()

    models = list(MODELS) if a.all else (a.models or ["gpt4o"])
    varieties = VARIETIES if a.all else (a.varieties or ["msa"])

    run(models, varieties, a.conditions, a.limit, a.temperature, a.sleep,
        a.workers)
