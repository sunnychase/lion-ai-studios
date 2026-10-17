"""negation.py — phrases where a prompt names something in order to forbid it.

The studio's video and image models run without a negative prompt, and they tend to show whatever a prompt names: "no cars"
puts cars in the street, "never look at the camera" invites a look at the camera. This module finds such phrases so the page
can warn before a render ("say what is there instead").

    find(text, limit=8)  -> [{"phrase": "no photographs", "thing": "photographs"}, ...]
    check(text)          -> {"negations": [...], "reason": "..." or ""}

Standard library only. Words inside double quotes are dialogue and are left alone; idioms such as "no doubt", "not only",
"no matter" and "no more than" are skipped.
"""
import re

REASON = ("These phrases name something in order to forbid it. The models run without a negative prompt and tend to show what "
          "a prompt names, so describe what is there instead (\"an empty street\", \"one continuous shot\").")

# words that open a forbidding phrase; "n't" verbs (don't, isn't, can't...) are matched separately
OPENERS = {"no", "not", "never", "without", "nobody", "none", "nothing", "neither", "nor", "cannot", "nowhere"}
PEOPLE = {"nobody", "no one", "no-one"}

# opener + following word(s) that are figures of speech, not a forbidden thing
IDIOMS = (
    "no doubt", "no matter", "no more than", "no less than", "no longer", "no sooner", "no wonder", "no way of",
    "not only", "not just", "not until", "not to mention", "not least", "not quite", "not too", "not very", "not so",
    "not really", "not yet", "nothing but", "nothing less than", "nothing more than", "none other than", "never mind",
    "never before", "without a doubt", "without fail", "can't help", "cannot help", "neither here nor there",
)

# a phrase stops at clause punctuation, at these words, or after MAX_WORDS
STOPS = {"and", "but", "while", "so", "then", "because", "which", "that", "as", "when", "until", "where", "though", "although",
         "yet", "instead", "rather", "only", "just", "except"}
MAX_WORDS = 5
LEADING = {"a", "an", "any", "the", "one", "single", "some", "more", "other"}

_QUOTED = re.compile(r"\"[^\"\n]*\"|“[^”\n]*”")
_CLAUSES = re.compile(r"[.!?;:,()\[\]\n—–]+|\s-\s")
_TOKEN = re.compile(r"[a-z0-9]+(?:['’][a-z]+)*(?:-[a-z0-9]+)*")


def _is_opener(tok: str) -> bool:
    return tok in OPENERS or tok.endswith("n't") or tok.endswith("n’t")


def _idiom_at(toks: list, i: int) -> bool:
    for idiom in IDIOMS:
        parts = idiom.split()
        if toks[i:i + len(parts)] == parts:
            return True
    return False


def _phrases(clause: str):
    toks = [t.replace("’", "'") for t in _TOKEN.findall(clause.lower())]
    i = 0
    while i < len(toks):
        t = toks[i]
        two = " ".join(toks[i:i + 2])
        if two in ("no one", "no-one") or t == "no-one":
            opener, start = two if t == "no" else t, i + (2 if t == "no" else 1)
        elif _is_opener(t):
            opener, start = t, i + 1
        else:
            i += 1
            continue
        if _idiom_at(toks, i):
            i = start
            continue
        rest = []
        for w in toks[start:]:
            if w in STOPS or _is_opener(w) or len(rest) >= MAX_WORDS:
                break
            rest.append(w)
        if opener == "nor" and not rest:
            i = start
            continue
        if not rest and opener not in PEOPLE and opener not in ("nobody", "nothing", "none"):
            i = start                                  # a bare "No," or "not." forbids nothing in particular
            continue
        thing = list(rest)
        while thing and thing[0] in LEADING:
            thing.pop(0)
        if opener in PEOPLE or opener == "nobody":
            thing_text = "people" + (" " + " ".join(thing) if thing else "")
        else:
            thing_text = " ".join(thing) or opener
        yield " ".join([opener] + rest), thing_text
        i = start + len(rest)


def find(text: str, limit: int = 8) -> list:
    """the forbidding phrases in TEXT, in order, without repeats: [{"phrase", "thing"}]"""
    if not text:
        return []
    plain = _QUOTED.sub(" . ", text)
    out, seen = [], set()
    for clause in _CLAUSES.split(plain):
        for phrase, thing in _phrases(clause):
            if phrase in seen:
                continue
            seen.add(phrase)
            out.append({"phrase": phrase, "thing": thing})
            if len(out) >= limit:
                return out
    return out


def check(text: str) -> dict:
    """what the page shows: {negations, reason}"""
    found = find(text)
    return {"negations": found, "reason": REASON if found else ""}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(check(" ".join(sys.argv[1:]) or sys.stdin.read()), indent=1))
