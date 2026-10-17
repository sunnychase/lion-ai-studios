"""wordcheck.py — did a spoken take say the words it was given?

The voice pipeline renders a line, Whisper transcribes the take, and check_text compares the two:

    check_text(text, heard) -> {action, words, matched, coverage, missing, longest_missing_run, ending_missing,
                                tail_words, reason, heard}

action is "redraw" when the take lost too much (coverage under 0.85, a run of 3 or more missing words, or the line's last
content word missing), "trim" when everything was said but extra words follow the end, and "ok" otherwise.

Both sides are normalised the same way (words_of): lower case, [bracketed directions] and <|speaker:N|> tags dropped,
punctuation stripped, numbers spelled out in English (ordinals and decimals too). The word lists are aligned with difflib;
inside a mismatched stretch a word still counts as said when it is spelled nearly the same (Whisper's "colour" for "color",
"alright" for "all right"). Standard library only.
"""
import difflib
import re

MIN_COVERAGE = 0.85
MAX_MISSING_RUN = 3          # this many missing words in a row is a dropped phrase
NEAR = 0.75                  # character similarity at which two words count as the same word

# words whose loss at the end of a line does not lose its meaning
FUNCTION_WORDS = set("""a an the and or but nor so of to in on at by for with from up as is are was were be been am it its this that
these those i me my you your he him his she her we us our they them their there here then than too very just oh ah um uh""".split())

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]
ORDINAL_END = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
SYMBOLS = {"&": " and ", "%": " percent ", "+": " plus ", "@": " at ", "=": " equals "}

_TAGS = re.compile(r"<\|[^|>]*\|>|<[^<>]{1,40}>|\[[^\]]*\]")
_MONEY = re.compile(r"([$£€])\s?(\d[\d,]*(?:\.\d+)?)")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?")


def _under_thousand(n: int) -> list:
    out = []
    if n >= 100:
        out += [ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        out.append(TENS[n // 10])
        n %= 10
        if n:
            out.append(ONES[n])
    elif n or not out:
        out.append(ONES[n])
    return out


def number_words(n: int) -> list:
    """an integer in English words: 2026 -> ['two', 'thousand', 'twenty', 'six']; 1984 reads as a year (nineteen eighty four)"""
    if n < 0:
        return ["minus"] + number_words(-n)
    if 1100 <= n <= 1999 and n % 100:
        return _under_thousand(n // 100) + _under_thousand(n % 100)
    out = []
    for size, name in SCALES:
        if n >= size:
            out += number_words(n // size) + [name]
            n %= size
    if n or not out:
        out += _under_thousand(n)
    return out


def _ordinal(words: list) -> list:
    last = words[-1]
    if last in ORDINAL_END:
        last = ORDINAL_END[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return words[:-1] + [last]


def _spell(m) -> str:
    tok = m.group(0)
    ordinal = tok[-2:] in ("st", "nd", "rd", "th")
    if ordinal:
        tok = tok[:-2]
    whole, _, frac = tok.replace(",", "").partition(".")
    words = number_words(int(whole)) if whole else ["zero"]
    if ordinal and not frac:
        words = _ordinal(words)
    if frac:
        words += ["point"] + [ONES[int(d)] for d in frac]
    return " " + " ".join(words) + " "


def _money(m) -> str:
    unit = {"$": "dollars", "£": "pounds", "€": "euros"}[m.group(1)]
    return f" {m.group(2)} {unit} "


def words_of(text: str) -> list:
    """TEXT as a list of plain lower-case words, the form both sides of the check are compared in"""
    s = _TAGS.sub(" ", text or "").lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = _MONEY.sub(_money, s)
    for sym, word in SYMBOLS.items():
        s = s.replace(sym, word)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)              # 12,000 -> 12000
    s = _NUMBER.sub(_spell, s)
    s = s.replace("'", "")                                   # don't -> dont, on both sides alike
    s = re.sub(r"[^a-z0-9]+", " ", s)                        # hyphens, dashes and all other punctuation split words
    return s.split()


def _similar(a: str, b: str) -> bool:
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= NEAR


def _pair_block(ref: list, hyp: list) -> list:
    """inside a mismatched stretch: (ref index, hyp index) pairs for words said nearly alike, in order. Also lets two
    words on one side stand for one on the other ("all right" / "alright")."""
    pairs, j = [], 0
    i = 0
    while i < len(ref):
        hit = None
        for k in range(j, len(hyp)):
            if _similar(ref[i], hyp[k]):
                hit = (1, k, 1)
            elif k + 1 < len(hyp) and _similar(ref[i], hyp[k] + hyp[k + 1]):
                hit = (1, k, 2)
            elif i + 1 < len(ref) and _similar(ref[i] + ref[i + 1], hyp[k]):
                hit = (2, k, 1)
            if hit:
                break
        if hit:
            nr, k, nh = hit
            for x in range(nr):
                pairs.append((i + x, k + nh - 1))
            i += nr
            j = k + nh
        else:
            i += 1
    return pairs


def align(ref: list, hyp: list) -> dict:
    """ref index -> hyp index for every reference word that was heard"""
    found = {}
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for d in range(i2 - i1):
                found[i1 + d] = j1 + d
        elif op == "replace":
            for a, b in _pair_block(ref[i1:i2], hyp[j1:j2]):
                found[i1 + a] = j1 + b
    return found


def _last_content(ref: list) -> int:
    for i in range(len(ref) - 1, -1, -1):
        if ref[i] not in FUNCTION_WORDS:
            return i
    return len(ref) - 1


def check_text(text: str, transcript: str) -> dict:
    """compare the line that was asked for with what Whisper heard; see the module docstring for the keys"""
    ref, hyp = words_of(text), words_of(transcript)
    heard = (transcript or "").strip()[:400]
    base = {"words": len(ref), "heard": heard}
    if not ref:
        return {**base, "action": "ok", "matched": 0, "coverage": 1.0, "missing": [], "longest_missing_run": 0,
                "ending_missing": False, "tail_words": [], "reason": "no words to check"}
    if not hyp:
        return {**base, "action": "redraw", "matched": 0, "coverage": 0.0, "missing": ref[:40], "longest_missing_run": len(ref),
                "ending_missing": True, "tail_words": [], "reason": "nothing was heard"}

    found = align(ref, hyp)
    missing_idx = [i for i in range(len(ref)) if i not in found]
    run = longest = 0
    for i in range(len(ref)):
        run = run + 1 if i not in found else 0
        longest = max(longest, run)
    coverage = round(len(found) / len(ref), 3)
    ending_missing = _last_content(ref) not in found
    last_heard = max(found.values()) if found else -1
    tail = hyp[last_heard + 1:] if found and not ending_missing else []

    problems = []
    if coverage < MIN_COVERAGE:
        problems.append(f"only {len(found)} of {len(ref)} words heard ({coverage:.0%})")
    if longest >= MAX_MISSING_RUN:
        problems.append(f"{longest} words in a row missing")
    if ending_missing:
        problems.append(f"the ending (\"{ref[_last_content(ref)]}\") was not heard")
    if problems:
        action, reason = "redraw", "; ".join(problems)
    elif tail:
        action, reason = "trim", f"all words said; {len(tail)} extra word(s) after the end"
    else:
        action = "ok"
        reason = "all words said" if not missing_idx else f"{len(found)} of {len(ref)} words heard, the ending intact"
    return {**base, "action": action, "matched": len(found), "coverage": coverage, "missing": [ref[i] for i in missing_idx][:40],
            "longest_missing_run": longest, "ending_missing": ending_missing, "tail_words": tail, "reason": reason}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(check_text(sys.argv[1], sys.argv[2]), indent=1))
