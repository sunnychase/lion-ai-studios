"""negation.py — things a prompt names in order to forbid them ("no photographs", "without a cut", "never look at the camera").

Adapted from drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold (app/negation.py, Apache-2.0, commit 0677c2c); modified for Lion AI
Studios (our own sentences, our field names). See NOTICE.md.

MiniMax H3 / FastH3 run here at CFG 1 (distilled: the empty side of the guidance is never used), and the image models run at CFG 1 too, so every
word of a prompt is something to show: a prompt that says "no photographs" tends to hang photographs on the wall, "no cut" does not stop cuts.
`check` lists such phrases so the page can show them before the job is queued. Nothing here changes the user's prompt.
"""
import re

REASON = ("The video and image models here have no negative prompt: a thing the prompt names tends to appear, even when the prompt says there "
          "is none of it. Take the words out and say what is there instead (\"plain dark walls\" for \"no photographs\").")
OWN = ("Nobody speaks.", "No voices, no speech.", "No speech.")            # sentences the studio may still pass through on purpose
IDIOMS = ("not only", "no doubt", "no matter", "not just", "no longer than", "no more than", "no less than", "no sooner")
WORD = r"[A-Za-z][A-Za-z'’-]*"
FOUND = re.compile(r"(?<![A-Za-z'’-])(no one|nobody|no|not|never|without|none of|neither|nor|cannot|" + WORD + r"n['’]t)"
                   r"(?![A-Za-z'’-])((?:[ \t]+" + WORD + r"){0,4})", re.IGNORECASE)
FILLER = {"a", "an", "the", "any", "of", "to", "be", "is", "are", "at", "in", "on", "it", "its", "their", "his", "her", "and", "or",
          "ever", "one", "other", "more", "new", "appear", "appears", "anywhere", "there", "that", "this", "with", "for", "into", "from"}


def find(text, limit=8):
    """the negated things in text: [{phrase, thing}] in reading order, each phrase once"""
    for own in OWN:
        text = re.sub(re.escape(own), " ", text, flags=re.IGNORECASE)
    out, seen = [], set()
    for m in FOUND.finditer(text):
        neg, rest = m.group(1), m.group(2).split()
        if any((neg + " " + " ".join(rest)).lower().startswith(i) for i in IDIOMS): continue
        kept, thing = [], []
        for w in rest:
            kept.append(w)
            if w.lower() not in FILLER: thing.append(w)
            if len(thing) == 3: break
        if not thing and neg.lower() not in ("nobody", "no one"): continue
        while kept and kept[-1].lower() in FILLER: kept.pop()
        phrase = " ".join([neg, *kept])
        if phrase.lower() in seen: continue
        seen.add(phrase.lower()); out.append({"phrase": phrase, "thing": " ".join(thing)})
        if len(out) == limit: break
    return out


def check(text):
    """what the page shows: {negations, reason}"""
    found = find(text or "")
    return {"negations": found, "reason": REASON if found else ""}
