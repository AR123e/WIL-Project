"""
Text normalisation shared by the chunk index and the query side, so both always agree.

Why this module exists: the first version removed "what", "is" AND "chess" as stop-words and used a
hand-made stemmer. "What is chess?" therefore became an EMPTY query (every score 0 -> "I don't know"), and
"rules" was stemmed to "rul". Here the domain word "chess" is kept, a real Porter stemmer is used, and
British/American spellings are unified.
"""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer

_ps = PorterStemmer()

# generic function words only - NOT domain words such as "chess", "play", "move"
STOP = set("""a an and are as at be but by can could do does for from had has have how i if in into is it its
me my of on or our should so than that the their them then there these they this to us was we were what when
where which who whom why will with would you your tell about please explain give""".split())

# words that appear in every question of this domain: useless as retrieval/gating evidence (they only match
# generic text such as "chess is played all over the world"), so they are dropped from QUERIES
DOMAIN_WORDS = {"chess"}

# British/American variants (the FIDE text is British: "centre", "defence", "colour")
SPELLING = {"center": "centre", "centers": "centres", "centered": "centred", "defense": "defence",
            "defenses": "defences", "color": "colour", "colors": "colours", "favorite": "favourite"}

# tiny, safe synonym table applied to QUERIES only (after stemming): FIDE calls the rules "Laws", and beginners
# say "get better" where the KB says "improve"
ALIASES = {"rule": ["law"], "law": ["rule"], "better": ["improv"], "stronger": ["improv"], "progress": ["improv"]}

_WORD = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=None)
def stem(word):
    return _ps.stem(word)


def words(text):
    return [SPELLING.get(w, w) for w in _WORD.findall(text.lower())]


def tokenize(text, expand=False, query=False):
    """Content tokens: lower-case, spelling-normalised, stop-words removed, Porter-stemmed.
    query=True also drops domain words ("chess"). May be empty for queries made only of function/domain words
    (e.g. "what is it?"); whole-game questions such as "what is chess?" are handled by classify_query instead."""
    toks = [stem(w) for w in words(text) if w not in STOP and not (query and w in DOMAIN_WORDS)]
    if expand:
        extra = [a for t in toks for a in ALIASES.get(t, [])]
        toks = toks + [a for a in extra if a not in toks]
    return toks


# ---- query intent --------------------------------------------------------------------------------
# Overview questions ("what is chess?", "how do I play chess?") are about the whole game, not one chunk,
# so they are routed to the beginner overview instead of relying on keyword overlap.
_PUNCT = re.compile(r"[^a-z0-9' ]+")

_INTRO = [
    r"^(so |ok |okay |hi |hey )?what('?s| is| are) (the game of |a game of |the |a )?chess( game)?( all about| exactly| like| really)?$",
    r"^(please )?(tell me about|teach me|introduce me to|give me an intro(duction)? to|give me an overview of|explain|describe|define|overview of|intro(duction)? to) (the game of |the )?chess( to me| for beginners| to a beginner)?$",
    r"^(please )?explain (how )?chess works$",
    r"^what do i need to know about chess$",
]
_HOWTO = [
    r"\bhow (do|can|should|to|would) (i |you |we |one )?(play|learn|start|begin|get started)( playing| with| learning)? chess\b",
    r"\b(rules|basics|fundamentals) (of|for) (the game of )?chess\b",
    r"\bchess (rules|basics|for beginners)\b",
    r"^(what are )?the basic rules$",
    r"\b(i am|i'?m) (a )?(new|beginner|complete beginner|totally new)\b.*\bchess\b",
    r"\bwhere (do|should|can) i (start|begin)\b",
]


def classify_query(query):
    """Return 'intro', 'howto' or None."""
    q = _PUNCT.sub(" ", query.lower())
    q = re.sub(r"\s+", " ", q).strip()
    if any(re.search(p, q) for p in _INTRO):
        return "intro"
    if any(re.search(p, q) for p in _HOWTO):
        return "howto"
    return None
