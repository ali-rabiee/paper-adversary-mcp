"""Check that a quoted passage really occurs in a source text.

Reviewer agents quote prior papers (and the submission) "verbatim"; this module
checks such quotes deterministically against the Markdown our PDF/HTML
extraction produces. Quote and source are normalized identically (HTML comments
such as page markers and anchors removed, NFKC, casefold, quotes, dashes and
spaces unified, a little LaTeX), keeping a map back to the original offsets.

A quote is "verified" only when its words are the source's words, up to case,
punctuation, spacing and hyphenation at a line or page break. Otherwise a fuzzy
pass aligns word tokens around shared 4-grams; a near miss that changes a
negation, a number or a quantifier is reported as not found. By default only
verified quotes count as evidence, and a quote with an ellipsis never does.
All scanning is linear in the input, so untrusted text cannot make it slow.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from array import array
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import accumulate

ACCEPT_APPROXIMATE = 0.92  # a threshold for Match.accepted_at; Match.accepted counts exact matches only
MIN_WORDS_DEFAULT = 8
MAX_WORDS_DEFAULT = 80
MAX_QUOTE_WORDS = 400  # longer quotes are refused without matching
_MAX_QUOTE_CHARS = 40 * MAX_QUOTE_WORDS
_MIN_COVERAGE = 0.85  # a fuzzy match covering less of the quote is reported as not_found
_MAX_GAP = 600  # normalized characters allowed between consecutive ellipsis segments
_MAX_CANDIDATES = 200  # fuzzy anchors examined per search
_MAX_VOTES = 20_000  # 4-gram hits counted per search, rarest 4-grams first
_MAX_BREAK = 200  # longest text between the halves of a word hyphenated at a line or page break
_RANK = {"not_found": 0, "approximate": 1, "verified": 2}
_BLOCKING = ("too_short", "ellipsis", "critical_difference")  # flags that rule out acceptance

_GREEK = dict(zip("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma "
                  "tau upsilon phi chi psi omega".split(), "αβγδεζηθικλμνξοπρστυφχψω"))
_LATEX = {  # command -> replacement; formatting commands vanish and keep their (brace-stripped) argument
    **dict.fromkeys(("mathbf mathrm mathcal mathit mathsf mathtt mathbb mathfrak mathscr boldsymbol bm textbf "
                     "textit textrm textsf texttt textsc emph text operatorname mbox hat bar tilde vec dot ddot "
                     "widehat widetilde overline underline left right").split(), ""),
    **_GREEK, **{name.capitalize(): letter for name, letter in _GREEK.items()},  # casefold lowercases anyway
    **{"var" + name: _GREEK[name] for name in ("epsilon", "theta", "phi", "pi", "rho", "sigma", "kappa")},
}
_SPECIAL = re.compile(r"\\(?P<cmd>%s)(?![A-Za-z])|[${}]|(?P<run>[\x00-\x7f]?[^\x00-\x7f]+)"
                      % "|".join(sorted(_LATEX, key=len, reverse=True)))
_CHARMAP = str.maketrans({
    **dict.fromkeys("\u2018\u2019\u201a\u201b\u2032", "'"), **dict.fromkeys("\u201c\u201d\u201e\u2033", '"'),
    **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"),
    **dict.fromkeys("\u00a0\u202f\u205f" + "".join(map(chr, range(0x2000, 0x200B))), " "),
    **dict.fromkeys("\u00ad\u200b\u200c\u200d\u2060\ufeff"),  # deleted
})
# Words are runs of letters and digits; a decimal number is one word, so "1.5" never squashes into "15".
_TOKEN = re.compile(r"\d+(?:\.\d+)+|[^\W_]+")
_ELLIPSIS = re.compile(r"\[\s*(?:\.\s*){3}\]|(?:\.\s*){2}\.")  # after NFKC, "\u2026" is "..."
_PAGE = re.compile(r"\s*page\s+(\d+)\s*")  # the inside of <!-- page 5 -->
_ANCHOR = re.compile(r"\s*anchor\s+(\S+)\s*")  # the inside of <!-- anchor S3.p2 -->
_HYPHEN_BREAK = re.compile(r"[-\u2010\u2011\u00ad](?:\s|<!--\s*(?:page|anchor)\s[^>]*-->)+")
_SENTENCE = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)\]]*\s+(?=[A-Z\"\u201c\u2018])|\n\s*\n\s*")

_NEGATIONS = frozenset("not no never without cannot nor neither none nothing non".split())
_QUANTIFIERS = frozenset(("only all every each any some more less fewer most least higher lower larger smaller "
                          "better worse increase increases increasing increased decrease decreases decreasing "
                          "decreased always sometimes").split())
_NEG_PREFIXES = ("non", "un", "in", "im", "il", "ir", "dis", "anti", "a")
_YEAR = re.compile(r"(?:19|20)\d\d[a-z]?")

_CLAIM_PAGE = re.compile(r"(?<![\w.])(?:pages?|pgs?|pp?)\.?\s*(\d+)(?:\s*[-\u2013\u2014]\s*(\d+))?", re.I)
_CLAIM_SECTION = re.compile(
    r"(?:§§?\s*|\b(?:sections?|sects?|secs?|appendix|app)\b\.?\s*)((?:\d+|[A-Z])(?:\.\d+)*)\b", re.I)
_CLAIM_ANCHOR = re.compile(r"[#¶]\s*([A-Za-z][\w.]*\w)")
_SECTION_NUMBER = re.compile(r"\s*(?:appendix\s+)?((?:\d+|[A-Z])(?:\.\d+)*)\b", re.I)


def _comment_spans(text: str) -> list[tuple[int, int]]:
    """HTML comments, found in one linear pass; an unclosed "<!--" is ordinary text."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while (start := text.find("<!--", pos)) != -1 and (close := text.find("-->", start + 4)) != -1:
        spans.append((start, close + 3))
        pos = close + 3
    return spans


def _strip_comments(text: str) -> str:
    pieces, pos = [], 0
    for a, b in _comment_spans(text):
        pieces.append(text[pos:a])
        pos = b
    return "".join(pieces) + text[pos:]


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().translate(_CHARMAP)


_fold_cluster = lru_cache(maxsize=4096)(_fold)  # one character, or a letter with its combining accents


def _normalize_map(text: str, comments: list[tuple[int, int]] | None = None,
                   code: str = "q") -> tuple[str, tuple[array, ...]]:
    """Normalized text plus its map back to `text` (arrays of typecode `code`): segment k starts at normalized
    offset n[k] and covers the original text [a[k], b[k]). A segment whose two lengths agree maps character by
    character; any other (a ligature, a LaTeX command, a letter with a combining accent) maps as a whole.
    """
    out: list[str] = []
    seg_n, seg_a, seg_b = array(code), array(code), array(code)
    n = 0

    def seg(length: int, a: int, b: int) -> None:  # `length` normalized characters come from text[a:b]
        nonlocal n
        if not length:
            return
        if seg_n and length == b - a and seg_b[-1] == a and n - seg_n[-1] == a - seg_a[-1]:
            seg_b[-1] = b  # extend a character-for-character segment
        else:
            seg_n.append(n)
            seg_a.append(a)
            seg_b.append(b)
        n += length

    def emit(piece: str, a: int, b: int) -> None:
        out.append(piece)
        seg(len(piece), a, b)

    pos = 0
    for c_start, c_end in [*(_comment_spans(text) if comments is None else comments), (len(text), len(text))]:
        for m in _SPECIAL.finditer(text, pos, c_start):
            emit(text[pos : m.start()].lower(), pos, m.start())  # plain ASCII between specials
            pos = m.end()
            if m.group("cmd"):
                emit(_LATEX[m.group("cmd")], m.start(), pos)
            elif m.group("run"):  # non-ASCII, with the character before it (which a combining accent may modify)
                run = m.group("run")
                folded = _fold(run)
                if len(folded) == len(run):  # character for character: map linearly
                    emit(folded, m.start(), pos)
                    continue
                pieces, i = [], 0
                while i < len(run):
                    j = i + 1
                    while j < len(run) and unicodedata.combining(run[j]):
                        j += 1
                    pieces.append(_fold_cluster(run[i:j]))
                    seg(len(pieces[-1]), m.start() + i, m.start() + j)
                    i = j
                out.append("".join(pieces))
            # $ and braces produce nothing
        emit(text[pos:c_start].lower(), pos, c_start)
        pos = c_end  # comments produce nothing
    return "".join(out), (seg_n, seg_a, seg_b)


def normalize(text: str) -> str:
    """The normalized form matching works on, whitespace collapsed (for display and debugging)."""
    return " ".join(_normalize_map(text)[0].split())


def _critical(tokens: list[str], i: int, j: int) -> list[str]:
    """Negations (as one class), quantifiers and numbers among tokens[i:j], sorted."""
    out = []
    for k in range(i, j):
        tok = tokens[k]
        if tok in _NEGATIONS or (tok == "t" and k > 0 and tokens[k - 1].endswith("n")):  # doesn't -> doesn, t
            out.append("not")
        elif tok in _QUANTIFIERS or any(c.isdigit() for c in tok):
            out.append(tok)
    return sorted(out)


def _nested(x: str, y: str, part) -> bool:
    return x != y and (part(x, y) or part(y, x))


def _prefix_negated(x: str, y: str) -> bool:
    """'possible' vs 'impossible', 'convex' vs 'nonconvex', 'typical' vs 'atypical'."""
    short, long = sorted((x, y), key=len)
    return len(short) >= 4 and long.endswith(short) and long[: -len(short)] in _NEG_PREFIXES


def _critical_diffs(a: list[str], b: list[str], opcodes: list[tuple]) -> list[str]:
    """The non-equal stretches of an alignment that change a negation, a quantifier or a number."""
    out = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal" and (_critical(a, i1, i2) != _critical(b, j1, j2)
                               or any(_prefix_negated(x, y) for x in a[i1:i2] for y in b[j1:j2])):
            k = 1 if i1 and j1 else 0  # one shared word of context
            out.append(f"quote '{' '.join(a[i1 - k : i2])}' vs source '{' '.join(b[j1 - k : j2])}'")
    return out


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_comments(text))


def _widen(text: str, start: int, end: int) -> tuple[int, int]:
    """Take in the $ of a formula the span starts or ends inside ("$\\alpha_t$ controls ...")."""
    m = re.search(r"\$\S*\Z", text[max(0, start - 40) : start])
    n = re.match(r"\S*?\$", text[end : end + 40])
    return start - (len(m.group()) if m else 0), end + (n.end() if n else 0)


def _markers(pattern: re.Pattern, text: str, comments: list[tuple[int, int]], conv) -> tuple[list[int], list]:
    """Positions and values of marker comments such as <!-- page 5 --> or <!-- anchor S3.p2 -->."""
    found = [(a, conv(m.group(1))) for a, b in comments if (m := pattern.fullmatch(text, a + 4, b - 3))]
    return [p for p, _ in found], [v for _, v in found]


@dataclass
class Match:
    status: str  # "verified" | "approximate" | "not_found"
    score: float  # 0..1 (1.0 for exact)
    coverage: float  # matched quote tokens / quote tokens (1.0 for exact)
    start: int | None  # offsets into the ORIGINAL source text of the matched span
    end: int | None
    canonical: str | None  # source[start:end] with markers removed and whitespace collapsed
    context: str | None  # canonical plus up to one sentence before and after (markers removed)
    location: dict | None  # describe_location(...) of the span
    flags: list[str]  # subset of: too_short, too_long, ellipsis, critical_difference, location_mismatch
    note: str | None = None

    @property
    def accepted(self) -> bool:
        """Exact evidence only: verified, and not too short, elided or critically different."""
        return self.status == "verified" and not any(f in self.flags for f in _BLOCKING)

    def accepted_at(self, approximate_score: float) -> bool:
        """Like `accepted`, but an approximate match scoring at least `approximate_score` also counts."""
        return self.accepted or (self.status == "approximate" and self.score >= approximate_score
                                 and not any(f in self.flags for f in _BLOCKING))

    def to_dict(self) -> dict:
        return {**asdict(self), "score": round(self.score, 4), "coverage": round(self.coverage, 4),
                "accepted": self.accepted}


@dataclass
class _Hit:
    status: str
    score: float
    matched: int  # quote tokens matched
    total: int  # quote tokens
    i: int  # source tokens [i, j)
    j: int
    note: str | None = None
    critical: bool = False


def _rank(chain: list[_Hit]) -> tuple[int, float, float]:
    return (min(_RANK[h.status] for h in chain), sum(h.matched for h in chain) / sum(h.total for h in chain),
            min(h.score for h in chain))


class SourceText:
    """A source prepared once (normalized streams, offset maps, a 4-gram index) and matched many times."""

    def __init__(self, text: str, sections: list[dict] | None = None):
        self.text = text
        self.sections = sorted(sections or [], key=lambda s: s["start"])
        code = "i" if len(text) < 1 << 25 else "q"  # 32-bit offsets when they fit, even after a 54-fold expansion
        comments = _comment_spans(text)
        self._comments = (array(code, (a for a, _ in comments)), array(code, (b for _, b in comments)))
        self._norm, (self._seg_n, self._seg_a, self._seg_b) = _normalize_map(text, comments, code)
        # Words (interned) and their normalized offsets; the squashed stream is the words concatenated.
        self._toks = list(map(sys.intern, _TOKEN.findall(self._norm)))
        self._tok_starts = array(code, (m.start() for m in _TOKEN.finditer(self._norm)))
        self._squashed = "".join(self._toks)
        self._sq_starts = array(code, accumulate(map(len, self._toks), initial=0))  # ends with len(squashed)
        # 4-gram index in two arrays: each position's 4-gram hash, and the positions sorted by that hash.
        self._gram_hash = array("q", map(hash, zip(self._toks, self._toks[1:], self._toks[2:], self._toks[3:])))
        self._gram_pos = array(code, sorted(range(len(self._gram_hash)), key=self._gram_hash.__getitem__))
        self._pages = _markers(_PAGE, text, comments, int)
        self._anchors = _markers(_ANCHOR, text, comments, str)

    # -- offsets and locations

    def _orig(self, n: int, end: bool = False) -> int:
        """Original offset where normalized character n starts (or, with end=True, ends)."""
        k = bisect_right(self._seg_n, n) - 1
        ns, a, b = self._seg_n[k], self._seg_a[k], self._seg_b[k]
        ne = self._seg_n[k + 1] if k + 1 < len(self._seg_n) else len(self._norm)
        if ne - ns == b - a:
            return a + n - ns + (1 if end else 0)
        return b if end else a

    def _nend(self, j: int) -> int:
        """Normalized offset just after token j-1."""
        return self._tok_starts[j - 1] + len(self._toks[j - 1])

    def _span(self, i: int, j: int) -> tuple[int, int]:
        return self._orig(self._tok_starts[i]), self._orig(self._nend(j) - 1, end=True)

    def _page_at(self, pos: int) -> int | None:
        positions, pages = self._pages
        # Text before the first marker (a title block) counts as the first page, as in ingest.
        return pages[max(0, bisect_right(positions, pos) - 1)] if pages else None

    def _anchor_at(self, pos: int) -> str | None:
        positions, ids = self._anchors
        k = bisect_right(positions, pos) - 1
        return ids[k] if k >= 0 else None

    def _section_at(self, pos: int) -> dict | None:
        found = None
        for sec in self.sections:  # sorted by start, so the last one containing pos is the deepest
            if sec["start"] <= pos and (sec.get("end") is None or pos < sec["end"]):
                found = sec
        return found

    def _outside_comment(self, pos: int) -> int:
        """`pos`, moved back to the start of an HTML comment containing it."""
        starts, ends = self._comments
        k = bisect_right(starts, pos) - 1
        return starts[k] if k >= 0 and pos < ends[k] else pos

    def _context(self, start: int, end: int, reach: int = 400) -> str:
        """The span plus up to one sentence before and after it (headings skipped), markers removed."""
        text = self.text
        a, b = self._outside_comment(max(0, start - reach)), self._outside_comment(min(len(text), end + reach))
        before, span = _strip_comments(text[a:start]), _strip_comments(text[start:end])
        full = before + span + _strip_comments(text[end:b])
        cuts = [0, *(m.end() for m in _SENTENCE.finditer(full)), len(full)]  # sentence and paragraph starts
        i = bisect_right(cuts, len(before)) - 1
        j = bisect_left(cuts, len(before) + len(span))
        lo = cuts[i - 1] if i > 0 and not full[cuts[i - 1] : cuts[i]].lstrip().startswith("#") else cuts[i]
        hi = cuts[j + 1] if j + 1 < len(cuts) and not full[cuts[j] : cuts[j + 1]].lstrip().startswith("#") else cuts[j]
        return " ".join(full[lo:hi].split())

    # -- matching

    def _joinable(self, k: int) -> bool:
        """Tokens k and k+1 are one word hyphenated at a line or page break: "contin-\\nuous", "de- <!-- page 3
        --> fined". Extraction joins lines with spaces, so a hyphen followed by any whitespace counts."""
        a, b = self._orig(self._nend(k + 1) - 1, end=True), self._orig(self._tok_starts[k + 1])
        return (b - a <= _MAX_BREAK and self._toks[k].isalpha() and self._toks[k + 1].isalpha()
                and _HYPHEN_BREAK.fullmatch(self.text, a, b) is not None)

    def _same_words(self, qt: list[str], i: int, j: int) -> bool:
        """The quote's words are source tokens i..j-1, the halves of a hyphenated word counting as one."""
        p, k = 0, i
        while p < len(qt) and k < j:
            if qt[p] == self._toks[k]:
                p, k = p + 1, k + 1
            elif k + 1 < j and qt[p] == self._toks[k] + self._toks[k + 1] and self._joinable(k):
                p, k = p + 1, k + 2
            else:
                return False
        return p == len(qt) and k == j

    def _exact(self, qt: list[str], lo: int, hi: int, limit: int) -> tuple[list[tuple[int, int]], int]:
        """Occurrences of the quote's words starting at tokens lo..hi-1: up to `limit` token spans, and how many
        there are. The letters-and-digits stream proposes candidates; the words decide ("103" is not "10^{-3}")."""
        sq, starts = "".join(qt), self._sq_starts
        spans: list[tuple[int, int]] = []
        count = steps = 0
        if not sq or lo >= hi:
            return spans, 0
        stop = starts[hi - 1] + len(sq)  # an occurrence must start at or before token hi-1
        pos = self._squashed.find(sq, starts[lo], stop)
        while pos != -1 and steps < 1000:
            steps += 1
            i, j = bisect_left(starts, pos), bisect_left(starts, pos + len(sq))
            if starts[i] == pos and starts[j] == pos + len(sq) and self._same_words(qt, i, j):
                count += 1
                if len(spans) < limit:
                    spans.append((i, j))
                pos = self._squashed.find(sq, pos + len(sq), stop)
            else:
                pos = self._squashed.find(sq, pos + 1, stop)
        return spans, count

    def _fuzzy(self, qt: list[str], lo: int, hi: int) -> tuple[int, int] | None:
        """Token span of the best window (most coverage, then score) for quote tokens qt, starting at lo..hi-1."""
        n, key = len(qt), self._gram_hash.__getitem__
        ranges = []
        for q in range(n - 3):
            h = hash((qt[q], qt[q + 1], qt[q + 2], qt[q + 3]))
            ranges.append((bisect_left(self._gram_pos, h, key=key), bisect_right(self._gram_pos, h, key=key), q))
        votes: dict[int, int] = {}  # implied quote start -> number of shared 4-grams
        budget = _MAX_VOTES
        for a, b, q in sorted(ranges, key=lambda r: r[1] - r[0]):  # rarest 4-grams first
            budget -= b - a
            if budget < 0:
                break
            for p in self._gram_pos[a:b]:
                votes[p - q] = votes.get(p - q, 0) + 1
        pad, max_width = n // 3 + 2, int(1.3 * n)
        limit = max(10, min(_MAX_CANDIDATES, 16000 // max(n, 1)))
        best, examined = None, []
        for d in sorted(votes, key=lambda d: (-votes[d], d)):
            if len(examined) >= limit:
                break
            if d + n <= lo or d - pad >= hi or any(abs(d - e) <= pad // 2 for e in examined):
                continue
            examined.append(d)
            w0 = max(lo, d - pad)
            blocks = SequenceMatcher(None, qt, self._toks[w0 : d + n + pad], autojunk=False).get_matching_blocks()
            blocks = blocks[:-1]
            # Trim the window to runs of matching blocks no wider than 1.3x the quote.
            for x, first in enumerate(blocks):
                if w0 + first.b >= hi:
                    break
                matched = 0
                for last in blocks[x:]:
                    width = last.b + last.size - first.b
                    if width > max_width:
                        break
                    matched += last.size
                    rank = (matched / n, 2 * matched / (n + width))
                    if best is None or rank > best[0]:
                        best = (rank, w0 + first.b, w0 + last.b + last.size)
        return (best[1], best[2]) if best else None

    def _aside(self, i: int, j: int) -> bool:
        """Source tokens i..j-1 are an aside a quote may skip: lines of their own (a heading, footnote or caption
        between quoted lines) or a bracketed citation such as "(Ho et al., 2020)" or "[12, 15]"."""
        if i == 0 or j >= len(self._toks):
            return False
        before, after = self._norm[self._nend(i) : self._tok_starts[i]], self._norm[self._nend(j) : self._tok_starts[j]]
        if "\n" in before and "\n" in after:
            return True
        words, opening = self._toks[i:j], before.rstrip()[-1:]
        return (opening in ("(", "[") and after.lstrip()[:1] in (")", "]")
                and not any(w in _NEGATIONS or w in _QUANTIFIERS for w in words)
                and (opening == "[" or any(_YEAR.fullmatch(w) for w in words)))

    def _fuzzy_hit(self, qt: list[str], i: int, j: int) -> _Hit:
        blocks = SequenceMatcher(None, qt, self._toks[i:j], autojunk=False).get_matching_blocks()
        if len(blocks) > 1:
            # An unmatched quote word at an edge is compared with the source word next to the window when one
            # contains the other: a quote starting inside "impossible" must not pass as "possible".
            lead, trail = blocks[0].a, len(qt) - blocks[-2].a - blocks[-2].size
            if lead and i > 0 and _nested(qt[lead - 1], self._toks[i - 1], str.endswith):
                i -= 1
            if trail and j < len(self._toks) and _nested(qt[-trail], self._toks[j], str.startswith):
                j += 1
        window = self._toks[i:j]
        sm = SequenceMatcher(None, qt, window, autojunk=False)
        # Words split differently ("α t" for "αt") count as covered, though not for the score; the critical
        # checks below still catch "1 5" for "15" and "a typical" for "atypical".
        matched = sum(b.size for b in sm.get_matching_blocks()) + sum(
            i2 - i1 for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag == "replace" and "".join(qt[i1:i2]) == "".join(window[j1:j2]))
        # A skipped aside still costs score, but its numbers are not a misquote.
        ops = [op for op in sm.get_opcodes() if not (op[0] == "insert" and self._aside(i + op[3], i + op[4]))]
        diffs = _critical_diffs(qt, window, ops)
        ok = matched / len(qt) >= _MIN_COVERAGE and not diffs
        note = "differs in a negation/number/quantifier: " + "; ".join(diffs) if diffs else None
        return _Hit("approximate" if ok else "not_found", sm.ratio(), matched, len(qt), i, j, note, bool(diffs))

    def _find(self, qt: list[str], lo: int, hi: int, limit: int) -> tuple[list[_Hit], int]:
        """Exact hits if there are any (and their count), else the best fuzzy hit, else nothing."""
        spans, count = self._exact(qt, lo, hi, limit)
        if spans:
            return [_Hit("verified", 1.0, len(qt), len(qt), i, j) for i, j in spans], count
        best = self._fuzzy(qt, lo, hi)
        return ([self._fuzzy_hit(qt, *best)] if best else []), 0

    def _chain(self, segs: list[list[str]], lo: int, hi: int) -> list[_Hit] | None:
        """Best in-order matches of ellipsis segments, each starting within _MAX_GAP of the previous end.
        Segments under 4 words can only match exactly."""
        best = None
        for hit in self._find(segs[0], lo, hi, limit=50)[0]:
            rest: list[_Hit] | None = []
            if len(segs) > 1:
                rest = self._chain(segs[1:], hit.j, bisect_right(self._tok_starts, self._nend(hit.j) + _MAX_GAP))
            if rest is not None and (best is None or _rank([hit, *rest]) > _rank(best)):
                best = [hit, *rest]
                if _rank(best)[0] == _RANK["verified"]:
                    break
        return best

    def _place_before(self, segs: list[list[str]], i: int) -> list[_Hit] | None:
        """Exact matches, in order, for short leading ellipsis segments that end within _MAX_GAP before token i."""
        hits: list[_Hit] = []
        for seg in reversed(segs):
            floor = self._tok_starts[i] - _MAX_GAP
            spans = [s for s in self._exact(seg, bisect_left(self._tok_starts, floor - 200), i, 200)[0]
                     if s[1] <= i and self._nend(s[1]) >= floor]
            if not spans:
                return None
            hits.insert(0, _Hit("verified", 1.0, len(seg), len(seg), *spans[-1]))
            i = spans[-1][0]
        return hits

    def _omits_negation(self, chain: list[_Hit]) -> bool:
        """The source text skipped between consecutive segments contains a negation."""
        return any("not" in _critical(self._toks, a.j, b.i) for a, b in zip(chain, chain[1:]))


def _unmatched(flags: list[str], note: str) -> Match:
    return Match("not_found", 0.0, 0.0, None, None, None, None, None, flags, note)


def match_passage(quote: str, source: SourceText, *, min_words: int = MIN_WORDS_DEFAULT,
                  max_words: int = MAX_WORDS_DEFAULT, claimed_location: str | None = None) -> Match:
    """Locate `quote` in `source`: verified (its exact words), approximate, or not_found."""
    if len(quote) > _MAX_QUOTE_CHARS:
        return _unmatched(["too_long"], f"the quote is longer than {_MAX_QUOTE_CHARS} characters; not matched")
    parts = [_TOKEN.findall(p) for p in _ELLIPSIS.split(_normalize_map(quote)[0])]
    words, elided = sum(len(p) for p in parts), len(parts) > 1
    flags = [flag for flag, on in (("too_short", words < min_words),
                                   ("too_long", words > min(max_words, MAX_QUOTE_WORDS)), ("ellipsis", elided)) if on]
    if words > MAX_QUOTE_WORDS:
        return _unmatched(flags, f"the quote has more than {MAX_QUOTE_WORDS} words; not matched")
    segs = [p for p in parts if p]
    notes: list[str] = []
    chain = None
    if not segs:
        notes.append("the quote has no words to match")
    elif not elided:
        hits, count = source._find(segs[0], 0, len(source._toks), limit=1)
        chain = hits or None
        if count > 1:
            notes.append(f"occurs {count} times")
    else:
        # The first segment of 3+ words anchors the chain; shorter segments before it are placed afterwards.
        first = next((k for k, s in enumerate(segs) if len(s) >= 3), None)
        chain = source._chain(segs[first:], 0, len(source._toks)) if first is not None else None
        lead = source._place_before(segs[:first], chain[0].i) if chain else None
        chain = lead + chain if chain and lead is not None else None
        if chain is None:
            notes.append("no segment of 3+ words between the ellipses" if first is None else
                         f"the segments do not occur in order, each within {_MAX_GAP} characters of the previous")
    if chain is None:
        return _unmatched(flags, "; ".join(notes) or "no similar passage found")
    status = min((h.status for h in chain), key=_RANK.__getitem__)
    critical = any(h.critical for h in chain)
    notes += [h.note for h in chain if h.note]
    if elided:  # an elided quote is at best approximate, and the text it leaves out is inspected
        status = "approximate" if status == "verified" else status
        if source._omits_negation(chain):
            status, critical = "not_found", True
            notes.append("the text left out at an ellipsis contains a negation")
    if critical:
        flags.append("critical_difference")
    start, end = _widen(source.text, *source._span(chain[0].i, chain[-1].j))
    location = describe_location(source, start, end)
    if claimed_location and _location_mismatch(claimed_location, location, bool(source._anchors[0])):
        flags.append("location_mismatch")
    return Match(status, min(h.score for h in chain), sum(h.matched for h in chain) / sum(h.total for h in chain),
                 start, end, _clean(source.text[start:end]).strip(), source._context(start, end), location,
                 flags, "; ".join(notes) or None)


def describe_location(source: SourceText, start: int, end: int) -> dict:
    """Pages, paragraph anchor and (deepest) section of the original span [start, end), plus a short label."""
    page_start, page_end = source._page_at(start), source._page_at(max(start, end - 1))
    anchor, section = source._anchor_at(start), source._section_at(start)
    parts = [f"{section.get('id') or ''} {section.get('title') or ''}".strip() if section else ""]
    if page_start is not None:
        parts.append(f"p. {page_start}" if page_start == page_end else f"pp. {page_start}\u2013{page_end}")
    if anchor:
        parts.append(f"¶ {anchor}")
    return {"page_start": page_start, "page_end": page_end, "anchor": anchor,
            "section_id": section.get("id") if section else None,
            "section_title": section.get("title") if section else None,
            "label": ", ".join(p for p in parts if p) or f"chars {start}\u2013{end}"}


def _same_branch(a: str, b: str) -> bool:
    """One dotted label is a prefix of the other: "3" and "3.2", "S3" and "S3.p2"."""
    pa, pb = a.lower().split("."), b.lower().split(".")
    k = min(len(pa), len(pb))
    return pa[:k] == pb[:k]


def _location_mismatch(claim: str, loc: dict, has_anchors: bool) -> bool:
    """The claim names pages, a section number or an anchor, and none agrees with where the span was found."""
    pages: set[int] = set()
    for m in _CLAIM_PAGE.finditer(claim):
        a, b = sorted((int(m.group(1)), int(m.group(2) or m.group(1))))
        pages.update(range(a, min(b, a + 100) + 1))
    if pages and loc["page_start"] is not None and not any(loc["page_start"] <= p <= loc["page_end"] for p in pages):
        return True
    number = _SECTION_NUMBER.match(loc["section_title"] or "")
    claimed = _CLAIM_SECTION.findall(claim)
    if claimed and number and not any(_same_branch(c, number.group(1)) for c in claimed):
        return True
    anchors = _CLAIM_ANCHOR.findall(claim)
    return bool(anchors and has_anchors and not any(loc["anchor"] and _same_branch(a, loc["anchor"]) for a in anchors))
