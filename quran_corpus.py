"""Local Hafs corpus and bounded, lazy phonetic references for all 114 surahs."""
from functools import lru_cache
from threading import Lock
from collections import Counter, defaultdict
from bisect import bisect_right
import re
import unicodedata
from rapidfuzz.distance import OSA

from quran_transcript import Aya, NoPhonemesSearchResult, PhoneticSearch, chunck_phonemes, quran_phonetizer

from recitation import MOSHAF, Reference, basic_form, make_reference
from word_alignment import lexical


_ARABIC_WORD = re.compile(r"[\u0621-\u063a\u0640-\u065f\u0670\u0671\u06d6-\u06ed]+")
_PERSIAN = str.maketrans({"ی": "ي", "ک": "ك"})
_INTRODUCTION = ("اعوذ", "بالله", "من", "الشيطان", "الرجيم")
# The spoken names of disconnected opening letters can be separate ASR words.
_INITIAL_NAMES = {
    "ا": "الف", "ل": "لام", "م": "ميم", "ص": "صاد", "ر": "راء", "ك": "كاف",
    "ه": "هاء", "ي": "ياء", "ع": "عين", "ط": "طاء", "س": "سين", "ح": "حاء",
    "ق": "قاف", "ن": "نون",
}
_INITIAL_SURAH = {2, 3, 7, 10, 11, 12, 13, 14, 15, 19, 20, 26, 27, 28, 29, 30,
                  31, 32, 36, 38, 40, 41, 42, 43, 44, 45, 46, 50, 68}


def _query_words(text):
    # Preserve the source's diacritic order (lexical uses the official spelling
    # for superscript alifs). Compatibility glyphs occur in some ASR imports.
    if any("\ufb50" <= ch <= "\ufeff" for ch in text):
        text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PERSIAN)
    return tuple(key for raw in _ARABIC_WORD.findall(text) if (key := lexical(raw)))


def _close_word(left, right):
    if left == right:
        return True
    left = left.removeprefix("ال")
    right = right.removeprefix("ال")
    return min(len(left), len(right)) >= 3 and OSA.distance(left, right, score_cutoff=1) <= 1


class _VerseTextIndex:
    """Look up spoken words in Quran order, including across ayah boundaries."""

    def __init__(self):
        aya = Aya(1, 1)
        self.words = []
        self.starts = []
        self.refs = []
        self.postings = defaultdict(list)
        self.pairs = defaultdict(list)
        self.initials = defaultdict(list)
        aliases = defaultdict(set)
        different_spellings = []
        for surah in range(1, 115):
            for number in range(1, aya.set_new(surah, 1).get().num_ayat_in_sura + 1):
                item = aya.set_new(surah, number).get()
                self.starts.append(len(self.words))
                self.refs.append((surah, number, item.sura_name, item.uthmani))
                modern = _query_words(item.imlaey)
                uthmani = _query_words(item.uthmani)
                if uthmani != modern:
                    different_spellings.append((uthmani[-24:], modern[-24:]))
                if (number == 1 and surah in _INITIAL_SURAH) or (surah, number) == (42, 2):
                    self.initials[modern[0]].append(len(self.starts) - 1)
                if len(modern) == len(uthmani):
                    for other, key in zip(uthmani, modern):
                        if other != key:
                            aliases[other].add(key)
                for key in modern:
                    pos = len(self.words)
                    self.postings[key].append(pos)
                    if pos:
                        self.pairs[self.words[-1], key].append(pos - 1)
                    self.words.append(key)
        self.words = tuple(self.words)
        self.aliases = {key: next(iter(values)) for key, values in aliases.items()
                        if len(values) == 1 and key not in self.postings}
        # The two scripts sometimes use different word boundaries (e.g. يا موسى).
        self.alternate_spellings = defaultdict(set)
        for old, modern in different_spellings:
            old = tuple(self.aliases.get(word, word) for word in old)
            if old != modern:
                self.alternate_spellings[old].add(modern)
        self.initial_names = [(tuple(_INITIAL_NAMES[ch] for ch in key), key)
                              for key in sorted(self.initials, key=len, reverse=True)]
        self.lookup = lru_cache(maxsize=512)(self._lookup)

    def verse(self, position):
        return bisect_right(self.starts, position) - 1

    def whole(self, start, end):
        verse = self.verse(start)
        stop = self.starts[verse + 1] if verse + 1 < len(self.starts) else len(self.words)
        return start == self.starts[verse] and end == stop

    def endpoint(self, start, end, query):
        verse = self.verse(end - 1)
        boundary = self.starts[verse]
        if start >= boundary:
            return verse
        # An isolated conjunction from the next verse is a poor reason to
        # advance. Require two independently supported words in the new ayah,
        # or its complete text when it genuinely consists of just one word.
        suffix = self.words[boundary:end]
        if len(suffix) == 1:
            if self.whole(boundary, end) and query[-1] == suffix[0]:
                return verse
            return verse - 1
        heard = iter(query[-(len(suffix) + 2):])
        supported = 0
        for word in suffix:
            if any(_close_word(word, candidate) for candidate in heard):
                supported += 1
                if supported == 2:
                    return verse
            else:
                break
        return verse - 1

    def exact(self, query):
        n = len(query)
        if n == 1:
            anchors = self.postings.get(query[0], ())
            positions = list(anchors) if any(self.whole(p, p + 1) for p in anchors) else []
        else:
            pairs = enumerate(zip(query, query[1:]))
            seeds = ((i, self.pairs.get(pair, ())) for i, pair in pairs)
            offset, anchors = min(seeds, key=lambda item: len(item[1]))
            starts = (position - offset for position in anchors if position >= offset)
            positions = [start for start in starts if self.words[start:start + n] == query]
            if n == 2 and positions and not any(self.whole(p, p + n) for p in positions):
                if sum(map(len, query)) < 5 or min(len(self.postings[w]) for w in query) > 50:
                    return ()
        # Prefer complete verse examples in the visible shortlist, but count
        # every indistinguishable occurrence in a different verse.
        positions.sort(key=lambda p: (not self.whole(p, p + n), p))
        return tuple(dict.fromkeys(self.endpoint(p, p + n, query) for p in positions))

    def fuzzy(self, query):
        n = len(query)
        if n < 3 or len(set(query)) < 3:
            return ()
        budget = min(3, max(1, (n + 1) // 4))
        # Moving a word boundary can cost two token edits even when the
        # consonant string changes by just one letter.
        span_budget = max(budget, 2)
        # A paired seed is useful even when every individual word is common.
        # Votes prioritize consistent positions. No per-query Quran-wide scan.
        seeds = []
        for i, word in enumerate(query):
            postings = self.postings.get(word, ())
            if 0 < len(postings) <= 100:
                seeds.append((len(postings), i, postings, 1))
        for i, pair in enumerate(zip(query, query[1:])):
            places = self.pairs.get(pair, ())
            if 0 < len(places) <= 100:
                seeds.append((len(places), i, places, 2))
        starts = Counter()
        for _, i, places, weight in sorted(seeds, key=lambda seed: seed[0])[:8]:
            for pos in places:
                starts[pos - i] += weight
        if not starts:
            return ()
        ranked = starts.most_common()
        if len(ranked) > 64 and ranked[63][1] == ranked[0][1]:
            return ()
        offsets = {start + shift for start, _ in ranked[:64]
                   for shift in range(-span_budget, span_budget + 1)
                   if 0 <= start + shift < len(self.words)}
        characters = ''.join(query)
        available = set(query)
        best = {}
        for start in offsets:
            for length in range(max(2, n - span_budget), n + span_budget + 1):
                end = start + length
                if end > len(self.words):
                    continue
                segment = self.words[start:end]
                distance = OSA.distance(query, segment, score_cutoff=span_budget)
                shared = available.intersection(segment)
                if distance > span_budget or len(shared) < 2:
                    continue
                char_cost = OSA.distance(characters, ''.join(segment))
                boundary_variant = char_cost <= 1
                if not boundary_variant and (distance > budget or len(shared) < min(3, n - 1)):
                    continue
                # The score must have evidence in the current speech as well
                # as the prefix, particularly for a new verse at a boundary.
                if not any(_close_word(q, w) for q in query[-2:] for w in segment[-2:]):
                    continue
                same_words = len(shared) == len(available)
                if n == 3 and ((char_cost > 2 and not same_words) or
                               min(len(self.postings[w]) for w in shared) > 65):
                    continue
                if n >= 4 and not same_words and char_cost > max(6, len(characters) // 3):
                    continue
                verse = self.endpoint(start, end, query)
                score = (1 if boundary_variant else distance, char_cost)
                if verse not in best or score < best[verse]:
                    best[verse] = score
        if not best:
            return ()
        edits, char_cost = min(best.values())
        ranked_verses = sorted(best.items(), key=lambda row: (row[1], row[0]))
        return tuple(verse for verse, score in ranked_verses
                     if score[0] == edits and score[1] <= char_cost + 1)

    def _lookup(self, query):
        if not query:
            return ()
        # The customary invocation before recitation sounds close to a clause
        # in 16:98. Only the words actually following that invocation locate an ayah.
        for i in range(len(query) - len(_INTRODUCTION), -1, -1):
            if query[i:i + len(_INTRODUCTION)] == _INTRODUCTION:
                query = query[i + len(_INTRODUCTION):]
                if not query:
                    return ()
                break
        # Decode the letter names only where they form a known opening. A
        # spoken "alif lam mim" alone is ambiguous between its six openings,
        # not between every interrogative أَلَمْ in the corpus.
        for names, key in self.initial_names:
            if query[:len(names)] == names:
                if len(query) == len(names):
                    return tuple(self.initials[key])
                query = (key,) + query[len(names):]
                break
        # A new recitation can follow an unrelated one without starting a new
        # microphone session. Require a substantial new suffix before relocking.
        sizes = [len(query), *(size for size in (8, 5, 4) if size < len(query))]
        for size in sizes:
            part = query[-size:]
            candidates = self.exact(part)
            for spelling in sorted(self.alternate_spellings.get(part, ())):
                candidates = tuple(dict.fromkeys((*candidates, *self.exact(spelling))))
            if not candidates:
                candidates = self.fuzzy(part)
            if candidates:
                return candidates
        return ()

    def locate(self, text):
        suffix = text[-2048:] if isinstance(text, str) else ""
        query = tuple(self.aliases.get(word, word) for word in _query_words(suffix)[-24:])
        found = self.lookup(query)
        candidates = []
        for index in found[:3]:
            surah, ayah, name, verse = self.refs[index]
            candidates.append(dict(surah=surah, ayah=ayah, surah_name=name, text=verse))
        status = "probable" if len(found) == 1 else "ambiguous" if found else "none"
        return dict(status=status, candidates=candidates, total_candidates=len(found))


@lru_cache(maxsize=1)
def _verse_text_index():
    return _VerseTextIndex()


class QuranCorpus:
    def __init__(self, cache_size=128):
        self._aya = Aya(1, 1)
        self.surahs = []
        for surah in range(1, 115):
            item = self._aya.set_new(surah, 1).get()
            self.surahs.append(dict(surah=surah, name=item.sura_name,
                                    ayah_count=item.num_ayat_in_sura))
        self._counts = {item["surah"]: item["ayah_count"] for item in self.surahs}
        self._reference = lru_cache(maxsize=cache_size)(self._make_reference)
        self._searcher = None
        self._search_lock = Lock()
        self._word_spellings = None
        self._passages = {}
        self._postings = defaultdict(list)
        self._word_phones = lru_cache(maxsize=256)(self._lookup_word_phones)

    def word_phones(self, heard_word):
        """A conservative pronunciation for checking an already-transcribed word."""
        return self._word_phones(lexical(heard_word))

    def _load_word_spellings(self):
        with self._search_lock:
            if self._word_spellings is None:
                spellings = {}
                for surah,count in self._counts.items():
                    for ayah in range(1,count+1):
                        item = self._aya.set_new(surah,ayah).get()
                        keys=tuple(lexical(word) for word in item.uthmani_words)
                        self._passages[surah,ayah]=keys
                        for pos,(key,word) in enumerate(zip(keys,item.uthmani_words)):
                            spellings.setdefault(key, set()).add(word)
                            self._postings[key].append((surah,ayah,pos))
                self._word_spellings = spellings

    def spoken_passage(self, tokens):
        """Propose one local comparison region from raw ASR. No text/feedback is emitted."""
        query=tuple(lexical(t['text']) for t in tokens)
        if not 3<=len(query)<=10 or len(set(query))<3:
            return None
        self._load_word_spellings()
        seeds=sorted(((len(self._postings.get(w,())),i,w) for i,w in enumerate(query)
                      if 0<len(self._postings.get(w,()))<=45),key=lambda seed:seed[0])
        if not seeds:
            return None
        budget=2 if len(query)>=6 else 1
        starts=set()
        for _,index,word in seeds[:3]:
            for s,a,p in self._postings[word]:
                starts.update((s,a,start) for start in range(max(0,p-index-budget),p-index+budget+1))
        best=budget+1
        places=set()
        for s,a,start in starts:
            verse=self._passages[s,a]
            for length in range(len(query)-budget,len(query)+budget+1):
                if start+length>len(verse):
                    continue
                segment=verse[start:start+length]
                if len(set(segment).intersection(query))<2+budget:
                    continue
                distance=OSA.distance(query,segment,score_cutoff=budget)
                if distance<best:
                    best,places=distance,set()
                if distance==best and distance<=budget:
                    places.add((s,a,start))
        if not places:
            return None
        first,last=min(places),max(places)
        # An internal deletion can shift two equally good starts by one word.
        # Refrains at different positions remain unresolved.
        if first[:2]!=last[:2] or last[2]-first[2]>best:
            return None
        return first

    def prepare_live(self):
        """Warm local lookup indexes before accepting time-sensitive audio."""
        self._load_searcher()
        self._load_word_spellings()
        _verse_text_index()

    def text_location(self, text):
        """Display-only reference candidates from the most recent raw Arabic text."""
        return _verse_text_index().locate(text)

    def _lookup_word_phones(self, key):
        if not key:
            return ()
        self._load_word_spellings()
        # Vowel variants may have the same consonantal identity. If the
        # spelling permits different consonants or long vowels, abstain.
        template, skeleton = (), ()
        for word in sorted(self._word_spellings.get(key, ())):
            phones = tuple(chunck_phonemes(quran_phonetizer(word,MOSHAF,remove_spaces=True).phonemes))
            letters = tuple(basic_form(p)[:1] for p in phones)
            if len(letters)<3 or (skeleton and letters!=skeleton):
                return ()
            if not template:
                template, skeleton = phones, letters
        return template

    def _validate_position(self, surah, ayah):
        if type(surah) is not int or surah not in self._counts:
            raise ValueError("شمارهٔ سوره باید بین ۱ تا ۱۱۴ باشد.")
        if type(ayah) is not int or not 1 <= ayah <= self._counts[surah]:
            raise ValueError("شمارهٔ آیه در این سوره معتبر نیست.")

    def _make_reference(self, surah, ayah):
        text = self._aya.set_new(surah, ayah).get().uthmani
        return make_reference(ayah, text, surah=surah)

    def reference(self, surah, ayah) -> Reference:
        self._validate_position(surah, ayah)
        return self._reference(surah, ayah)

    def verses(self, surah, start=1, limit=10):
        self._validate_position(surah, start)
        if type(limit) is not int or limit < 1:
            raise ValueError("تعداد آیات باید یک عدد مثبت باشد.")
        # Text browsing must not phonetize a whole surah or populate the LRU.
        result = []
        for ayah in range(start, min(start + limit, self._counts[surah] + 1)):
            item = self._aya.set_new(surah, ayah).get()
            result.append(dict(surah=surah, ayah=ayah, text=item.uthmani,
                               words=item.uthmani_words))
        return result

    def _load_searcher(self):
        with self._search_lock:
            if self._searcher is None:
                self._searcher = PhoneticSearch()
            return self._searcher

    def search(self, phonemes, error_ratio=0.1):
        if not isinstance(phonemes, str) or not phonemes.strip():
            return []
        if not 0 <= error_ratio <= 1:
            raise ValueError("error_ratio must be between 0 and 1")
        try:
            return self._load_searcher().search(phonemes, error_ratio=error_ratio)
        except NoPhonemesSearchResult:
            return []

    def validate(self):
        """Check every local text/metadata record without filling phonetic caches."""
        total = 0
        for surah in self.surahs:
            for ayah in range(1, surah["ayah_count"] + 1):
                item = self._aya.set_new(surah["surah"], ayah).get()
                if not item.uthmani or not item.imlaey or not item.uthmani_words:
                    raise RuntimeError(f"Incomplete local Quran text: {surah['surah']}:{ayah}")
                total += 1
        if len(self.surahs) != 114 or total != 6236:
            raise RuntimeError("The local Hafs corpus must contain 114 surahs and 6236 ayahs")
        return total
