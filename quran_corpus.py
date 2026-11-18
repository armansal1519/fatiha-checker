"""Local Hafs corpus and bounded, lazy phonetic references for all 114 surahs."""
from functools import lru_cache
from threading import Lock
from collections import defaultdict
from rapidfuzz.distance import OSA

from quran_transcript import Aya, NoPhonemesSearchResult, PhoneticSearch, chunck_phonemes, quran_phonetizer

from recitation import MOSHAF, Reference, basic_form, make_reference
from word_alignment import lexical


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
