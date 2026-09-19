"""Bounded rolling-window alignment. Live colors are provisional observations.

The model is unchanged. Unread prefixes/suffixes and clipped edge words are
never turned into deletion errors. A client cursor disambiguates repeated text.
"""
from collections import defaultdict
from time import perf_counter

from quran_transcript import chunck_phonemes

from recitation import basic_form, describe, make_reference

WINDOW_SECONDS = 10
MAX_LIVE_SECONDS = 12


class LiveFollower:
    def __init__(self, references=None):
        self.references = references or {i: make_reference(i) for i in range(1, 8)}
        self.model_reference = make_reference(1, " ".join(r.text for r in self.references.values()))
        self.words, self.groups, self.owners = [], [], []
        self.word_ranges, self.ayah_starts = [], {}
        self.optional_wasl = set()
        self.end_vowels = {}
        for ayah, ref in self.references.items():
            word_offset = len(self.words)
            self.ayah_starts[ayah] = word_offset
            phone_to_char = {}
            for char, mapping in enumerate(ref.phonetic.mappings):
                if mapping is not None:
                    for pos in range(*mapping.pos):
                        phone_to_char.setdefault(pos, char)
            phone_pos = 0
            for group in chunck_phonemes(ref.phonetic.phonemes):
                char = phone_to_char.get(phone_pos, phone_to_char.get(phone_pos - 1, 0))
                word = next((i for i, (a, b) in enumerate(ref.word_spans) if a <= char < b), len(ref.words) - 1)
                if phone_pos == 0 and ref.words[0].startswith("ٱ"):
                    self.optional_wasl.add(len(self.groups))
                self.groups.append(group)
                self.owners.append(word_offset + word)
                phone_pos += len(group)
            if ayah < 7:
                self.end_vowels[len(self.groups) - 1] = ref.text[-1] if ref.text[-1] in "َُِ" else ""
            for i, word in enumerate(ref.words):
                self.words.append(dict(index=word_offset + i, ayah=ayah, word_index=i, word=word))
        for word in range(len(self.words)):
            indices = [i for i, owner in enumerate(self.owners) if owner == word]
            self.word_ranges.append((min(indices), max(indices) + 1))

    def align(self, predicted, cursor=0, start_ayah=1, settled=False):
        heard = chunck_phonemes(predicted.replace(" ", "").strip())
        floor = self.ayah_starts[start_ayah]
        cursor = max(cursor, floor)
        empty = dict(status="listening", cursor=cursor, updates=[], current_word=None,
                     message="برای دنبال‌کردن قرائت، چند کلمهٔ پیوسته بخوانید.")
        if len(heard) < 3:
            return empty
        # A ten-second window includes prior context; do not search distant
        # repeated phrases or retroactively color words before the chosen start.
        first_word = max(floor, cursor - 14)
        last_word = min(len(self.words), cursor + 12)
        left = self.word_ranges[first_word][0]
        right = self.word_ranges[last_word - 1][1]
        expected = self.groups[left:right]
        n, m = len(expected), len(heard)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        moves = [[None] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i
            moves[i][0] = "insert"
            h = basic_form(heard[i - 1])
            for j in range(1, n + 1):
                e = basic_form(expected[j - 1])
                same_base = e[:1] == h[:1]
                cost = 0 if e == h else 0.15 if same_base else 1.1
                candidates = [(dp[i - 1][j - 1] + cost, "pair"),
                              (dp[i][j - 1] + 1, "delete"),
                              (dp[i - 1][j] + 1, "insert")]
                # Chunking joins identical consonants across a verse boundary
                # (e.g. الرحيمْ مَالِك). Retain both word owners in that case.
                if j >= 2 and self.owners[left + j - 2] != self.owners[left + j - 1]:
                    if basic_form(expected[j - 2] + expected[j - 1]) == h:
                        candidates.insert(0, (dp[i - 1][j - 2], "merge"))
                dp[i][j], moves[i][j] = min(candidates, key=lambda item: item[0])
        ends = [j for j in range(1, n + 1) if self.owners[left + j - 1] >= max(floor, cursor - 2)]
        end = min(ends, key=lambda j: (dp[m][j], abs(self.owners[left + j - 1] + 1 - cursor)))
        cost = dp[m][end]
        i, j, path = m, end, []
        while i > 0:
            move = moves[i][j]
            if move == "pair":
                path.append((left + j - 1, heard[i - 1], "replace")); i -= 1; j -= 1
            elif move == "merge":
                path.append((left + j - 1, expected[j - 1], "replace"))
                path.append((left + j - 2, expected[j - 2], "replace")); i -= 1; j -= 2
            elif move == "delete":
                path.append((left + j - 1, "", "delete")); j -= 1
            else:
                path.append((min(left + j, right - 1), heard[i - 1], "insert")); i -= 1
        path.reverse()
        ref_path = [p for p, _, op in path if op != "insert"]
        if not ref_path:
            return empty
        a, b = min(ref_path), max(ref_path) + 1
        matches = sum(basic_form(self.groups[p])[:1] == basic_form(h)[:1] for p, h, op in path if op == "replace")
        if cost / max(m, b - a) > 0.38 or matches < max(3, m * 0.45):
            return dict(empty, message="جای خواندن هنوز روشن نیست؛ چند کلمهٔ بعد را پیوسته بخوانید.")
        first, last = self.owners[a], self.owners[b - 1]
        errors = defaultdict(list)
        for p, h, operation in path:
            e = "" if operation == "insert" else self.groups[p]
            if basic_form(e) == basic_form(h):
                continue
            # Accept either a stop or the proper connecting vowel at ayah ends.
            if p in self.end_vowels and basic_form(h) == basic_form(e) + self.end_vowels[p]:
                continue
            # Hamzat al-wasl can disappear when linked to the previous ayah.
            if p in self.optional_wasl and operation == "delete" and p > a:
                continue
            kind, message = describe(e, h, operation)
            errors[self.owners[p]].append(dict(type=kind, message=message))
        updates = []
        for word in range(first, last + 1):
            start, stop = self.word_ranges[word]
            if start < a or stop > b:
                continue  # Edge word is clipped: no verdict.
            # Do not judge the moving right edge until a pause/stop or a later
            # word provides context. Missing future words are never errors.
            if word == last and not settled:
                continue
            updates.append(dict(**self.words[word], state="difference" if errors[word] else "match", errors=errors[word]))
        reached = max(cursor, last + 1)
        complete = last == len(self.words) - 1 and any(u["index"] == last for u in updates)
        return dict(status="tracking", cursor=reached, updates=updates, current_word=last,
                    complete=complete, message="قرائت دنبال می‌شود؛ رنگ‌ها با رسیدن صدای بیشتر بازبینی می‌شوند.")

    def analyze(self, service, wave, cursor, start_ayah=1, settled=False):
        started = perf_counter()
        predicted = service.predict_live(wave)
        predicted_at = perf_counter()
        result = self.align(predicted, cursor=cursor, start_ayah=start_ayah, settled=settled)
        result["timings"] = dict(inference_ms=round((predicted_at-started)*1000, 1),
                                 alignment_ms=round((perf_counter()-predicted_at)*1000, 1))
        return result
