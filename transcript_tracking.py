"""Time-based feedback on unmodified ASR words, never canonical substitution.

Audio is owned by the transport. This module retains only text and compact
phoneme intervals so an initially ambiguous reference can be checked later.
"""
from collections import Counter
from copy import deepcopy
from math import isfinite

from quran_transcript import chunck_phonemes
import Levenshtein
from recitation import basic_form, describe
from word_alignment import align_words, lexical, missing_onset

RATE = 16000
OVERLAP_MS = 2000


_UNSEARCHED = object()


def valid_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _consonant_change(expected, heard):
    """Single written consonant substitution, excluding spelling-only alif/hamza variants."""
    left,right=lexical(expected),lexical(heard)
    if len(left)!=len(right) or len(left)<3:
        return None
    changed=[(a,b) for a,b in zip(left,right) if a!=b]
    consonants='بتثجحخدذرزسشصضطظعغفقكلمنه'
    if len(changed)==1 and all(ch in consonants for ch in changed[0]):
        return changed[0]
    return None


def _agreed_letter(raw, reference, errors):
    letters=[e for e in errors if e['type']=='letter']
    if len(letters)!=1:
        return False
    e=letters[0]
    pair=(basic_form(e['expected'])[:1],basic_form(e['heard'])[:1])
    consonants='ءبتثجحخدذرزسشصضطظعغفقكلمنهوي'
    if pair[0]==pair[1] or not all(p and p in consonants for p in pair):
        return False
    return lexical(raw)==lexical(reference) or _consonant_change(reference,raw)==pair


def aligned_groups(expected, heard, prefer_owner=None):
    """Semi-global phone alignment: unread reference edges incur no deletions."""
    n, m = len(expected), len(heard)
    if not n or m < 3:
        return None
    expected_forms = [basic_form(g['text']) for g in expected]
    heard_forms = [basic_form(g['text']) for g in heard]
    boundary_forms = [None] + [
        basic_form(a['text'] + b['text']) if a['owner'] != b['owner'] else None
        for a, b in zip(expected, expected[1:])]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    moves = [bytearray(n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = float(i)
        moves[i][0] = 3
        h = heard_forms[i - 1]
        for j in range(1, n + 1):
            e = expected_forms[j - 1]
            cost = 0 if e == h else .15 if e[:1] == h[:1] else 1.1
            choices = [(dp[i-1][j-1] + cost, 1), (dp[i][j-1] + 1, 2), (dp[i-1][j] + 1, 3)]
            # One decoded group can straddle a word/verse boundary.
            if boundary_forms[j-1] == h:
                choices.insert(0, (dp[i-1][j-2], 4))
            dp[i][j], moves[i][j] = min(choices, key=lambda pair: pair[0])
    end = min(range(1, n+1), key=lambda j: (dp[m][j],
              abs(expected[j-1]['owner']-prefer_owner) if prefer_owner is not None else 0))
    cost = dp[m][end]
    i, j, path = m, end, []
    while i:
        move = moves[i][j]
        if move == 1:
            path.append((j-1, i-1, 'replace')); i -= 1; j -= 1
        elif move == 2:
            path.append((j-1, None, 'delete')); j -= 1
        elif move == 4:
            path.extend([(j-1, i-1, 'merge'), (j-2, i-1, 'merge')]); i -= 1; j -= 2
        else:
            path.append((min(j, n-1), i-1, 'insert')); i -= 1
    path.reverse()
    paired = [(e,h) for e,h,op in path if h is not None and op != 'insert']
    matches = sum(expected_forms[e][:1] == heard_forms[h][:1] for e,h in paired)
    if cost/max(m, len(paired), 1) > .38 or matches < max(3, m*.45):
        return None
    return path


class TranscriptTracker:
    def __init__(self, corpus, start_surah=None, start_ayah=None):
        self.corpus = corpus
        self.tokens = []
        self.committed_count = 0
        self._preview_prefix_count = 0
        self._next_id = 0
        self._pending = {}
        self._word_pending = {}
        self._asr_seen = {}
        self._acoustic_supported = set()
        self._published = {}
        self._gaps = {}
        self._history = []
        self._anchor = None
        self._last_anchor_ms = None
        self._manual = None
        self._protected_ids = set()
        self._word_origin = None
        self.current_reference = None
        self._last_message = 'چند کلمهٔ پیوسته بخوانید تا محل قرائت روشن شود.'
        if start_surah is not None:
            self.corpus.reference(start_surah, start_ayah)
            self._manual = (start_surah, start_ayah)

    def summary(self):
        reviewed = sum(t['state'] in {'match', 'difference'} for t in self.tokens)
        uncertain = len(self.tokens)-reviewed
        return dict(total=len(self.tokens), reviewed=reviewed,
                    differences=sum(t['state']=='difference' for t in self.tokens),
                    uncertain=uncertain, unreviewed=uncertain, missing=len(self._gaps))

    def analysis_start(self, proposed_start):
        """Keep a complete mutable word in the next acoustic context.

        The transport still owns/limits audio. This is only a boundary hint;
        it never rewrites text or invents word timing.
        """
        start_ms = proposed_start*1000/RATE
        crossing = [t['start_ms'] for t in self.tokens[self.committed_count:]
                    if valid_time(t['start_ms']) and valid_time(t['end_ms'])
                    and t['start_ms'] < start_ms < t['end_ms']]
        return min(proposed_start, max(0, int(min(crossing)*RATE/1000))) if crossing else proposed_start

    def preview_transcript(self, result, start_sample, end_sample, settled=False):
        """Keep a visible provisional repeat while the independent check runs."""
        words=result.get('words',[])
        start_ms=start_sample*1000/RATE
        for i in range(max(1,self.committed_count),len(self.tokens)):
            first,second=self.tokens[i-1:i+1]
            key=lexical(first['text'])
            if (not key or key!=lexical(second['text']) or not valid_time(first['start_ms'])
                    or not valid_time(second['end_ms']) or first['start_ms']<start_ms):
                continue
            nearby=[lexical(w['text']) for w in words if valid_time(w.get('start_ms'))
                    and valid_time(w.get('end_ms')) and w['end_ms']+start_ms>=first['start_ms']
                    and w['start_ms']+start_ms<=second['end_ms']+200]
            if nearby.count(key)<2:
                return None
        return self.update_transcript(result,start_sample,end_sample,defer_commit=True)

    def commit_preview(self,end_sample,settled):
        previous=self.committed_count
        self.committed_count=max(previous,self._preview_prefix_count)
        self._preview_prefix_count=0
        cutoff=end_sample*1000/RATE-OVERLAP_MS
        for token in self.tokens[self.committed_count:]:
            if not settled and (not valid_time(token['end_ms']) or token['end_ms']>cutoff):
                break
            self.committed_count+=1
        if previous==self.committed_count:
            return None
        return dict(replace_from=len(self.tokens),tokens=[],committed_count=self.committed_count,
                    summary=self.summary())

    def update_transcript(self, result, start_sample, end_sample, settled=False, *, defer_commit=False):
        self._preview_prefix_count=0
        activity=result.get('audio_activity',{})
        if activity.get('status') in {'no_speech','uncertain'}:
            # No detected speech is not a revised ASR hypothesis. Retain the
            # last heard words instead of erasing them during trailing noise.
            if settled:
                self.committed_count=len(self.tokens)
            return dict(replace_from=len(self.tokens),tokens=[],committed_count=self.committed_count,
                        summary=self.summary())
        start_ms, end_ms = start_sample*1000/RATE, end_sample*1000/RATE
        revision=result.get('repeat_verified_from_ms')
        prefix = self.tokens[:self.committed_count]
        old_tail = self.tokens[self.committed_count:]
        # A word whose beginning left the window is no longer fully observable.
        # Preserve that occurrence even if its final few milliseconds overlap
        # the new window. A clipped ASR hypothesis cannot replace a whole word.
        while old_tail:
            old=old_tail[0]
            before_window=(valid_time(old['end_ms']) and old['end_ms'] <= start_ms
                           or valid_time(old['start_ms']) and old['start_ms'] < start_ms)
            untimed_leaving=(old['end_ms'] is None and old.get('_window_start_ms',start_ms)<start_ms
                             and old.get('_window_end_ms',end_ms)<=start_ms+OVERLAP_MS)
            seen,last_observation=self._asr_seen.get(old['id'],(0,-1))
            supported_stable=(seen>=2 and old['id'] in self._acoustic_supported
                              and valid_time(old['end_ms']) and old['end_ms']<=end_ms-1000
                              and (not valid_time(revision) or old['end_ms']<=start_ms+revision))
            if not (before_window or untimed_leaving or supported_stable): break
            prefix.append(old_tail.pop(0))
        replace_from = len(prefix)
        last_end = max((t['end_ms'] for t in prefix if valid_time(t['end_ms'])), default=-1)
        words = result.get('words', []) if result.get('speech', True) else []
        if not words and result.get('text', '').strip() and result.get('speech', True):
            # Retain an un-timed ASR phrase, clearly ungradable, rather than
            # inventing word boundaries or replacing it with Quran text.
            words = [dict(text=result['text'], start_ms=None, end_ms=None)]
        tail, used = [], set()
        for word in words:
            raw = word.get('text')
            if not isinstance(raw, str) or not raw.strip():
                continue
            a, b = word.get('start_ms'), word.get('end_ms')
            timed = valid_time(a) and valid_time(b) and 0 <= a <= b <= end_ms-start_ms+300
            a, b = (round(start_ms+a, 2), round(start_ms+b, 2)) if timed else (None, None)
            if timed and b <= last_end + 30:
                continue  # already committed occurrence at this audio time
            if timed and prefix:
                last = prefix[-1]
                if valid_time(last.get('start_ms')) and valid_time(last.get('end_ms')):
                    overlap = min(b,last['end_ms'])-max(a,last['start_ms'])
                    duration = min(b-a,last['end_ms']-last['start_ms'])
                    if (lexical(raw)==lexical(last['text']) and duration > 0
                            and overlap/duration >= .5):
                        # Same acoustic occurrence seen again in overlapping
                        # context. Adjacent real repeats have separate times.
                        continue
            previous = None
            if timed:
                for old in old_tail:
                    if old['id'] in used or old['text'].strip() != raw.strip() or not valid_time(old['start_ms']):
                        continue
                    intersection = min(b,old['end_ms'])-max(a,old['start_ms'])
                    if intersection > 0 or abs((a+b-old['start_ms']-old['end_ms'])/2) <= 150:
                        previous = old; used.add(old['id']); break
            if previous:
                token = dict(previous, text=raw, start_ms=a, end_ms=b)
            else:
                self._next_id += 1
                token = dict(id=f'w{self._next_id}', text=raw, start_ms=a, end_ms=b,
                             state='pending' if timed else 'uncertain', errors=[], reference=None)
            token['_window_start_ms']=start_ms
            token['_window_end_ms']=end_ms
            count,last_observation=self._asr_seen.get(token['id'],(0,-1))
            if end_sample>last_observation:
                self._asr_seen[token['id']]=(count+1,end_sample)
            tail.append(token)
        self.tokens = prefix+tail
        alive = {t['id'] for t in self.tokens}
        self._pending = {key:value for key,value in self._pending.items() if key in alive}
        self._word_pending = {key:value for key,value in self._word_pending.items() if key in alive}
        self._asr_seen = {key:value for key,value in self._asr_seen.items() if key in alive}
        self._acoustic_supported.intersection_update(alive)
        self._published = {key:value for key,value in self._published.items() if key in alive}
        self._gaps = {key:g for key,g in self._gaps.items()
                      if g.get('before_token_id') in alive and g.get('after_token_id') in alive}
        if defer_commit:
            # Protected words survive the preview. They become immutable once
            # the independent decoder confirms the same acoustic window.
            self._preview_prefix_count=len(prefix)
        else:
            self.committed_count=len(prefix)
            self.commit_preview(end_sample,settled)
        return dict(replace_from=replace_from, tokens=deepcopy(tail), committed_count=self.committed_count,
                    summary=self.summary())

    def _verse_step(self, surah, ayah, direction):
        count = self.corpus.surahs[surah-1]['ayah_count']
        if direction > 0:
            if ayah < count: return surah, ayah+1
            return (surah+1, 1) if surah < 114 else None
        if ayah > 1: return surah, ayah-1
        return (surah-1, self.corpus.surahs[surah-2]['ayah_count']) if surah > 1 else None

    def _window(self, surah, ayah, word_index=0, end=None, manual=False):
        positions = [(surah, ayah)]
        previous = (surah,ayah)
        for _ in range(6 if not manual else 0):
            previous = self._verse_step(*previous,-1)
            if previous: positions.insert(0,previous)
            else: break
        current = (surah,ayah)
        for _ in range(4):
            current = self._verse_step(*current, 1)
            if not current: break
            positions.append(current)
            if end and current >= end: break
        words, groups = [], []
        for s,a in positions:
            ref = self.corpus.reference(s,a)
            offset = len(words)
            for i,text in enumerate(ref.words):
                words.append(dict(surah=s, ayah=a, word_index=i, word=text))
            phone_map = {}
            for ci,mapping in enumerate(ref.phonetic.mappings):
                if mapping is not None:
                    for ph in range(*mapping.pos): phone_map.setdefault(ph,ci)
            pos = 0
            pieces = chunck_phonemes(ref.phonetic.phonemes)
            for gi,text in enumerate(pieces):
                char = phone_map.get(pos,phone_map.get(pos-1,0))
                wi = next((i for i,(lo,hi) in enumerate(ref.word_spans) if lo <= char < hi),len(ref.words)-1)
                groups.append(dict(text=text, owner=offset+wi,
                                   end_vowel=(ref.text[-1:] if ref.text[-1:] in 'َُِ' else '') if gi==len(pieces)-1 else '',
                                   optional_wasl=gi==0 and ref.words[0].startswith('ٱ')))
                pos += len(text)
        center = next((i for i,w in enumerate(words) if (w['surah'],w['ayah'],w['word_index']) >= (surah,ayah,word_index)),0)
        # Include a whole selected long ayah on first manual localization.
        left = max(0,center-50)
        right = len(words) if manual else min(len(words),center+50)
        selected = [g for g in groups if left <= g['owner'] < right]
        return words, selected

    def _corroborated(self, words, groups, heard, path):
        """A chance phone match in ordinary speech is not a recitation anchor."""
        if not path:
            return False
        if not self.tokens:  # A location probe alone cannot publish a word verdict.
            return True
        first, last = heard[0]['start_ms'], heard[-1]['end_ms']
        tokens = [t for t in self.tokens if valid_time(t['start_ms']) and valid_time(t['end_ms'])
                  and t['start_ms'] < last and first < t['end_ms']]
        by_word = {}
        for token in tokens:
            by_word.setdefault(lexical(token['text']),[]).append(token)
        agreed = set()
        for ref, obs, op in path:
            if obs is None or op == 'insert':
                continue
            word = lexical(words[groups[ref]['owner']]['word'])
            center = (heard[obs]['start_ms'] + heard[obs]['end_ms']) / 2
            for token in by_word.get(word, ()):
                if token['start_ms'] <= center <= token['end_ms']:
                    agreed.add(token['id'])
        return bool(agreed) and len(agreed) >= min(2, len(tokens))

    def _spoken_location(self, heard):
        search=getattr(self.corpus,'spoken_passage',None)
        if search is None:
            return None
        first,last=heard[0]['start_ms'],heard[-1]['end_ms']
        live=[t for t in self.tokens[-32:] if valid_time(t['start_ms']) and valid_time(t['end_ms'])
              and t['start_ms']<last and t['end_ms']>first][-10:]
        place=search(live)
        if place is None:
            return None
        surah,ayah,start=place
        words,expected=self._window(surah,ayah,start)
        owners={lexical(w['word']):i for i,w in enumerate(words) if (w['surah'],w['ayah'])==place[:2]
                and start<=w['word_index']<=start+len(live)+1}
        by_owner={}
        for group in expected:
            by_owner.setdefault(group['owner'],[]).append(group['text'])
        own={t['id']:by_owner.get(owners.get(lexical(t['text'])),[]) for t in live}
        from word_alignment import align_word_phone_evidence
        evidence=align_word_phone_evidence(live,heard,own,slack_ms=300)
        witnesses={lexical(t['text']) for t in live if evidence[t['id']]['status']=='supported'
                   and len(own[t['id']])>=3}
        if len(witnesses)<2:
            return None
        # Hand off to the occurrence-level judge. Each additional word/gap
        # still needs its own time-aligned acoustic evidence before publication.
        return words,expected,None

    def _locate(self, heard, previous=None):
        if not heard:
            return None
        continuation=None
        if self._anchor is not None:
            w = self._anchor
            words,expected = self._window(w['surah'],w['ayah'],w['word_index'])
            previous_owner=next((i for i,word in enumerate(words)
                                 if (word['surah'],word['ayah'],word['word_index'])==
                                    (w['surah'],w['ayah'],w['word_index'])),None)
            path = aligned_groups(expected,heard,prefer_owner=previous_owner)
            if path is not None and (self._manual or self._corroborated(words,expected,heard,path)):
                return words,expected,path
            if (self._last_anchor_ms is not None and heard[0]['start_ms']<=self._last_anchor_ms+1000
                    and self._last_anchor_ms<heard[-1]['end_ms']<=self._last_anchor_ms+8000):
                # A rolled tail may contain just the second member of a swap.
                # Continue the recent position only with fresh identity evidence
                # for a complete local word; prefer a new global lock if found.
                from word_alignment import align_word_phone_evidence
                nearby={lexical(item['word']):i for i,item in enumerate(words)
                        if (item['surah'],item['ayah'])==(w['surah'],w['ayah'])
                        and abs(item['word_index']-w['word_index'])<=8}
                live=[t for t in self.tokens[-64:] if valid_time(t['start_ms']) and valid_time(t['end_ms'])
                      and t['start_ms']<heard[-1]['end_ms'] and t['end_ms']>heard[0]['start_ms']]
                templates={t['id']:[g['text'] for g in expected if g['owner']==nearby[lexical(t['text'])]]
                           for t in live if lexical(t['text']) in nearby}
                proofs=align_word_phone_evidence(live,heard,templates,slack_ms=300)
                for token in live:
                    proof=proofs[token['id']]
                    if (token['end_ms']>self._last_anchor_ms-300 and len(proof['expected'])>=3
                            and proof['status']=='supported'):
                        continuation=(words,expected,None)
                        break
        if self._manual is not None:
            words,expected = self._window(*self._manual,manual=True)
            path = aligned_groups(expected,heard)
            return words,expected,path
        query = ''.join(g['text'] for g in heard)
        if len(heard) < 6:
            return continuation
        def position(span):
            return span.sura_idx, span.aya_idx, span.uthmani_word_idx

        candidates = self.corpus.search(query,error_ratio=.15)
        by_position = {position(r.start):r for r in candidates}
        if len(by_position) != 1 and previous:
            # Independent past phones must precede the new observation. Reusing
            # a duplicate overlap as extra evidence would resolve ties falsely.
            prefix = [g for g in previous if g['end_ms'] <= heard[0]['start_ms']][-16:]
            if (len(prefix) >= 6 and heard[0]['start_ms'] - prefix[-1]['end_ms'] <= 750
                    and heard[-1]['end_ms'] > previous[-1]['end_ms']):
                joined = ''.join(g['text'] for g in prefix) + query
                joint_hits = self.corpus.search(joined,error_ratio=0)
                joints = {position(r.end):r for r in joint_hits}
                if len(joints) == 1:
                    # Center on the CURRENT end, even if the start of the hit
                    # belongs to the preceding ayah.
                    by_position = joints
        if not by_position:
            # A repeated/missing/swapped word can break a whole-phrase
            # search. Look for shorter exact acoustic matches.
            for size in (8,6):
                if len(heard)<size: continue
                starts = {0,max(0,(len(heard)-size)//2),len(heard)-size}
                fragments = []
                for start in sorted(starts):
                    fragment=''.join(g['text'] for g in heard[start:start+size])
                    hits = {position(r.start):r for r in self.corpus.search(fragment,error_ratio=0)}
                    if hits:
                        fragments.append(hits)
                if fragments:
                    by_position = {p:r for hits in fragments for p,r in hits.items()}
                    # Distinct fragments at nearby successive words can all
                    # witness one reading. Repeated copies of the same fragment
                    # remain ambiguous, even when they share an ayah number.
                    positions = [next(iter(hits)) for hits in fragments if len(hits)==1]
                    if (len(positions)==len(fragments) and len({p[:2] for p in positions})==1
                            and max(p[2] for p in positions)-min(p[2] for p in positions)<len(heard)):
                        by_position = {positions[0]:fragments[0][positions[0]]}
                    break
        if len(by_position) != 1:
            self._last_message = 'چند محل مشابه پیدا شد؛ ادامه دهید یا محل شروع را دستی انتخاب کنید.' if by_position else 'محل خواندن هنوز روشن نیست؛ چند کلمهٔ دیگر بخوانید.'
            return continuation or self._spoken_location(heard)
        (surah,ayah,word),r = next(iter(by_position.items()))
        words,expected = self._window(surah,ayah,word,end=(r.end.sura_idx,r.end.aya_idx))
        preferred=next((i for i,w in enumerate(words) if
                        (w['surah'],w['ayah'],w['word_index'])==position(r.end)),None)
        path = aligned_groups(expected,heard,prefer_owner=preferred)
        if self._corroborated(words,expected,heard,path):
            return words,expected,path
        # A swapped/repeated clause can have a unique acoustic location and no
        # monotonic mapping to the reference phones. Pass the candidate to the
        # independent word-order check only when the raw ASR also supplies two
        # distinct local words. Each final verdict still needs its own phones.
        nearby = {lexical(w['word']) for w in words
                  if (w['surah'],w['ayah'])==(surah,ayah) and abs(w['word_index']-word)<=len(heard)}
        raw = {lexical(t['text']) for t in self.tokens
               if valid_time(t['start_ms']) and valid_time(t['end_ms'])
               and t['start_ms']<heard[-1]['end_ms'] and t['end_ms']>heard[0]['start_ms']}
        if len(nearby & raw)>=2:
            return words,expected,None
        return continuation or self._spoken_location(heard)

    def _observe(self, token, state, errors, reference, endpoint, settled, *, word=False):
        signature = (state,tuple((e.get('type'),e.get('expected'),e.get('heard')) for e in errors),
                     tuple(reference.get(k) for k in ('surah','ayah','word_index')) if reference else None)
        pending=self._word_pending if word else self._pending
        old,count,end = pending.get(token['id'],(None,0,-1))
        count = count+1 if old==signature and endpoint>end else 1
        pending[token['id']] = signature,count,endpoint
        if state=='uncertain' or settled or count>=2:
            token.update(state=state, errors=errors, reference=reference)
        elif token['state'] in {'match','difference'} and old != signature:
            # Withdraw a contradicted old verdict immediately; don't retain green.
            token.update(state='uncertain', errors=[], reference=reference)
        else:
            token['reference'] = reference
        if token['state'] not in {'match','difference'}:
            self._gaps={key:gap for key,gap in self._gaps.items()
                        if token['id'] not in {gap['before_token_id'],gap['after_token_id']}}

    def _check_phonetic(self, prediction, start_sample, end_sample, settled, located=_UNSEARCHED, letter_evidence=None):
        offset = start_sample*1000/RATE
        end_ms = end_sample*1000/RATE
        heard=[]
        for g in prediction.get('groups',[]):
            a,b = g.get('start_ms'),g.get('end_ms')
            if isinstance(g.get('text'),str) and valid_time(a) and valid_time(b) and 0 <= a <= b <= end_ms-offset+300:
                heard.append(dict(text=g['text'],start_ms=offset+a,end_ms=offset+b))
        candidates = [t for t in self.tokens if t['id'] not in self._protected_ids
                      and valid_time(t['start_ms']) and t['end_ms']>offset and t['start_ms']<end_ms]
        if not candidates:
            return False
        if located is _UNSEARCHED:
            located = self._locate(heard) if heard else None
        if located is None:
            for token in candidates:
                self._observe(token,'uncertain',[],None,end_sample,True)
            return False
        words,expected,path = located
        if not path:
            for token in candidates:
                self._observe(token,'uncertain',[],None,end_sample,True)
            return False
        matched = [(e,h) for e,h,op in path if h is not None and op!='insert']
        if not matched: return False
        first_owner = min(expected[e]['owner'] for e,_ in matched)
        last_owner = max(expected[e]['owner'] for e,_ in matched)
        self._anchor = dict(words[last_owner])
        anchor_ref = words[first_owner]
        ref = self.corpus.reference(anchor_ref['surah'],anchor_ref['ayah'])
        self.current_reference = dict(surah=anchor_ref['surah'],ayah=anchor_ref['ayah'],text=ref.text)
        by_heard = {}
        deleted = {}
        for e,h,operation in path:
            if h is None:
                deleted.setdefault(expected[e]['owner'],[]).append(e)
            else:
                by_heard.setdefault(h,[]).append((e,operation))
        owner_tokens = {}
        for token in candidates:
            indices = [i for i,g in enumerate(heard)
                       if token['start_ms']-60 <= (g['start_ms']+g['end_ms'])/2 <= token['end_ms']+60]
            owners = Counter(expected[e]['owner'] for h in indices for e,op in by_heard.get(h,[]) if op!='insert')
            if not indices or not owners:
                insertion = indices and all(by_heard.get(h) and all(op=='insert' for _,op in by_heard[h]) for h in indices)
                before = [(e,h) for e,h in matched if h<min(indices)] if indices else []
                after = [(e,h) for e,h in matched if h>max(indices)] if indices else []
                if insertion and before and after:
                    reference=dict(words[expected[after[0][0]]['owner']])
                    self._observe(token,'difference',[dict(type='insertion',operation='insert',expected='',heard=token['text'],
                                  message='این واژه میان دو بخشِ تطبیق‌یافته اضافه تشخیص داده شد.')],reference,end_sample,settled)
                    continue
                self._observe(token,'uncertain',[],None,end_sample,True)
                continue
            owner,count = owners.most_common(1)[0]
            if count/max(1,sum(owners.values())) < .6:
                self._observe(token,'uncertain',[],None,end_sample,True)
                continue
            reference = dict(words[owner])
            owner_tokens.setdefault(owner,[]).append(token)
            errors=[]
            for h in indices:
                for e,op in by_heard.get(h,[]):
                    if expected[e]['owner'] != owner and op!='insert': continue
                    exp = '' if op=='insert' else expected[e]['text']
                    actual = heard[h]['text']
                    if op=='merge' or basic_form(exp)==basic_form(actual): continue
                    if expected[e]['end_vowel'] and basic_form(actual)==basic_form(exp)+expected[e]['end_vowel']: continue
                    kind,message = describe(exp,actual,op)
                    errors.append(dict(type=kind,operation=op,expected=exp,heard=actual,message=message))
            for e in deleted.get(owner,[]):
                # The reference's initial hamzat al-wasl is optional when linked.
                if expected[e]['optional_wasl']: continue
                kind,message=describe(expected[e]['text'],'','delete')
                errors.append(dict(type=kind,operation='delete',expected=expected[e]['text'],heard='',message=message))
            # ASR and phones can disagree: never judge a word solely by spelling.
            if lexical(token['text']) != lexical(reference['word']) and not errors:
                self._observe(token,'uncertain',[dict(type='transcription',expected=reference['word'],heard=token['text'],
                              message='رونویسی واژه و شواهد آوایی سازگار نیستند؛ این بخش نامشخص است.')],reference,end_sample,True)
                continue
            full_owner = [i for i,g in enumerate(expected) if g['owner']==owner]
            covered_owner = {e for e,h,op in path if expected[e]['owner']==owner}
            clipped = not set(full_owner).issubset(covered_owner)
            owner_evidence={h for e,h,op in path if h is not None and op!='insert' and expected[e]['owner']==owner}
            clipped = clipped or not owner_evidence.issubset(set(indices))
            if 0 in owner_evidence and missing_onset(expected[full_owner[0]]['text'],heard[0]['text']):
                clipped = True
            moving_edge = not settled and token['end_ms'] > end_ms-500
            if clipped or moving_edge:
                self._observe(token,'uncertain',[],reference,end_sample,True)
                continue
            # De-duplicate descriptions, without removing repeated spoken tokens.
            errors=list({(e['type'],e['expected'],e['heard']):e for e in errors}.values())
            if (letter_evidence is not None and len(full_owner)>=3 and len(errors)==1
                    and _agreed_letter(token['text'],reference['word'],errors)):
                # This complete CTC owner can survive a failed *lexical* alignment.
                # Never save a second copy of the sound or change the heard spelling.
                letter_evidence[token['id']]=(reference,errors)
            self._observe(token,'difference' if errors else 'match',errors,reference,end_sample,settled)
        # Omissions are annotations between confirmed spoken anchors, not tokens.
        for key,gap in list(self._gaps.items()):
            anchors=[t for t in candidates if t['id'] in {gap['before_token_id'],gap['after_token_id']}]
            if len(anchors)==2: del self._gaps[key]
        for owner,indices in deleted.items():
            if not first_owner < owner < last_owner or owner in owner_tokens: continue
            full=[i for i,g in enumerate(expected) if g['owner']==owner]
            if set(indices)!=set(full): continue
            left=max((o for o in owner_tokens if o<owner),default=None)
            right=min((o for o in owner_tokens if o>owner),default=None)
            if left is None or right is None: continue
            before=owner_tokens[left][-1]; after=owner_tokens[right][0]
            if before['state'] not in {'match','difference'} or after['state'] not in {'match','difference'}: continue
            reference=words[owner]
            key=f"g:{before['id']}:{after['id']}:{reference['surah']}:{reference['ayah']}:{reference['word_index']}"
            self._gaps[key]=dict(id=key,before_token_id=before['id'],after_token_id=after['id'],
                                 expected=reference['word'],reference=reference,
                                 message=f"واژهٔ «{reference['word']}» میان این دو بخش پیدا نشد؛ احتمال جاافتادگی وجود دارد.")
        self._last_message='قرائت دنبال می‌شود؛ واژه‌های نامشخص با رسیدن شواهد بیشتر بررسی می‌شوند.'
        return True

    @staticmethod
    def _absolute_groups(prediction, start_sample):
        offset=start_sample*1000/RATE
        return [dict(text=g['text'],start_ms=offset+g['start_ms'],end_ms=offset+g['end_ms'])
                for g in prediction.get('groups',[]) if isinstance(g.get('text'),str)
                and valid_time(g.get('start_ms')) and valid_time(g.get('end_ms'))]

    @staticmethod
    def _word_phone_errors(expected, actual):
        """Preserve full vowel/shadda evidence after assigning acoustic owners."""
        errors=[]
        def append(e,h,operation):
            if basic_form(e)==basic_form(h): return
            kind,message=describe(e,h,operation)
            errors.append(dict(type=kind,operation=operation,expected=e,heard=h,message=message))
        left=[basic_form(g)[:1] for g in expected]
        right=[basic_form(g)[:1] for g in actual]
        for operation,a,b,c,d in Levenshtein.opcodes(left,right):
            paired=min(b-a,d-c)
            if operation in {'equal','replace'}:
                for k in range(paired): append(expected[a+k],actual[c+k],'replace')
                for k in range(a+paired,b): append(expected[k],'','delete')
                for k in range(c+paired,d): append('',actual[k],'insert')
            elif operation=='delete':
                for k in range(a,b): append(expected[k],'','delete')
            else:
                for k in range(c,d): append('',actual[k],'insert')
        return errors

    def _check_words(self, words, expected, start_sample, end_sample, settled, letter_evidence=None):
        """Check positional structure separately from spoken acoustic identity.

        A correctly pronounced word in the wrong slot is still a word error.
        Acoustic alignment follows the ASR occurrence order, never the expected
        Quran order, so a repeat or swap cannot migrate to a convenient owner.
        """
        from word_alignment import align_word_phone_evidence
        candidates=[t for t in self.tokens[-64:] if t['id'] not in self._protected_ids
                    and valid_time(t['start_ms']) and valid_time(t['end_ms'])]
        if not candidates: return False
        if self._word_origin is None:
            origin=self._manual or ((self.current_reference['surah'],self.current_reference['ayah'])
                                    if self.current_reference else None)
            if origin: self._word_origin=(*origin,0)
        preferred=next((i for i,w in enumerate(words) if
                        (w['surah'],w['ayah'],w['word_index'])==self._word_origin),None)
        alignment=align_words(words,candidates,preferred_owner=preferred)
        by_id={t['id']:t for t in candidates}
        own_groups={}
        changed_letters={}
        external_ids=set()
        by_owner = {}
        for g in expected:
            by_owner.setdefault(g['owner'],[]).append(g['text'])
        for entry in alignment['tokens']:
            owner=entry['source_owner']
            slot=entry['owner']
            # A free reference edge can represent the final near-spelled word
            # as an insertion. Its adjacent pair of matched words fixes a
            # candidate position; the timed phones must independently confirm it.
            if slot is None and entry['op']=='insert':
                idx=entry['index']
                others=alignment['tokens']
                if idx>=2 and all(e['op']=='match' for e in others[idx-2:idx]):
                    a,b=others[idx-2:idx]
                    proposed=b['owner']+1
                    if (a['owner']+1==b['owner'] and proposed<len(words)
                            and (words[b['owner']]['surah'],words[b['owner']]['ayah'])==
                            (words[proposed]['surah'],words[proposed]['ayah'])):
                        slot=proposed
            contrast=_consonant_change(words[slot]['word'],entry['text']) if slot is not None else None
            if entry['op'] in {'substitute','insert'} and contrast:
                canonical=by_owner.get(slot,[])
                sites=[i for i,group in enumerate(canonical) if basic_form(group)[:1]==contrast[0]]
                if len(sites)==1:
                    changed_letters[entry['id']]=slot
                    # Reserve phones for this word even if the spoken spelling
                    # (often a learner's nonword) is absent from the lexicon.
                    own_groups[entry['id']]=canonical
                    continue
            if owner is None:
                sources=[i for i,w in enumerate(words) if lexical(w['word'])==lexical(entry['text'])]
                if sources:
                    owner=min(sources,key=lambda i:abs(i-(entry['owner'] if entry['owner'] is not None else preferred or 0)))
            if owner is not None:
                own_groups[entry['id']]=by_owner.get(owner,[])
            if (not own_groups.get(entry['id']) and entry['op'] in {'substitute','insert'}
                    and entry['confidence']=='anchored'):
                lookup = getattr(self.corpus,'word_phones',None)
                if lookup:
                    phones = lookup(entry['text'])
                    slot = by_owner.get(entry['owner'],[])
                    if phones and [basic_form(p)[:1] for p in phones] != [basic_form(p)[:1] for p in slot]:
                        own_groups[entry['id']] = phones
                        external_ids.add(entry['id'])
        # Select the widest containing acoustic context, not whichever model
        # prediction happens to produce the fewest errors. Each window assigns
        # every CTC group to at most one heard occurrence.
        evidence={}; best={}; contexts=[]
        for record in self._history[-16:]:
            lo,hi=record['start']*1000/RATE,record['end']*1000/RATE
            included=[t for t in candidates if t['end_ms']>lo and t['start_ms']<hi]
            if not included: continue
            index=len(contexts)
            contexts.append((record,included))
            for token in included:
                if lo <= token['start_ms']+60 and hi >= token['end_ms']-60:
                    rank=(hi-lo,hi,-index)
                    if token['id'] not in best or rank>best[token['id']]:
                        best[token['id']]=rank
        # Rank on time coverage first. Align the full set of neighbours only
        # for contexts that actually contribute evidence to at least one word.
        chosen={}
        for token_id,(_,_,negative_index) in best.items():
            chosen.setdefault(-negative_index,[]).append(token_id)
        for index,token_ids in chosen.items():
            record,included=contexts[index]
            groups=self._absolute_groups(record['prediction'],record['start'])
            # Whisper word boundaries and the CTC frame grid are independent;
            # allow a small frame-sized margin in addition to the word-time
            # uncertainty. Full phonetic insertions remain in the evidence.
            observed=align_word_phone_evidence(included,groups,own_groups,slack_ms=300)
            for token_id in external_ids & observed.keys():
                proof = observed[token_id]
                # Another Quran word absent from this passage is plausible
                # only if all of its consonants have independent CTC evidence.
                # Standalone dictionary vowels are not pronunciation verdicts.
                heard_letters = [basic_form(p)[:1] for p in proof['actual']]
                own_letters = [basic_form(p)[:1] for p in proof['expected']]
                if heard_letters != own_letters:
                    proof['status'] = 'uncertain'
            for token_id in token_ids:
                evidence[token_id]=observed[token_id]
        lo,hi=start_sample*1000/RATE,end_sample*1000/RATE
        # Context words help the sequence alignment, but an old acoustic check
        # cannot mutate future words, and expiring cached evidence is not a new
        # contradiction of an already reviewed earlier word.
        affected={t['id'] for t in candidates if t['start_ms']<hi and
                  (t['end_ms']>lo or (t['state'] not in {'match','difference'}
                   and evidence.get(t['id'],{}).get('status')=='supported'))}
        # Complete a previously provisional substitution only when its
        # reciprocal partner has now actually been heard and both identities
        # are independently supported. Never reopen unrelated old verdicts.
        swaps={(e['owner'],e['source_owner']):e for e in alignment['tokens']
               if e['op']=='transpose' and e['confidence']=='anchored'
               and evidence.get(e['id'],{}).get('status')=='supported'}
        for (slot,source),entry in swaps.items():
            partner=swaps.get((source,slot))
            old=by_id[entry['id']]
            if (partner and partner['id'] in affected and old['start_ms']<hi
                    and any(e.get('type')=='substitution' for e in old['errors'])):
                affected.add(entry['id'])
        for token_id in affected:
            if evidence.get(token_id,{}).get('status')=='supported':
                self._acoustic_supported.add(token_id)
            else:
                self._acoustic_supported.discard(token_id)
        self._gaps={key:gap for key,gap in self._gaps.items()
                    if not {gap['before_token_id'],gap['after_token_id']}.issubset(affected)}
        reviewed=[]
        supported_matches={entry['id'] for entry in alignment['tokens']
                           if entry['op']=='match' and entry['confidence']=='anchored'
                           and evidence.get(entry['id'],{}).get('status')=='supported'}
        def adjacent_match(entry, word):
            for step in (-1,1):
                index=entry['index']+step
                if not 0<=index<len(alignment['tokens']):
                    continue
                near=alignment['tokens'][index]
                if near['op']=='match' and near['owner'] is not None:
                    anchor=words[near['owner']]
                    if (anchor['surah'],anchor['ayah'],anchor['word_index'])==(
                            word['surah'],word['ayah'],word['word_index']+step):
                        return True
            return False
        for entry in alignment['tokens']:
            token=by_id[entry['id']]
            if token['id'] not in affected: continue
            proof=evidence.get(token['id'],{})
            owner=entry['owner']
            reference=dict(words[owner]) if owner is not None else None
            if token['id'] in changed_letters:
                slot=changed_letters[token['id']]
                reference=dict(words[slot])
                actual=proof.get('actual',[])
                canonical=own_groups[token['id']]
                found=self._word_phone_errors(canonical,actual)
                anchored=len(alignment['anchors'])>=2 and adjacent_match(entry,reference)
                if (anchored and proof.get('status')=='supported'
                        and all(p['operation']!='insert' for p in proof.get('alignment',[]))
                        and not proof.get('deleted_expected_indices') and len(found)==1
                        and _agreed_letter(token['text'],reference['word'],found)
                        and (settled or token['end_ms']<=hi-500)):
                    self._observe(token,'difference',found,reference,end_sample,settled,word=True)
                    reviewed.append(slot)
                else:
                    self._observe(token,'uncertain',[dict(type='transcription',
                        expected=reference['word'],heard=token['text'].strip(),
                        message='املای شنیده‌شده با مرجع فرق دارد؛ آوا یا جای این حرف هنوز به‌اندازهٔ کافی روشن نیست.')],
                        reference,end_sample,True,word=True)
                continue
            if entry['confidence']!='anchored' or proof.get('status')!='supported':
                saved=(letter_evidence or {}).get(token['id'])
                if (saved and entry['op'] in {'match','substitute','insert'} and
                        (owner is None or saved[0]==reference) and len(alignment['anchors'])>=2
                        and (entry['op']=='match' or adjacent_match(entry,saved[0]))):
                    self._observe(token,'difference',saved[1],saved[0],end_sample,settled,word=True)
                    continue
                self._observe(token,'uncertain',[],reference,end_sample,True,word=True)
                continue
            moving_edge=not settled and token['end_ms']>end_sample*1000/RATE-500
            if moving_edge:
                self._observe(token,'uncertain',[],reference,end_sample,True,word=True)
                continue
            op=entry['op']
            if op in {'insert','repeat'}:
                if reference is None and entry['source_owner'] is not None:
                    reference=dict(words[entry['source_owner']])
                errors=[dict(type='repetition' if op=='repeat' else 'insertion',operation='insert',
                    expected='',heard=token['text'].strip(),
                    message='این واژه دوباره خوانده شده است.' if op=='repeat' else 'این واژه میان دو بخشِ تطبیق‌یافته اضافه خوانده شده است.')]
            elif op in {'transpose','substitute'}:
                errors=[dict(type='transposition' if op=='transpose' else 'substitution',operation='replace',
                    expected=reference['word'],heard=token['text'].strip(),
                    message=f"در این جایگاه «{token['text'].strip()}» به‌جای «{reference['word']}» خوانده شده است.")]
            elif op=='match':
                errors=self._word_phone_errors(proof['expected'],proof['actual'])
                pieces=proof.get('alignment',[])
                if pieces and (pieces[0]['operation']=='insert' or pieces[-1]['operation']=='insert'):
                    # A CTC-only phone at a word join has no reliable owner.
                    # Keep the independent evidence, but do not attach a red
                    # pronunciation verdict to a neighbouring ASR word.
                    self._observe(token,'uncertain',[dict(type='boundary',expected=reference['word'],
                        heard=token['text'].strip(),message='مرز آوایی این کلمه روشن نیست؛ شواهد تلفظ برای علامت‌گذاری کافی نیست.')],
                        reference,end_sample,True,word=True)
                    continue
            else:
                self._observe(token,'uncertain',[],reference,end_sample,True,word=True)
                continue
            # One exact independent acoustic reading can confirm an interior
            # word already stable across two fresh ASR observations. Keep the
            # normal confirmation rule for differences and uncertain edges.
            early_match=(not errors and op=='match' and proof.get('exact')
                         and len(supported_matches)>=2 and self._asr_seen.get(token['id'],(0,))[0]>=2
                         and token['end_ms']<=hi-750 and (not start_sample or token['start_ms']>=lo+300))
            self._observe(token,'difference' if errors else 'match',errors,reference,end_sample,
                          settled or early_match,word=True)
            if owner is not None: reviewed.append(owner)
        for gap in alignment['gaps']:
            before,after=by_id[gap['before_id']],by_id[gap['after_id']]
            if not {before['id'],after['id']}&affected: continue
            if (before['state'] not in {'match','difference'} or after['state'] not in {'match','difference'}
                    or evidence.get(before['id'],{}).get('status')!='supported'
                    or evidence.get(after['id'],{}).get('status')!='supported'): continue
            reference=dict(words[gap['owner']])
            key=f"g:{before['id']}:{after['id']}:{reference['surah']}:{reference['ayah']}:{reference['word_index']}"
            self._gaps[key]=dict(id=key,before_token_id=before['id'],after_token_id=after['id'],
                expected=reference['word'],reference=reference,
                message=f"واژهٔ «{reference['word']}» میان این دو کلمه خوانده نشده است.")
        if reviewed:
            current_owners=[entry['owner'] for entry in alignment['tokens']
                if entry['owner'] is not None and by_id[entry['id']]['state'] in {'match','difference'}
                and by_id[entry['id']]['end_ms']>start_sample*1000/RATE
                and by_id[entry['id']]['start_ms']<end_sample*1000/RATE]
            if current_owners:
                self._anchor=dict(words[max(current_owners)])
                first=words[min(current_owners)]
                ref=self.corpus.reference(first['surah'],first['ayah'])
                self.current_reference=dict(surah=first['surah'],ayah=first['ayah'],text=ref.text)
        return bool(reviewed or self._gaps)

    def _check(self, prediction, start_sample, end_sample, settled):
        heard=self._absolute_groups(prediction,start_sample)
        located=self._locate(heard) if heard else None
        letters={}
        phonetic=self._check_phonetic(prediction,start_sample,end_sample,settled,located=located,letter_evidence=letters)
        if located is None: return phonetic
        words,expected,_=located
        return self._check_words(words,expected,start_sample,end_sample,settled,letters) or phonetic

    def _feedback(self, status):
        updates=[]
        for token in self.tokens:
            metadata={k:token[k] for k in ('id','state','errors','reference')}
            if self._published.get(token['id']) != metadata:
                metadata=deepcopy(metadata)
                updates.append(metadata)
                self._published[token['id']]=deepcopy(metadata)
        return dict(updates=updates,
                    gaps=deepcopy(list(self._gaps.values())),reference=deepcopy(self.current_reference),
                    status=status,message=self._last_message,summary=self.summary())

    def update_phonemes(self, prediction, start_sample, end_sample, settled=False):
        if prediction.get('audio_activity',{}).get('status') in {'no_speech','uncertain'}:
            # Skipping a non-speech model call supplies no contrary phonetic
            # evidence. Keep previous word verdicts and omission anchors.
            return self._feedback('tracking' if self.current_reference else 'listening')
        # No audio, tensors or logits retained. Up to 30min of sparse evidence.
        record=dict(prediction=deepcopy(prediction),start=start_sample,end=end_sample,settled=settled)
        self._history.append(record)
        heard=self._absolute_groups(prediction,start_sample)
        previous = None
        if self._anchor is None and self._manual is None and len(self._history) >= 2:
            old = self._history[-2]
            if old['start'] < start_sample and old['end'] < end_sample:
                previous = self._absolute_groups(old['prediction'],old['start'])
        location=self._locate(heard,previous) if heard else None
        letters={}
        located=self._check_phonetic(prediction,start_sample,end_sample,settled,located=location,letter_evidence=letters)
        # Once a reference is known, revisit a bounded number of previously
        # unresolved windows. Text is immutable; only annotation metadata changes.
        if located:
            current_anchor=deepcopy(self._anchor)
            current_reference=deepcopy(self.current_reference)
            rechecked=0
            for old in reversed(self._history[:-1]):
                if old.get('resolved'): continue
                old['resolved']=self._check_phonetic(old['prediction'],old['start'],old['end'],old['settled'])
                rechecked+=1
                if rechecked>=4: break
            self._anchor=current_anchor
            self.current_reference=current_reference
        # Apply sequence verdicts once with the newest boundary. Replaying an
        # unresolved old acoustic window must not make newer words look like
        # partial future edges or retract their omission anchors.
        if location is not None:
            words,expected,_=location
            located=self._check_words(words,expected,start_sample,end_sample,settled,letters) or located
        if located and self._anchor is not None:
            self._last_anchor_ms=end_sample*1000/RATE
        record['resolved']=located
        return self._feedback('tracking' if located else 'listening')

    def select_reference(self, surah, ayah):
        if type(surah) is not int or type(ayah) is not int:
            raise ValueError('شمارهٔ سوره یا آیه معتبر نیست.')
        self.corpus.reference(surah,ayah)
        self._manual=(surah,ayah)
        self._anchor=None
        self._last_anchor_ms=None
        self._word_origin=(surah,ayah,0)
        # A manual change starts a new reference epoch. It must not erase an
        # already reviewed earlier reading. Pending/ambiguous words remain
        # eligible for retrospective checks with the newly selected reference.
        self._protected_ids.update(t['id'] for i,t in enumerate(self.tokens)
                                   if i<self.committed_count and t['state'] in {'match','difference'})
        affected={t['id'] for t in self.tokens if t['id'] not in self._protected_ids}
        self._pending={key:value for key,value in self._pending.items() if key not in affected}
        self._word_pending={key:value for key,value in self._word_pending.items() if key not in affected}
        for token in self.tokens:
            if token['id'] in affected:
                token.update(state='uncertain',errors=[],reference=None)
        self._gaps={key:gap for key,gap in self._gaps.items()
                    if not {gap['before_token_id'],gap['after_token_id']}&affected}
        ref=self.corpus.reference(surah,ayah)
        self.current_reference=dict(surah=surah,ayah=ayah,text=ref.text)
        located=False
        for old in self._history[-4:]:
            if affected:
                located=self._check(old['prediction'],old['start'],old['end'],old['settled']) or located
        if not located:
            self._last_message='محل جدید انتخاب شد؛ به خواندن ادامه دهید.'
        return self._feedback('tracking' if located else 'listening')
