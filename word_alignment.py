"""Order-aware lexical structure; acoustic verification remains independent.

No function rewrites a heard token. ``confidence`` is a categorical description
of structural anchoring, never a pronunciation probability or final verdict.
"""
from math import isfinite
import re
from collections.abc import Mapping
from functools import lru_cache

from recitation import basic_form


def _letters(text):
    text = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06edـ]", "", text)
    text = text.replace("ٱ", "ا").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    return re.sub(r"[^\u0621-\u063a\u0641-\u064a]", "", text)


@lru_cache(maxsize=1)
def _superscript_alifs():
    """The same small alif is written in كاذبة but omitted in الرحمن."""
    from quran_transcript import Aya
    aya = Aya(1, 1)
    spellings = {}
    for surah in range(1, 115):
        count = aya.set_new(surah, 1).get().num_ayat_in_sura
        for ayah in range(1, count + 1):
            verse = aya.set_new(surah, ayah).get()
            if len(verse.uthmani_words) != len(verse.imlaey_words):
                continue
            for source, modern in zip(verse.uthmani_words, verse.imlaey_words):
                if 'ٰ' in source and _letters(source.replace('ٰ', 'ا')) == _letters(modern):
                    spellings[source] = _letters(modern)
    return spellings


def lexical(text):
    text = text.strip()
    if 'ٰ' in text:
        modern = _superscript_alifs().get(text)
        if modern is not None:
            return modern
    return _letters(text)


def _text(item, reference=False):
    if isinstance(item, str):
        return item
    return item.get("word" if reference else "text", "")


def _verse_ranges(reference):
    """Identify complete supplied ayahs; no text-specific reread exceptions."""
    result = {}
    start = 0
    while start < len(reference):
        first = reference[start]
        if not isinstance(first, dict) or not {"surah", "ayah", "word_index"} <= first.keys():
            start += 1
            continue
        key = first["surah"], first["ayah"]
        end = start + 1
        while end < len(reference) and isinstance(reference[end], dict) and (
                reference[end].get("surah"), reference[end].get("ayah")) == key:
            end += 1
        indices = [item.get("word_index") for item in reference[start:end]]
        if indices == list(range(end - start)):
            result[end] = start
        start = end
    return result


def align_words(reference_words, heard_tokens, *, preferred_owner=None,
                allow_reread=True, max_transposition_span=6):
    """Align every heard occurrence against a reference subsequence.

    Owners are indexes into ``reference_words``. A transposed token's ``owner``
    is its expected positional slot; ``source_owner`` identifies the word it
    actually resembles, for a SEPARATE acoustic identity check. Unread reference
    edges are free and never produce omission gaps. Complete consecutive ayah
    rereads are supported without interpreting every repeated word as an error.

    Returns ``tokens`` (one entry per input, same id/raw text), ``gaps``,
    ``anchors``, ``ambiguous``, ``cost`` and the selected reference ``span``.
    """
    reference = list(reference_words)
    heard = list(heard_tokens)
    rkeys = [lexical(_text(item, True)) for item in reference]
    hkeys = [lexical(_text(item)) for item in heard]
    n, m = len(reference), len(heard)
    if not n or not m:
        return dict(tokens=[dict(id=item.get("id", str(i)), text=_text(item), index=i,
                                 owner=None, source_owner=None, op="uncertain", confidence="unanchored")
                            for i, item in enumerate(heard)], gaps=[], anchors=[], ambiguous=True,
                    cost=float(m), span=None)
    # score=(edit cost, negative exact anchor count, sum of heard anchor indexes).
    # Last component preserves the FIRST occurrence when an equal word repeats.
    scores = [[None] * (n + 1) for _ in range(m + 1)]
    moves = [[None] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        scores[0][j] = (0.0, 0, 0)
    for i in range(1, m + 1):
        scores[i][0] = (float(i), 0, 0)
        moves[i][0] = ("insert", 1, 0)
    verses = _verse_ranges(reference) if allow_reread else {}

    def extend(score, cost, exact=0, position_sum=0):
        return score[0] + cost, score[1] - exact, score[2] + position_sum

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            equal = bool(hkeys[i-1]) and hkeys[i-1] == rkeys[j-1]
            choices = [
                (extend(scores[i-1][j-1], 0 if equal else 1, int(equal), (i-1) if equal else 0),
                 ("match" if equal else "substitute", 1, 1)),
                (extend(scores[i-1][j], 1), ("insert", 1, 0)),
                (extend(scores[i][j-1], 1), ("delete", 0, 1)),
            ]
            for span in range(2, min(i, j, max_transposition_span) + 1):
                hs, rs = hkeys[i-span:i], rkeys[j-span:j]
                if (rs[0] and rs[-1] and rs[0] != rs[-1] and hs[0] == rs[-1]
                        and hs[-1] == rs[0] and hs[1:-1] == rs[1:-1]):
                    choices.append((extend(scores[i-span][j-span], .75, span-2,
                                           sum(range(i-span+1, i-1))), ("transpose", span, span)))
            if j in verses:
                start = verses[j]
                size = j - start
                verse = rkeys[start:j]
                if i >= 2 * size and hkeys[i-2*size:i-size] == verse and hkeys[i-size:i] == verse:
                    choices.append((extend(scores[i-size][j], 0, size, sum(range(i-size, i))),
                                    ("reread", size, 0)))
            scores[i][j], moves[i][j] = min(choices, key=lambda item: item[0])
    best_primary = min((scores[m][j][0], scores[m][j][1]) for j in range(1, n+1))
    endpoints = [j for j in range(1, n+1) if scores[m][j][:2] == best_primary]
    if preferred_owner is not None:
        distance = min(abs(j - 1 - preferred_owner) for j in endpoints)
        endpoints = [j for j in endpoints if abs(j - 1 - preferred_owner) == distance]
    end = min(endpoints, key=lambda j: scores[m][j])
    ambiguous = len(endpoints) > 1
    i, j, entries, deleted = m, end, [], []
    while i:
        operation, di, dj = moves[i][j]
        if operation == "delete":
            deleted.append(j-1)
        elif operation == "reread":
            first = j - di
            for k in range(di):
                entries.append(dict(index=i-di+k, owner=first+k, source_owner=first+k,
                                    op="match", reread=True))
        elif operation == "transpose":
            for k in range(di):
                owner = j - di + k
                source = j-1 if k == 0 else j-di if k == di-1 else owner
                entries.append(dict(index=i-di+k, owner=owner, source_owner=source,
                                    op="transpose" if k in {0,di-1} else "match"))
        elif operation == "insert":
            entries.append(dict(index=i-1, owner=None, source_owner=None, op="insert"))
        else:
            entries.append(dict(index=i-1, owner=j-1, source_owner=(j-1 if operation == "match" else None),
                                op=operation))
        i -= di
        j -= dj
    start = j
    entries.sort(key=lambda entry: entry["index"])
    anchors = [entry for entry in entries if entry["op"] == "match"]
    for entry in entries:
        k = entry["index"]
        entry.update(id=heard[k].get("id", str(k)), text=_text(heard[k]))
        before = [anchor for anchor in anchors if anchor["index"] < k]
        after = [anchor for anchor in anchors if anchor["index"] > k]
        if entry["op"] == "insert":
            if k and hkeys[k] and hkeys[k] == hkeys[k-1]:
                previous = entries[k-1]
                if previous.get("source_owner") is not None:
                    entry["op"] = "repeat"
                    entry["source_owner"] = previous["source_owner"]
            if entry["source_owner"] is None:
                sources = [owner for owner, key in enumerate(rkeys) if key and key == hkeys[k]]
                entry["source_owner"] = sources[0] if len(sources) == 1 else None
        if ambiguous:
            confidence = "ambiguous"
        elif entry["op"] == "match":
            confidence = "anchored" if len(anchors) >= 2 else "unanchored"
        elif entry["op"] == "transpose":
            confidence = "anchored"
        else:
            confidence = "anchored" if before and after else "unanchored"
        entry["confidence"] = confidence
        if entry["owner"] is not None and isinstance(reference[entry["owner"]], dict):
            entry["reference"] = dict(reference[entry["owner"]])
    # When a run of identical reference words has only one heard occurrence,
    # either side of that occurrence could contain the omission. Preserve its
    # raw text, but do not invent a gap position or a definite reference owner.
    ambiguous_gap_owners = set()
    deleted_owners = set(deleted)
    for at, anchor in enumerate(anchors):
        owner = anchor["owner"]
        left, right = owner, owner + 1
        while left and rkeys[left-1] == rkeys[owner]:
            left -= 1
        while right < n and rkeys[right] == rkeys[owner]:
            right += 1
        if right-left < 2 or not 0 < at < len(anchors)-1:
            continue
        before, after = anchors[at-1], anchors[at+1]
        run_deletions = set(range(left, right)) & deleted_owners
        if (len(run_deletions) == right-left-1
                and before["owner"] < left and after["owner"] >= right
                and before["index"] + 1 == anchor["index"] == after["index"] - 1):
            anchor["confidence"] = "ambiguous"
            ambiguous_gap_owners.update(run_deletions)
    gaps = []
    if not ambiguous:
        for owner in sorted(deleted_owners - ambiguous_gap_owners):
            before = [a for a in anchors if a["owner"] < owner]
            after = [a for a in anchors if a["owner"] > owner]
            if not before or not after:
                continue
            left, right = max(before, key=lambda a: a["index"]), min(after, key=lambda a: a["index"])
            if left["index"] >= right["index"]:
                continue
            gaps.append(dict(owner=owner, before_id=left["id"], after_id=right["id"],
                             confidence="anchored"))
    return dict(tokens=entries, gaps=gaps, anchors=[a["id"] for a in anchors],
                ambiguous=ambiguous, cost=scores[m][end][0], span=(start, end))


def word_phone_evidence(token, timed_groups, expected_groups, *, slack_ms=60, max_edit_ratio=.35):
    """Check the heard word's OWN identity, independent of its expected slot.

    ``expected_groups`` accepts strings or ``{text: ...}`` groups. Timed groups
    and token bounds must share the same (relative OR absolute) clock. Minor
    phonetic differences can support identity; ``supported`` never means the
    pronunciation is correct. All actual groups/differences remain available.
    """
    expected = [g if isinstance(g, str) else g["text"] for g in expected_groups]
    base = dict(status="uncertain", indices=[], actual=[], expected=expected,
                edit_ratio=None, exact=False, reason="missing_word_timing")
    a, b = token.get("start_ms"), token.get("end_ms")
    valid = lambda value: isinstance(value, (int,float)) and not isinstance(value,bool) and isfinite(value)
    if not valid(a) or not valid(b) or b <= a or not expected:
        return base
    indices = [i for i,g in enumerate(timed_groups)
               if valid(g.get("start_ms")) and valid(g.get("end_ms"))
               and a-slack_ms <= (g["start_ms"]+g["end_ms"])/2 <= b+slack_ms]
    actual = [timed_groups[i]["text"] for i in indices]
    base.update(indices=indices, actual=actual, reason="missing_phone_evidence")
    if not actual:
        return base
    # A CTС group crossing the ASR boundary is not complete evidence for green.
    if any(timed_groups[i]["start_ms"] < a-slack_ms or timed_groups[i]["end_ms"] > b+slack_ms
           for i in indices):
        base["reason"] = "phone_group_crosses_word_boundary"
        return base
    left = [basic_form(g) for g in expected]
    right = [basic_form(g) for g in actual]
    dp = [[0] * (len(left)+1) for _ in range(len(right)+1)]
    for j in range(len(left)+1): dp[0][j] = j
    for i in range(1,len(right)+1):
        dp[i][0] = i
        for j in range(1,len(left)+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                           dp[i-1][j-1] + (0 if right[i-1]==left[j-1] else
                                          .25 if right[i-1][:1]==left[j-1][:1] else 1))
    ratio = dp[-1][-1]/max(len(left),len(right),1)
    exact = left == right
    # If expected edge phones disappeared outside ASR's word bounds, identity
    # may be plausible but the missing earlier/later error must not turn green.
    if len(right) < len(left) and (left[:len(right)] == right or left[-len(right):] == right):
        base.update(edit_ratio=ratio, exact=exact, reason="incomplete_word_phone_coverage")
        return base
    base.update(status="supported" if ratio <= max_edit_ratio else "disagreement",
                edit_ratio=ratio, exact=exact,
                reason="word_identity_supported" if ratio <= max_edit_ratio else "asr_phone_disagreement")
    return base


def missing_onset(expected, heard):
    """A vowel-only first CTC cluster carries no evidence of a replacement consonant."""
    left, right = basic_form(expected), basic_form(heard)
    vowels = 'َُِ'
    return bool(left and left[0] not in vowels and right and all(ch in vowels for ch in right))


def align_word_phone_evidence(tokens, timed_groups, expected_by_token, *, slack_ms=250,
                             max_edit_ratio=.35):
    """Partition phones monotonically by the ORDER OF SPOKEN ASR occurrences.

    ``expected_by_token`` is either ``{token_id: [str|{text:...}, ...]}`` or a
    list aligned with tokens. It describes each heard word's own phonemes, not
    the reference slot where it was supposed to occur. Every heard phone is
    assigned at most once; insertions remain in actual evidence. Returned values
    use the same identity-only contract as ``word_phone_evidence``.
    """
    tokens, heard = list(tokens), list(timed_groups)
    valid = lambda v: isinstance(v,(int,float)) and not isinstance(v,bool) and isfinite(v)
    expected, owners, own, starts = [], [], [], []
    for k, token in enumerate(tokens):
        supplied = expected_by_token.get(token["id"], []) if isinstance(expected_by_token,Mapping) else expected_by_token[k]
        group_list = [g if isinstance(g,str) else g["text"] for g in (supplied or [])]
        own.append(group_list)
        starts.append(len(expected))
        for group in group_list:
            expected.append(group)
            owners.append(k)
    n, m = len(expected), len(heard)
    result = {token["id"]:dict(status="uncertain", indices=[], actual=[], expected=own[k],
                               edit_ratio=None, exact=False, reason="missing_phone_evidence")
              for k,token in enumerate(tokens)}
    if not n or not m:
        return result

    def distance(k, h):
        a,b = tokens[k].get("start_ms"),tokens[k].get("end_ms")
        c,d = heard[h].get("start_ms"),heard[h].get("end_ms")
        if not all(valid(v) for v in (a,b,c,d)):
            return 0
        center=(c+d)/2
        return max(a-center,center-b,0)

    expected_forms = [basic_form(group) for group in expected]
    heard_forms = [basic_form(group["text"]) for group in heard]

    dp=[[0.0]*(n+1) for _ in range(m+1)]
    moves=[bytearray(n+1) for _ in range(m+1)]
    for j in range(n+1): dp[0][j]=j; moves[0][j]=2
    for i in range(1,m+1):
        dp[i][0]=i; moves[i][0]=3
        right = heard_forms[i-1]
        timing_cost = [min(distance(k,i-1)/1000,.5)*.2 for k in range(len(tokens))]
        for j in range(1,n+1):
            left = expected_forms[j-1]
            phonetic = 0 if left==right else .25 if left[:1]==right[:1] else 1
            substitution = phonetic + timing_cost[owners[j-1]]
            choices=[(dp[i-1][j-1]+substitution,1),
                     (dp[i][j-1]+1,2),(dp[i-1][j]+1,3)]
            dp[i][j],moves[i][j]=min(choices,key=lambda pair:pair[0])
    i,j,path=m,n,[]
    while i or j:
        move=moves[i][j]
        if i and j and move==1:
            path.append((j-1,i-1,"replace")); i-=1; j-=1
        elif j and (not i or move==2):
            path.append((j-1,None,"delete")); j-=1
        else:
            path.append((min(j,n-1),i-1,"insert")); i-=1
    path.reverse()
    assigned={k:[] for k in range(len(tokens))}
    deleted={k:[] for k in range(len(tokens))}
    for at,(e,h,operation) in enumerate(path):
        if operation=="delete":
            deleted[owners[e]].append(e)
            continue
        if operation!="insert":
            assigned[owners[e]].append((e,h,operation))
            continue
        before=next((owners[a] for a,b,op in reversed(path[:at]) if b is not None and op!="insert"),None)
        after=next((owners[a] for a,b,op in path[at+1:] if b is not None and op!="insert"),None)
        possible={k for k in (before,after) if k is not None}
        if not possible:
            possible={owners[e]}
        owner=min(possible,key=lambda k:(distance(k,h),k))
        assigned[owner].append((None,h,"insert"))
    for k,token in enumerate(tokens):
        entry=result[token["id"]]
        pieces=sorted(assigned[k],key=lambda piece:piece[1])
        indices=[h for _,h,_ in pieces]
        actual=[heard[h]["text"] for h in indices]
        entry.update(indices=indices,actual=actual,
                     alignment=[dict(expected_index=(e-starts[k] if e is not None else None),
                                     heard_index=h,operation=operation)
                                for e,h,operation in pieces],
                     deleted_expected_indices=[e-starts[k] for e in deleted[k]])
        a,b=token.get("start_ms"),token.get("end_ms")
        if not valid(a) or not valid(b) or b<=a:
            entry["reason"]="missing_word_timing"
            continue
        if not own[k] or not indices:
            continue
        if any(not valid(heard[h].get("start_ms")) or not valid(heard[h].get("end_ms"))
               or heard[h]["end_ms"]<heard[h]["start_ms"] for h in indices):
            entry["reason"]="invalid_phone_timing"
            continue
        outside=any(heard[h]["start_ms"]<a-slack_ms or heard[h]["end_ms"]>b+slack_ms for h in indices)
        if outside:
            entry["reason"]="incomplete_word_phone_coverage"
            continue
        expected_indices=[e for e,owner in enumerate(owners) if owner==k]
        paired={e for e,_,op in pieces if e is not None and op!="insert"}
        # A deletion of an edge phone can be actual mispronunciation or a clipped
        # ASR/CTC word. Do not green either case without complete edge evidence.
        first_expected, first_heard, _ = pieces[0]
        clipped_onset = (first_expected == expected_indices[0] and first_heard == 0
                         and missing_onset(expected[first_expected], heard[first_heard]['text']))
        if clipped_onset or expected_indices[0] not in paired or expected_indices[-1] not in paired:
            entry["reason"]="incomplete_word_phone_coverage"
            continue
        edits=len(deleted[k])
        for e,h,operation in pieces:
            if operation=="insert": edits+=1
            else:
                left,right=expected_forms[e],heard_forms[h]
                edits+=0 if left==right else .25 if left[:1]==right[:1] else 1
        ratio=edits/max(len(own[k]),len(actual),1)
        exact=[basic_form(g) for g in actual]==[basic_form(g) for g in own[k]]
        entry.update(status="supported" if ratio<=max_edit_ratio else "disagreement",
                     edit_ratio=ratio,exact=exact,
                     reason="word_identity_supported" if ratio<=max_edit_ratio else "asr_phone_disagreement")
    return result
