"""Recover independently decoded word occurrences from repeated acoustic intervals."""
from math import isfinite

from recitation import basic_form
from word_alignment import lexical

RATE = 16000
MAX_CHECKS = 2


def bounds(item):
    a,b=item.get('start_ms'),item.get('end_ms')
    if all(isinstance(x,(float,int)) and isfinite(x) for x in (a,b)) and 0<=a<b:
        return a,b
    return None


def repeat_windows(words, groups):
    # Map the first consonant or long vowel of an observed CTC group to its
    # written letter. Abstain on ambiguous spellings. No verse is consulted.
    letters={'ء':'ا','ۥ':'و','ۦ':'ي'}
    sound=[basic_form(g.get('text',''))[:1] for g in groups]
    written=[letters.get(s,s) for s in sound]
    for i,word in enumerate(words):
        span=bounds(word)
        key=lexical(word['text'])
        if not span or len(key)<2 or len(set(key))<2:
            continue
        if i and lexical(words[i-1]['text'])==key:
            continue
        n=len(key)
        for j in range(len(groups)-2*n+1):
            if ''.join(written[j:j+n])!=key or sound[j:j+n]!=sound[j+n:j+2*n]:
                continue
            fragment=groups[j:j+2*n]
            times=[bounds(g) for g in fragment]
            if any(t is None for t in times) or any(a[1]>b[0] for a,b in zip(times,times[1:])):
                continue
            a,b=times[0][0],times[n-1][1]
            c,d=times[n][0],times[-1][1]
            if not span[0]-120 <= (a+b)/2 <= span[1]+120:
                continue
            # Joined recitation can lengthen a consonant on just one pass.
            if not 0 <= c-b <= 850 or min(b-a,d-c)<160 or max(b-a,d-c)>3*min(b-a,d-c):
                continue
            # An adjacent whole word with a different identity must not be
            # silently overwritten by a putative duplicated syllable.
            following=words[i+1] if i+1<len(words) else None
            if following and lexical(following['text'])!=key:
                other=bounds(following)
                if other and other[1]<=d+100:
                    continue
            yield i,key,(a,b,c,d)
            break


def recover_repetitions(transcript, prediction, wave, decode):
    """Keep primary words and add only independently heard second occurrences.

    CTC supplies distinct times, never transcript text. When the primary ASR
    collapsed two matching phonetic cycles, an unprompted decode beginning
    between them must emit the same word again on the second interval.
    """
    original=transcript.get('words',[])
    groups=prediction.get('groups',[])
    if not original or not groups:
        return transcript
    corrected=list(original)
    checks=0
    verified_from=None
    for i,key,(a,b,c,d) in reversed(list(repeat_windows(original,groups))):
        following=original[i+1] if i+1<len(original) else None
        if following and lexical(following['text'])==key:
            # Both occurrences already have distinct ASR tokens. Refine their
            # coverage with the two disjoint CTC intervals for word feedback.
            span=bounds(following)
            if not span or min(span[1],d)<=max(span[0],c):
                continue
            second=dict(following)
            next_index=i+2
        else:
            if checks>=MAX_CHECKS:
                continue
            checks+=1
            start=round((b+c)/2*RATE/1000)
            end=min(len(wave),round((d+1800)*RATE/1000))
            second_pass=decode(wave[start:end]).get('words',[])
            if not second_pass or lexical(second_pass[0]['text'])!=key:
                continue
            stamp=bounds(second_pass[0])
            if not stamp or stamp[0]+start/16>=d or stamp[1]+start/16<c:
                continue
            if following:
                anchor=lexical(following['text'])
                if len(second_pass)<2 or lexical(second_pass[1]['text'])!=anchor:
                    continue
            second=dict(second_pass[0])
            # A decoder can omit the initial inter-word space at a clip start.
            if not second['text'][0].isspace() and not original[i]['text'][-1].isspace():
                second['text']=' '+second['text']
            corrected.insert(i+1,second)
            next_index=i+2
        corrected[i]=dict(original[i],start_ms=a,end_ms=b)
        corrected[i+1]=dict(second,start_ms=c,end_ms=d)
        verified_from=a if verified_from is None else min(verified_from,a)
        # The original decoder often assigns all repeated audio to the next
        # word's leading silence. Preserve that word and bound its timestamp.
        if next_index<len(corrected):
            neighbor=corrected[next_index]
            span=bounds(neighbor)
            if span and span[0]<d<span[1]:
                corrected[next_index]=dict(neighbor,start_ms=d)
    if corrected==original:
        return transcript
    return dict(transcript,words=corrected,text=''.join(w['text'] for w in corrected),
                repeat_verified_from_ms=verified_from)
