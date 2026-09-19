"""Model inference and a deliberately limited, text-dependent feedback layer."""
import logging
from hashlib import blake2b
import os
import re
from dataclasses import dataclass
from pathlib import Path

from model_assets import MODEL_ID, MODEL_REVISION, configure_cache, ensure_model

BASE = Path(__file__).resolve().parent
configure_cache()

import Levenshtein
from quran_transcript import Aya, MoshafAttributes, chunck_phonemes, explain_error, quran_phonetizer

SUPPORTED_SURAHS = (1,)
VOWELS = {"َ": "فتحه", "ِ": "کسره", "ُ": "ضمه"}
LONG_VOWELS = "اۦۥ"
MOSHAF = MoshafAttributes(rewaya="hafs", madd_monfasel_len=2, madd_mottasel_len=4,
                         madd_mottasel_waqf=4, madd_aared_len=2)


@dataclass
class Reference:
    ayah: int
    text: str
    phonetic: object
    words: list[str]
    word_spans: list[tuple[int, int]]


def make_reference(ayah: int, text: str | None = None) -> Reference:
    text = text if text is not None else Aya(1, ayah).get().uthmani
    # quran_phonetizer applies its end-of-segment waqf operations automatically.
    phonetic = quran_phonetizer(text, MOSHAF, remove_spaces=True)
    matches = list(re.finditer(r"\S+", text))
    return Reference(ayah, text, phonetic, [m.group() for m in matches], [m.span() for m in matches])


def basic_form(group: str) -> str:
    """Ignore advanced length/attribute differences while retaining vowel presence."""
    group = group.replace("ڇ", "").replace("ۜ", "")
    group = group.replace("ں", "ن").replace("۾", "م")
    # Presence/type of a long vowel matters; its beat count does not in v1.
    group = re.sub(r"([اۦۥ])\1+", r"\1", group)
    # Extended nasalization is outside v1; single vs double remains significant.
    group = re.sub(r"([نم])\1{2,}", r"\1\1", group)
    return group


def readable(group: str) -> str:
    return basic_form(group).replace("ۦ", "ی").replace("ۥ", "و")


def describe(expected: str, heard: str, operation: str) -> tuple[str, str]:
    if operation == "delete":
        return "deletion", f"بخش «{readable(expected)}» در خروجی مدل پیدا نشد؛ احتمال حذف یا تلفظ نامشخص وجود دارد."
    if operation == "insert":
        return "insertion", f"نزدیک این کلمه، بخش اضافهٔ «{readable(heard)}» تشخیص داده شد."
    exp_base = re.sub("[َُِ]", "", basic_form(expected))
    heard_base = re.sub("[َُِ]", "", basic_form(heard))
    if exp_base == heard_base:
        exp_vowel = VOWELS.get(expected[-1:], "سکون")
        heard_vowel = VOWELS.get(heard[-1:], "سکون")
        return "vowel", f"در این بخش، {heard_vowel} به‌جای {exp_vowel} تشخیص داده شد."
    if exp_base and heard_base and set(exp_base) == set(heard_base) and len(exp_base) != len(heard_base):
        return "gemination", "تشدید این بخش با مرجع تفاوت دارد؛ آن را دوباره بررسی کنید."
    return "letter", f"در این بخش، «{readable(heard)}» به‌جای «{readable(expected)}» تشخیص داده شد."


def compare_prediction(ref: Reference, predicted: str) -> list[dict]:
    predicted = predicted.replace(" ", "").strip()
    # Preserve the library's text mapping/error metadata. Supplement its group
    # alignment: v0.6.1 zips unequal replacement spans, dropping residual groups.
    upstream = explain_error(ref.text, ref.phonetic.phonemes, predicted, ref.phonetic.mappings)
    ref_groups = chunck_phonemes(ref.phonetic.phonemes)
    pred_groups = chunck_phonemes(predicted)
    phone_to_char = {}
    for char_idx, mapping in enumerate(ref.phonetic.mappings):
        if mapping is not None:
            for position in range(*mapping.pos):
                phone_to_char.setdefault(position, char_idx)
    starts, cursor = [], 0
    for group in ref_groups:
        starts.append(cursor)
        cursor += len(group)
    upstream_by_span = {(e.ph_pos[0], e.expected_ph, e.preditected_ph): e for e in upstream}
    found = []

    def append_error(ri, expected, heard, op):
        if basic_form(expected) == basic_form(heard):
            return
        phone_start = starts[ri] if ri < len(starts) else len(ref.phonetic.phonemes)
        hint = upstream_by_span.get((phone_start, expected, heard))
        if hint is not None and hint.uthmani_pos[0] < len(ref.text):
            char_start = hint.uthmani_pos[0]
        else:
            char_start = phone_to_char.get(phone_start, phone_to_char.get(phone_start - 1, 0))
        word_index = next((i for i, (a, b) in enumerate(ref.word_spans) if a <= char_start < b), len(ref.words) - 1)
        kind, message = describe(expected, heard, op)
        found.append(dict(word_index=word_index, word=ref.words[word_index], type=kind,
                          operation=op, expected=readable(expected), heard=readable(heard), message=message))

    # Anchor on base phones so vowel and gemination differences remain local.
    ref_keys = [basic_form(g)[:1] for g in ref_groups]
    pred_keys = [basic_form(g)[:1] for g in pred_groups]
    for op, a, b, c, d in Levenshtein.opcodes(ref_keys, pred_keys):
        if op == "equal":
            for ri, pi in zip(range(a, b), range(c, d)):
                append_error(ri, ref_groups[ri], pred_groups[pi], "replace")
        elif op == "delete":
            for ri in range(a, b):
                append_error(ri, ref_groups[ri], "", "delete")
        elif op == "insert":
            for pi in range(c, d):
                append_error(a, "", pred_groups[pi], "insert")
        else:
            paired = min(b - a, d - c)
            for offset in range(paired):
                append_error(a + offset, ref_groups[a + offset], pred_groups[c + offset], "replace")
            for ri in range(a + paired, b):
                append_error(ri, ref_groups[ri], "", "delete")
            for pi in range(c + paired, d):
                append_error(b, "", pred_groups[pi], "insert")
    return found


class RecitationService:
    def __init__(self):
        self.references = {i: make_reference(i) for i in range(1, 8)}
        self.verses = [dict(ayah=i, text=r.text, words=r.words) for i, r in self.references.items()]
        self.ready = False
        self.model = None
        self._live_prediction = None

    def load(self):
        if self.ready:
            return
        try:
            import torch
            from quran_muaalem import Muaalem
            torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
            logging.info("Loading %s on CPU. First run downloads the model to %s.", MODEL_ID, os.environ["HF_HOME"])
            model_path = ensure_model()
            # Use the complete local snapshot so later starts need no network,
            # including metadata requests. Never load a different model revision.
            self.model = Muaalem(model_name_or_path=model_path, device="cpu", dtype=torch.float32)
            self.model.model.eval()
            self.ready = True
            logging.info("Model ready. Open http://127.0.0.1:5000")
        except Exception:
            self.model = None
            logging.exception("Model could not be loaded. Check the download and installed dependencies.")

    def predict(self, wave, ref):
        if not self.ready:
            raise RuntimeError("Model is not ready")
        return self.model([wave], [ref.phonetic], sampling_rate=16000)[0].phonemes.text

    def predict_live(self, wave):
        """Same Muaalem weights, features and official phoneme decoder.

        Live feedback consumes no sifat. Skip reference tokenization and the
        costly advanced-attribute alignment in Muaalem.__call__ (pinned 0.2.2).
        Do not quantize, change precision, or change the phoneme decoding rule.
        Called only under the shared server inference lock.
        """
        if not self.ready:
            raise RuntimeError("Model is not ready")
        import torch
        from quran_muaalem.decode import phonemes_level_greedy_decode

        # Pausing/stopping can review identical audio with different alignment
        # rules. Reuse its prediction, retaining only a digest and text, no audio.
        digest = blake2b(wave.tobytes(), digest_size=16).digest()
        if self._live_prediction and self._live_prediction[0] == digest:
            return self._live_prediction[1]
        with torch.inference_mode():
            features = self.model.processor([wave], sampling_rate=16000, return_tensors="pt")
            features = {k: v.to(self.model.device, dtype=self.model.dtype) for k, v in features.items()}
            logits = self.model.model(**features, return_dict=False)[0]["phonemes"]
            probs = logits.softmax(dim=-1).cpu().to(torch.float32)
            predicted = phonemes_level_greedy_decode(
                probs, self.model.multi_level_tokenizer.id_to_vocab["phonemes"])[0].text
        self._live_prediction = digest, predicted
        return predicted

    def analyze(self, wave, ayah):
        ref = self.references[ayah]
        predicted = self.predict(wave, ref)
        base_result = dict(surah=1, ayah=ayah, errors=[])
        if not predicted.strip():
            return dict(base_result, status="needs_retry", message="خواندن آیه تشخیص داده نشد. در محیطی آرام دوباره ضبط کنید.")
        # Gross mismatch is a retry, not dozens of purported pronunciation errors.
        expected_core = "".join(basic_form(g) for g in chunck_phonemes(ref.phonetic.phonemes))
        heard_core = "".join(basic_form(g) for g in chunck_phonemes(predicted))
        mismatch = Levenshtein.distance(expected_core, heard_core) / max(len(expected_core), len(heard_core), 1)
        if mismatch > 0.6:
            return dict(base_result, status="needs_retry", message="این ضبط با آیهٔ انتخاب‌شده تطبیق کافی ندارد. فقط همین آیه را دوباره بخوانید.")
        errors = compare_prediction(ref, predicted)
        base_result["errors"] = errors
        return dict(base_result, status="differences" if errors else "no_difference",
                    message="در کلمه‌های مشخص‌شده اختلاف احتمالی پیدا شد؛ توضیح هر بخش را بررسی کنید." if errors else "در موارد بررسی‌شده اختلافی پیدا نشد")
