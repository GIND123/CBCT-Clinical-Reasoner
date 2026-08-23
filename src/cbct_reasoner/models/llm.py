"""Optional findings-to-narrative renderer (LoRA fine-tune of a small instruct LM).

The template renderer concatenates prototype sentences, which is factually safe
but reads like a checklist. ToothFairy4 references are dictated prose, and Phase 2
is judged by maxillofacial surgeons in a head-to-head arena where fluency is
visible. A paraphraser that keeps the finding set fixed and only improves the
prose can win those matchups without touching factual content.

The framing is deliberate and defensive: **this model never decides what is
true.** It receives an already-selected finding list and rewrites it. Anything it
adds is a hallucination that RadFact precision will punish, so
``verify_faithfulness`` checks the output's concept profile against the input's
and rejects generations that introduce or drop findings. The renderer is only
adopted if it beats the template on out-of-fold data - see ``compare_renderers``.

Requires the ``llm`` extra: ``pip install '.[llm]'``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cbct_reasoner.config import LlmConfig
from cbct_reasoner.ontology import concept_profile
from cbct_reasoner.text import join_report, split_phrases

SYSTEM_PROMPT = (
    "You are a maxillofacial radiologist. Rewrite the given list of CBCT findings "
    "as a single flowing clinical report paragraph. Use every finding exactly once. "
    "Do not add findings, measurements, tooth numbers, or laterality that are not "
    "in the list. Do not add recommendations. Preserve negations."
)


def build_prompt(findings: Sequence[str]) -> list[dict[str, str]]:
    listing = "\n".join(f"- {finding}" for finding in findings)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Findings:\n{listing}\n\nReport:"},
    ]


@dataclass(frozen=True, slots=True)
class SftExample:
    findings: tuple[str, ...]
    report: str

    def to_dict(self) -> dict[str, object]:
        return {"findings": list(self.findings), "report": self.report}


def build_sft_dataset(
    references: Sequence[str], *, shuffle_findings: bool = True, seed: int = 2026
) -> list[SftExample]:
    """Turn each reference report into a (finding list -> narrative) pair.

    The finding list is the report's own phrases, so the target is exactly
    recoverable. Shuffling the input order teaches the model to impose report
    structure rather than echo the order it was handed.
    """
    import random

    rng = random.Random(seed)
    examples: list[SftExample] = []
    for reference in references:
        phrases = split_phrases(reference)
        if len(phrases) < 2:
            continue
        ordered = list(phrases)
        if shuffle_findings:
            rng.shuffle(ordered)
        examples.append(SftExample(findings=tuple(ordered), report=reference))
    return examples


def save_sft_dataset(examples: Sequence[SftExample], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            stream.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    return output


def verify_faithfulness(findings: Sequence[str], generated: str) -> tuple[bool, dict[str, object]]:
    """Reject a generation that changed the finding set.

    Compares ontology concept profiles rather than strings, so legitimate
    rephrasing passes while an added finding, a dropped finding, or a flipped
    negation fails.
    """
    source = concept_profile(findings)
    target = concept_profile(split_phrases(generated))
    added = {k: v for k, v in target.items() if k not in source}
    dropped = {k: v for k, v in source.items() if k not in target}
    flipped = {
        key: (source[key], target[key])
        for key in set(source) & set(target)
        if {source[key], target[key]} == {"present", "absent"}
    }
    ok = not added and not flipped and len(dropped) <= 0
    return ok, {"added": added, "dropped": dropped, "flipped": flipped}


class NarrativeRenderer:
    """Wraps a base model plus LoRA adapter for findings-to-prose rewriting."""

    def __init__(self, model, tokenizer, config: LlmConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def load(
        cls, config: LlmConfig, *, adapter_path: str | Path | None = None, device: str | None = None
    ) -> NarrativeRenderer:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            dtype=torch.bfloat16 if resolved == "cuda" else torch.float32,
        )
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.to(resolved).eval()
        return cls(model, tokenizer, config)

    def render(self, findings: Sequence[str], *, fallback: str | None = None) -> str:
        """Rewrite findings as prose, falling back to the template on any failure."""
        import torch

        template = fallback if fallback is not None else join_report(list(findings))
        if not findings:
            return template
        try:
            prompt = self.tokenizer.apply_chat_template(
                build_prompt(findings), tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            generated = self.tokenizer.decode(
                output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
        except Exception as error:  # pragma: no cover - defensive
            print(f"warning: narrative generation failed ({error}); using the template", flush=True)
            return template

        ok, detail = verify_faithfulness(findings, generated)
        if not ok:
            print(f"warning: rejected unfaithful generation {detail}", flush=True)
            return template
        return generated


def train_adapter(
    examples: Sequence[SftExample],
    config: LlmConfig,
    output_dir: str | Path,
    *,
    device: str | None = None,
) -> Path:
    """LoRA supervised fine-tune. Requires the ``llm`` extra and a GPU in practice."""
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, dtype=torch.bfloat16 if resolved == "cuda" else torch.float32
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.to(resolved).train()

    def encode(example: SftExample) -> dict[str, torch.Tensor]:
        prompt = tokenizer.apply_chat_template(
            build_prompt(example.findings), tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(example.report + tokenizer.eos_token, add_special_tokens=False)[
            "input_ids"
        ]
        input_ids = (prompt_ids + answer_ids)[: config.max_length]
        # Loss is computed on the report only; the prompt is context, not a target.
        labels = ([-100] * len(prompt_ids) + answer_ids)[: config.max_length]
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
        }

    encoded = [encode(example) for example in examples]

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in batch)
        pad_id = tokenizer.pad_token_id
        return {
            "input_ids": torch.stack(
                [
                    torch.cat(
                        [item["input_ids"], torch.full((width - len(item["input_ids"]),), pad_id)]
                    )
                    for item in batch
                ]
            ),
            "labels": torch.stack(
                [
                    torch.cat([item["labels"], torch.full((width - len(item["labels"]),), -100)])
                    for item in batch
                ]
            ),
        }

    loader = DataLoader(encoded, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total = len(loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total), total)

    for epoch in range(config.epochs):
        running = 0.0
        for batch in loader:
            batch = {key: value.to(resolved) for key, value in batch.items()}
            attention = (batch["input_ids"] != tokenizer.pad_token_id).long()
            loss = model(**batch, attention_mask=attention).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())
        print(
            f"[llm] epoch {epoch + 1}/{config.epochs} loss={running / max(1, len(loader)):.4f}",
            flush=True,
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(destination))
    tokenizer.save_pretrained(str(destination))
    return destination


def compare_renderers(
    findings_by_case: dict[str, Sequence[str]],
    references_by_case: dict[str, str],
    renderer: NarrativeRenderer,
) -> dict[str, object]:
    """Score template rendering against LLM rendering on the same finding sets.

    Adopt the LLM only if it wins here. Identical inputs isolate the effect of
    prose quality from the effect of finding selection.
    """
    from cbct_reasoner.metrics.score import score_reports

    case_ids = sorted(set(findings_by_case) & set(references_by_case))
    template = {case_id: join_report(list(findings_by_case[case_id])) for case_id in case_ids}
    narrative = {
        case_id: renderer.render(findings_by_case[case_id], fallback=template[case_id])
        for case_id in case_ids
    }
    references = {case_id: references_by_case[case_id] for case_id in case_ids}
    template_score = score_reports(template, references)
    narrative_score = score_reports(narrative, references)
    return {
        "template": template_score.to_dict(),
        "narrative": narrative_score.to_dict(),
        "adopt_narrative": narrative_score.final > template_score.final,
        "delta": narrative_score.final - template_score.final,
    }
