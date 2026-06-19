"""Response-LM tokenization with an instruction/response mask.

See design doc §4: for `response_lm` the loss (and gradient signature) covers
only the response tokens; instruction tokens are masked out.
"""

from __future__ import annotations

from typing import Any


def encode_response_lm(
    tokenizer: Any, instruction: str, response: str, *, max_len: int = 1024
) -> dict[str, list[int]]:
    """Tokenize instruction + response and mark which positions are response.

    Returns a dict with `input_ids`, `attention_mask`, `response_mask` (all the
    same length). The instruction is tokenized with special tokens (BOS), the
    response without, and an EOS is appended when the tokenizer defines one.
    The response_mask is 1 on response tokens (and the trailing EOS), 0 on the
    instruction.
    """
    instr_ids = list(tokenizer(instruction, add_special_tokens=True)["input_ids"])
    resp_ids = list(tokenizer(response, add_special_tokens=False)["input_ids"])

    ids = instr_ids + resp_ids
    mask = [0] * len(instr_ids) + [1] * len(resp_ids)

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))
        mask.append(1)

    ids = ids[:max_len]
    mask = mask[:max_len]
    if not any(mask):  # guarantee at least one supervised token
        mask[-1] = 1
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "response_mask": mask}
