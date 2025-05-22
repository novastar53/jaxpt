import logging

import numpy as np


logger = logging.getLogger(__name__)


def format_as_gpt4_chatml_and_tokenize(
    tokenizer,
    question="What is photosynthesis?",
    system_prompt="You are a helpful assistant.",
    start=False,
    logger=logger,
):
    if start is True:
        x = [
            "<|im_start|>system\n" + system_prompt + "<|im_end|>\n",
        ]
    else:
        x = []

    x += [
        "<|im_start|>user\n" + question + "<|im_end|>\n",
        "<|im_start|>assistant\n",
    ]
    x = "".join(x)
    logger.debug(f"Formatted input:\n{x}")
    tok_x = np.array(tokenizer.encode(x), dtype=np.uint16)
    logger.debug(f"Tokenized input of length {len(tok_x)} generated.")
    return tok_x
