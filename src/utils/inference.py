"""
Utility functions for running model inference. 
"""

from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING, Dict, List

import torch
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    # avoid circular imports
    from src.trainer import CustomTrainer


def compute_trainer_perplexity(
    batch: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizerFast,
    trainer: CustomTrainer,
) -> List[float]:
    """

    A helper fucntion for computing the perplexity of a batch of data. Assumes that the data
    has been tokenized and batched using the same tokenizer that was used for training the model.

    Args:
        * batch: a batch of data that should contain a key "input_ids" as well as
            "special_tokens_mask" (by default this is returned by the tokenizer we uses).
        * tokenizer: a tokenizer object that was used for tokenizing the input ids, we use this
            only to determine the mask token id.
        * trainer: a trainer object that was used for training the model
    Returns:
        * perplexity (float): The perplexity of the n-gram

    """

    mask_idx = tokenizer.mask_token_id

    assert (
        mask_idx is not None
    ), "The tokenizer must have a mask token and a pad token"

    input_ids = batch["input_ids"]

    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)

    # (Batch, #repetitions dimension, seq len)
    input_ids = input_ids.unsqueeze(1).to(trainer.args.device)

    repeat_ids = input_ids.repeat([1, seq_len, 1])

    mask = (
        torch.ones(input_ids.size(-1), device=trainer.args.device).diag(0)
    ).repeat([batch_size, 1, 1])

    # Setting the diagonal for each batch to MASK token id (0)
    masked_input = repeat_ids.masked_fill(mask == 1, mask_idx)

    # For each batch, set the labels to be the original input ids (all others to ignore_index=-100)
    labels = repeat_ids.masked_fill(masked_input != mask_idx, -100)

    # For each batch, if the label is a special token, set it to -100 (ignore_index)
    special_tokens_mask = (
        batch["special_tokens_mask"].unsqueeze(1).to(trainer.args.device)
    )
    labels = labels.masked_fill(special_tokens_mask == 1, -100)

    # combining the repeated input ids dimension (2nd dim) with the batch dim (1st dim)
    # NOTE this gives an effective batch size = batch_size * seq_len
    masked_input = masked_input.view(-1, seq_len)
    labels = labels.view(-1, seq_len)

    base_model_outputs = trainer.model(input_ids=masked_input)
    base_model_hidden_states = base_model_outputs[0]

    # NOTE: The 'mlm' unit is always in the objective curriculum
    # (this is checked by ObjectiveCurriculum.__init__)
    loss = trainer.objective_curriculum.units["mlm"].compute_loss(
        base_model_hidden_states,
        {},  # No Input dict required for perplexity, just labels
        override_lables=labels,
        loss_kwargs={
            "reduction": "none",
        },
    )

    # loss is a tensor (batch * seq_len, seq_len), where in the second dimension only at most one
    # token should be non-zero (the masked token). We sum over the second dimension to get the
    # loss for each token in each batch

    loss = loss.sum(dim=-1)

    # Now loss is a vector of (batch * seq_len) length, we reshape it to (batch, seq_len)
    loss = loss.view(batch_size, seq_len)

    # Now averaging over the seq_len dimension to get the average loss for each batch
    mean_loss = loss.mean(dim=-1)

    # batch perplexity is a vector of length batch_size
    batch_perplexity = torch.exp(mean_loss)

    # converting batch perplexity to a list of floats
    batch_perplexity = batch_perplexity.cpu().tolist()

    return batch_perplexity