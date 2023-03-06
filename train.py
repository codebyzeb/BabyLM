"""Train a RoBERTa model on the BabyLM dataset."""

import logging
import os
import textstat

# config-related imports
import hydra

# training pipeline imports
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import TrainingArguments

# wandb for logging metrics
import wandb
from src.config import BabyLMConfig
from src.models import load_model
from src.objective import load_objective_collator
from src.preprocessing import DataPreprocessor
from src.tokenizer import load_tokenizer
from src.trainer import CustomTrainer
from src.utils.setup import set_seed

# from transformers import Trainer, TrainingArguments


# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=BabyLMConfig)

# A logger for this file
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: BabyLMConfig):
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing keys in config: \n {missing_keys}")

    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.experiment.seed)

    # Loading dataset
    logger.info("Loading dataset")
    dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        use_auth_token=os.environ["HF_READ_TOKEN"],
    )

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(cfg, dataset)

    # Load model
    logger.info("Initializing model")
    model = load_model(cfg)

    # Preprocess data
    logger.info("Preprocessing data")
    
    data_preprocessor = DataPreprocessor(cfg, tokenizer)
    processed_dataset = dataset.map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=["text"],
    )

    objective_collator = load_objective_collator(cfg, tokenizer)

    # Setting up wandb
    if cfg.experiment.dry_run:
        os.environ["WANDB_MODE"] = "disabled"
    else:
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project=cfg.experiment.group,
            name=cfg.experiment.name,
            entity="baby-lm",
        )

    # Set up training arguments
    # TODO: If we are using wandb sweeps, note that we will need to think about how we store/
    # initialize the name of the current experiment so that it doesn't interfere with the name
    # of other experiments, and also so that we can store checkpoints of that run on HF hub;
    # alternatively maybe we use ray tune which is natively supported by Trainer
    training_args = TrainingArguments(
        output_dir=f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        do_predict=False,
        per_device_train_batch_size=cfg.trainer.batch_size,  # NOTE: We can should maybe use auto_find_batch_size
        learning_rate=cfg.trainer.lr,
        max_steps=cfg.trainer.max_training_steps,
        warmup_steps=cfg.trainer.num_warmup_steps,
        seed=cfg.experiment.seed,
        save_steps=1,
        report_to="wandb"
        if not cfg.experiment.dry_run
        else None,  # wandb deactivated for dry runs
        hub_strategy="every_save",
        push_to_hub=not cfg.experiment.dry_run,
        hub_model_id=f"CamBabyTrainers/{cfg.experiment.group}-{cfg.model.name}-model"
        if not cfg.experiment.dry_run
        else None,
        hub_token=os.environ["HF_WRITE_TOKEN"]
        if not cfg.experiment.dry_run
        else None,
    )

    # Set up trainer
    trainer = CustomTrainer(
        experiment_group=cfg.experiment.group,
        experiment_name=cfg.experiment.name,
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=objective_collator,
    )

    trainer.train(resume_from_checkpoint=cfg.model.resume_checkpoint_path)


if __name__ == "__main__":
    main()
