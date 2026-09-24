"""Compose selected Beauty train configs with Hydra and emit resolved audit fields."""

import argparse
import importlib.util
import json
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
import omegaconf
from omegaconf import OmegaConf

CASES = {
    "id_only": ("id_only/train_beauty", "configs/experiment/id_only/train_beauty.yaml", "configs/model/id_only.yaml"),
    "text_only": ("text_only/train_beauty", "configs/experiment/text_only/train_beauty.yaml", "configs/model/text_only.yaml"),
    "text_only_ablate_negatives": (
        "text_only/ablate_negatives/train_beauty",
        "configs/experiment/text_only/ablate_negatives/train_beauty.yaml",
        "configs/model/text_only.yaml",
    ),
}

FIELDS = (
    "model.id_loss_weight",
    "model.text_loss_weight",
    "model.full_batch_id_negatives",
    "model.full_batch_text_negatives",
    "model.text_loss_function._target_",
    "model.text_loss_function.normalize",
    "model.text_loss_function.contrastive_tau",
    "model.d_model",
    "callbacks.model_checkpoint.monitor",
    "callbacks.model_checkpoint.mode",
    "callbacks.model_checkpoint.save_top_k",
    "callbacks.model_checkpoint.every_n_train_steps",
    "trainer.val_check_interval",
    "trainer.max_steps",
    "trainer.accelerator",
)


def register_project_resolvers(ensrec_root: Path):
    path = ensrec_root / "src" / "utils" / "custom_hydra_resolvers.py"
    spec = importlib.util.spec_from_file_location("ensrec_custom_hydra_resolvers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load custom resolver module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def compose_fields(ensrec_root: Path):
    register_project_resolvers(ensrec_root)
    cases = {}
    for name, (experiment, experiment_source, model_source) in CASES.items():
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(ensrec_root / "configs"),
            job_name="beauty_config_audit",
        ):
            cfg = compose(
                config_name="train.yaml",
                overrides=[f"experiment={experiment}", "trainer=gpu"],
            )
            fields = {field: OmegaConf.select(cfg, field) for field in FIELDS}
        cases[name] = {
            "experiment_source": experiment_source,
            "base_model_source": model_source,
            "callback_defaults_source": "configs/callbacks/model_checkpoint.yaml",
            "overrides": [f"experiment={experiment}", "trainer=gpu"],
            "fields": fields,
        }
    return {
        "method": (
            "Hydra 1.3 compose() with project custom resolvers; selected fields resolved. "
            "The full job config is not serialized because compose() has no runtime output_dir."
        ),
        "hydra_version": hydra.__version__,
        "omegaconf_version": omegaconf.__version__,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensrec-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compose_fields(args.ensrec_root.resolve())
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
