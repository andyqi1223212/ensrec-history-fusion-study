"""Report source-aware effective values for the Beauty ID/Text config fields."""

import argparse
import json
from pathlib import Path

import yaml

CASES = {
    "id_only": (
        "experiment/id_only/train_beauty.yaml",
        "model/id_only.yaml",
    ),
    "text_only": (
        "experiment/text_only/train_beauty.yaml",
        "model/text_only.yaml",
    ),
    "text_only_ablate_negatives": (
        "experiment/text_only/ablate_negatives/train_beauty.yaml",
        "model/text_only.yaml",
    ),
}

FIELDS = (
    "_target_",
    "id_loss_weight",
    "text_loss_weight",
    "full_batch_id_negatives",
    "full_batch_text_negatives",
    "text_loss_function._target_",
    "text_loss_function.normalize",
    "text_loss_function.contrastive_tau",
    "d_model",
    "monitor",
    "save_top_k",
    "mode",
    "every_n_train_steps",
    "max_steps",
    "val_check_interval",
    "trainer_group",
)


def get_path(mapping, dotted_path):
    value = mapping
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def yaml_file(root: Path, relative_path: str):
    path = (root / "configs" / relative_path).resolve()
    return path, yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def display_source(root: Path, path):
    return str(path.relative_to(root.resolve())) if path else None


def effective_case(root: Path, experiment_relative: str, model_relative: str):
    exp_path, exp = yaml_file(root, experiment_relative)
    model_path, model = yaml_file(root, model_relative)
    exp_model = exp.get("model", {})
    callback_path, callbacks = yaml_file(root, "callbacks/model_checkpoint.yaml")
    exp_checkpoint = get_path(exp, "callbacks.model_checkpoint") or {}
    base_checkpoint = get_path(callbacks, "model_checkpoint") or callbacks
    exp_trainer = exp.get("trainer", {})
    result = {}
    for field in FIELDS:
        if field in {"monitor", "save_top_k", "mode", "every_n_train_steps"}:
            overlay = exp_checkpoint.get(field)
            base = base_checkpoint.get(field)
            value = overlay if overlay is not None else base
            source = (
                exp_path if overlay is not None else callback_path if base is not None else None
            )
        elif field in {"max_steps", "val_check_interval"}:
            value = exp_trainer.get(field)
            source = exp_path if field in exp_trainer else None
        elif field == "trainer_group":
            value = next(
                (
                    default["override /trainer"]
                    for default in exp.get("defaults", [])
                    if isinstance(default, dict) and "override /trainer" in default
                ),
                None,
            )
            source = exp_path if value is not None else None
        else:
            overlay = get_path(exp_model, field)
            base = get_path(model, field)
            value = overlay if overlay is not None else base
            source = (
                exp_path if overlay is not None else model_path if base is not None else None
            )
        result[field] = {"value": value, "source": display_source(root, source)}
    return {
        "experiment_source": display_source(root, exp_path),
        "experiment_defaults": exp.get("defaults"),
        "base_model_source": display_source(root, model_path),
        "fields": result,
    }


def audit(root: Path):
    return {
        "status": "source-aware static overlay; Hydra composition not executed",
        "cases": {
            name: effective_case(root, *relative_paths)
            for name, relative_paths in CASES.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensrec-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.ensrec_root)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
