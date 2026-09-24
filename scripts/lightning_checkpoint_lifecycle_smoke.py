"""Exercise Lightning validation/checkpoint/test/validate hooks on a tiny CPU model."""

import argparse
import json
import tempfile
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset


class TinyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.validation_steps = []
        self.loaded_checkpoint_steps = []
        self.test_loaded_step = None
        self.final_validate_loaded_step = None
        self.capture_fit_validation = True
        self.fit_validation_scores = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.weight * x, y)

    def validation_step(self, batch, batch_idx):
        score = float(self.global_step)
        if self.capture_fit_validation:
            self.validation_steps.append(int(self.global_step))
            self.fit_validation_scores.append(score)
        self.log("val/probe_score", score, on_step=False, on_epoch=True, logger=False)
        if self.loaded_checkpoint_steps:
            self.final_validate_loaded_step = self.loaded_checkpoint_steps[-1]

    def test_step(self, batch, batch_idx):
        self.test_loaded_step = self.loaded_checkpoint_steps[-1] if self.loaded_checkpoint_steps else None

    def on_load_checkpoint(self, checkpoint):
        self.loaded_checkpoint_steps.append(int(checkpoint["global_step"]))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--every-n-train-steps", type=int, default=2001)
    parser.add_argument("--expect-no-checkpoint", action="store_true")
    parser.add_argument(
        "--save-on-validation-end",
        action="store_true",
        help="Disable step cadence and set save_on_train_epoch_end=False.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="ensrec-lightning-smoke-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # One sample per step keeps the framework hook schedule easy to inspect.
    x = torch.ones((2001, 1))
    y = torch.zeros((2001, 1))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=1, num_workers=0)
    eval_loader = DataLoader(TensorDataset(torch.ones((1, 1)), torch.zeros((1, 1))), batch_size=1)
    every_n_train_steps = None if args.save_on_validation_end else args.every_n_train_steps
    save_on_train_epoch_end = False if args.save_on_validation_end else None
    checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="probe-{step:04d}",
        monitor="val/probe_score",
        mode="max",
        save_top_k=1,
        every_n_train_steps=every_n_train_steps,
        save_on_train_epoch_end=save_on_train_epoch_end,
        auto_insert_metric_name=False,
    )
    model = TinyModule()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=-1,
        max_steps=2001,
        val_check_interval=2000,
        num_sanity_val_steps=0,
        limit_train_batches=2001,
        limit_val_batches=1,
        limit_test_batches=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
    model.capture_fit_validation = False
    best_path = checkpoint.best_model_path
    if not best_path:
        report = {
            "scope": "synthetic CPU Lightning lifecycle only; not an EnsRec training run",
            "lightning_version": pl.__version__,
            "every_n_train_steps": every_n_train_steps,
            "save_on_train_epoch_end": save_on_train_epoch_end,
            "validation_steps_during_fit": model.validation_steps,
            "validation_scores_during_fit": model.fit_validation_scores,
            "trainer_global_step_after_fit": int(trainer.global_step),
            "best_model_path": None,
            "best_model_score": None,
        }
        print(json.dumps(report, indent=2))
        if args.expect_no_checkpoint:
            return
        raise RuntimeError("ModelCheckpoint did not create best_model_path")
    saved = torch.load(best_path, map_location="cpu", weights_only=False)
    saved_global_step = int(saved["global_step"])
    best_score = float(checkpoint.best_model_score)

    trainer.test(model, dataloaders=eval_loader, ckpt_path=best_path)
    test_loaded_step = model.test_loaded_step
    trainer.validate(model, dataloaders=eval_loader, ckpt_path=best_path)
    validate_loaded_step = model.final_validate_loaded_step

    report = {
        "scope": "synthetic CPU Lightning lifecycle only; not an EnsRec training run",
        "lightning_version": pl.__version__,
        "every_n_train_steps": every_n_train_steps,
        "save_on_train_epoch_end": save_on_train_epoch_end,
        "validation_steps_during_fit": model.validation_steps,
        "validation_scores_during_fit": model.fit_validation_scores,
        "best_model_path": best_path,
        "best_model_score": best_score,
        "trainer_global_step_after_fit": int(trainer.global_step),
        "saved_checkpoint_global_step": saved_global_step,
        "test_ckpt_path_argument": best_path,
        "test_loaded_checkpoint_global_step": test_loaded_step,
        "final_validate_ckpt_path_argument": best_path,
        "final_validate_loaded_checkpoint_global_step": validate_loaded_step,
    }
    print(json.dumps(report, indent=2))
    if model.validation_steps != [2000]:
        raise AssertionError(f"Expected one fit validation at step 2000, got {model.validation_steps}")
    expected_checkpoint_step = 2000 if args.save_on_validation_end else args.every_n_train_steps
    if expected_checkpoint_step is None:
        expected_checkpoint_step = 2000
    if saved_global_step != expected_checkpoint_step:
        raise AssertionError(
            f"Expected checkpoint global_step {expected_checkpoint_step}, got {saved_global_step}"
        )
    expected_score = 2000.0
    if model.fit_validation_scores != [expected_score] or best_score != expected_score:
        raise AssertionError(
            f"Expected checkpoint score {expected_score} from validation at step 2000; "
            f"got scores={model.fit_validation_scores}, best_score={best_score}"
        )
    if test_loaded_step != saved_global_step or validate_loaded_step != saved_global_step:
        raise AssertionError("Test and final validate did not load the saved checkpoint step")


if __name__ == "__main__":
    main()
