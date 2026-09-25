"""PMF training and evaluation with polarity logits and magnitude."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...models.subNets.PolMagHead import compose_intensity, polarity_from_intensity
from ...utils import dict_to_str


logger = logging.getLogger("MMSA")


class POLMAG:
    def __init__(self, args):
        self.args = args
        self.best_epoch = None
        self.best_validation = None

    def _batch(self, batch_data):
        device = self.args.device
        text = batch_data["text"].to(device)
        audio = batch_data["audio"].to(device)
        vision = batch_data["vision"].to(device)
        y = batch_data["labels"]["M"].view(-1).to(device)
        padding_mask = batch_data["padding_mask"].to(device)
        return text, audio, vision, y, padding_mask

    def _optimizer(self, model):
        return optim.Adam(model.parameters(), lr=self.args.learning_rate,
                          weight_decay=self.args.get("weight_decay", 0.0))

    @torch.no_grad()
    def _fit_normalization(self, model, train_dataset):
        pooled = []
        loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            text, audio, vision, _, padding_mask = self._batch(batch)
            pooled.append(model.Model.pooled_features(text, audio, vision, padding_mask).cpu())
        model.Model.fit_normalization(torch.cat(pooled, dim=0))
        logger.info("PMF standardization fitted on %d training examples only.", len(train_dataset))

    def do_train(self, model, dataloader, return_epoch_results=False):
        device = self.args.device
        self._fit_normalization(model, dataloader["train"].dataset)
        ce = nn.CrossEntropyLoss()
        l1 = nn.L1Loss()
        optimizer = self._optimizer(model)
        best_f1 = -1.0
        best_epoch = 0
        epoch = 0
        while True:
            epoch += 1
            model.train()
            running = 0.0
            for batch_data in tqdm(dataloader["train"], leave=False):
                text, audio, vision, y, padding_mask = self._batch(batch_data)
                target = polarity_from_intensity(y)
                out = model(text, audio, vision, padding_mask=padding_mask)
                loss_cls = ce(out["polarity"], target)
                mask = target != 1
                if mask.any():
                    loss_mag = l1(out["magnitude"][mask], y[mask].abs())
                else:
                    loss_mag = out["magnitude"].sum() * 0.0
                loss = loss_cls + loss_mag
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
            valid = self.do_test(model, dataloader["valid"], mode="VAL")
            train_loss = running / len(dataloader["train"])
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epoch - best_epoch}/{epoch}] "
                f"loss: {train_loss:.4f} {dict_to_str(valid)}"
            )
            # Preserve the original four-decimal MacroF1 checkpoint rule.
            selection_f1 = round(valid["MacroF1"], 4)
            if selection_f1 > best_f1:
                best_f1 = selection_f1
                best_epoch = epoch
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(device)
                self.best_epoch = epoch
                self.best_validation = valid.copy()
                logger.info(f"BEST-({self.args.model_name}) epoch={epoch} {dict_to_str(valid)}")
            if epoch - best_epoch >= self.args.early_stop:
                return None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        pred_y, true_y, pred_c, true_c = [], [], [], []
        with torch.no_grad():
            for batch_data in dataloader:
                text, audio, vision, y, padding_mask = self._batch(batch_data)
                out = model(text, audio, vision, padding_mask=padding_mask)
                polarity = out["polarity"].argmax(dim=-1)
                composed = compose_intensity(polarity, out["magnitude"])
                pred_y.append(composed.detach().cpu().numpy())
                true_y.append(y.detach().cpu().numpy())
                pred_c.append(polarity.detach().cpu().numpy())
                true_c.append(polarity_from_intensity(y).detach().cpu().numpy())
        pred_y = np.concatenate(pred_y)
        true_y = np.concatenate(true_y)
        pred_c = np.concatenate(pred_c)
        true_c = np.concatenate(true_c)
        # Both intensity metrics use the final signed prediction on ALL samples,
        # including true/predicted neutral cases. Undefined Pearson stays missing.
        corr = float("nan")
        if len(pred_y) > 1 and np.ptp(pred_y) > 0 and np.ptp(true_y) > 0:
            corr = float(np.corrcoef(pred_y, true_y)[0, 1])
        results = {
            "Acc3": float(accuracy_score(true_c, pred_c)),
            "MacroF1": float(f1_score(true_c, pred_c, average="macro", labels=[0, 1, 2], zero_division=0)),
            "MAE": float(np.mean(np.abs(pred_y - true_y))),
            "Corr": corr,
        }
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(results)}")
        return results
