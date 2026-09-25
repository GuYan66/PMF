"""Portable PMF training entry point; all paths are explicit."""
import argparse
import json
from pathlib import Path
import torch
from MMSA import MMSA_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test2", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and forward pass without training or saving files")
    options = parser.parse_args()
    for path in (options.data, options.test2, options.config):
        if path is not None and not path.is_file():
            parser.error(f"File not found: {path}")
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if options.device == "auto" else options.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Use cpu, cuda:N or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    config = json.loads(options.config.read_text(encoding="utf-8")) if options.config else {}
    config.update(device=device)
    if options.test2:
        config["test2_feature"] = str(options.test2.resolve())
    if options.dry_run:
        from MMSA.config import get_config_regression
        from MMSA.data_loader import MMDataLoader
        from MMSA.models import AMIO
        from MMSA.trains import ATIO
        args = get_config_regression("pmf", "mosei")
        args.update(config)
        args.update(custom_feature=str(options.data.resolve()), feature_T=None, feature_A=None, feature_V=None, train_mode="regression")
        loaders = MMDataLoader(args, options.num_workers)
        model = AMIO(args).to(device)
        trainer = ATIO().getTrain(args)
        trainer._fit_normalization(model, loaders["train"].dataset)
        model.eval()
        text, audio, vision, _, mask = trainer._batch(next(iter(loaders["train"])))
        with torch.no_grad():
            out = model(text, audio, vision, padding_mask=mask)
        print(json.dumps({"mode":"dry_run", "device":str(device), "splits":{k:len(v.dataset) for k,v in loaders.items()},
                          "parameters":sum(p.numel() for p in model.parameters()),
                          "polarity_shape":list(out["polarity"].shape), "magnitude_shape":list(out["magnitude"].shape)}, indent=2))
        return
    output = options.output_dir.resolve()
    MMSA_run("pmf", "mosei", config=config, custom_feature=str(options.data.resolve()), seeds=options.seeds,
             gpu_ids=[device.index or 0] if device.type == "cuda" else [], num_workers=options.num_workers,
             model_save_dir=output/"saved_models", res_save_dir=output/"results", log_dir=output/"logs")


if __name__ == "__main__":
    main()
