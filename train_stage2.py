import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl

from ldm.util import instantiate_from_config


def build_trainer(lightning_cfg, devices):
    callbacks = []
    if lightning_cfg and "callbacks" in lightning_cfg:
        for _, cb_cfg in lightning_cfg.callbacks.items():
            callbacks.append(instantiate_from_config(cb_cfg))

    trainer_kwargs = {
        "callbacks": callbacks,
        "devices": devices,
        "accelerator": "gpu" if devices else "cpu",
    }
    if lightning_cfg and "trainer" in lightning_cfg:
        trainer_kwargs.update(OmegaConf.to_container(lightning_cfg.trainer, resolve=True))
    return pl.Trainer(**trainer_kwargs)


def freeze_unet_except_film(model):
    """
    Freeze diffusion UNet weights, 학습 대상은 FiLM 어댑터만 남긴다.
    """
    diff_unet = model.model.diffusion_model
    for name, p in diff_unet.named_parameters():
        if "film_mlps" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="config yaml path")
    parser.add_argument("--gpus", type=str, default="0", help="CUDA devices, e.g. '0,1'")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path for resuming")
    args = parser.parse_args()

    config = OmegaConf.load(args.base)

    model = instantiate_from_config(config.model)
    # 학습률 설정
    if hasattr(config.model, "base_learning_rate"):
        model.learning_rate = config.model.base_learning_rate

    # UNet은 동결, FiLM 어댑터만 학습
    freeze_unet_except_film(model)

    data = instantiate_from_config(config.data)

    devices = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    trainer = build_trainer(config.get("lightning", None), devices)

    trainer.fit(model, data, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
