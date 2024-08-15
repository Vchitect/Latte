import os
import torch
import argparse
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from models import get_models
from datasets import get_dataset
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from utils import update_ema, requires_grad, text_preprocessing, get_experiment_dir, clip_grad_norm_
from copy import deepcopy
from einops import rearrange


class LatteTrainingModule(LightningModule):
    def __init__(self, args):
        super(LatteTrainingModule, self).__init__()
        self.args = args
        self.model = get_models(args)
        self.ema = deepcopy(self.model)
        requires_grad(self.ema, False)

        self.diffusion = create_diffusion(timestep_respacing="")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0)
        self.lr_scheduler = None

        # Load pretrained model if specified
        if args.pretrained:
            # Old load script, only load EMA
            checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
            if "ema" in checkpoint:  # supports checkpoints from train.py
                print('Using ema ckpt!')
                checkpoint = checkpoint["ema"]

            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f'Successfully loaded {len(pretrained_dict) / len(checkpoint.items()) * 100:.2f}% of pretrained model weights.')
            self.train_steps = int(args.pretrained.split("/")[-1].split('.')[0])

        # Freeze VAE
        self.vae.requires_grad_(False)

    def training_step(self, batch, batch_idx):
        x = batch['video'].to(self.device)
        video_name = batch['video_name']

        with torch.no_grad():
            b, _, _, _, _ = x.shape
            x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
            x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()

        model_kwargs = dict(y=video_name if self.args.extras == 2 else None)

        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
        loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        model_kwargs_mask = model_kwargs.copy()
        model_kwargs_mask['enable_mask'] = True
        loss_dict_mask = self.diffusion.training_losses(self.model, x, t, model_kwargs_mask)
        loss += loss_dict_mask["loss"].mean()

        if self.global_step < self.args.start_clip_iter:
            clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm, clip_grad=False)
        else:
            clip_grad_norm_(self.model.parameters(), self.args.clip_max_norm, clip_grad=True)

        update_ema(self.ema, self.model)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        self.lr_scheduler = get_scheduler(
            name="constant",
            optimizer=self.opt,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )
        return [self.opt], [self.lr_scheduler]


class EMACheckpointCallback(Callback):
    def __init__(self, model, ema, opt, args, checkpoint_dir, save_every_n_steps):
        super().__init__()
        self.model = model
        self.ema = ema
        self.opt = opt
        self.args = args
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Save checkpoint every `save_every_n_steps` steps
        if trainer.global_step % self.save_every_n_steps == 0 and trainer.global_step > 0:
            rank = trainer.global_rank
            if rank == 0:
                # New save script, save all models
                checkpoint = {
                        "model": self.model.module.state_dict(),
                        "ema": self.ema.state_dict(),
                        "opt": self.opt.state_dict(),
                        "args": args
                    }
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{trainer.global_step:07d}.pt")
                torch.save(checkpoint, checkpoint_path)
                pl_module.log(f"Saved checkpoint to {checkpoint_path}")


def main(args):
    seed = args.global_seed
    torch.manual_seed(seed)

    # Setup an experiment folder and logger
    experiment_dir = get_experiment_dir(args.results_dir, args)
    tb_logger = TensorBoardLogger(experiment_dir, name="latte")

    # Setup model checkpointing
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ema_checkpoint_callback = EMACheckpointCallback(
        # EMA will be set later in the training module
        model=None,
        ema=None,
        opt=None,
        args=None,
        checkpoint_dir=checkpoint_dir,
        save_every_n_steps=args.ckpt_every
    )

    # Create the dataset and dataloader
    dataset = get_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    sample_size = args.image_size // 8
    args.latent_size = sample_size

    # Initialize the training module
    model = LatteTrainingModule(args)
    ema_checkpoint_callback.model = model.model
    ema_checkpoint_callback.ema = model.ema  # Set the EMA model in the callback
    ema_checkpoint_callback.opt = model.opt
    ema_checkpoint_callback.args = args

    # Trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=[0],    # Specify GPU ids
        strategy=DDPStrategy(find_unused_parameters=True),
        max_steps=args.max_train_steps,
        logger=tb_logger,
        callbacks=[ema_checkpoint_callback, LearningRateMonitor()],
        log_every_n_steps=args.log_every,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_dataloaders=loader, ckpt_path=args.resume_from_checkpoint if args.resume_from_checkpoint else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ffs/ffs_train.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))