import argparse
import importlib
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Train UDBM stage-2 restoration model.")
    parser.add_argument("--variant", choices=["S", "M", "L"], default="L")
    parser.add_argument("--gpu", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1.")
    parser.add_argument("--dataroot", type=str, default="./datasets/all_in_one")
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument("--load_size", type=int, default=268)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--direction", type=str, default="AtoB")
    parser.add_argument("--preprocess", type=str, default="crop")
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--bsize", type=int, default=2)
    parser.add_argument("--ckpt_path_s1", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sampling_timesteps", type=int, default=4)
    parser.add_argument("--train_num_steps", type=int, default=600000)
    parser.add_argument("--train_batch_size", type=int, default=10, help="Kept for compatibility; task DataLoader batch sizes control the real training batch.")
    parser.add_argument("--gradient_accumulate_every", type=int, default=2)
    parser.add_argument("--task_batch_sizes", type=str, default=None, help="Comma-separated fog,light,rain,snow,blur batch sizes. Default is 16,4,8,8,4 for grad=1 and 8,2,4,4,2 for grad=2.")
    parser.add_argument("--save_and_sample_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--resume_milestone", type=int, default=None)
    parser.add_argument("--results_folder", type=str, default=None)
    return parser.parse_args()


def build_universal_train_dataset(opt, image_size):
    from data.universal_dataset import AlignedDataset_all

    return [
        AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task="fog"),
        AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task="light_only"),
        AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task="rain"),
        AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task="snow"),
        AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task="blur"),
    ]


def main():
    opt = parse_args()
    if opt.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if opt.ckpt_path_s1 is None:
        opt.ckpt_path_s1 = f"./ckpt_universal/udbm_{opt.variant.lower()}_s1/model-600.pt"

    module = importlib.import_module(f"src.model_udbm_s2_{opt.variant.lower()}")
    UDBMBridge, UDBMTrainer, RestorationDenoiser, set_seed = (
        module.UncertaintyAwareDiffusionBridge,
        module.UDBMTrainer,
        module.RestorationDenoiser,
        module.set_seed,
    )

    sys.stdout.flush()
    set_seed(10)

    num_unet = 1
    objective = "pred_res"
    test_res_or_noise = "res"
    condition = True
    sum_scale = 0.01
    delta_end = 1.8e-3
    num_samples = 1
    results_folder = opt.results_folder or f"./ckpt_universal/udbm_{opt.variant.lower()}_s2"

    dataset = build_universal_train_dataset(opt, opt.image_size)

    restoration_denoiser = RestorationDenoiser(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        objective=objective,
        test_res_or_noise=test_res_or_noise,
    )
    udbm_bridge = UDBMBridge(
        restoration_denoiser,
        image_size=opt.image_size,
        timesteps=1000,
        delta_end=delta_end,
        sampling_timesteps=opt.sampling_timesteps,
        objective=objective,
        loss_type="l1",
        condition=condition,
        sum_scale=sum_scale,
        test_res_or_noise=test_res_or_noise,
    )
    trainer = UDBMTrainer(
        udbm_bridge,
        dataset,
        opt,
        train_batch_size=opt.train_batch_size,
        num_samples=num_samples,
        train_lr=opt.lr,
        train_num_steps=opt.train_num_steps,
        gradient_accumulate_every=opt.gradient_accumulate_every,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        results_folder=results_folder,
        condition=condition,
        save_and_sample_every=opt.save_and_sample_every,
        num_unet=num_unet,
    )
    if opt.resume_milestone is not None:
        trainer.load(opt.resume_milestone)
    trainer.train()


if __name__ == "__main__":
    main()
