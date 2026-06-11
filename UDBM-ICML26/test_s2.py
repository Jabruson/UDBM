import argparse
import importlib
import os
import sys

import numpy as np


TASK_MAPPING = {
    "rain": ["rain1", "rain2", "rain3", "rain4", "rain5"],
    "snow": ["snow1", "snow2"],
    "real_dark": ["real_dark_mef", "real_dark_dice", "real_dark_npe"],
    "real_blur": ["real_hide", "real_j", "real_r"],
    "cd11": ["l", "h", "r", "s", "lh", "lr", "ls", "hr", "hs", "lhr", "lhs"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test UDBM stage-2 restoration model.")
    parser.add_argument("--variant", choices=["S", "M", "L"], default="L")
    parser.add_argument("--gpu", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES value, e.g. 0.")
    parser.add_argument("--dataroot", type=str, default="./datasets/all_in_one")
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--direction", type=str, default="AtoB")
    parser.add_argument("--preprocess", type=str, default="none")
    parser.add_argument("--no_flip", type=bool, default=True)
    parser.add_argument("--bsize", type=int, default=2)
    parser.add_argument("--ckpt_path_s1", type=str, default=None)
    parser.add_argument("--results_folder", type=str, default=None, help="Stage-2 checkpoint folder.")
    parser.add_argument("--result_dir", type=str, default="./result", help="Folder for restored images and metric outputs.")
    parser.add_argument("--milestone", type=int, default=600, help="Stage-2 checkpoint milestone, e.g. model-600.pt.")
    parser.add_argument("--tasks", nargs="+", default=["light_only", "rain", "blur", "fog", "snow"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--sampling_timesteps", type=int, default=1)
    return parser.parse_args()


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
    train_batch_size = 1
    num_samples = 1
    train_num_steps = 100000
    results_folder = opt.results_folder or f"./ckpt_universal/udbm_{opt.variant.lower()}_s2"

    restoration_denoiser = RestorationDenoiser(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        objective=objective,
        test_res_or_noise=test_res_or_noise,
        test_mode=False,
    )
    udbm_bridge = UDBMBridge(
        restoration_denoiser,
        image_size=opt.image_size,
        timesteps=1000,
        delta_end=delta_end,
        sampling_timesteps=opt.sampling_timesteps,
        ddim_sampling_eta=0.0,
        objective=objective,
        loss_type="l1",
        condition=condition,
        sum_scale=sum_scale,
        test_res_or_noise=test_res_or_noise,
    )

    final_summary = {}
    print(f"Start processing tasks: {opt.tasks}")

    for main_task in opt.tasks:
        print(f"\n{'=' * 20} Processing Main Task: {main_task} {'=' * 20}")
        sub_tasks = TASK_MAPPING.get(main_task, [main_task])
        main_task_metrics = []

        for sub_task in sub_tasks:
            from data.universal_dataset import AlignedDataset_all

            print(f"--- Running sub-task: {sub_task} ---")
            dataset = AlignedDataset_all(
                opt,
                opt.image_size,
                augment_flip=False,
                equalizeHist=True,
                crop_patch=False,
                generation=False,
                task=sub_task,
            )
            trainer = UDBMTrainer(
                udbm_bridge,
                dataset,
                opt,
                train_batch_size=train_batch_size,
                num_samples=num_samples,
                train_lr=2e-4,
                train_num_steps=train_num_steps,
                gradient_accumulate_every=2,
                ema_decay=0.995,
                amp=False,
                convert_image_to="RGB",
                results_folder=results_folder,
                condition=condition,
                save_and_sample_every=1000,
                num_unet=num_unet,
            )
            if trainer.accelerator.is_local_main_process:
                trainer.load(opt.milestone)
                trainer.set_results_folder(opt.result_dir)
                current_metric = trainer.test(last=True, task=sub_task)
                main_task_metrics.append(current_metric)
                print(f"Finished {sub_task}: {current_metric}")

        if main_task_metrics:
            avg_dict = {}
            for key in main_task_metrics[0].keys():
                avg_dict[key] = np.mean([d[key] for d in main_task_metrics if key in d])
            final_summary[main_task] = avg_dict
            print(f"\n>>> Average Scores for [{main_task}]:")
            for key, value in avg_dict.items():
                print(f"{key}: {value:.4f}")
        else:
            print(f"No metrics collected for {main_task}")

    print("\n" + "#" * 30)
    print("FINAL SUMMARY REPORT")
    print("#" * 30)
    for task_name, scores in final_summary.items():
        score_str = ", ".join([f"{key}: {value:.4f}" for key, value in scores.items()])
        print(f"{task_name}: {score_str}")


if __name__ == "__main__":
    main()
