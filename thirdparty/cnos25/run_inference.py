# Derived from https://github.com/Vision-Kek/cnos25 (commit d92e3d3, MIT License; see thirdparty/cnos25/LICENSE).
# Modified by POSE-Lab, National Technical University of Athens (NovaSplat), 2026.
import logging
import os
import os.path as osp
from pathlib import Path
import subprocess
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def _resolve_template_dir(cfg: DictConfig) -> Path:
    return Path(str(cfg.data.reference_dataloader.template_dir))


def _resolve_descriptor_path(cfg: DictConfig) -> Path:
    cache_dir = Path(str(cfg.model.descriptor.cfg.cache_dir))
    cache_file = str(cfg.model.descriptor.cfg.cache_file)
    return cache_dir / cache_file


def _ensure_reference_descriptors(cfg: DictConfig) -> None:
    template_dir = _resolve_template_dir(cfg)
    descriptor_path = _resolve_descriptor_path(cfg)

    if descriptor_path.exists():
        logging.info("Using reference descriptors at %s", descriptor_path)
        return

    if not template_dir.exists():
        raise FileNotFoundError(
            "Missing rendered template directory for CNOS25 inference: "
            f"{template_dir}. Run onboarding first or override "
            "data.reference_dataloader.template_dir and "
            "model.descriptor.cfg.cache_dir to point at the generated templates."
        )

    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Reference descriptor cache not found at %s. Extracting descriptors from %s.",
        descriptor_path,
        template_dir,
    )
    command = [
        sys.executable,
        "-m",
        "src.scripts.extract_template_descriptors",
        f"dataset_name={cfg.dataset_name}",
        f"onboarding={cfg.onboarding}",
        f"data.root_dir={cfg.data.root_dir}",
        f"local.bop_data_root={cfg.local.bop_data_root}",
        f"local.dinov3_repo={cfg.local.dinov3_repo}",
        f"local.dinov3_checkpoint={cfg.local.dinov3_checkpoint}",
        f"data.reference_dataloader.template_dir={template_dir}",
        f"descriptor_model.cfg.cache_dir={descriptor_path.parent}",
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to extract CNOS25 reference descriptors automatically.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not descriptor_path.exists():
        raise FileNotFoundError(
            "Descriptor extraction completed without producing the expected file: "
            f"{descriptor_path}"
        )
    logging.info("Descriptor extraction completed: %s", descriptor_path)


def inference_loop(cfg, model, query_dataloader):
    bopeval = instantiate(cfg.bopeval)

    for sample in tqdm(query_dataloader):
        sample = sample[0]  # bs=1
        if cfg.dataset_name == "hot3d":
            sample["image"] = sample["image"].convert("RGB")
        start = time.time()
        # Proposal stage
        proposals = model.proposal_fwd_pass(sample["image"])

        # Classification stage
        proposal_descriptors = model.dino_fwd_pass(sample["image"], proposals)
        score_per_proposal, assigned_idx_object = model.classify(proposal_descriptors)

        torch.cuda.synchronize()  # for accurate runtime measure
        runtime = time.time() - start
        bopeval.save_detections_to_npzfile(
            sample=sample,
            boxes=proposals.boxes[: cfg.n_max_save].cpu(),
            class_idcs=assigned_idx_object[: cfg.n_max_save].cpu(),
            scores=score_per_proposal[: cfg.n_max_save].cpu(),
            runtime=runtime,
            box_fmt="xyxy",
            masks=proposals.masks[: cfg.n_max_save].cpu() if cfg.save_masks else None,
        )

    bopeval.generate_json_from_npzs()
    if cfg.split != "test":
        bopeval.measure_AP()

    if cfg.postprocess is not None:
        logging.info("Postprocessing...")
        hydra.utils.instantiate(
            cfg.postprocess,
            result_json=osp.join(
                bopeval.result_dir, f"{bopeval.bopstyle_experiment_name}.json"
            ),
            save_masks=cfg.save_masks,
        )
        if cfg.split != "test":
            bopeval.measure_AP(
                resultfile_name=f"nms-{bopeval.bopstyle_experiment_name}"
            )
        logging.info("Postprocessing Done.")


@hydra.main(version_base=None, config_path="configs", config_name="run_inference")
def run_inference(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)  # allows adding new keys
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg.runtime.output_dir
    logging.info("The outputs of hydra will be stored in: %s", output_path)

    logging.info("Initializing query dataloader...")
    logging.info("cfg.data.root_dir=%s", cfg.data.root_dir)
    if cfg.dataset_name == "hot3d":
        from src.dataloader import bop_hot3d

        query_dataset = bop_hot3d.BOPHOT3DTest(
            root_dir=osp.join(cfg.data.root_dir, "hot3d"), split=cfg.split
        )
    else:
        query_dataloader_config = cfg.data.query_dataloader.copy()
        query_dataloader_config.dataset_name = cfg.dataset_name
        root_dir = Path(str(query_dataloader_config.root_dir))
        if root_dir.name != cfg.dataset_name:
            root_dir = root_dir / cfg.dataset_name
        query_dataloader_config.root_dir = str(root_dir)
        query_dataset = instantiate(query_dataloader_config)
    logging.info("Initializing query dataloader Done.")

    logging.info("Initializing model...")
    _ensure_reference_descriptors(cfg)
    model = instantiate(cfg.model)
    logging.info("Initializing model Done.")

    no_collate = lambda batch: batch  # don't convert to tensor/device
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=1,
        num_workers=cfg.local.num_workers,
        collate_fn=no_collate,
        shuffle=False,
    )
    if "static" in cfg.onboarding:
        logging.info("Using static onboarding images")
    else:
        raise NotImplementedError("Not implemented yet.")

    logging.info("---" * 20)
    inference_loop(cfg, model, query_dataloader)
    logging.info("---" * 20)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()
