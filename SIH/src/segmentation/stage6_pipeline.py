import json
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from .dataset import UAVidDataset
from .model import SegmentationModel
from .trainer import train_segmentation
from .inference import run_inference
from .evaluator import evaluate_segmentation


def run_stage6(train_root, val_root, output_dir=None, config_manager=None):
    if config_manager is None:
        config_manager = ConfigManager()

    if output_dir is None:
        base_out = config_manager.get_path("paths.outputs_dir")
        output_dir = base_out / "stage6_segmentation"
    else:
        output_dir = Path(output_dir).resolve()

    log_dir = config_manager.get_path("paths.logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("stage6_segmentation", log_file=log_dir / "stage6.log")

    logger.info("Starting Stage 6 — UAVid Semantic Segmentation")

    model_name = config_manager.get("segmentation.model", "deeplabv3_mobilenet_v3_large")
    num_classes = config_manager.get("segmentation.num_classes", 8)
    input_size = config_manager.get("segmentation.input_size", [512, 512])
    dev_subset = config_manager.get("segmentation.dev_subset_size", 50)
    inference_only = config_manager.get("segmentation.inference_only", False)

    input_size = tuple(input_size)

    seg_out = output_dir / "segmentation"
    seg_out.mkdir(parents=True, exist_ok=True)

    train_root = Path(train_root).resolve()
    val_root = Path(val_root).resolve()

    logger.info(f"Loading training subset ({dev_subset} images) from {train_root}")
    train_dataset = UAVidDataset(
        split_root=train_root,
        input_size=input_size,
        augment=True,
        max_samples=dev_subset,
    )
    logger.info(f"Train subset size: {len(train_dataset)}")

    logger.info(f"Loading validation split from {val_root}")
    val_dataset = UAVidDataset(
        split_root=val_root,
        input_size=input_size,
        augment=False,
        max_samples=None,
    )
    logger.info(f"Val size: {len(val_dataset)}")

    logger.info(f"Building model: {model_name}")
    model = SegmentationModel(model_name=model_name, num_classes=num_classes)

    train_result = {}
    if not inference_only:
        logger.info("Training segmentation model...")
        train_result = train_segmentation(
            model=model,
            train_dataset=train_dataset,
            output_dir=seg_out,
            config_manager=config_manager,
            logger=logger,
        )
    else:
        logger.info("inference_only=True, skipping training.")
        ckpt = seg_out / "checkpoints" / "model_final.pth"
        if ckpt.exists():
            model.load(ckpt)
            logger.info(f"Loaded checkpoint: {ckpt}")
        else:
            logger.warning("No checkpoint found for inference_only mode. Using untrained model.")

    logger.info("Running inference on validation split...")
    manifest = run_inference(
        model=model,
        dataset=val_dataset,
        output_dir=seg_out,
        batch_size=1,
        logger=logger,
    )

    logger.info("Evaluating predictions against GT labels...")
    eval_result = evaluate_segmentation(
        manifest=manifest,
        gt_root=str(val_root),
        output_dir=seg_out,
        logger=logger,
    )

    report = {
        "status": "SUCCESS",
        "model": model_name,
        "num_classes": num_classes,
        "train_subset_size": len(train_dataset),
        "val_size": len(val_dataset),
        "training": train_result,
        "evaluation": eval_result,
        "output_dir": str(seg_out),
    }

    report_path = seg_out / "segmentation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Stage 6 complete. Report: {report_path}")
    return report
