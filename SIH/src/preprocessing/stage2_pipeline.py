from pathlib import Path
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from .frame_extractor import VideoPreprocessor

def run_stage2(video_path, output_dir=None, config_manager=None):
    if config_manager is None:
        config_manager = ConfigManager()

    log_dir = config_manager.get_path("paths.logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("stage2_preprocessing", log_file=log_dir / "stage2.log")

    vpath = Path(video_path).resolve()
    logger.info(f"Starting Stage 2 Video Preprocessing for: {vpath}")

    preprocessor = VideoPreprocessor(config_manager=config_manager)
    manifest_data = preprocessor.process_video(vpath, output_dir=output_dir)

    sampling = manifest_data["sampling"]
    logger.info(f"Video preprocessing completed successfully.")
    logger.info(f"Total video frames: {sampling['total_video_frames']}")
    logger.info(f"Extracted frames: {sampling['extracted_frames_count']}")
    logger.info(f"Selected frames: {sampling['selected_frames_count']}")
    logger.info(f"Rejected frames: {sampling['rejected_frames_count']}")
    logger.info(f"Selection rate: {sampling['selection_rate'] * 100:.2f}%")
    logger.info(f"Manifest written to: {manifest_data['output_paths']['manifest_path']}")

    return manifest_data
