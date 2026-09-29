import pytest
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.utils.system_info import get_system_info, print_system_report
from src.cli.main_cli import run_cli

def test_config_manager(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        "project:\n  name: TestProject\n  stage: 1\npaths:\n  data_dir: test_data\nlogging:\n  level: DEBUG\n",
        encoding="utf-8"
    )
    cfg = ConfigManager(config_path=config_file, root_dir=tmp_path)
    assert cfg.get("project.name") == "TestProject"
    assert cfg.get("project.stage") == 1
    assert cfg.get("logging.level") == "DEBUG"
    
    data_path = cfg.get_path("paths.data_dir")
    assert data_path == tmp_path / "test_data"

def test_config_directory_setup(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        "paths:\n  data_dir: data\n  input_dir: data/input\n  logs_dir: logs\n",
        encoding="utf-8"
    )
    cfg = ConfigManager(config_path=config_file, root_dir=tmp_path)
    created = cfg.setup_directories()
    assert len(created) > 0
    assert (tmp_path / "data").exists()
    assert (tmp_path / "data" / "input").exists()
    assert (tmp_path / "logs").exists()

def test_system_info():
    info = get_system_info()
    assert "os" in info
    assert "python_version" in info
    assert "cpu_count_physical" in info
    assert "total_ram_gb" in info
    assert "torch" in info
    assert "nvidia_smi" in info
    assert "colmap" in info
    assert "ffmpeg" in info
    
    report = print_system_report(info)
    assert "SYSTEM & ENVIRONMENT DIAGNOSTIC REPORT" in report
    assert info["python_version"] in report

def test_logger(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logger("test_logger", log_file=log_file, console=False)
    logger.info("Test log entry")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log entry" in content

def test_cli_version(capsys):
    ret = run_cli(["version"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Drone-to-Hybrid-3D-Digital-Twin version" in captured.out

def test_cli_inspect_env(capsys):
    ret = run_cli(["inspect-env"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "SYSTEM & ENVIRONMENT DIAGNOSTIC REPORT" in captured.out

def test_cli_init_workspace():
    ret = run_cli(["init-workspace"])
    assert ret == 0

def test_cli_check_config():
    ret = run_cli(["check-config"])
    assert ret == 0
