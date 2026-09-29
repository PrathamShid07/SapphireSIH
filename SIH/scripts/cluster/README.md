# SPIT GPU Cluster Guide for UAV Digital Twin

Cluster hardware: Dual AMD EPYC (112 cores), 2x NVIDIA RTX A6000, 250 GB RAM.

---

## 1. Local Testing Rule

> **Rule:** Always test each stage locally first. Push only code to GitHub — never dataset or outputs.

---

## 2. One-Time Setup on Cluster

SSH into the cluster, then:

```bash
git clone <your-repo-url>
cd SIH
chmod +x scripts/cluster/*.sh

# Set up Python environment + CUDA PyTorch
bash scripts/cluster/setup_spit_env.sh

# Copy data from your local machine (run this on your PC)
scp -r data/input/ sarangpatil23@cluster.gpu.spit.ac.in:~/SIH/data/input/
scp -r data/raw/   sarangpatil23@cluster.gpu.spit.ac.in:~/SIH/data/raw/
```

Data only needs to be copied **once** — it stays on the cluster permanently.

---

## 3. Two-Part Pipeline Split

The pipeline is split into two jobs for efficiency:

```
Job 1: Stages 2-5 (Photogrammetry)    → GPU accelerated, no dataset needed
Job 2: Stages 6-12 (AI + Fusion)      → Needs UAVid dataset for Stage 6
```

### Job 1 — Stages 2-5: Frame extraction → Dense reconstruction

```bash
sbatch scripts/cluster/submit_stages1to5.sh data/input/sample_drone.mp4 outputs/my_run
```

This runs:
| Stage | Description |
|-------|-------------|
| 2 | Frame extraction & deduplication |
| 3 | GPS/IMU metadata parsing |
| 4 | COLMAP sparse SfM reconstruction |
| 5 | MVS dense point cloud |

### Job 2 — Stages 6-12: AI detection → Final textured mesh

> Run only after Job 1 completes successfully.

```bash
sbatch scripts/cluster/submit_stages6to12.sh data/input/sample_drone.mp4 outputs/my_run
```

This runs:
| Stage | Description |
|-------|-------------|
| 6 | Semantic segmentation training (UAVid) |
| 7 | Vehicle detection & tracking |
| 8 | Dynamic masking & clean SfM |
| 9 | 3D object localization |
| 10 | Asset library management |
| 11 | Scene fusion |
| 12 | Texturing, UV unwrap, LOD decimation |

---

## 4. Run a Single Stage (for debugging)

```bash
# Verify GPU is detected
sbatch scripts/cluster/submit_gpu.sh inspect-env

# Run any single command
sbatch scripts/cluster/submit_gpu.sh preprocess-video --video data/input/sample_drone.mp4
sbatch scripts/cluster/submit_gpu.sh track-vehicles --video data/input/sample_drone.mp4
```

---

## 5. Monitoring Jobs

```bash
# Check your job queue
squeue -u $USER

# Watch live GPU usage
watch -n 1 nvidia-smi

# Follow job logs
tail -f logs/slurm_stages1to5_*.out
tail -f logs/slurm_stages6to12_*.out
tail -f logs/slurm_*.err
```

---

## 6. After Jobs Complete

Pull outputs back to your local machine:

```bash
# On your PC
scp -r sarangpatil23@cluster.gpu.spit.ac.in:~/SIH/outputs/ ./outputs/
```
