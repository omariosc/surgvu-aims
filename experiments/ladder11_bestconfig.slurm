#!/bin/bash
#SBATCH --job-name=surgvu_l11best
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder11_%j.out
#SBATCH --error=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder11_%j.err
# LEVER 3 (combine the two winners -> FINAL Cat-1 detector pick). Chained afterok on
# LEVER 1 (ladder9) so the thr0.6/0.7 cam dirs + their mAP result lines exist. This
# cell DISCOVERS the best CAM threshold among {0.5,0.6,0.7} from the logged 40-ep
# yolo26m base-config mAPs, then trains yolo26m at THAT threshold with the LEVER 2
# longer+aug recipe (120 ep + heavy aug). One final best-config cell.
# KILL CRITERION: if best-config mAP <= the thr0.5_yolo26m baseline 0.7746 -> the
#   combined winners do not stack; the wall is the Grad-CAM localisation supervision.
set -e
mkdir -p /scratch/sc20osc/miccai-2026/SurgVU/logs
export HF_HOME=/scratch/sc20osc/hf_cache
export TORCH_HOME=/scratch/sc20osc/hf_cache/torch
export OMP_NUM_THREADS=4
export YOLO_CONFIG_DIR=/scratch/sc20osc/hf_cache/ultralytics
PY=/scratch/sc20osc/surgvu_env/bin/python
EXP=/users/sc20osc/SurgVU/experiments
D=/scratch/sc20osc/miccai-2026/SurgVU/data
CKPT=/scratch/sc20osc/miccai-2026/SurgVU/models/spine_full/spine_best.pt
RESULTS=/users/sc20osc/AUTONOMOUS_RESULTS.md
YOLO=/scratch/sc20osc/AILET/from_pc/weights/yolo26m.pt
YTAG=$(basename $YOLO .pt)

echo "=== node $(hostname)  ladder11 best-config ==="; nvidia-smi -L
cd $EXP

# --- discover best CAM threshold among 0.5/0.6/0.7 from the logged base-config mAPs ---
# thr0.5_yolo26m baseline is 0.7746 (logged); thr0.6/0.7 land from ladder9.
BEST_THR=0.5
BEST_MAP=0.7746
for T in 0.6 0.7; do
  M=$(grep -oE "detector-variant thr${T}_${YTAG} .*mAP@\[\.5:\.95\]=[0-9.]+" $RESULTS \
      | grep -oE 'mAP@\[\.5:\.95\]=[0-9.]+' | tail -1 | cut -d= -f2)
  if [ -n "$M" ]; then
    GT=$($PY -c "print(1 if $M > $BEST_MAP else 0)" 2>/dev/null)
    if [ "$GT" = "1" ]; then BEST_THR=$T; BEST_MAP=$M; fi
  fi
done
echo "=== discovered best CAM threshold = $BEST_THR (base-config mAP=$BEST_MAP) ==="

THR=$BEST_THR
CELL="thr${THR}_${YTAG}_BEST_long120aug"
CAM=/scratch/sc20osc/miccai-2026/SurgVU/models/cam_thr${THR}
OUT=/scratch/sc20osc/miccai-2026/SurgVU/models/detector_$CELL
PARSE=/scratch/sc20osc/miccai-2026/SurgVU/logs/ladder11_${SLURM_JOB_ID}.parse
mkdir -p $OUT

# ensure the cam dir for the chosen threshold exists (ladder9 builds 0.6/0.7; 0.5 preexists)
if [ ! -f $CAM/pseudo_boxes.json ]; then
  echo "=== cam_thr${THR} missing -> regenerating (600 frames) ==="
  $PY $EXP/cam_pseudoboxes.py --ckpt $CKPT --manifest $D/frame_manifest_full.csv \
      --out_dir $CAM --n 600 --thr $THR 2>&1 | tee $PARSE
else
  echo "=== reusing existing $CAM/pseudo_boxes.json ==="; touch $PARSE
fi

echo "=== train YOLO ($YTAG) @ thr=$THR, 120 ep + heavy aug (FINAL best-config) ==="
set +e
$PY $EXP/train_detector.py --pseudo $CAM/pseudo_boxes.json --img_dir $CAM \
    --out_root $OUT --weights $YOLO --epochs 120 --patience 40 \
    --mosaic 1.0 --mixup 0.15 --hsv_h 0.015 --hsv_s 0.7 --hsv_v 0.4 \
    --degrees 10.0 --translate 0.1 --scale 0.5 --fliplr 0.5 2>&1 | tee -a $PARSE
set -e

MAP=$(grep -oE 'DETECTOR_mAP50_95=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
MAP50=$(grep -oE 'DETECTOR_mAP50=[0-9.nan]+' $PARSE | tail -1 | cut -d= -f2)
NBOX=$(grep -oE '[0-9]+ pseudo-boxes' $PARSE | tail -1 | grep -oE '^[0-9]+')
[ -z "$MAP" ] && MAP="nan"; [ -z "$MAP50" ] && MAP50="nan"
VERD=""
if [ "$MAP" != "nan" ]; then
  CMP=$($PY -c "print('CONFIRMED' if $MAP > 0.7746 else 'REFUTED')" 2>/dev/null)
  VERD=" [$CMP vs thr0.5_yolo26m baseline 0.7746]"
fi
echo "[SurgVU][${SLURM_JOB_ID}] Cat-1 detector-variant ${CELL} (${NBOX:-?} pseudo-boxes, case-disjoint self-consistency, NO human GT) -> mAP@[.5:.95]=${MAP} mAP50=${MAP50}${VERD}" >> $RESULTS
echo "=== appended to $RESULTS ==="; tail -1 $RESULTS
echo "=== $CELL DONE $(date) ==="
