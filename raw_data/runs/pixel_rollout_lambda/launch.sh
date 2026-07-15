SWM=/data1/likun-share/junjxu/.stable_worldmodel
DEC=/data1/likun-share/junjxu/runs/decoder_viz/universal_proj
OUT=/data1/likun-share/junjxu/runs/pixel_rollout_lambda
LOG=$OUT/logs
PY=/home/likun-share/junjxu/wm/le-wm/.venv/bin/python
SC=/home/likun-share/junjxu/wm/phyworld/scripts/rollout_pixel_eval.py
px(){  # tag ckpt gpu
  ( CUDA_VISIBLE_DEVICES=$3 STABLEWM_HOME=$SWM HF_HOME=/data1/likun-share/junjxu/.cache_huggingface \
    $PY $SC --domain uniform_motion --ckpt $SWM/$2 --decoder $DEC/udecoder_$1.pt \
    --tag $1 --max-trajs 800 --out $OUT ) > $LOG/$1.log 2>&1
  echo "[done $(date +%H:%M:%S)] $1 exit=$?" >> $LOG/master.log
}
px lam0   uniform_motion_paperinit_id1k/uniform_motion_paperinit_id1k_epoch_20_object.ckpt   3 &
px lam0p3 uniform_posonly_w0p3_id1k/uniform_posonly_w0p3_id1k_epoch_20_object.ckpt           1 &
px lam1   uniform_motion_piwm_probe_id1k/uniform_motion_piwm_probe_id1k_epoch_20_object.ckpt 0 &
px lam3   uniform_posonly_w3p0_id1k/uniform_posonly_w3p0_id1k_epoch_20_object.ckpt           4 &
px lam10  uniform_posonly_w10p0_id1k/uniform_posonly_w10p0_id1k_epoch_20_object.ckpt         7 &
wait
echo "=== PIXEL ROLLOUT SWEEP DONE $(date) ===" >> $LOG/master.log
