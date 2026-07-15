SWM=/data1/likun-share/junjxu/.stable_worldmodel
OUT=/data1/likun-share/junjxu/runs/ood_rollout_lambda
LOG=$OUT/logs
PY=/home/likun-share/junjxu/wm/le-wm/.venv/bin/python
SCRIPT=/home/likun-share/junjxu/wm/phyworld/scripts/rollout_eval_id1k.py
roll(){  # $1=tag $2=ckpt_subpath $3=gpu
  ( CUDA_VISIBLE_DEVICES=$3 STABLEWM_HOME=$SWM HF_HOME=/data1/likun-share/junjxu/.cache_huggingface \
    $PY $SCRIPT --domain uniform_motion --max-trajs 1152 --tag $1 \
    --ckpt $SWM/$2 ) > $LOG/rollout_$1.log 2>&1
  echo "[done $(date +%H:%M:%S)] $1 exit=$?" >> $LOG/master.log
}
roll lam0    uniform_motion_paperinit_id1k/uniform_motion_paperinit_id1k_epoch_20_object.ckpt   3 &
roll lam0p3  uniform_posonly_w0p3_id1k/uniform_posonly_w0p3_id1k_epoch_20_object.ckpt           1 &
roll lam1    uniform_motion_piwm_probe_id1k/uniform_motion_piwm_probe_id1k_epoch_20_object.ckpt 0 &
roll lam3    uniform_posonly_w3p0_id1k/uniform_posonly_w3p0_id1k_epoch_20_object.ckpt           4 &
roll lam10   uniform_posonly_w10p0_id1k/uniform_posonly_w10p0_id1k_epoch_20_object.ckpt         7 &
wait
echo "=== OOD ROLLOUT SWEEP DONE $(date) ===" >> $LOG/master.log
