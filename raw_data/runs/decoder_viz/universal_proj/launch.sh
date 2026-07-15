SWM=/data1/likun-share/junjxu/.stable_worldmodel
OUT=/data1/likun-share/junjxu/runs/decoder_viz/universal_proj
LOG=$OUT/logs
PY=/home/likun-share/junjxu/wm/le-wm/.venv/bin/python
SC=/home/likun-share/junjxu/wm/le-wm/decode_viz/train_universal_decoder.py
udec(){  # tag ckpt gpu
  ( CUDA_VISIBLE_DEVICES=$3 STABLEWM_HOME=$SWM HF_HOME=/data1/likun-share/junjxu/.cache_huggingface \
    $PY $SC --domain uniform_motion --ckpt $SWM/$2 --emb-source proj --epochs 40 \
    --tag $1 --out $OUT ) > $LOG/$1.log 2>&1
  echo "[done $(date +%H:%M:%S)] $1 exit=$?" >> $LOG/master.log
}
udec lam0   uniform_motion_paperinit_id1k/uniform_motion_paperinit_id1k_epoch_20_object.ckpt   3 &
udec lam0p3 uniform_posonly_w0p3_id1k/uniform_posonly_w0p3_id1k_epoch_20_object.ckpt           1 &
udec lam1   uniform_motion_piwm_probe_id1k/uniform_motion_piwm_probe_id1k_epoch_20_object.ckpt 0 &
udec lam3   uniform_posonly_w3p0_id1k/uniform_posonly_w3p0_id1k_epoch_20_object.ckpt           4 &
udec lam10  uniform_posonly_w10p0_id1k/uniform_posonly_w10p0_id1k_epoch_20_object.ckpt         7 &
wait
echo "=== PROJ UDEC SWEEP DONE $(date) ===" >> $LOG/master.log
