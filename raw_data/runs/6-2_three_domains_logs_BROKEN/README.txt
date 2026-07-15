=== 监控 / 控制 sweep 的常用命令 ===

PID / 进程状态:
  pgrep -af "rerun_three_domains|train\.py"
  ps -o pid,ppid,etime,cmd -p $(pgrep -f rerun_three_domains | head -1)

GPU 使用:
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

总进度 (12 jobs):
  cat /data1/likun-share/junjxu/runs/6-2_three_domains_logs/orchestrator.log

单 job 训练日志:
  ls /data1/likun-share/junjxu/runs/6-2_three_domains_logs/train_*.log
  tail -f /data1/likun-share/junjxu/runs/6-2_three_domains_logs/train_parabola_paperinit_id1k.log

单 job eval 日志（训练完后才有）:
  ls /data1/likun-share/junjxu/runs/6-2_three_domains_logs/rollout_*.log

杀掉整个 sweep（如需要）:
  pkill -9 -s $(ps -o sid= -p $(pgrep -f rerun_three_domains | head -1))

预计时间: 20 epoch × ~50s = ~17 min/job. 12 job / 8 GPU ≈ 2 轮 ≈ 35-45 分钟。
ckpt 目录: /data1/likun-share/junjxu/.stable_worldmodel/<name>_id1k/
