export PYTHONPATH=$PYTHONPATH:../../../../../../

# python run_pred.py \
#   --rm_model_path /ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-10b-sft-v9-mean-reward-flash-fp16-oasst_mix_v1/checkpoint-1500/ \
#   --test_path /ossfs/workspace/mnt/tangjian.dtj/data/RM/oasst_mix_v1/test.jsonl \
#   --out_path /ossfs/workspace/mnt/tangjian.dtj/calibration/glm-10b-sft-mean-reward-oasst_mix_v1/calibration_score.jsonl

rm_model=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-sft-v9-loss-2-last-reward-flash-fp16-oasst_mix_v1/checkpoint-3500/
output_dir=/ossfs/workspace/mnt/tangjian.dtj/calibration/glm-5b-sft-loss-2-last-reward-oasst_mix_v1

rm_model=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-10b-sft-v9-loss-2-last-reward-flash-fp16-oasst_mix_v1/checkpoint-3500/
output_dir=/ossfs/workspace/mnt/tangjian.dtj/calibration/glm-10b-sft-loss-2-last-reward-oasst_mix_v1

rm_model=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-sft-v9-loss-1-last-reward-flash-fp16-oasst_mix_v1/checkpoint-3500/
output_dir=/ossfs/workspace/mnt/tangjian.dtj/calibration/glm-5b-sft-loss-1-last-reward-oasst_mix_v1


rm_model=/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-10b-sft-v9-loss-1-last-reward-flash-fp16-oasst_mix_v1/checkpoint-1500/
output_dir=/ossfs/workspace/mnt/tangjian.dtj/calibration/glm-10b-sft-loss-1-last-reward-oasst_mix_v1

rm_model="/ossfs/workspace/mnt/tangjian.dtj/model/rw_model/glm-5b-exp-v1/checkpoint-1000/"
output_dir="/ossfs/workspace/mnt/tangjian.dtj/calibration/glm-5b-exp-v1"

rm_model=/mnt/xiaohao.wzh/rm/model/rw_model/glm-10b-2k-v9-bf16-20230506-140819-checkpoint-133230-rm-v8_k6_100k-use_mean-no_position-bf16
output_dir=/mnt/xiaohao.wzh/rm/calibration/v8_k6_100k/glm-10b-2k-v9-bf16-20230506-140819-checkpoint-133230-rm-v8_k6_100k-use_mean-no_position-bf16

python run_pred.py \
  --rm_model_path ${rm_model} \
  --test_path /mnt/chatgpt/data/RM/v8_k6_100k/test.jsonl \
  --out_path ${output_dir} \
  --use_mean_value

python plot_calibration.py \
 --score_path ${output_dir}
 