from alps.framework.engine import GpuConf
from alps.framework.engine import ResourceConf
from alps.pytorch.api.runner.engine import TorchK8sEngine
from alps.pytorch.api.base.submit import submit_torch_task


def main():
    # 配置每个节点gpu数量
    gpu_count = 8
    # 配置节点数量
    # num_nodes = 1 # 2b
    num_nodes = 2  # 10b
    gpu = GpuConf(count=gpu_count)
    worker = ResourceConf(
        num=num_nodes,
        core=max(gpu_count * 4, 4),
        memory=1024 * 900,
        gpu=gpu,
        disk_quota=1024 * 200,
    )
    # worker = ResourceConf(num=1, core=4, memory=4096, disk_quota=10240)
    # 使用TorchK8sEngine
    k8s_engine = TorchK8sEngine(
        worker=worker,
        app="gbank",
        priority="high",  # 使用自己的app时可以使用high priority。使用公共资源池时不能填写priority。
        # image="reg.docker.alibaba-inc.com/aii/aistudio:3770122-20230402194748",
        # image="reg.docker.alibaba-inc.com/alipay-alps/alps_torch:torch1.12_rdma_trlx_flashatten",
        # image="reg.docker.alibaba-inc.com/aii/aistudio:4390135-20230519151242",
        image="reg.docker.alibaba-inc.com/aii/aistudio:4090122-20230519105032",
        deploy_tag="kubemaker",
        rdma=False,
        host_network=False
    )
    # 参数配置
    args = ["--norm_reward"]

    kwargs = {
        '--rm_model_path':
            '/mnt/tangjian.dtj/model/rw_model/glm-10b-2k-sft-v9-rm-v7-use-last/checkpoint-500',
        '--ppo_model_path': '/mnt/tangjian.dtj/pretrained_models/glm-10b-2k-sft-v9-checkpoint-133230',
        '--prompt_path': '/mnt/chatgpt/data/RL/rl_v10.csv',
        '--exp_cfg_path': 'antllm/examples/rlhf/rl/exps/exp.yml',
        '--save_dir': '/mnt/tangjian.dtj/model/rl_model/glm-2k-10b-10b-freeze-1-norm-use-last-2-nodes',
        '--log_dir': '/mnt/tangjian.dtj/model/rl_model/glm-2k-10b-10b-freeze-1-norm-use-last-2-nodes/logs',
        '--mask_type': '[gMASK]',
        '--num_head': 1
    }

    # 设置训练脚本路径
    entry_file = "./trlx_glm.py"

    # 设置用户工号
    user_number = "xxxx"

    # 提交任务
    submit_torch_task(
        entry_file=entry_file,
        engine=k8s_engine,
        args=args,
        kwargs=kwargs,
        user=user_number,
        exit_on_submit=False,
        accelerate_trlx=True,
        init_rc="configs/init.rc",
        submit_root="../../../../",
        config_file="configs/accelerate_config_multi_nodes.yaml",
        ignores=["adaspeech"],
        reward_model_single=True
    )


if __name__ == '__main__':
    main()
