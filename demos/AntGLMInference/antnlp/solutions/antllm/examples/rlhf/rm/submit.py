from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf
from pypai.conf import KMConf
from pypai.conf import GpuType


'''
 cpu 单位是核
 memory 单位是MB
 gpu_num gpu卡数
 num 节点数量
'''

worker_num = 8  # 可以从0到n
master = ExecConf(cpu=32, memory=819200, gpu_num=8, num=1, gpu_type=GpuType.A100)
worker = ExecConf(cpu=32, memory=819200, gpu_num=8, num=worker_num, gpu_type=GpuType.A100)
km_conf = KMConf(
    image="reg.docker.alibaba-inc.com/aii/aistudio:4090122-20230522120009",
)

#  挂载nas, 修改nas地址
mnt_cmd = "mkdir -p /mnt && mount -t nfs -o vers=3,nolock,proto=tcp nas:/ /mnt && ls /mnt"  # noqa

#  cd到代码执行路径
cd_cmd = "cd /mnt/tangjian.dtj/code/antnlp/solutions/antllm/examples/rlhf/rm"

#  执行脚本
run_cmd = "bash run_train_rm_multi_node.sh"

cmd = f"pip list && {mnt_cmd} && {cd_cmd} && {run_cmd}"


def gpujob():
    job = PythonJobBuilder(
        source_root=None,
        main_file="",
        command=cmd,
        km_conf=km_conf,
        k8s_app_name="gbank",
        k8s_priority="high",
        master=master,
        worker=worker,
        runtime="pytorch",
        rdma=True,
        host_network=True,
    )
    #  job.run()
    #  如果想提交任务后，可以直接退出
    job.run(enable_wait=False)


if __name__ == "__main__":
    gpujob()