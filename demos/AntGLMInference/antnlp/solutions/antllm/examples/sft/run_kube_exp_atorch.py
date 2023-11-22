import argparse
import os
import shutil
from datetime import datetime

from pypai.conf import ExecConf, KMConf
from pypai.job import PythonJobBuilder


def main():
    args = parse_args()

    image = "reg.docker.alibaba-inc.com/aii/aistudio:4300121-20230808211027"

    node_gpu_num = 8  # gpu number per node
    node_num = args.nodes    # node number
    cpu_core = 32     # pod cpu core
    memory = 900 * 1024   # pod memory size in MB
    disk = 1800 * 1024  # pod disk size in MB
    gpu_type = ""
    app = "gbank"
    k8s_priority = "high"

    dt = datetime.now()
    time_stamp = dt.strftime("%Y%m%d-%H%M%S")

    if args.resume_timestamp is None:
        args.resume_timestamp = time_stamp
        resume_from_checkpoint = 'false'
    else:
        resume_from_checkpoint = 'true'

    output_dir = os.path.join(args.output_dir, os.path.basename(
        args.script), args.resume_timestamp)

    if resume_from_checkpoint == 'false':
        source_dir = 'solutions/antllm'
        dest_dir = os.path.join(output_dir, 'solutions/antllm')
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        print(f'Copy {source_dir} to {dest_dir}')
        shutil.copytree(source_dir, dest_dir)

    print(f'output_dir: {output_dir}')

    command = f"""if [ ! -d /workspace ]; then mkdir /workspace; fi; \
    mount -t nfs -o vers=3,nolock,proto=tcp alipay-heyuan-31-bmc76.cn-heyuan-alipay.nas.aliyuncs.com:/ /workspace && \
    cd {output_dir} && \
    bash {args.script} {output_dir} {resume_from_checkpoint}"""

    print(f"using {node_num} nodes. command is {repr(command)}")
    master = ExecConf(
        num=1,
        cpu=cpu_core,
        memory=memory,
        gpu_num=node_gpu_num,
        gpu_type=gpu_type,
        disk_m=disk,
    )

    worker = None
    if node_num >= 1:
        worker = ExecConf(
            num=node_num,
            cpu=cpu_core,
            memory=memory,
            gpu_num=node_gpu_num,
            gpu_type=gpu_type,
            disk_m=disk,
        )

    print('km_conf')
    km_conf = KMConf(
        image=image,
        # cluster=cluster
    )

    print(f'Prepare job')
    source_root = '/tmp/jdjdjsjsjdjjdjdjdjsjdsjsj'
    os.makedirs(source_root, exist_ok=True)
    with open(os.path.join(source_root, 'test.py'), 'w') as fout:
        fout.write('a = 1\n')
    fout.close()
    job = PythonJobBuilder(source_root=source_root,
                           command=command,
                           main_file='',
                           master=master,
                           worker=worker,
                           k8s_priority=k8s_priority,
                           k8s_app_name=app,
                           km_conf=km_conf,
                           runtime='pytorch',
                           rdma=True,
                           host_network=True,
                           name=f"tianxuan-glm10-if-{time_stamp}",
                           )

    # job.run()
    print('Submit job')
    job.run(enable_wait=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('script', default=None, type=str,
                        action='store', help='script to execute')
    parser.add_argument('output_dir', default=None, type=str,
                        action='store', help='output dir')
    parser.add_argument(
        "--nodes", '-n', type=int, default=0, help="Num of nodes."
    )
    parser.add_argument(
        "--resume_timestamp", "-t", type=str, default=None, help="Timestamp to resume"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
