import os
import sys
import subprocess
import logging
from tools.util.kmitool_util import kmi_cryptor
from tools.util.download_util import DownloadUtil
from solutions.antllm.antllm.evaluation.scripts.check_dataset_format import CheckBenchmarkDatasets

_logger = logging.getLogger()


class DownloadUtilNew(DownloadUtil):
    """
    Utility class for downloading files from oss/util.
    """
    @classmethod
    def _download_oss(
        cls,
        url: str,
        filepath: str,
        ossutil="ossutil",
        oss_access_id=None,
        oss_access_key=None,
        oss_endpoint=None,
        kmitool_config=None,
        retry_times=10,
    ):
        command = ossutil
        if kmitool_config:
            oss_access_id, oss_access_key, oss_endpoint = cls._kmi_lookup(
                kmitool_config,
                oss_access_id=oss_access_id,
                oss_access_key=oss_access_key,
                oss_endpoint=oss_endpoint,
            )

        command += f" --retry-times={retry_times}"

        if oss_access_id:
            command += " -i {}".format(oss_access_id)
        if oss_access_key:
            command += " -k {}".format(oss_access_key)
        if oss_endpoint:
            command += " -e {}".format(oss_endpoint)

        _logger.info("running command: '%s'", command)
        filepath = os.path.dirname(filepath)
        command += " cp -rf {} {}".format(url, filepath)
        res = subprocess.run(command.split(), stdout=subprocess.DEVNULL)
        if res.returncode != 0:
            raise RuntimeError("failed executing {}".format(command))
        else:
            _logger.info("sucessfully downloaded {}".format(filepath))


def subprocess_popen(statement):
    p = subprocess.Popen(statement, shell=True, stdout=subprocess.PIPE)
    while p.poll() is None:
        if p.wait() != 0:
            print("command running error!")
            return False
        else:
            re = p.stdout.readlines()
            result = []
            for i in range(len(re)):
                res = re[i].decode('utf-8').strip('\r\n')
                result.append(res)
            return result


def FetchBenchmarkDataset(
        dataset_name,
        save_dir='./',
        des_file=None,
        oss_url='oss://antsys-adabrain/solutions/chatgpt/data/评测数据集',
        ossutil=None
):
    oss_id = kmi_cryptor.get_value("adabrain_oss_id")
    oss_key = kmi_cryptor.get_value("adabrain_oss_key")
    oss_endpoint = kmi_cryptor.get_value("adabrain_oss_host")

    if ossutil is not None:
        ossutil = ossutil
    else:
        result = subprocess_popen('which ossutil64')
        if isinstance(result, bool):
            print('Not found ossutil command! Please install it first or put its path under PATH variable!')
            sys.exit(0)
        else:
            ossutil = result[0]

    if dataset_name != 'all':
        dataset_url = '{}/{}'.format(oss_url, dataset_name)
    else:
        dataset_url = oss_url
    download_util = DownloadUtilNew()

    try:
        download_util.download(
            dataset_url,
            output_dir=save_dir,
            oss_access_id=oss_id,
            oss_access_key=oss_key,
            oss_endpoint=oss_endpoint,
            keep_archive=True,
            ossutil=ossutil
        )
        dataset_path = save_dir  # os.path.join(save_dir, dataset_name)
        if des_file is not None:
            obj = CheckBenchmarkDatasets(dataset_path=dataset_path, des_file=des_file)
            if dataset_name != 'all':
                obj.RunCheckSingleDataset(dataset_name=dataset_name)
            else:
                obj.RunCheck()
    except BaseException:
        raise BaseException('Not reached dataset {}.'.format(dataset_url))


if __name__ == '__main__':
    # dataset_path = '/mnt/experiment/chengzhao.wq/glm_train/dataset/EvalData/评测数据集'
    if len(sys.argv) < 2:
        FetchBenchmarkDataset('AFQMC', save_dir='./antllm_test/')
    elif len(sys.argv) == 2:
        FetchBenchmarkDataset(sys.argv[1], save_dir='./antllm_test/')
    elif len(sys.argv) == 3:
        FetchBenchmarkDataset(sys.argv[1], save_dir=sys.argv[2])
    elif len(sys.argv) == 4:
        FetchBenchmarkDataset(sys.argv[1], save_dir=sys.argv[2], des_file=sys.argv[3])
    else:
        print('Too many parameters. You should input 1 or 2 parameters, please check them! :)')
        sys.exit(0)
