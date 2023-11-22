import os
import sys
import json


class CheckBenchmarkDatasets(object):
    def __init__(self, dataset_path=None, des_file=None) -> None:
        assert dataset_path is not None
        assert des_file is not None
        self.dataset_path = dataset_path
        self.dataset_description = json.load(open(des_file, 'r'))

    def _check(self, folder_path, dataset_name):
        if 'zero-shot' in self.dataset_description[dataset_name]['method']:
            prompt_file = '{}/test_prompts.json'.format(folder_path)

            if not os.path.exists(prompt_file):
                print('test_prompts.json file for {} does not exist!'.format(
                    dataset_name))
                sys.exit(0)
            else:
                try:
                    prompt_samples = [json.loads(x)
                                      for x in open(prompt_file, 'r')]
                except BaseException:
                    print('{} format error!'.format(dataset_name))

                for line in prompt_samples:
                    if not ('input' in line and 'references' in line):
                        print('{} format error! Keyword input or references not in file'.format(
                            dataset_name))
                        break
                    else:
                        if not (
                            isinstance(line['input'], str) and isinstance(
                                line['references'], list)
                        ):
                            print(line)
                            print('{} format error!\
                                    Keyword input or references not has the standard format'.format(dataset_name))
                            break
        else:
            train_file_flag = False
            dev_file_flag = False
            test_file_flag = False

            for _file in os.listdir(folder_path):
                if 'train' in _file:
                    train_file_flag = True

                if 'dev' in _file or 'val' in _file:
                    dev_file_flag = True

                if 'test' in _file:
                    test_file_flag = True

            if not train_file_flag:
                print('{} does not have train file!'.format(dataset_name))

            if not dev_file_flag:
                print('{} does not have dev file!'.format(dataset_name))

            if not test_file_flag:
                print('{} does not have test file!'.format(dataset_name))

    def RunCheck(self):
        for dataset_name in self.dataset_description:
            folder_path = '{}/{}'.format(self.dataset_path, dataset_name)

            if not os.path.isdir(folder_path):
                print('{} folder not exist!'.format(dataset_name))
                continue

            self._check(folder_path, dataset_name)

        print('Done checking for all benchmark dataset! :)')

    def RunCheckSingleDataset(self, dataset_name=None, indir=None):
        assert dataset_name in self.dataset_description

        if indir is None:
            folder_path = '{}/{}'.format(self.dataset_path, dataset_name)
        else:
            folder_path = indir

        if not os.path.isdir(folder_path):
            print('{} folder not exist!'.format(dataset_name))
            sys.exit(0)

        self._check(folder_path, dataset_name)
        print('Done checking for {} dataset! :)'.format(dataset_name))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('No dataset path or dataset description file specified. Please check them!')
        sys.exit(0)
    elif len(sys.argv) == 3:
        dataset_path = sys.argv[1]
        # by default, it's put under the 'solutions/antllm/antllm/evaluation/configs/datasets_des.json'
        des_file = sys.argv[2]
        obj = CheckBenchmarkDatasets(
            dataset_path=dataset_path, des_file=des_file)
        obj.RunCheck()
    elif len(sys.argv) == 4:
        dataset_path = sys.argv[1]
        # by default, it's put under the 'solutions/antllm/antllm/evaluation/configs/datasets_des.json'
        des_file = sys.argv[2]
        dataset_name = sys.argv[3]
        obj = CheckBenchmarkDatasets(
            dataset_path=dataset_path, des_file=des_file)
        obj.RunCheckSingleDataset(dataset_name=dataset_name)
    else:
        print('Too many parameters, please check them! :)')
        sys.exit(0)
