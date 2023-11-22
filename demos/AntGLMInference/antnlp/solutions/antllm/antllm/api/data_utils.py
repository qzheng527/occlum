import logging
import tempfile
import os
import shutil
import adabench.utils.util as util

glogger = logging.getLogger('antllm')


def easy_upload(train_path="", validation_path="", test_path="", dataset_id="", **kwargs):
    r'''Help user use adabench upload more easily
        user only need to provide only a train.jsonl or dev.jsonl or test.jsonl
    Args
        train_path: a path points to where train.jsonl locates,
        and does not need to be named as train.jsonl. eg /home/admin/my_train.jsonl
        validation_path: a path points to where dev.jsonl locates,
        and does not need to be named as dev.jsonl. eg /home/admin/my_dev.jsonl
        test_path: a path points to where test.jsonl locates,
        and does not need to be named as test.jsonl. eg /home/admin/my_test.jsonl
    Return
        dataset_id if success else ""
    '''
    if not train_path and not validation_path and not test_path:
        glogger.error("you must give at least one file to be upload")
        raise Exception("you must give at least one file to be upload")
    with tempfile.TemporaryDirectory() as tmp_path:
        if train_path:
            train_file = os.path.join(tmp_path, "train.jsonl")
            shutil.copy(train_path, train_file)
        if validation_path:
            validation_file = os.path.join(tmp_path, "dev.jsonl")
            shutil.copy(validation_path, validation_file)
        if test_path:
            test_file = os.path.join(tmp_path, "test.jsonl")
            shutil.copy(test_path, test_file)
        return upload(tmp_path, dataset_id, **kwargs)


def distill_data_upload(teacher_finetune_train_path="", teacher_finetune_validation_path="",
                        distill_train_path="", distill_validation_path="",
                        student_finetune_train_path="", student_finetune_validation_path="",
                        dataset_id="", **kwargs):
    '''
    @param teacher_finetune_train_path: 大模型精调的训练数据
    @param teacher_finetune_validation_path: 大模型精调的验证数据
    @param distill_train_path:  进行蒸馏的无标签训练数据
    @param distill_validation_path: 进行蒸馏的无标签验证数据
    @param student_finetune_train_path:  student模型进行精调的训练数据
    @param student_finetune_validation_path: student模型进行精调的验证数据
    @param dataset_id:  数据集id
    @param kwargs:
    @return: dataset_id if success else ""
    '''
    if not distill_train_path or not distill_validation_path:
        glogger.error("you need to provide distill train file and distill validation file")
        raise Exception("you need to provide distill train file and distill validation file")

    with tempfile.TemporaryDirectory() as tmp_path:
        dir_name = os.path.join(tmp_path, "resource")
        os.mkdir(dir_name)
        if teacher_finetune_train_path:
            teacher_finetune_train_file = os.path.join(dir_name, "llm_finetune_train.jsonl")
            shutil.copy(teacher_finetune_train_path, teacher_finetune_train_file)
        if teacher_finetune_validation_path:
            teacher_finetune_validation_file = os.path.join(dir_name, "llm_finetune_eval.jsonl")
            shutil.copy(teacher_finetune_validation_path, teacher_finetune_validation_file)
        distill_train_file = os.path.join(dir_name, "distill_train.jsonl")
        shutil.copy(distill_train_path, distill_train_file)
        distill_eval_file = os.path.join(dir_name, "distill_eval.jsonl")
        shutil.copy(distill_validation_path, distill_eval_file)

        if student_finetune_train_path:
            student_finetune_train_file = os.path.join(dir_name, "student_train.jsonl")
            shutil.copy(student_finetune_train_path, student_finetune_train_file)
        if student_finetune_validation_path:
            student_finetune_validation_file = os.path.join(dir_name, "student_eval.jsonl")
            shutil.copy(student_finetune_validation_path, student_finetune_validation_file)

        return upload(tmp_path, dataset_id, **kwargs)


def upload(dataset_path,
           dataset_id="",
           version=1,
           author="",
           alg_task="supervised_finetune",
           title="",
           privilege='public',
           **kwargs
           ):
    r'''Upload dataset.

    Args:
        dataset_id: optional, dataset id
        dataset_path: dataset project path
        version: dataset version, default to 1
        author: optional, default value will be set to os.environ["USER"]
        alg_task: optional, default value will be set to supervised_finetune,
            be careful to use default value.
        title: optional, title is the name of dataset
            where dataset_id is identification, default is dataset_id.
        privilege: optional, if the dataset can be access by everyone,
            the is privilege is  public. default is public.
    Returns: dataset_id if success else None
    Raise: Exception if an error occurs
    '''
    from adabench.api import dataset_upload_v2
    return dataset_upload_v2(dataset_path,
                             dataset_id=dataset_id,
                             version=version,
                             author=util.get_user_name() if not author else author,
                             alg_task=alg_task,
                             title=title,
                             privilege=privilege,
                             channel="antllm",
                             origin="easy_upload",
                             **kwargs
                             )


def download(dataset_id, output_dir='.', version=None, splits=[], **kwargs):
    '''
    Args:
        dataset_id: dataset id
        output_dir: dataset project path
        version: dataset version, default to latest
        splits: to download dataset split types, default to all types. Choice is like 'train', 'dev', 'test'
    Returns: List of files for this dataset
    Raise: Exception if an error occurs
    '''
    user = kwargs['user'] if kwargs.get('user') else util.get_user_name()
    from adabench.api import dataset_download
    return dataset_download(dataset_id=dataset_id,
                            dpath=output_dir,
                            user=user,
                            version=version,
                            splits=splits,
                            )