from alps.io.base import OdpsConf
from alps.pytorch.api.io.dataset.odps_dataset import OdpsIterableDataset
from typing import List, Union


def create_dataset(
    table: str,
    fields: Union[str, List[str]],
    setup_fn,
    transform,
    is_train: bool,
    num_devices: int,
    batch_size: int,
    odps_conf: OdpsConf = None,
):
    # odps配置
    if odps_conf is None:
        odps_conf = OdpsConf(
            project="apmktalgo_dev",
            endpoint="http://service.odps.aliyun-inc.com/api",
        )
    # odps表名

    # odps表要读取的字段

    # 训练数据初始化
    train_dataset_test = OdpsIterableDataset(
        odps_conf=odps_conf,
        table=table,
        fields=fields,
        setup_fn=setup_fn,
        transform=transform,
        auto_shard=False,
        aistudio_reader_num_processes=4,
    )
    if not is_train:
        return train_dataset_test
        # 如果每个gpu分配的数据不一样，ddp模式下就会一直hang住,所以要让dataset的总量刚好是num_device的条数，
    train_dataset = OdpsIterableDataset(
        odps_conf=odps_conf,
        table=table,
        fields=fields,
        setup_fn=setup_fn,
        transform=transform,
        auto_shard=True,
        max_size=len(train_dataset_test)
        // (num_devices * batch_size)
        * num_devices
        * batch_size,  # 如果每个gpu分配的数据不一样，ddp模式下就会一直hang住
    )
    return train_dataset
