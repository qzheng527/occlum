import os
import json
import time
import base64
import uuid
import logging
import requests
import pandas as pd
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5 as Signature_pkcs1_v1_5
from Crypto.Hash import SHA

logger = logging.getLogger("common")

# HOST = 'https://itag-pre.alipay.com'
HOST = 'https://itag.alipay.com'
CREATE_DATASET_URL = f'{HOST}/mng/api/v1/dataproxy/createDatasetProxy'
APPEND_DATA_URL = f'{HOST}/mng/api/v1/dataproxy/appendData2Dataset'
CREATE_TASK_URL = f'{HOST}/mng/api/v1/task/createTask'


def post(url,
         data,
         headers=None,
         params=None,
         logger=logger,
         by_form=False,
         **kwargs):
    logger.info('Request POST to {}'.format(url))
    if by_form:
        response = requests.post(url,
                                 data=data,
                                 json=None,
                                 params=params,
                                 headers=headers,
                                 verify=False,
                                 **kwargs).json()
    else:
        response = requests.post(url,
                                 data=None,
                                 json=data,
                                 params=params,
                                 headers=headers,
                                 verify=False,
                                 **kwargs).json()
    return response


class Itag(object):
    def __init__(self, tnt_inst_id: str, private_key: str, user_id: str, buc_account_no: str):
        self.tntInstId = tnt_inst_id
        self.prk = "-----BEGIN RSA PRIVATE KEY-----\n" + private_key + "\n-----END RSA PRIVATE KEY-----"
        self.user_id = user_id
        self.bucAccountNo = buc_account_no
        self.source = "ALPHAD"  # 固定的参数
        self.batch_size = 1000  # itag要求每次最多传1000条数据
        self._logger = logger

    def _get_auth(self) -> str:
        self.reqId = str(uuid.uuid4())
        self.reqTimestamp = str(int(round(time.time() * 1000)))
        self.reqSourceApp = "antllm"
        message = "_".join([
            self.tntInstId,
            self.bucAccountNo,
            self.reqId,
            self.reqTimestamp,
            self.reqSourceApp
        ])
        # print(message)

        rsakey = RSA.importKey(self.prk)
        signer = Signature_pkcs1_v1_5.new(rsakey)
        digest = SHA.new()
        digest.update(message.encode("utf-8"))
        sign = signer.sign(digest)
        signature = base64.b64encode(sign)
        # print(signature.decode('utf-8'))

        return {
            "bucAccountNo": self.bucAccountNo,
            "alphaqTntInstId": self.tntInstId,
            "reqTimestamp": self.reqTimestamp,
            "reqSourceApp": self.reqSourceApp,
            "reqId": self.reqId,
            "token": signature
        }

    def _upload_file(self, filepath: str) -> str:
        basename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        fileds = [{"FieldName": column, "Type": "TEXT"} for column in df.columns]

        # step-1: 创建数据集
        body = {
            "TntInstId": self.tntInstId,
            "Source": self.source,
            "Name": basename,
            "SharedMode": "USER",  # ALL，TNT，USER
            "Schema": {
                "Fields": fileds
            },
            "User": {
                "userId": self.user_id,
                "accountType": "BUC",
                "accountNo": self.bucAccountNo
            }
        }
        header = self._get_auth()
        print(header)
        print(body)

        resp = post(CREATE_DATASET_URL, body, headers=header)
        print(resp)

        if resp['Code'] != 0:
            self._logger.error('Create Dataset failed, response: {}'.format(resp))
            return None
        dataset_id = resp['Result']

        # step-2: 传数据
        print(f'data len: {len(df)}')
        for i in range(0, len(df), self.batch_size):
            batch_df = df[i: i + self.batch_size]
            records = batch_df.to_json(orient="records", force_ascii=False)
            records = json.loads(records)
            body = {
                "TntInstId": self.tntInstId,
                "Source": self.source,
                "SourceBizId": dataset_id,
                "BatchNo": str(int(i / self.batch_size) + 1),
                "Data": [
                    {
                        "OutDataId": "fake_id",
                        "MetaInfos": record
                    } for record in records
                ],
                "User": {
                    "userId": self.user_id,
                    "accountType": "BUC",
                    "accountNo": self.bucAccountNo
                }
            }
            resp = {
                'Code': 0,
            }
            resp = post(APPEND_DATA_URL, body, headers=header)
            print(resp)

        if resp['Code'] != 0:
            self._logger.error('Append Data failed, response: {}'.format(resp))
            return None
        
        return dataset_id

    def create_task(self,
                    name: str,
                    filepath: str,
                    template_id: str,
                    biz_code: str,
                    biz_no: str,
                    **kwargs) -> str:
        dataset_id = self._upload_file(filepath)
        if not dataset_id:
            return None

        header = self._get_auth()
        print(header)

        data = {
            "TaskName": name,
            "BizInfo": {
                "BizCode": biz_code,
                "BizNo": biz_no
            },
            "AssignConfig": {
                "AssignType": "FIXED_SIZE",
                "AssignCount": kwargs.get("assignCount", 1)
            },
            "TaskWorkFlow": [
                {
                    "Users": [{
                        "UserId": user_id
                    } for user_id in kwargs.get("markUsers", "").split(",") if user_id],
                    "Groups": [{
                        "GroupId": group_id
                    } for group_id in kwargs.get("markUserGroups", "").split(",") if group_id],
                    "NodeName": "MARK",
                    "Exif": {}
                }
            ],
            "Admins": {
                "Users": [
                    {
                        "UserId": shared_user_id,
                    } for shared_user_id in kwargs.get("sharedUsers", self.user_id).split(",")
                ]
            },
            "TemplateId": str(template_id),
            "TaskType": "NORMAL",
            "DatasetProxyRelations": [
                {
                    "Source": "ALPHAD",
                    "SourceBizId": dataset_id,
                    "DatasetType": "LABEL"
                }
            ],
            "TntInstId": self.tntInstId,
            "User": {
                "userId": self.user_id,
                "accountType": "BUC",
                "accountNo": self.bucAccountNo
            }
        }
        check_info = {
            "Users": [{
                "UserId": user_id
            } for user_id in kwargs.get("checkUsers", "").split(",") if user_id],
            "Groups": [{
                "GroupId": group_id
            } for group_id in kwargs.get("checkUserGroups", "").split(",") if group_id],
            "NodeName": "CHECK",
            "Exif": {}
        }
        sample_info = {
            "Users": [{
                "UserId": user_id
            } for user_id in kwargs.get("samplingUsers", "").split(",") if user_id],
            "Groups": [{
                "GroupId": group_id
            } for group_id in kwargs.get("samplingUserGroups", "").split(",") if group_id],
            "NodeName": "SAMPLING",
            "Exif": {}
        }
        vote_config = {
            "MARK": {
                "VoteNum": kwargs.get("voteNum", 0),
                "MinVote": kwargs.get("minVote", 0)
            },
            "CHECK": {
                "VoteNum": kwargs.get("voteNum", 0),
                "MinVote": kwargs.get("minVote", 0)
            },
            "SAMPLING": {
                "VoteNum": kwargs.get("voteNum", 0),
                "MinVote": kwargs.get("minVote", 0)
            }
        }
        if check_info["Users"] or check_info["Groups"]:
            data["TaskWorkFlow"].append(check_info)
        if sample_info["Users"] or sample_info["Groups"]:
            data["TaskWorkFlow"].append(sample_info)
        if vote_config["MARK"]["VoteNum"] > 0:
            data["VoteConfigs"] = vote_config
        # print(data)

        resp = post(CREATE_TASK_URL, data, headers=header)

        if resp['Code'] != 0:
            self._logger.error('Create Task failed, response: {}'.format(resp))
            return None

        return resp['Result']['TaskId']
