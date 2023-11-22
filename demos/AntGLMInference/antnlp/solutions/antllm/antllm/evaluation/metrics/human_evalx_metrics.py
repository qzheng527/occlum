import random
import json
import threading
import time
from typing import Dict, List

import requests


"""
calculate pass1 for code evaluation
"""


class PassAt1():
    def __init__(self):
        self.eval_api_url = "https://anteval.antgroup-inc.cn/openapi/anteval/job/metrics"
        self.job_detail_url_prefix = "https://anteval.antgroup-inc.cn/openapi/anteval/job/detail?jobId="
        self.job_result_url_prefix = "https://anteval.antgroup-inc.cn/openapi/anteval/job/result/query?jobId="

    def generation_postprocess(self, generated_code, language):
        code = generated_code
        if language in ["python"]:
            generation = ""
            for line in code.split("\n"):
                if line and line[0] != ' ':
                    break
                generation += line + "\n"

        elif language in ["cpp"]:
            generation = ""
            for line in code.split("\n"):
                if line and line.startswith("int main"):
                    break
                generation += line + "\n"

        elif language in ["js", "java", "go"]:
            generation = ""
            for line in code.split("\n"):
                generation += line + "\n"
                if line == "}" or line == "};":
                    break
        else:
            raise Exception('{} not support!'.format(language))
        return generation

    def compute(self, predictions: List[List], references: List[List], extras: List[Dict]):
        output_res = dict()
        threads = []
        data = self.construct_data(predictions, extras)
        for lang, data_lang in data.items():
            t = threading.Thread(target=self.eval_code,
                                 args=(data_lang, lang, output_res))
            t.start()
            threads.append(t)

        for t in threads:
            t.join(timeout=1250)  # wait to run at most 1250 seconds
        output_res = {k: round(v * 100, 2) for k, v in output_res.items()}
        return output_res

    def construct_data(self, predictions, extras):
        data = dict()

        # in HumanEvalX output data, there is no "predictions" and "references" field. Instead,
        # it contains "prediction" and "canonical_solution" for the same usage.
        operators = ['209144', '396458', '362188', '119784', '112708']
        operators = ['119784']
        random.shuffle(operators)
        operator = operators[0]
        for i, extra in enumerate(extras):
            language = extra['language']
            if language not in data:
                res = {
                    "fromApp": "GLM",
                    "token": "10b96af9-3468-3e63-97db-8472909699dc",
                    "operator": operator,
                    'jobName': 'nlp_code_eval',
                    "evalConfig": {
                        "evalType": "TEXT_TO_CODE",
                        "metrics": ["pass@1"],
                        "lang": language,
                        "caseSetName": "humaneval-x-" + language,
                        "dataMaxCount": "200",
                        "paramsMap": {
                            "eval_mode": "metric_compute"
                        }
                    },
                    "modelConfig": {
                        "modelName": "AntGLM-10B-SFT",
                        "modelVersion": "20230602"
                    },
                    "genarations": []
                }
                data[language] = res

            prediction = self.generation_postprocess(
                predictions[i][0], language=language)

            data[language]['genarations'].append({'task_id': extra['task_id'], 'generation': prediction,
                                                  'reference': extra['canonical_solution'], 'prompt': extra['input'],
                                                  'dataset': 'humaneval-x'})
        return data

    def eval_code(self, data_lang, lang, output_res):
        headers = {"Content-Type": "application/json"}
        res = requests.post(self.eval_api_url, json=data_lang, headers=headers)
        print(f'{lang}: {res.text}')
        if res.status_code == 200:
            result = json.loads(res.text)
            if result['success']:
                while True:
                    time.sleep(5)
                    job_id = result['data']['jobId']
                    res_job_detail = requests.get(
                        self.job_detail_url_prefix + str(job_id))
                    res_job_detail = json.loads(res_job_detail.text)
                    if res_job_detail["success"]:
                        # time.sleep(1200)  # wait 1200 seconds to let code run
                        while True:
                            res_job_result = requests.get(
                                self.job_result_url_prefix + str(job_id))
                            res_job_result = json.loads(res_job_result.text)
                            try:
                                passAt1 = json.loads(res_job_result["data"]["datas"][0])[
                                    "pass@1"]["pass_at_k"]["pass@1"]
                                output_res[lang] = passAt1
                                break
                            except Exception:
                                time.sleep(5)
                        break
                    else:
                        pass


if __name__ == "__main__":
    pass
    # extras = [{"task_id": "Java/0"},
    #           {"task_id": "Java/1"}]
    # pass1 = PassAt1()
    # pass1.compute(None, None, extras)
