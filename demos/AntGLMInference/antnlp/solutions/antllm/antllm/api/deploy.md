# 微调模型部署
目前模型部署的话分为基座模型部署和lora权重更新
* 基座模型支持部署到zark和maya两种方式
* 基座模型部署完成后, 可以把本地或者云端训练的lora权重热更新到基座上, 目前仅支持maya方式, zark功能开发中预计很快上线
## 部署自定义模型基座到maya
基座只支持nas盘和http(s)两种云端模式。
nas盘格式: nas://{domain-name}:/path/to/llm
http(s)格式： http://{domain-name}/path/to/llm.tar 文件名任意取，只能为.tar或.tar.gz包，包内可直接存放大模型文件或包含一个目录，大模型文件放置于子目录内。
demo部署代码：
```
from antllm.api.deploy import DeployManager, DeployStatus, GPUType
model = 'local-model-path'
mng = DeployManager()
task_id = mng.deploy_base(
    base_llm='nas://alipayshnas-0007-jxd68.cn-shanghai-eu13-a01.nas.aliyuncs.com:/llm/glm5b_sft',
    scene_name='base', # 必填，部署后的场景名
    version='v1', #  必填，部署后的版本名
    biz_domain='Intelligent.Interaction', # 必填，部署的aistudio租户名英文ID，可通过aistudio IDE右上角我的租户查看
    gpu_type=GPUType.A10, # 必填，部署的GPU卡类型
    pre_idc='stl', # 选填，部署的预发机房，默认会从空闲机房自动选择IDC
    prod_idc='stl' # 选填，部署的生产机房，默认会从空闲机房自动选择IDC
)
deploy_info = mng.deploy_status(task_id)
if deploy_info.status == DeployStatus.success:
    print('deploy success')
else:
    print(deploy_info.message)
```
默认部署的aistudio推理代码（参数:app_path）是app/antglm/api_template，你可以模仿这个代码新拷贝一份，增加自己的前后处理逻辑。


## 部署自定义模型基座到zark
* 部署10b模型到zark上支持双卡a10模式(不用a100也可成功部署), 附带tgi推理加速
### 创建应用
在下面的链接中填写的应用名称/实例名称分别对应scene_name version
点击创建即可
http://11.166.224.204:7777/services/pmas2/zark_config.html
### 申请部署
填写刚才的应用名称和实例名称,点击申请,等待审批通过即可部署
http://zarkmeta.alipay.com/zark_dashboard/app_apply
### 部署代码
以部署10b模型为例,
下面demo部署代码会部署10b模型到双卡a10上：
```
from antllm.api.deploy import DeployManager, DeployStatus, GPUType
mng = DeployManager()
pmas_addr = mng.deploy_base(
    base_llm='https://alipay/AntGLM-10B-SFT-Detoxcity-20230602.tar.gz', # 必填, 模型地址,必须为http(s)格式, 注意该地址只是demo不可用
    scene_name='test', # 必填, 部署的app name
    version='tgi', #  必填, 部署的instance name
    biz_domain='Intelligent.Interaction', # 必填, 部署单元
    gpu_type=GPUType.A10, # 必填, 部署的GPU卡类型
    prod_idc='stl,ea179', # 必填, 部署的生产机房
    platform="zark“, # 必填, 部署到zark平台
    gpu_count=2, # 选填, gpu卡数
    replica=1, # 选填, 副本数
    worker_count=1, # 选填, python worker数量
)
print(pmas_addr)
```
上面吐出的pmas_addr用浏览器打开, 可以查看部署状态, 有问题请咨询zark值班, 值班群3050008388
* 其他可用参数说明
* * cpu_count, 选填, cpu个数, 默认为20
* * mem_count, 选填, 内存, 默认为51200
* * disk_count, 选填, 硬盘, 默认为100
* * env, 选填, 部署环境, 默认为prod
* * start args, 选填, 模型启动参数
* * worker_image, 选填, zark底座镜像
* * app_image, 选填, tgi镜像版本
* * startup_timeout, 选填, 副本启动超时时间, 默认为1800s


## 部署peft微调模型
可热更新本地或云端训练的lora权重到maya
* 更新智能交互提供的公共大模型底座，这种模型所有人共享一个大模型服务。
* * 关键部署参数为：
model: 要部署的模型信息。可为本地训练产出的lora权重目录，或remote训练的task id。
adapter_name: lora权重名称。在线预测时需要提供。
* 更新私有的大模型服务的底座lora权重
* * 关键部署参数为：
model: 要部署的模型信息。可为本地训练产出的lora权重目录，或remote训练的task id。
adapter_name: lora权重名称。在线预测时需要提供。
scene_name: 目标大模型服务名称
version: 目标大模型服务版本

1) 默认部署云端训练的模型到公共大模型代码如下：
```
from antllm.api.deploy import DeployManager, DeployParams, DeployStatus
model = 'remote-train-task-id'
mng = DeployManager()
task_id = mng.deploy(model, 'adapter-1')
print(task_id)
deploy_info = mng.deploy_status(task_id)
if deploy_info.status == DeployStatus.success:
    print('deploy success')
else:
    print(deploy_info.message)
```

2) 如果是更新的lora权重在本地，model参数请填写本地lora权重路径，另外，DeployParams需要增加base_llm参数，以指定基座大模型名称。
当前可选的基座大模型名称列表可通过如下代码获取， ret的keys()即为可选的大模型名称集合:
```
mng = DeployManager()
ret = mng.get_base_llms()

ret: 基座模型信息，其中key就是基座模型名称。scene_name/version就是默认部署的基座模型服务名称。例如:
{
    "glm10b_rlhf_20230602": {
        "ais_public_service": {
            "scene_name": "glm10b_rlhf_20230602",
            "version": "v1"
        },
        "image": "reg.docker.alibaba-inc.com/aii/aistudio:aistudio-102991717-855067390-1687184787085"
    },
    "glm10b_sft_20230602": {
        "ais_public_service": {
            "scene_name": "glm10b_sft_20230602",
            "version": "v1"
        },
        "image": "reg.docker.alibaba-inc.com/aii/aistudio:aistudio-102276384-662218608-1687183793710"
    }
}
```

最终本地lora权重更新代码如下:
```
from antllm.api.deploy import DeployManager, DeployParams, DeployStatus
model = 'local-lora-path'
params = DeployParams(
    base_llm='glm10b_rlhf_20230602'  # 基座模型名称
)
mng = DeployManager()
task_id = mng.deploy(model, 'adapter-1', params)  # adapter名字，用于区别唯一的微调权重，建议更名为其他
print(task_id)
deploy_info = mng.deploy_status(task_id)
if deploy_info.status == DeployStatus.success:
    print('deploy success')
else:
    print(deploy_info.message)
```


3) 如果想热更新自己的大模型服务的lora权重：
```
from antllm.api.deploy import DeployManager, DeployParams, DeployStatus
model = 'local-lora-path'
params = DeployParams(
    model='local-lora-path'
    scene_name='your-scene-name'  # 服务名称
    version='your-version'  # 服务版本
)
mng = DeployManager()
task_id = mng.deploy(model, 'adapter-1', params)  # adapter名字，用于区别唯一的微调权重，建议更名为其他
print(task_id)
deploy_info = mng.deploy_status(task_id)
if deploy_info.status == DeployStatus.success:
    print('deploy success')
else:
    print(deploy_info.message)
```

4) 关键部署结果信息
```
mng = DeployManager()
deploy_info = mng.deploy_status(task_id)
其中deploy_info.service_id， 为本次部署对应的服务ID，在接下来的服务访问中会有用。
其中deploy_info.scene_name， 为本次部署对应的场景名，在接下来的服务访问中会有用。
其中deploy_info.version， 为本次部署对应的版本，在接下来的服务访问中会有用。
```


## 服务访问
### maya

 - 方式1（推荐）:通过RemoteCompletion访问，请上述小节方式

 - 方式2:通过原始HTTP请求访问

1）明确服务ID（即服务唯一标识）与版本。
2）访问方式：
生产url：'https://paiplusinference.alipay.com/inference/{服务唯一标识}/{version}
HTTP头：'Content-Type: application/json', 'MPS-app-name: {scene_name}', 'MPS-http-version: 1.0'
body如下，所有内容放入features.data中，data的value是个json str。当前只有query, adapter_name两个参数，如果adapter name不填写则使用基座模型推理.
{
    "features":{
        "data" : "{\"query\": \"where are you from , bro?\", \"adapter_name\":  \"default\"}"
    }
}

整体可访问通的请求如下:

```
curl -X POST 'https://paiplusinference.alipay.com/inference/c7992997f76af6b3_lx_platform/antglm_5b' \
-H 'Content-Type: application/json' \
-H 'MPS-app-name: lx_platform' \
-H 'MPS-http-version: 1.0' \
-d '{
    "features":{
        "data" : "{\"query\": \"where are you from , bro?\", \"adapter_name\":  \"linxi_test_0612_7\"}"
    }
}'
```
### zark
在线服务实例的URL pattern是： http://{zark服务域名}/{app_name}/{instance_name}/{service}
生产环境的域名为zark.sh.global.alipay.com
app_name对应部署时候的参数scene_name, instance_name应version.
service可以选择流式和非流式两种, generate/generate_strem
整体可访问通的请求如下
* 非流式
```
curl http://zark.sh.global.alipay.com/${app_name}/${instance_name}/generate   
-X POST      
-d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}'
-H 'Content-Type: application/json'
```
* 流式
```
curl http://zark.sh.global.alipay.com/${app_name}/${instance_name}/generate_stream   
-X POST      
-d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}'
-H 'Content-Type: application/json'
```
* python代码, 请先执行pip install text-generation

```python
>>> from text_generation import Client

>>> client = Client("https://zark.sh.global.alipay.com/${app_name}/${instance_name}")
>>> client.generate("Why is the sky blue?").generated_text
' Rayleigh scattering'

>>> result = ""
>>> for response in client.generate_stream("Why is the sky blue?"):
>>>     if not response.token.special:
>>>         result += response.token.text
>>> result
' Rayleigh scattering'
```

更多可以参考https://yuque.antfin-inc.com/zark/uev5ey/gsii9d