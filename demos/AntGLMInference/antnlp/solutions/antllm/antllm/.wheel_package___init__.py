# coding=utf-8
# @Author: jianiu.lj
# @Date: 2023-07-07

# antllm 作为 antnlp 的底层库，不依赖其它 antnlp 的代码，打的包中的代码已把 solutions.antllm 这级去掉，可以直接使用antllm库的内容。
# 所以暂时不再使用此文件
# import os
# import sys
#
# if '_init_flag' not in dir():
#     # AntNLP OCB 中的部分代码，是写的全路径 import，比如 import solutions.antllm.xxx，需要把本库的路径加到sys.path中
#     lib_path = os.path.dirname(os.path.abspath(__file__))
#     # 让用户可以直接 import antllm.api.xxx，能比较简单的使用
#     api_path = os.path.join(lib_path, "solutions/antllm")
#     sys.path.insert(0, lib_path)
#     sys.path.insert(0, api_path)
#     # 需要 reload，否则import还是会失败
#     _init_flag = True
#     from importlib import reload
#     reload(sys.modules[__name__])
