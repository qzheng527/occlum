# coding=utf-8
# @Author: jianiu.lj
# @Date: 2023-07-12


class ErrorCode:
    # common error
    INVALID_PARAM = '10001'
    MISSING_PARAM = '10002'
    FILEFORMAT_ERROR = '10003'
    # job dispatcher error
    JOB_PREPARE_ERROR = '20001'
    NETWORK_ERROR = '20002'
    AUTHORITY_ERROR = '20003'

    # deploy error
    BASE_LLM_NOT_EXIST = '30001'
    TriggerDeployError = '30002'
    GetDeployStatusError = '30003'
    GetBaseLLMError = '30004'


class AntLLMError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self):
        return "错误码 %s: %s" % (
            self.code,
            self.message
        )


class InvalidParamError(AntLLMError):
    def __init__(self, message):
        super().__init__(ErrorCode.INVALID_PARAM, message)


class MissingParamError(AntLLMError):
    def __init__(self, param_name):
        message = f'Missing required parameter: {param_name}'
        super().__init__(ErrorCode.MISSING_PARAM, message)


class JobPrepareError(AntLLMError):
    def __init__(self, message):
        super().__init__(ErrorCode.JOB_PREPARE_ERROR, message)


class BaseLLMNotExistError(AntLLMError):
    def __init__(self, message):
        super().__init__(ErrorCode.BASE_LLM_NOT_EXIST, message)


class TriggerDeployError(AntLLMError):
    def __init__(self, message):
        super().__init__(ErrorCode.TriggerDeployError, message)


class GetDeployStatusError(AntLLMError):
    def __init__(self, message):
        super().__init__(ErrorCode.GetDeployStatusError, message)


class GetBaseLLMError(AntLLMError):
    def __init__(self, message):
        super().__init__(ErrorCode.GetBaseLLMError, message)


class FileFormatError(AntLLMError):
    def __init__(self, message):
        super().__init__(ErrorCode.FILEFORMAT_ERROR, message)