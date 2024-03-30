"""
    以下所有配置也都支持利用环境变量覆写，环境变量配置格式见docker-compose.yml。
    读取优先级：环境变量 > config_private.py > config.py
    --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    All the following configurations also support using environment variables to override,
    and the environment variable configuration format can be seen in docker-compose.yml.
    Configuration reading priority: environment variable > config_private.py > config.py
"""

# [step 1]>> API_KEY = "sk-123456789xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx123456789"。极少数情况下，还需要填写组织（格式如org-123456789abcdefghijklmno的），请向下翻，找 API_ORG 设置项
API_KEY = "sk-Wv9xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxMoh9"  

# [step 2]>> 改为True应用代理，如果直接在海外服务器部署，此处不修改；如果使用本地或无地域限制的大模型时，此处也不需要修改
USE_PROXY = True
if USE_PROXY:
    """
    代理网络的地址，打开你的代理软件查看代理协议(socks5h / http)、地址(localhost)和端口(11284)
    填写格式是 [协议]://  [地址] :[端口]，填写之前不要忘记把USE_PROXY改成True，如果直接在海外服务器部署，此处不修改
            <配置教程&视频教程> https://github.com/binary-husky/gpt_academic/issues/1>
    [协议] 常见协议无非socks5h/http; 例如 v2**y 和 ss* 的默认本地协议是socks5h; 而cl**h 的默认本地协议是http
    [地址] 填localhost或者127.0.0.1（localhost意思是代理软件安装在本机上）
    [端口] 在代理软件的设置里找。虽然不同的代理软件界面不一样，但端口号都应该在最显眼的位置上
    """
    proxies = {
        #          [协议]://  [地址]  :[端口]
        "http":  "http://127.0.0.1:2307",  # 再例如  "http":  "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:2307",  # 再例如  "https": "http://127.0.0.1:7890",
    }
else:
    proxies = None


# ------------------------------------ 以下配置可以优化体验, 但大部分场合下并不需要修改 ------------------------------------

# 重新URL重新定向，实现更换API_URL的作用（高危设置! 常规情况下不要修改! 通过修改此设置，您将把您的API-KEY和对话隐私完全暴露给您设定的中间人！）
# 格式: API_URL_REDIRECT = {"https://api.openai.com/v1/chat/completions": "在这里填写重定向的api.openai.com的URL"}
# 举例: API_URL_REDIRECT = {"https://api.openai.com/v1/chat/completions": "https://reverse-proxy-url/v1/chat/completions"}
API_URL_REDIRECT = {}


# 多线程函数插件中，默认允许多少路线程同时访问OpenAI。Free trial users的限制是每分钟3次，Pay-as-you-go users的限制是每分钟3500次
DEFAULT_WORKER_NUM = 3


# 模型选择是 (注意: LLM_MODEL是默认选中的模型, 它*必须*被包含在AVAIL_LLM_MODELS列表中 )
LLM_MODEL = "gpt-4" # 可选 ↓↓↓
AVAIL_LLM_MODELS = ["gpt-4-1106-preview", "gpt-4-turbo-preview", "gpt-4-vision-preview",
                    "gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k", "gpt-3.5-turbo", "azure-gpt-3.5",
                    "gpt-4", "gpt-4-32k", "azure-gpt-4", "glm-4", "glm-3-turbo",
                    "gemini-pro", "chatglm3", "claude-2"]
# P.S. 其他可用的模型还包括 [
# "moss", "qwen-turbo", "qwen-plus", "qwen-max"
# "zhipuai", "qianfan", "deepseekcoder", "llama2", "qwen-local", "gpt-3.5-turbo-0613",
# "gpt-3.5-turbo-16k-0613",  "gpt-3.5-random", "api2d-gpt-3.5-turbo", 'api2d-gpt-3.5-turbo-16k',
# "spark", "sparkv2", "sparkv3", "chatglm_onnx", "claude-1-100k", "claude-2", "internlm", "jittorllms_pangualpha", "jittorllms_llama"
# ]


# 设置gradio的并行线程数（不需要修改）
CONCURRENT_COUNT = 100


# 是否在提交时自动清空输入框
AUTO_CLEAR_TXT = False


# 加一个live2d装饰
ADD_WAIFU = False


# 设置用户名和密码（不需要修改）（相关功能不稳定，与gradio版本和网络都相关，如果本地使用不建议加这个）
# [("username", "password"), ("username2", "password2"), ...]
AUTHENTICATION = []


# 如果需要在二级路径下运行（常规情况下，不要修改!!）（需要配合修改main.py才能生效!）
CUSTOM_PATH = "/"


# HTTPS 秘钥和证书（不需要修改）
SSL_KEYFILE = ""
SSL_CERTFILE = ""



MATHPIX_APPKEY = ""


# 自定义API KEY格式
CUSTOM_API_KEY_PATTERN = ""




# HUGGINGFACE的TOKEN，下载LLAMA时起作用 https://huggingface.co/docs/hub/security-tokens
HUGGINGFACE_ACCESS_TOKEN = "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"


# GROBID服务器地址（填写多个可以均衡负载），用于高质量地读取PDF文档
# 获取方法：复制以下空间https://huggingface.co/spaces/qingxu98/grobid，设为public，然后GROBID_URL = "https://(你的hf用户名如qingxu98)-(你的填写的空间名如grobid).hf.space"
GROBID_URLS = [
    "https://qingxu98-grobid.hf.space","https://qingxu98-grobid2.hf.space","https://qingxu98-grobid3.hf.space",
    "https://qingxu98-grobid4.hf.space","https://qingxu98-grobid5.hf.space", "https://qingxu98-grobid6.hf.space",
    "https://qingxu98-grobid7.hf.space", "https://qingxu98-grobid8.hf.space",
]


# 是否允许通过自然语言描述修改本页的配置，该功能具有一定的危险性，默认关闭
ALLOW_RESET_CONFIG = False


# 在使用AutoGen插件时，是否使用Docker容器运行代码
AUTOGEN_USE_DOCKER = False


# 临时的上传文件夹位置，请勿修改
PATH_PRIVATE_UPLOAD = "private_upload"


# 日志文件夹的位置，请勿修改
PATH_LOGGING = "gpt_log"


# 除了连接OpenAI之外，还有哪些场合允许使用代理，请勿修改
WHEN_TO_USE_PROXY = ["Download_LLM", "Download_Gradio_Theme", "Connect_Grobid",
                     "Warmup_Modules", "Nougat_Download", "AutoGen"]



