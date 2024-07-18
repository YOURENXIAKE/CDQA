import torch.cuda
import torch.backends
import os
import logging
import uuid

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

# 在以下字典中修改属性值，以指定本地embedding模型存储位置
# 如将 "text2vec": "GanymedeNil/text2vec-large-chinese" 修改为 "text2vec": "User/Downloads/text2vec-large-chinese"
# 此处请写绝对路径
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "/home/zrchen/yongyou/embmodel",
    "m3e-small": "moka-ai/m3e-small",
    "m3e-base": "moka-ai/m3e-base",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# supported LLM models
# llm_model_dict 处理了loader的一些预设行为，如加载位置，模型名称，模型处理器实例
# 在以下字典中修改属性值，以指定本地 LLM 模型存储位置
# 如将 "chatglm-6b" 的 "local_model_path" 由 None 修改为 "User/Downloads/chatglm-6b"
# 此处请写绝对路径
# llm_model_dict = {
#     "chatglm-6b-int4-qe": {
#         "name": "chatglm-6b-int4-qe",
#         "pretrained_model_name": "THUDM/chatglm-6b-int4-qe",
#         "local_model_path": None,
#         "provides": "ChatGLM"
#     },
#     "huozi": {
#         "name": "huozi",
#         "pretrained_model_name": "/home/zrchen/bloom/hf_iter_0006103",
#         "local_model_path": None,
#         "provides": "HuoZiLLM"
#     },
#     "chatglm-6b-int4": {
#         "name": "chatglm-6b-int4",  # 把这个位置也改了
#         "pretrained_model_name": "/users12/zrchen/yongyou/chatglm-4bit",
#         "local_model_path": None,
#         "provides": "ChatGLM"
#     },
#     "chatglm-6b-int8": {
#         "name": "chatglm-6b-int8",
#         "pretrained_model_name": "THUDM/chatglm-6b-int8",
#         "local_model_path": None,
#         "provides": "ChatGLM"
#     },
#     "chatglm-6b": {
#         "name": "chatglm-6b",
#         "pretrained_model_name": "THUDM/chatglm-6b",
#         "local_model_path": None,
#         "provides": "ChatGLM"
#     },

#     "chatyuan": {
#         "name": "chatyuan",
#         "pretrained_model_name": "ClueAI/ChatYuan-large-v2",
#         "local_model_path": None,
#         "provides": None
#     },
#     "moss": {
#         "name": "moss",
#         "pretrained_model_name": "fnlp/moss-moon-003-sft",
#         "local_model_path": None,
#         "provides": "MOSSLLM"
#     },
#     "vicuna-13b-hf": {
#         "name": "vicuna-13b-hf",
#         "pretrained_model_name": "vicuna-13b-hf",
#         "local_model_path": None,
#         "provides": "LLamaLLM"
#     },

#     # 通过 fastchat 调用的模型请参考如下格式
#     "fastchat-chatglm-6b": {
#         "name": "chatglm-6b",  # "name"修改为fastchat服务中的"model_name"
#         "pretrained_model_name": "chatglm-6b",
#         "local_model_path": None,
#         "provides": "FastChatOpenAILLM",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLM"
#         "api_base_url": "http://localhost:8000/v1"  # "name"修改为fastchat服务中的"api_base_url"
#     },

#     # 通过 fastchat 调用的模型请参考如下格式
#     "fastchat-vicuna-13b-hf": {
#         "name": "vicuna-13b-hf",  # "name"修改为fastchat服务中的"model_name"
#         "pretrained_model_name": "vicuna-13b-hf",
#         "local_model_path": None,
#         "provides": "FastChatOpenAILLM",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLM"
#         "api_base_url": "http://localhost:8000/v1"  # "name"修改为fastchat服务中的"api_base_url"
#     },
# }
llm_model_dict = {
    "huozi": {
        "name": "huozi",
        "pretrained_model_name": "/home/zrchen/bloom/hf_iter_0006103",
        "local_model_path": None,
        "provides": "HuoZiLLM"
    },
    "chatglm-6b-int4": {
        "name": "chatglm-6b-int4",  # 把这个位置也改了
        "pretrained_model_name": "/home/zrchen/yongyou/chatglm-4bit",
        "local_model_path": None,
        "provides": "ChatGLM"
    },
    "chatglm2-6b": {
        "name": "chatglm2-6b",  # 把这个位置也改了
        "pretrained_model_name": "/home/zrchen/yongyou/chatglm2-6b",
        "local_model_path": None,
        "provides": "ChatGLM"
    },
    "chatglm2-6b-32k": {
        "name": "chatglm2-6b-32k",  # 把这个位置也改了
        "pretrained_model_name": "/home/zrchen/yongyou/chatglm2-6b-32k",
        "local_model_path": None,
        "provides": "ChatGLM"
    },
    "chatglm2-6b-int4": {
        "name": "chatglm2-6b-int4",  # 把这个位置也改了
        "pretrained_model_name": "/home/zrchen/yongyou/chatglm2-6b-int4",
        "local_model_path": None,
        "provides": "ChatGLM"
    },
    "qwen-int4": {
        "name": "qwen-int4",  # 把这个位置也改了
        "pretrained_model_name": "/home/zrchen/yongyou/qwen70b/Qwen-72B-Chat-Int4",
        "local_model_path": None,
        "provides": "QWenLLM"
    },
    "qwen": {
        "name": "qwen",  # 把这个位置也改了
        "pretrained_model_name": "/home/zrchen/yongyou/qwen70b/Qwen-72B-Chat",
        "local_model_path": None,
        "provides": "QWenLLM"
    },
    "fastchat-huozi": {
        "name": "fastchat-huozi",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "fastchat-huozi",
        "local_model_path": None,
        "provides": "FastChatHuoZiLLM",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLM"
        "api_base_url": "https://huozi.8wss.com/api/v1"  # "name"修改为fastchat服务中的"api_base_url"
    },
    "fastchat-qwen2-72b": {
        "name": "fastchat-qwen",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "fastchat-qwen2-72b",
        "local_model_path": None,
        "provides": "FastChatOpenAILLM",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLM"
        "api_base_url": "http://localhost:8000/v1"  # "name"修改为fastchat服务中的"api_base_url"
    },
    "fastchat-qwen2-7b": {
        "name": "fastchat-llama",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "fastchat-qwen2-7b",
        "local_model_path": None,
        "provides": "FastChatQwen27bLLM",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLM"
        "api_base_url": "https://api.siliconflow.cn/v1"  # "name"修改为fastchat服务中的"api_base_url"
    },
}

# LLM 名称 把这个改了 chatglm-6b-int4
LLM_MODEL = "fastchat-qwen2-7b"
# 量化加载8bit 模型
LOAD_IN_8BIT = False
# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = False
# 本地lora存放的位置
LORA_DIR = "loras/"

# LLM lora path，默认为空，如果有请直接指定文件夹路径
LLM_LORA_PATH = ""
USE_LORA = True if LLM_LORA_PATH else False

# LLM streaming reponse
STREAMING = False

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
# PROMPT_TEMPLATE = """你是一个聊天机器人。我会给你一段信息，请你根据这些信息生成回答。我为你提供的信息：
# {context}

# 根据上述已知信息，请用简洁和专业的来回答我的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 。一个例子：我提出的问题为：绿地集团成立于哪一年。你的回答：答案：绿地集团成立于1998年；依据：[1]绿地集团是一所综合性的公司，成立于1998年。我的问题是：{question}"""
PROMPT_TEMPLATE_HISTORY = """我会给你一段信息，请你根据这些信息生成回答。我为你提供的信息：
{context}

根据上述已知信息，请用简洁和专业的来回答我的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。我的问题是：{question}"""
# PROMPT_TEMPLATE = """我会给你一段信息，请你根据这些信息生成回答。我为你提供的信息：
# {context}

# 根据上述已知信息，请用简洁和专业的来回答我的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。我的问题是：{question}"""
# PROMPT_DIVIDE_TASK_QWEN = """
# 设定：你是一个将复杂问题分解为子任务的专家。我会给你输入一个复杂问题，请你将其分解为若干具有顺序结构的子任务。
# 目标：将问题分解为解决该问题的若干子任务。
# 输入格式：问题。
# 输出格式：*1.子任务1*,*2.子任务2*,*3.子任务3*......
# 示例：
# 1)输入：“M7级别的职员去法国的报销标准”。输出：*1.查找M7级别职员去各级地区的报销标准*,*2.查找上海是几级地区*,*3.结合前两个子任务的答案，回答M7级别的职员去上海的报销标准*"""
PROMPT_DIVIDE_TASK_QWEN = """
设定：
您是一位专家，负责将复杂问题分解为有序的子任务。当我提出一个复杂问题时，请您按照以下格式将其分解为一系列子任务。

目标：
- 使用以下格式将复杂问题分解为子任务。

输入格式：
- 问题：[具体问题描述]

输出格式（请严格遵循）：
- 子任务列表：
  1. [子任务1描述]
  2. [子任务2描述]
  3. [子任务3描述]
  ...

示例1：
输入：
- 问题：“M7级别的职员去法国的报销标准是什么？”

输出：
- 子任务列表：
  1. 查找M7级别职员的一般报销标准。
  2. 确定法国属于哪一级别的地区。
  3. 结合前两个子任务的信息，确定M7级别职员去法国的具体报销标准。

示例2：
输入：
- 问题：“借款包含哪些？”
输出：
- 子任务列表：
  1. 查找投标保证金借款标准。
  2. 查找履约保证金借款标准。
  3. 查找租房押金标准。
  4. 查找差旅费借款标准。
  5. 查找公共费用类借款标准。
  6. 查找活动类借款标准。

示例3：
输入：
- 问题：“去西宁出差每天的差旅补贴？”
输出：
- 子任务列表：
  1. 判断西宁属于哪个省。
  2. 查找境内差旅补贴标准。


示例4：
输入：
- 问题：“我出差回来之后，应该什么时候去报销？”
输出：
- 子任务列表：
  1. 查找差旅费报销期限相关标准。
  2. 根据出差时间得出报销期限。

示例5：
输入：
- 问题：“发票丢了能报销吗？”
输出：
- 子任务列表：
  1. 查找个人支付的发票丢失的相关报销标准。

示例6：
输入：
- 问题：“我能否再借公司钱吗？”
输出：
- 子任务列表：
  1. 查找借款需遵循的基本原则。

示例7：
输入：
- 问题：“某借款的还款时间？”
输出：
- 子任务列表：
  1. 查找各种租金借款的基本原则(如房屋租金借款)。

示例8：
输入：
- 问题：“出差租房买XXX的钱可以报销吗？”
输出：
- 子任务列表：
  1. 查找租房时，每人可一次性报销的采购费用。

示例9：
输入：
- 问题：“租房的哪些费用能报销？”
输出：
- 子任务列表：
  1. 查找租房时，每人可以报销哪些费用。

示例10：
输入：
- 问题：“XX职级常驻XX的定额交通费？”
输出：
- 子任务列表：
  1. 查找交通费标准列表以及定额交通费标准。
  2. 判断是否常驻西藏和青海地区。
  3. 根据职级得出定额交通费。

示例11：
输入：
- 问题：“XX职级搭乘XX出差的标准”
输出：
- 子任务列表：
  1. 查找差旅交通工具标准。

示例12：
输入：
- 问题：“XX职级员工去XXX地区出差的酒店报销标准”
输出：
- 子任务列表：
  1. 判断XXX属于几类地区。
  2. 查找差旅标准列表中大陆地区差旅标准。
  3. 查找海外差旅标准。

示例13：
输入：
- 问题：“XX职级在XXX的定额交通费标准”
输出：
- 子任务列表：
  1. 判断XXX是几类地区。
  2. 查找交通费标准列表中的定额交通费标准。

示例14：
输入：
- 问题：“XX职级的通勤交通费标准”
输出：
- 子任务列表：
  1. 查找通勤交通费标准。

示例15：
输入：
- 问题：“XX是几类地区？”
输出：
- 子任务列表：
  1. 查找大陆地区城市分类设置标准。
  2. 查找海外差旅标准。

请注意，每个子任务后面都有一个句号，并且子任务之间没有额外的空行。

"""

PROMPT_DIVIDE_TASK = """
目标：将问题分解为解决该问题的若干子任务。
输入格式：问题。
输出格式：*1.子任务1*,*2.子任务2*,*3.子任务3*......
示例：
1)输入：“天津地区的补助是多少”。输出：*1.查找天津地区属于几级地区*,*2.每个地区对应的补助*
2)输入：“河北地区的补助是多少”。输出：*1.查找河北地区属于几级地区*,*2.每个地区对应的补助*
3)输入：{question}。输出:"""

PROMPT_COT = """
目标：根据输入的子任务及其相关资料，回答问题，请 step by step 进行思考，并给出推理过程。
输入格式：问题,*1.子任务1,参考资料:*,*2.子任务2,参考资料:*,*3.子任务3,参考资料*
输出格式：*答案*,*推理过程*
输入：问题是{question}，子任务是{chain}。输出："""
# PROMPT_COT_QW = """
# 设定：你是一个根据若干子任务及其参考资料，回答最终问题的专家。我会给你输入问题以及分解的子任务及其参考资料，请你根据子任务及其参考资料，step by step 进行思考，并给出推理过程，回答问题。
# 目标：根据子任务及其参考资料，step by step 进行思考，回答问题，并给出推理过程。
# 目标：根据输入的子任务及其相关资料，回答问题，请 step by step 进行思考
# 输入格式：问题是,子任务是*1.子任务1,参考资料:*,*2.子任务2,参考资料:*,*3.子任务3,参考资料*
# 输出格式：*答案*,*推理过程*"""

PROMPT_COT_QW_STEP = """
设定：
您是一位专家，负责根据问题，前序子任务及其答案，逐步思考并回答当前子任务。我将为您提供当前子任务以及参考资料、前序子任务以及相应的答案，请您依据这些信息，逐步推理并给出答案。

目标：
- 根据问题，前序子任务及其回答，当前子任务及其参考资料，逐步思考并回答当前子任务。
- 提供完整的推理过程。

输入格式：
- 问题是：[问题描述]
- 当前子任务是：[子任务描述]
- 当前子任务的参考资料：[资料]
- 前序子任务包括：
  1. 子任务1，答案：[答案1]
  2. 子任务2，答案：[答案2]
  3. 子任务3，答案：[答案3]

输出格式：
- 答案：[当前子任务答案]
- 推理过程：[详细推理步骤]

输入格式：
- 问题是：如何计算半径为2的圆的面积？
- 当前子任务是：找到圆的半径。
- 当前子任务的参考资料：圆面积的标准公式是 A = πr²。
- 前序子任务包括：
  1. 子任务1：找到计算圆面积的公式。答案：圆面积的标准公式是 A = πr²，其中 A 是面积，r 是半径，π 是圆周率。

输出格式：
- 答案：当前圆的半径为2。
- 推理过程：根据题目描述，当前圆的半径为2。

"""

PROMPT_COT_QW_STEP_FIN = """
设定：
您是一位专家，负责根据一系列由问题分解而成的所有子任务及其答案，综合思考并回答问题。我将为您提供所有由问题分解而成的子任务及其答案，请您依据这些信息，综合推理并给出最终答案。

目标：
- 根据所有子任务及其回答，综合思考并回答最终问题。
- 提供完整的推理过程。

输入格式：
- 问题是：[问题描述]
- 子任务及其答案包括：
  1. 子任务1，答案：[答案1]
  2. 子任务2，答案：[答案2]
  3. 子任务3，答案：[答案3]
  ...
  n. 子任务n，答案：[答案n]

输出格式：
- 答案：[问题答案]
- 推理过程：[详细推理步骤]

样例：
输入：
- 问题是：如何计算一个半径是5的圆的面积？
- 子任务及其答案包括：
  1. 子任务1：确定圆的半径。答案：圆的半径是5厘米。
  2. 子任务2：找到计算圆面积的公式。答案：找到计算圆面积的公式是 A = πr²。

输出格式：
- 答案：圆的面积是 78.5平方厘米。
- 推理过程：根据子任务1的答案，我们知道圆的半径是5厘米。根据子任务2的答案，我们知道计算圆面积的公式是 A = πr²。将半径5厘米代入公式中，我们得到 A = π * 5² = π * 25。使用π约等于3.14，我们计算得到圆的面积是 3.14 * 25 = 78.5平方厘米。

"""


PROMPT_COT_QW = """
设定：
您是一位专家，负责根据提供的子任务及其参考资料，逐步思考并回答最终问题。我将为您提供问题、子任务以及相应的参考资料，请您依据这些信息，逐步推理并给出答案。

目标：
- 根据子任务及其参考资料，逐步思考并回答问题。
- 提供完整的推理过程。

输入格式：
- 问题是：[问题描述]
- 子任务包括：
  1. 子任务1，参考资料：[资料1]
  2. 子任务2，参考资料：[资料2]
  3. 子任务3，参考资料：[资料3]

输出格式：
- 答案：[最终答案]
- 推理过程：[详细推理步骤]

样例：
输入：
- 问题是：地球到月球的距离是多少？
- 子任务包括：
  1. 子任务1，参考资料：NASA的官方网站提供的数据
  2. 子任务2，参考资料：天文学教科书中的相关章节
  3. 子任务3，参考资料：最近的科学期刊文章

输出：
- 答案：地球到月球的平均距离约为384,400公里。
- 推理过程：
  1. 根据NASA的官方网站，地球和月球之间的平均距离是384,400公里。
  2. 天文学教科书确认了这一数据，并解释了测量这一距离的方法。
  3. 最近的科学期刊文章讨论了地球和月球距离的微小变化，但平均值仍然是384,400公里。

"""

PROMPT_TEMPLATE = """我会给你一段信息，请你根据这些信息生成回答。
此外，我会提供给你一些回答特定任务的步骤。如果用户问"某地区的补助是多少"，你可以根据下面的步骤进行回答：
1)查找该地区属于几级地区。
2)根据该地区所属的地区查找对应的补助。
例如，用户提问"天津地区的补助是多少",你的思考步骤如下:
1)因为天津不属于一级地区以及二级地区所列的城市，所以属于三级地区。
2)根据信息，三级地区的补助为200，所以最终答案是200。

我为你提供的信息：
{context}

根据上述已知信息，请用简洁和专业的来回答我的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。我的问题是：{question}"""

'''
这里是回答该问题的相关信息，请首先提供你的解释，然后用“因此，答案是”之后加上你的答案。
[{“人员类别”:“董事会成员”，“香港/澳门(港币)”:“实报实销”,“新加坡(新币)”:“实报实销”，“马来西亚(马币)”:“实报实销”,“台湾(新台币)":“实报实销”，“海外A类和8类地区(美元)”:“实报实销”，“海外c类地区(美元)”:“实报实销”，{“人员类别":“/P8及以上”,“香港/澳门(港币)”:“1,5”,“新加坡(新币)”:“28”,“马来西亚(马币)”:“3”，“台湾(新台币)”:“2,580”，“海外A类和B类地区(美元)”:“128”,“海外(类地区(美元)”:“100”)，{“人员类别”:“其他员工”，“香港/澳门(港币)”:“1,088”,“新加坡(新币)":"150”，“马来西亚(马币)”:“280”，“台湾(新台币)”:“2,200”,“海外A类和B类地区(美元)":“100”,“海外c类地区(美元)":“88”}]海外A类地区:美国、日本、韩国、英国、德国、瑙土、法围。海外B类地区:北欧其它国家(已列入一类地区欧洲田家之外)、加拿大、香港、澳门、澳大利亚、新加坡、台湾、新西兰、俄罗斯。 海外类地区;亚洲其它国家(已列入海外A类和B类地区的亚洲国家和地区之外)、东欧国家、非洲国家、南美国家。问题:M7标准的职员去上海的住宿标准为多少?解释:法国属于海外A类地区，同时M4/P8及以上职员去海外A类和B类地区的住宿标准为128美元，因此，答案是128美元。
这里是回答该问题的相关信息，请首先提供你的解释，然后用“因此，答案是”之后加上你的答案。“人员类别”:“董事会成员”,“香港/澳门(港币)”:“实报实销”,“新加坡(新币)”:“实报实销”,“马来西亚(马币)”:“实报实销”，“台湾(新台币)”:“实报实销”，“海外A类和B类地区 (美元)”:“实报实销”，“海外(类地区(美元)”:“实报实销”，{"人员类别":"/P8及以上”，“香港/澳门(港币)":“1,580”,“新加坡(新币)":“28”，“马来西亚(马币)":“350”，“台湾(新台币)”:“2,50”，“海外A类和B类地区(美元)”:“120”,“海外(类地区(美元)”:“188”，““人员类别":“其他员工”,“香港/澳门(港币)”:“1,00”,“新加坡(新币)”:“15”,“马来西亚 (马币)”:“200”，“台湾(新台币)”:“2,20”,“海外A类和B类地区(美元)”:“108”,“海外c类地区(美元)”:“88”}]海外A类地区;美国、日本、韩田、英田、德田、瑞士、法国。 海外B类地区:北欧其它国家(已列入一类地区欧洲国家之外)、加拿大、香港、澳门、澳大利亚、新加坡、台湾、新西兰、俄罗斯。 海外c类地区:亚洲其它国家(已列入海外A类和B类地区的亚洲国家和地区之外)、东欧田家、非洲国家、南美国家。问题:董事会成员去德国的住宿标准为多少?解释:德国属于海外A类地区，同时董事会成员去海外A类和B类地区的住宿标准为实报实销，因此，答案是实报实销。
这里是回答该问题的相关信息，请首先提供你的解释，然后用“因此，答案是”之后加上你的答案(table[{"城市分类":"一类地区”，"包含":“北京、上海、深圳、广州”，{“城市分类”:"二类地区”，"包含”:"直辖市、省会城市及大连、青岛、苏州、无锡、宁波、桂林、珠海、厦门、三亚、东先”}，{"城市分类":"三类地区"，"包含":"除一二类地区以外的国内其他城市"}]</table>”<table>[{"报销标准分类":"一类”，“职级对应":"集团董事会成员”，"交通费 一类地区”:"1000","交通费 二类地区”:"1000"，"交通费 三类地区”:"1000")，《"报销标准分类”:"二类"，"职级对应”:"M8-M18 /P12及以上”，"交通费 一类地区":"1808”“交通费 二类地区":“1600”,"交通费 三类地区":"1000”}，{"报销标准分类":"三类”，"职級对应":"7 /P11”，"交通费，一类地区”:“100”,“交通费 二类地区”:"100”,“交通费 三类地区":"180”)，{“报销标准分类":"四类”，"职级对应”:"H6/P10及一级部门负责人”，“交通费 一类地区”:"108”，“交通费 二类地区":"108”，“交通费 三类地区”:"100”)，“报销标准分类”:“五类”,，"职级对应”:“H5/P9及二级部门负责人”，“交通费 一类地区":“60”，“交通费 二类地区”:“6”，“交通费 三类地区”:“60，("报销标准分类":"六类”，"职级对应”:"H3-M/P7-P8及三级部门负责人”，"交通费 一类地区":“45”,“交通费 二类地区":“40”，“交通费 三类地区”:“350”)，{“报销标准分类":"七"，"职级对应”:"其他级别的营销序列和专业服务序列"，，"交通费 一类地区”:"≤450”,"交通费 二类地区":"<400”,"交通费 三类地区":"s350”}，{”报销标准分类":"八类”，"职级对应":"其他人员”，“交通费 一类地区":“≤400”，"交通费 二类地区":"≤250"，"交通费 三类地区”:“≤200"}]</table>问题:M7职级去北京的交通费标准为多少?
'''
# PROMPT_TEMPLATE = """
# 你是回答问题机器人，需要根据提供的信息，生成有依据的回答。回答包括"答案"、"依据"两部分。
# "答案"是你根据提供的信息回答的完整的句子。"依据"是所提供的信息中与问题最相关的片段，不能和答案相同，必须是提供的信息的原文。
# 输出格式：
# <
# 答案:
# 依据:
# >

# 样例1:
# <
# 提供的信息：
# 据雅生活招股说明书， 绿地控股从2018年到2022年会尽可能聘请雅生活作为物业管理服务的供应商， 每年交付建筑面积不少于700万平方米， 并且每年额外开发的 300万平方米物业的物业服务商选择上也给予雅生活优先权。 绿地控股 2015-2018 年年平均竣工面积约为 2047 万平。绿地集团是一所综合性的公司，成立于1998年。公司为雅居乐物业子公司，在管面积主要来源于雅居乐销售交付的住宅物业.
# 问题是：
# 绿地集团成立于哪一年?

# 模型输出:
# 答案:绿地集团成立于1998年。
# 依据:绿地集团是一所综合性的公司，成立于1998年。公司为雅居乐物业子公司，在管面积主要来源于雅居乐销售交付的住宅物业.
# >

# 样例2:
# <
# 提供的信息：
# 绿地集团是一家成立于1992年的综合性公司，总部位于上海。公司为雅居乐物业子公司，在管面积主要来源于雅居乐销售交付的住宅物业。截至2020年2月，绿地集团广泛布局中国、美国、澳大利亚、加拿大、英国、德国、日本、韩国、马来西亚、柬埔寨、越南等国家，并在管项目1031个，覆盖全国25个省、直辖市和自治区，83个城市。绿地集团的主营业务包括物业管理服务、业主增值服务和非业主增值服务，其中物业管理服务为公司收入的主要来源。
# 问题是：
# 绿地集团总部位于哪里?

# 模型输出:
# 答案:绿地集团总部位于上海。
# 依据:绿地集团是一家成立于1992年的综合性公司，总部位于上海。公司为雅居乐物业子公司，在管面积主要来源于雅居乐销售交付的住宅物业。
# >

# 提供的信息：
# <
# {context}
# >
# 问题是：
# <
# {question}
# >

# 请注意，你的输出一定要包含"答案"以及"依据"两部分，并且"依据"必须是提供的信息的原文。
# """

# PROMPT_TEMPLATE1 = """
# 从我提供给你的文档，生成能够回答我给你的问题的背景信息。文档如下：。问题如下：{question}。
# """

# {"type": "question answering", "task": "step1", "pid": 1, "prompt": "Provide a background document from Wikipedia to answer the given question. \n\n {query} \n\n"}
# {"type": "question answering", "task": "step1", "pid": 2, "prompt": "Generate a background document from Wikipedia to answer the given question. \n\n {query} \n\n"}
# {"type": "question answering", "task": "step2", "pid": 1, "prompt": "Refer to the passage below and answer the following question with just one entity. \n\n Passage: {background} \n\n Question: {query} \n\n The answer is"}

# PROMPT_TEMPLATE = """<指令>你是回答问题机器人，需要根据提供的信息，生成回答，并在回答中的每一句话都标注出和这句话最相关的已知信息的序号 </指令>
# <样例>已知信息:[1]绿地集团是一家成立于1992年的综合性公司，总部位于上海。
# [2]公司为雅居乐物业子公司，在管面积主要来源于雅居乐销售交付的住宅物业。截至2020年2月，绿地集团广泛布局中国、美国、澳大利亚、加拿大、英国、德国、日本、韩国、马来西亚、柬埔寨、越南等国家，并在管项目1031个。
# [3]覆盖全国25个省、直辖市和自治区，83个城市。绿地集团的主营业务包括物业管理服务、业主增值服务和非业主增值服务，其中物业管理服务为公司收入的主要来源。
# 问题：
# 请介绍绿地集团?
# 回答问题机器人的思考过程：
# 1.首先，生成答案：绿地集团是一家成立于1992年的综合性公司，总部位于上海。公司在管项目1031个。其中物业管理服务为公司收入的主要来源。
# 2.然后，第一句话"绿地集团是一家成立于1992年的综合性公司，总部位于上海"和用户输入的已知信息[1]最相关。
# 3.然后，第二句话"公司在管项目1031个"和用户输入的已知信息[2]最相关。
# 4.然后，第三句话"其中物业管理服务为公司收入的主要来源"和用户输入的已知信息[3]最相关。
# 5.因此，最后将每句话最相关的已知信息嵌入到原始答案，得到最终输出：绿地集团是一家成立于1992年的综合性公司，总部位于上海[1]。公司在管项目1031个[2]。其中物业管理服务为公司收入的主要来源[3]。
# 6.回答问题机器人最终输出：绿地集团是一家成立于1992年的综合性公司，总部位于上海[1]。公司在管项目1031个[2]。其中物业管理服务为公司收入的主要来源[3]。
# </样例>
# <样例>已知信息:[1]绿地集团是一家成立于1992年的综合性公司，总部位于上海。
# [2]公司为雅居乐物业子公司，在管面积主要来源于雅居乐销售交付的住宅物业。截至2020年2月，绿地集团广泛布局中国、美国、澳大利亚、加拿大、英国、德国、日本、韩国、马来西亚、柬埔寨、越南等国家，并在管项目1031个。
# [3]覆盖全国25个省、直辖市和自治区，83个城市。绿地集团的主营业务包括物业管理服务、业主增值服务和非业主增值服务，其中物业管理服务为公司收入的主要来源。
# 问题：
# 绿地集团成立于哪一年？
# 回答问题机器人的思考过程：
# 1.首先，生成答案：绿地集团成立于1992年。
# 2.然后，第一句话"绿地集团成立于1992年"和用户输入的已知信息[1]最相关。
# 3.因此，最后将每句话最相关的已知信息嵌入到原始答案，得到最终输出：绿地集团成立于1992年[1]。
# 4.回答问题机器人最终输出：绿地集团成立于1992年[1]。
# </样例>
# <已知信息>{context}</已知信息>

# <问题>{question}</问题>回答问题机器人的思考过程："""

# 缓存知识库数量
CACHED_VS_NUM = 1

# 文本分句长度
SENTENCE_SIZE = 500

# 匹配后单段上下文长度
CHUNK_SIZE = 500

# 传入LLM的历史记录长度
# LLM_HISTORY_LEN = 3
LLM_HISTORY_LEN = 3

# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 3

# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

FLAG_USER_NAME = uuid.uuid4().hex

logger.info(f"""
loading model config
llm device: {LLM_DEVICE}
embedding device: {EMBEDDING_DEVICE}
dir: {os.path.dirname(os.path.dirname(__file__))}
flagging username: {FLAG_USER_NAME}
""")

# 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False

# Bing 搜索必备变量
# 使用 Bing 搜索需要使用 Bing Subscription Key,需要在azure port中申请试用bing search
# 具体申请方式请见
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource
# 使用python创建bing api 搜索实例详见:
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
# 注意不是bing Webmaster Tools的api key，

BING_SUBSCRIPTION_KEY = ""