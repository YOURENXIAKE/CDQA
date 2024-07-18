import os
from tqdm import tqdm
os.environ['NUMEXPR_MAX_THREADS'] = '16' # 核心数
# Show reply with source text from input document
REPLY_WITH_SOURCE = False
from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
import re
# def divide_task_divide_resp(resp):
#     pattern = re.compile(r'\*(.+)\*')
#     match_obj = pattern.findall(resp)
#     print(match_obj)
#     if len(match_obj) == 0:
#         return None
#     else:
#         ret_list = []
#         for i in range(0,len(match_obj)):
#             ret_list.append(match_obj[i].replace('*',''))
#         return ret_list

def divide_task_divide_resp(resp):
    tasks = re.findall(r'\d+\.\s(.*?。)', resp)
    return tasks

def main():

    llm_model_ins = shared.loaderLLM()
    llm_model_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    vs_path = None
    while not vs_path:
        # filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
        # 知识文件路径 更改 测试一下一个house_data, 一个房地产领域的知识文件
        filepath = "/home/zrchen/yongyou/pdfminer.txt"

        # 判断 filepath 是否为空，如果为空的话，重新让用户输入,防止用户误触回车
        if not filepath:
            continue
        vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath)
    history = []
    tokenizer = local_doc_qa.llm.checkPoint.tokenizer

    ###
    # test = ['报销的基本原则？','福利费报销需要什么发票？','什么情况下，抬头不是单位全称的发票可以报销？','香港是几类地区？','菲律宾是几类地区？','海淀区是几类地区？']
    test = ['报销的基本原则？','福利费报销需要什么发票？','什么情况下，抬头不是单位全称的发票可以报销？','香港是几类地区？','菲律宾是几类地区？','海淀区是几类地区？','去日本的差旅补贴标准？','去西宁出差每天的差旅补贴？','我出差时间是3个月，应该什么时候去报销？','我自己打车的发票丢了能报销吗？','我上次找公司借的钱没还，还能再借吗？','办公场地租赁发生借款的还款时间？','出差租房买被子的钱可以报销吗？','租房的哪些费用能报销？','咨询费报销的要求？','M5职级常驻拉萨的定额交通费？','人力资源部门M5职级员工的通讯费？','报销舞弊会罚多少钱？','M9职级搭乘飞机出差的标准？','P13职级员工搭乘火车出差的标准？','M8职级员工去上海出差的酒店报销标准？','M5职级员工去香港出差的酒店报销标准？','一人多职时的报销标准？','出差乘坐火车哪些情况下可以购买卧铺？','M5职级在上海的定额交通费标准?','一类地区的通勤交通费标准?']
    ###

    # while True:
    for q in tqdm(test):
        # query = input("Input your question 请输入问题：")
        query = q
        last_print_len = 0

        # first step：规化任务
        for resp in local_doc_qa.get_knowledge_based_answer_divide_task(query=query,
                                                                     vs_path=vs_path,
                                                                     chat_history=history,
                                                                     streaming=False):
                print(resp)
        ret_list = divide_task_divide_resp(resp)        


        for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                     vs_path=vs_path,
                                                                     chat_history=history,ret_list = ret_list,
                                                                     streaming=STREAMING):
            if STREAMING:
                print(resp["result"][last_print_len:], end="", flush=True)
                last_print_len = len(resp["result"])
            else:
                print(resp["result"])
        if REPLY_WITH_SOURCE:
            source_text = [f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                           # f"""相关度：{doc.metadata['score']}\n\n"""
                           for inum, doc in
                           enumerate(resp["source_documents"])]
            print("\n\n" + "\n\n".join(source_text))


if __name__ == "__main__":
    # 通过cli.py调用cli_demo时需要在cli.py里初始化模型，否则会报错：
    # langchain-ChatGLM: error: unrecognized arguments: start cli
    # 为此需要先将
    # args = None
    # args = parser.parse_args()
    # args_dict = vars(args)
    # shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    # 语句从main函数里取出放到函数外部
    # 然后在cli.py里初始化
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    main()
