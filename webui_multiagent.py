import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
import gradio as gr
from gradio_kmpdf import kmpdf
import shutil

from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import re
from AgentSystem import AgentSystem
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
def divide_task_divide_resp(resp):
    tasks = re.findall(r'\d+\.\s(.*?。)', resp)
    return tasks

def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst


embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()

flag_csv_logger = gr.CSVLogger()

def get_answer_pdf_single(query, vs_path, history, mode, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
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
            relate_docs = resp["relate_docs"]
            answer = resp["result"]
    source = "\n\n"
    source += "".join(
        [f"""<details> <summary>子任务 [{i + 1}] {re_k}</summary>\n"""
            f"""{relate_docs[i]}\n"""
            f"""</details>"""
            for i, re_k in
            enumerate(ret_list)])
    history[-1][-1] += source            
    yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)


def get_answer_pdf_mt(query, vs_path, history, mode, score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    agent_system = AgentSystem(local_doc_qa,vs_path) 
    for q in [query]:
      try:
        agent_system.clear_all()
        # query = input("Input your question 请输入问题：")
        query = q
        last_print_len = 0
        # first step: Multi-Agent Table Reasoning
        while True:
          first_step_ans = agent_system.process_task(query)
          print(first_step_ans)
          if 'EPC' in first_step_ans:
            ans = None
            # # 进入第二步
            # query = agent_system.reson_agent.process_output(first_step_ans)
            # print(query)
            resp = local_doc_qa.get_knowledge_based_answer_divide_task_agent(q,agent_system.single_qa_agent)
            print(resp)
            tasks = re.findall(r'\d+\.\s(.*?。)', resp)
            ret_list = tasks
            resp = local_doc_qa.get_knowledge_based_answer_agent(q, vs_path, agent_system.step2_agent,ret_list = ret_list)
            ans = resp["result"]
            print(ans)
            break
          else:
            print(first_step_ans)
            break
      except Exception as e:
        ans = None
        resp = local_doc_qa.get_knowledge_based_answer_divide_task_agent(q,agent_system.single_qa_agent)
        print(resp)
        tasks = re.findall(r'\d+\.\s(.*?。)', resp)
        ret_list = tasks
        resp = local_doc_qa.get_knowledge_based_answer_agent(q, vs_path, agent_system.step2_agent,ret_list = ret_list)
        ans = resp["result"]
        print(ans)
    yield ans, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)

# [query, vs_path, chatbot,multi_task_chatbot,multi_agent_qa,mix_qa, mode,stg_display],
# [chatbot,multi_task_chatbot,multi_agent_qa,mix_qa, query]
def get_answer(query, vs_path, main_chatbot,multi_task_chatbot,multi_agent_qa,mix_qa, mode, stg_display,score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE, streaming: bool = STREAMING):
    if mode == "知识库问答" and vs_path is not None and os.path.exists(vs_path) and "index.faiss" in os.listdir(
            vs_path):
        if "任务分解" in stg_display:
            main_chatbot.append([None,"任务分解策略已开启，右侧为您生成任务分解策略回答"])
            for resp in local_doc_qa.get_knowledge_based_answer_divide_task(query=query,
                                                                            vs_path=vs_path,
                                                                            chat_history=[],
                                                                            streaming=False):
                multi_task_chatbot.append(['第一步:任务分解',resp])
                print(resp)
                ret_list = divide_task_divide_resp(resp)
            for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                        vs_path=vs_path,
                                                                        chat_history=[],ret_list = ret_list,
                                                                        streaming=STREAMING):
                # if STREAMING:
                #     print(resp["result"][last_print_len:], end="", flush=True)
                #     last_print_len = len(resp["result"])
                # else:
                relate_docs = resp["relate_docs"]
                answer = resp["result"]
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>子任务 [{i + 1}] {re_k}</summary>\n"""
                    f"""{relate_docs[i]}\n"""
                    f"""</details>"""
                    for i, re_k in
                    enumerate(ret_list)])
            multi_task_chatbot.append(['第二步:多跳推理',answer])
            multi_task_chatbot.append(['推理依据',source])
            # history[-1][-1] += source       
        if "多智能体" in stg_display:
            main_chatbot.append([None,"多智能体策略已开启，右侧为您生成多智能体策略回答"])
            agent_system = AgentSystem(local_doc_qa,vs_path) 
            for q in [query]:
                try:
                    agent_system.clear_all()
                    # query = input("Input your question 请输入问题：")
                    query = q
                    last_print_len = 0
                    # first step: Multi-Agent Table Reasoning
                    first_step_ans,multi_agent_qa = agent_system.process_task(query,multi_agent_qa)
                    print(first_step_ans)
                    if 'EPC' in first_step_ans:
                        multi_agent_qa.append([None,"出现EPC,此策略无法回答问题"])
                except Exception as e:
                        multi_agent_qa.append([None,str(e)])
        if "混合方案" in stg_display:
            main_chatbot.append([None,"混合方案已开启，右侧为您生成多智能体策略回答"])
            agent_system = AgentSystem(local_doc_qa,vs_path) 
            for q in [query]:
                try:
                    agent_system.clear_all()
                    # query = input("Input your question 请输入问题：")
                    query = q
                    last_print_len = 0
                    # first step: Multi-Agent Table Reasoning
                    first_step_ans,mix_qa = agent_system.process_task(query,mix_qa)
                    print(first_step_ans)
                    if 'EPC' in first_step_ans:
                        mix_qa.append([None,"出现EPC,尝试进行混合方案"])
                        ans = None
                        resp = local_doc_qa.get_knowledge_based_answer_divide_task_agent(q,agent_system.single_qa_agent)
                        print(resp)
                        tasks = re.findall(r'\d+\.\s(.*?。)', resp)
                        ret_list = tasks
                        resp = local_doc_qa.get_knowledge_based_answer_agent(q, vs_path, agent_system.step2_agent,ret_list = ret_list)
                        ans = resp["result"]
                        mix_qa.append(['混合方案回答',ans])
                        break
                except Exception as e:
                    mix_qa.append([None,f"多智能体回答失效{e},尝试进行混合方案"])
                    ans = None
                    resp = local_doc_qa.get_knowledge_based_answer_divide_task_agent(q,agent_system.single_qa_agent)
                    print(resp)
                    tasks = re.findall(r'\d+\.\s(.*?。)', resp)
                    ret_list = tasks
                    resp = local_doc_qa.get_knowledge_based_answer_agent(q, vs_path, agent_system.step2_agent,ret_list = ret_list)
                    ans = resp["result"]
                    print(ans)
                    mix_qa.append(['混合方案回答',ans])
    return main_chatbot,multi_task_chatbot,multi_agent_qa,mix_qa, query


def init_model():
    args = parser.parse_args()

    args_dict = vars(args)
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)
    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        # generator = local_doc_qa.llm.generatorAnswer(prompt="你好", history=None,
        #                                                       streaming=False)
        # for answer_result in generator:
        #     print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        print(e)
        logger.error(e)
        reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply 


def reinit_model(llm_model, embedding_model, llm_history_len, no_remote_model, top_k,
                 history):
    use_ptuning_v2 = False
    use_lora = False
    try:
        logger.info(llm_model)
        llm_model_ins = shared.loaderLLM(llm_model, no_remote_model, use_ptuning_v2)
        llm_model_ins.history_len = llm_history_len
        local_doc_qa.init_cfg(llm_model=llm_model_ins,
                              embedding_model=embedding_model,
                              top_k=top_k)
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(model_status)
    except Exception as e:
        logger.error(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.info(model_status)
    return history + [[None, model_status]]

def get_vector_store_pdf(files, sentence_size, history, one_conent, one_content_segmentation):
    files = [files]
    vs_name = add_vs_name_pdf(chatbot_pdf)
    print(vs_name)
    vs_path = os.path.join(KB_ROOT_PATH, vs_name, "vector_store")
    filelist = []
    print(local_doc_qa)
    print(local_doc_qa.llm)
    print(local_doc_qa.embeddings)
    if local_doc_qa.llm and local_doc_qa.embeddings:
        if isinstance(files, list):
            print(2)
            for file in files:
                filename = os.path.split(file)[-1]
                # shutil.move(file, os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
                filelist.append(file)
                # filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
            print(filelist)
            print(vs_path)
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else:
            print(1)
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        print(file_status)
        vs_path = None
    logger.info(file_status)
    print(vs_path)
    return vs_path, files[0], history + [[None, file_status]], \
           gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path) if vs_path else [])
def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    filelist = []
    print(local_doc_qa)
    print(local_doc_qa.llm)
    print(local_doc_qa.embeddings)
    if local_doc_qa.llm and local_doc_qa.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
                filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else:
            vs_path, loaded_files = local_doc_qa.one_knowledge_add(vs_path, files, one_conent, one_content_segmentation,
                                                                   sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]], \
           gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path) if vs_path else [])

def change_vs_name_input_pdf(history):
    vs_id = "阿斯顿"
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history,\
                gr.update(choices=[]), gr.update(visible=False)
    else:
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        if "index.faiss" in os.listdir(vs_path):
            file_status = f"已加载知识库{vs_id}，请开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), \
                   gr.update(visible=True)
        else:
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请先上传文件后，再开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=[], value=[]), gr.update(visible=True, value=[])

def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history,\
                gr.update(choices=[]), gr.update(visible=False)
    else:
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        if "index.faiss" in os.listdir(vs_path):
            file_status = f"已加载知识库{vs_id}，请开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), \
                   gr.update(visible=True)
        else:
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请先上传文件后，再开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], \
                   gr.update(choices=[], value=[]), gr.update(visible=True, value=[])


knowledge_base_test_mode_info = ("【注意】\n\n"
                                 "以下是系统输出")

def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=False), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    elif mode == "知识库测试":
        return gr.update(visible=True), gr.update(visible=True), [[None,
                                                                   knowledge_base_test_mode_info]]
    else:
        return gr.update(visible=False), gr.update(visible=False), history

def change_stg(mode, history):
    td = False
    ma = False
    mx = False
    if '任务分解' in mode:
        td = True
    if '多智能体' in mode:
        ma = True
    if '混合方案' in mode:
        mx = True
    history.append([None,f"您的设置:\n任务分解:{td}\多智能体:{ma}\混合方案:{mx}\n"])
    return gr.update(visible=td), gr.update(visible=ma),gr.update(visible=mx), history

def change_chunk_conent(mode, label_conent, history):
    conent = ""
    if "chunk_conent" in label_conent:
        conent = "搜索结果上下文关联"
    elif "one_content_segmentation" in label_conent:  # 这里没用上，可以先留着
        conent = "内容分段入库"

    if mode:
        return gr.update(visible=True), history + [[None, f"【已开启{conent}】"]]
    else:
        return gr.update(visible=False), history + [[None, f"【已关闭{conent}】"]]


def add_vs_name(vs_name, chatbot):
    if vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "content")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "content"))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "vector_store")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "vector_store"))
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_name), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot, gr.update(visible=True)
def add_vs_name_pdf(chatbot):
    from datetime import datetime

    now = datetime.now()
    timestamp = now.timestamp()
    vs_name = 'pdf_test' + str(timestamp)
    if vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        print(vs_status)
        return chatbot
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "content")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "content"))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "vector_store")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "vector_store"))
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        print(vs_status)
        return vs_name

# 自动化加载固定文件间中文件
def reinit_vector_store(vs_id, history):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id, "vector_store"))
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                  label="文本入库分句长度限制",
                                  interactive=True, visible=True)
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(os.path.join(KB_ROOT_PATH, vs_id, "content"),
                                                                         vs_path, sentence_size)
        model_status = """知识库构建成功"""
    except Exception as e:
        logger.error(e)
        model_status = """知识库构建未成功"""
        logger.info(model_status)
    return history + [[None, model_status]]


def refresh_vs_list():
    return gr.update(choices=get_vs_list()), gr.update(choices=get_vs_list())

def delete_file(vs_id, files_to_delete, chatbot):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    content_path = os.path.join(KB_ROOT_PATH, vs_id, "content")
    docs_path = [os.path.join(content_path, file) for file in files_to_delete]
    status = local_doc_qa.delete_file_from_vector_store(vs_path=vs_path,
                                                        filepath=docs_path)
    if "fail" not in status:
        for doc_path in docs_path:
            if os.path.exists(doc_path):
                os.remove(doc_path)
    rested_files = local_doc_qa.list_file_from_vector_store(vs_path)
    if "fail" in status:
        vs_status = "文件删除失败。"
    elif len(rested_files)>0:
        vs_status = "文件删除成功。"
    else:
        vs_status = f"文件删除成功，知识库{vs_id}中无已上传文件，请先上传文件后，再开始提问。"
    logger.info(",".join(files_to_delete)+vs_status)
    chatbot = chatbot + [[None, vs_status]]
    return gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), chatbot


def delete_vs(vs_id, chatbot):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id))
        status = f"成功删除知识库{vs_id}"
        logger.info(status)
        chatbot = chatbot + [[None, status]]
        return gr.update(choices=get_vs_list(), value=get_vs_list()[0]), gr.update(visible=True), gr.update(visible=True), \
               gr.update(visible=False), chatbot, gr.update(visible=False)
    except Exception as e:
        logger.error(e)
        status = f"删除知识库{vs_id}失败"
        chatbot = chatbot + [[None, status]]
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=True), chatbot, gr.update(visible=True)


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 知识库系统
"""
default_vs = get_vs_list()[0] if len(get_vs_list()) > 1 else "为空"
init_message = f"""欢迎使用知识库系统
"""

# 初始化消息
model_status = init_model()

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    vs_path,vs_path_pdf, file_status, model_status = gr.State(
        os.path.join(KB_ROOT_PATH, get_vs_list()[0], "vector_store") if len(get_vs_list()) > 1 else ""),gr.State(
        os.path.join(KB_ROOT_PATH, "阿斯顿", "vector_store")), gr.State(""), gr.State(
        model_status)
    gr.Markdown(webui_title)
    with gr.Tab("知识库问答"):
        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False,height=600)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交",container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["LLM 对话", "知识库问答"],
                                label="请选择使用模式",
                                value="知识库问答", )
                stg_display = gr.CheckboxGroup(["任务分解", "多智能体", "混合方案"], label="求解策略选择", info="勾选求解策略")

                multi_task_qa = gr.Accordion("任务分解回答", visible=False)
                multi_agent_qa = gr.Accordion("多智能体回答", visible=False)
                mix_qa = gr.Accordion("混合方案回答", visible=False)
                knowledge_set = gr.Accordion("知识库设定", visible=False)
                vs_setting = gr.Accordion("配置知识库")
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chatbot])
                stg_display.change(fn=change_stg,inputs=[stg_display, chatbot],
                            outputs=[multi_task_qa, multi_agent_qa,mix_qa, chatbot])
                with multi_task_qa:
                    multi_task_chatbot = gr.Chatbot([[None, "这里是任务分解的回答"]],
                    elem_id="chat-box-mt",
                    show_label=False,height=400)
                with multi_agent_qa:
                    multi_agent_chatbot = gr.Chatbot([[None, "这里是多智能体的回答"]],
                    elem_id="chat-box-ma",
                    show_label=False,height=400)
                with mix_qa:
                    mix_chatbot = gr.Chatbot([[None, "这里是混合方案的回答"]],
                    elem_id="chat-box-mx",
                    show_label=False,height=400)
                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
                                            )
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    vs_delete = gr.Button("删除本知识库", visible=False)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg', ".csv"],
                                            file_count="multiple",
                                            show_label=False)
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("删除文件"):
                            files_to_delete = gr.CheckboxGroup(choices=[],
                                                             label="请从知识库已有文件中选择要删除的文件",
                                                             interactive=True)
                            delete_file_button = gr.Button("从知识库中删除选中文件")
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, chatbot],
                                 outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                    vs_delete.click(fn=delete_vs,
                                    inputs=[select_vs, chatbot],
                                    outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                    select_vs.change(fn=change_vs_name_input,
                                     inputs=[select_vs, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot, files_to_delete, vs_delete])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot, files_to_delete], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot, files_to_delete], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_answer,
                                 [query, vs_path, chatbot,multi_task_chatbot,multi_agent_chatbot,mix_chatbot, mode,stg_display],
                                 [chatbot,multi_task_chatbot,multi_agent_chatbot,mix_chatbot, query])
                    delete_file_button.click(delete_file,
                                             show_progress=True,
                                             inputs=[select_vs, files_to_delete, chatbot],
                                             outputs=[files_to_delete, chatbot])
        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot([[None, knowledge_base_test_mode_info]],
                                     elem_id="chat-box",
                                     show_label=False,height=600)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交",container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["知识库测试"],  # "知识库问答",
                                label="请选择使用模式",
                                value="知识库测试",
                                visible=False)
                knowledge_set = gr.Accordion("知识库设定", visible=True)
                vs_setting = gr.Accordion("配置知识库", visible=True)
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chatbot])
                with knowledge_set:
                    score_threshold = gr.Number(value=VECTOR_SEARCH_SCORE_THRESHOLD,
                                                label="知识相关度 Score 阈值，分值越低匹配度越高",
                                                precision=0,
                                                interactive=True)
                    vector_search_top_k = gr.Number(value=VECTOR_SEARCH_TOP_K, precision=0,
                                                    label="获取知识库内容条数", interactive=True)
                    chunk_conent = gr.Checkbox(value=False,
                                               label="是否启用上下文关联",
                                               interactive=True)
                    chunk_sizes = gr.Number(value=CHUNK_SIZE, precision=0,
                                            label="匹配单段内容的连接上下文后最大长度",
                                            interactive=True, visible=False)
                    chunk_conent.change(fn=change_chunk_conent,
                                        inputs=[chunk_conent, gr.Textbox(value="chunk_conent", visible=False), chatbot],
                                        outputs=[chunk_sizes, chatbot])
                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs_test = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None)
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加单条内容或文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf'],
                                            file_count="multiple",
                                            show_label=False
                                            )
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("添加单条内容"):
                            one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题", lines=1)
                            one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容", lines=5)
                            one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
                                                                   interactive=True)
                            load_conent_button = gr.Button("添加内容并加载知识库")
                    # 将上传的文件保存到content文件夹下,并更新下拉框
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs_test)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, chatbot],
                                 outputs=[select_vs_test, vs_name, vs_add, file2vs, chatbot])
                    select_vs_test.change(fn=change_vs_name_input,
                                     inputs=[select_vs_test, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs_test, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs_test, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot], )
                    load_conent_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs_test, one_title, sentence_size, chatbot,
                                                     one_conent, one_content_segmentation],
                                             outputs=[vs_path, files, chatbot], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode, score_threshold, vector_search_top_k, chunk_conent,
                                  chunk_sizes],
                                 [chatbot, query])
    with gr.Tab("PDF问答"):
        with gr.Row():
            with gr.Column(scale=5):
                chatbot_pdf = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                        elem_id="chat-box",
                                        show_label=False,height=750)
                query_pdf = gr.Textbox(show_label=False,
                                    placeholder="请输入提问内容，按回车进行提交",container=False)
                query_pdf.submit(get_answer_pdf_single,
                                    [query_pdf, vs_path_pdf, chatbot_pdf, mode],
                                    [chatbot_pdf, query_pdf])
            with gr.Column(scale=10):
                pdf = kmpdf(label="Upload a PDF", interactive=True)
                pdf.upload(get_vector_store_pdf,
                        show_progress=True,
                        inputs=[pdf, sentence_size, chatbot_pdf, vs_add, vs_add],
                        outputs=[vs_path_pdf, pdf, chatbot_pdf, files_to_delete])
                        # chatbot_pdf = None

    with gr.Tab("模型配置"):
        llm_model = gr.Radio(llm_model_dict_list,
                             label="LLM 模型",
                             value=LLM_MODEL,
                             interactive=True)
        no_remote_model = gr.Checkbox(shared.LoaderCheckPoint.no_remote_model,
                                      label="加载本地模型",
                                      interactive=True)

        llm_history_len = gr.Slider(0, 10,
                                    value=LLM_HISTORY_LEN,
                                    step=1,
                                    label="LLM 对话轮数",
                                    interactive=True)
        embedding_model = gr.Radio(embedding_model_dict_list,
                                   label="Embedding 模型",
                                   value=EMBEDDING_MODEL,
                                   interactive=True)
        top_k = gr.Slider(1, 20, value=VECTOR_SEARCH_TOP_K, step=1,
                          label="向量匹配 top k", interactive=True)
        load_model_button = gr.Button("重新加载模型")
        load_model_button.click(reinit_model, show_progress=True,
                                inputs=[llm_model, embedding_model, llm_history_len, no_remote_model, top_k, chatbot], outputs=chatbot)
    demo.load(
        fn=refresh_vs_list,
        inputs=None,
        outputs=[select_vs, select_vs_test],
        queue=True,
        show_progress=False,
    )
#  Set the concurrency_limit directly on event listeners e.g. btn.click(fn, ..., concurrency_limit=10) or gr.Interface(concurrency_limit=10). If necessary, the total number of workers can be configured via `max_threads` in launch().
# (demo
#  .queue(concurrency_count=3)
#  .launch(server_name='0.0.0.0',
#          server_port=7999,
#          show_api=False,
#          share=False,
#          inbrowser=False))
(demo
 .queue()
 .launch(server_name='0.0.0.0',
         server_port=7999,
         show_api=False,
         share=False,
         inbrowser=False))
# 雅居乐集团成立于哪一年？