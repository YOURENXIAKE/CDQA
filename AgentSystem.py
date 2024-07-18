import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
sql_ip = "0.0.0.0"

#########导入Multi-agent Table Reasoning
import re
import openai
from openai import OpenAI
import pymysql
from vllm_wrapper import vLLMWrapper

from qwen_prompt import TASK_DIVIDE,PATCHER,REASONNER,PROMPT_DIVIDE_TASK_QWEN,PROMPT_COT_QW
import ast

class BaseAgent():
  def __init__(self,prompt='',agent_output=False):
    self.agent_output = agent_output
    self.task_prompt = prompt
    self.history_list = self.initial_history()
    
  def dprint(self,msg):
    if self.agent_output:
      print(msg)

  def preprocess_input(self,resp):
    return resp

  def initial_history(self):
    return [{'role': 'system', 'content': self.task_prompt}]
  
  def update_history(self,message):
    self.history_list.append(message)

  def process_task(self,input_text):
    return None
  
  def clear_history(self):
    self.history_list = [{'role': 'system', 'content': self.task_prompt}]

  def process_single_qa(self,query,sys=None):
    if sys is None:
      self.update_history({'role': 'user','content': query})
      return self.generate_answer()
    else:
      self.history_list = [{'role': 'system', 'content': sys},{'role': 'user','content': query}]
      return self.generate_answer()

  def generate_answer(self,messages=None):
    if messages is None:
      messages = self.history_list
    client = OpenAI(
        api_key="sk-bclmwvpwfeloazhjrqxjbvywouxwatiiuxuigadwlbnyxjsz",
        base_url="https://api.siliconflow.cn/v1"
    )
    response = client.chat.completions.create(
      model="Qwen/Qwen2-7B-Instruct",
      messages=messages,
      stream=False,
      stop=[] # You can add custom stop words here, e.g., stop=["Observation:"] for ReAct prompting.
    )
    return (response.choices[0].message.content)
  
  def discard_agent(self):
    pass

class ReasonAgent(BaseAgent):
  def __init__(self,prompt='',agent_output=True):
    self.agent_output = agent_output
    self.task_prompt = prompt
    self.history_list = self.initial_history()
    self.reason_material = []
    self.reason_tasks = []

  def generate_input(self):
    text = ''
    for i in range(len(self.reason_tasks)):
      text += f"{i+1}:{self.reason_tasks[i]}:{self.reason_material[i]}\n"
    return text

  def update_material(self,data):
    self.reason_material.append(data)

  def process_output(self,resp):
    pattern = r"EPC\[(.*?)\]"
    matches = re.search(pattern, resp)
    if matches:
        sql_statement = matches.group(1)
        return sql_statement
    else:
        return resp
  def clear_history(self):
    self.history_list = [{'role': 'system', 'content': self.task_prompt}]
    self.reason_material = []

  def process_task(self,question,multi_agent_qa=None):
    
    input_text = self.generate_input() + f'推理的问题是:{question}'
    self.dprint(input_text)
    self.dprint('LLM问询了一次ReasonAgent!\n')
    self.update_history({'role': 'user','content': input_text})
    ret = self.generate_answer()
    multi_agent_qa.append(['ReasonAgent',ret])
    return ret,multi_agent_qa

class TaskDivideAgent(BaseAgent):
  def process_output(self,output):
    ret = ast.literal_eval(output)
    return ret  # 转换为list

  def process_task(self,question):
    self.dprint('LLM问询了一次TaskDivideAgent!\n')
    self.update_history({'role': 'user','content': question})
    ret = self.generate_answer()
    print(ret)
    return self.process_output(ret)

class SimpleQaAgent(BaseAgent):

  def preprocess_input(self,resp):
    pattern = r"AnswerOracle\[(.*?)\]"
    matches = re.search(pattern, resp)
    if matches:
        sql_statement = matches.group(1)
        return sql_statement
    else:
        return resp

  def process_task(self,input_text):
    input_text = self.preprocess_input(input_text)
    self.dprint('LLM问询了一次AnswerOracle!问题是:'+input_text+'\n')
    self.update_history({'role': 'user','content': input_text})
    ret = self.generate_answer()
    self.initial_history()
    return ret

class KnowledgeBaseAgent(BaseAgent):
  def __init__(self,prompt='',agent_output=False,local_doc_qa=None,vs_path=None):
    self.agent_output = agent_output
    self.task_prompt = prompt
    self.history_list = self.initial_history()
    self.doc_qa_system = local_doc_qa
    self.vs_path = vs_path
  def preprocess_input(self,resp):
    pattern = r"TextOracle\[(.*?)\]"
    matches = re.search(pattern, resp)

    if matches:
        sql_statement = matches.group(1)
        return sql_statement
    else:
        return resp

  def process_task(self,input_text):
    input_text = self.preprocess_input(input_text)
    self.dprint('LLM问询了一次TextOracle!问题是:'+input_text+'\n')
    self.update_history({'role': 'user','content': input_text})
    # kn = '海外A类地区：美国、日本、韩国、英国、德国、瑞士、法国;海外B类地区：北欧其它国家（已列入一类地区欧洲国家之外）、加拿大、香港、澳门、澳大利亚、新加坡、台湾、新西兰、俄罗斯;海外C类地区：亚洲其它国家（已列入海外A类和B类地区的亚洲国家和地区之外）、东欧国家、非洲国家、南美国家.'
    kn = self.doc_qa_system.retrival_relative_docs(input_text,self.vs_path)
    return kn

class OutputAgent(BaseAgent):

  def preprocess_input(self,resp):
    pattern = r"OutputOracle\[(.*?)\]"
    matches = re.search(pattern, resp)

    # 如果找到匹配项，则提取SQL语句
    if matches:
        sql_statement = matches.group(1)
        return sql_statement
    else:
        return resp

  def process_task(self,input_text): # ToDo
    input_text = self.preprocess_input(input_text)
    self.dprint('LLM问询了一次AnswerOracle!问题是:'+input_text+'\n')
    self.update_history({'role': 'user','content': input_text})
    return '河南省的省长是高书记'

class SQLAgent(BaseAgent):
  def __init__(self,prompt='',agent_output=True):
    self.agent_output = agent_output
    self.task_prompt = prompt
    self.history_list = self.initial_history()
    self.sql_conn, self.sql_cursor = self.initial_database()
  
  def initial_database(self):
    # 建立数据库连接'192.168.1.50'
    conn = pymysql.connect(
        host=sql_ip,		# 主机名（或IP地址）
        port=3310,				# 端口号，默认为3306
        user='root',			# 用户名
        password='PASSWORD',	# 密码
        charset='utf8mb4'  		# 设置字符编码
    )
    # 创建游标对象
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

    # 选择数据库
    conn.select_db("yongyou")

    return conn,cursor
  
  def generate_answer(self):
    sql = self.history_list[-1]['content']
    self.dprint('LLM使用了一次TableOracle!查询的sql语句是:'+sql+'\n')
    try: 
      self.sql_cursor.execute(sql)
      result : tuple = self.sql_cursor.fetchall()
      if len(result) == 0:
        if self.agent_output:
          i = input("LLM遇到了一个问题，你是否愿意帮助他？如果想让他自己思考，输入回车；否则输入提示信息:")
        else:
          i = ''
        if i == '':
          # helps = '未查询到任何结果！提示：你可以打印该表的所有数据，结合该表的列名及其含义，对应地看一下你所需要查询的数据。请你重新生成下一步操作:'
          helps = '未查询到任何结果！提示：将查询条件放宽，尝试使用LIKE操作符和通配符来匹配，如LIKE %key%。请你重新生成下一步操作:'
        else:
          helps = f'未查询到任何结果！错误原因是:{i}'
        return helps
      ret = set()
      for row in result:
          ret.add(row)
      return list(ret)

    except Exception as e:
      if self.agent_output:
        i = input("LLM遇到了一个问题，你是否愿意帮助他？如果想让他自己思考，输入回车；否则输入提示信息:")
      else:
        i = ''
      if i == '':
        helps = f'未查询到任何结果！报错是{e},请思考错误原因是?并重新生成下一步操作:'
      else:
        helps = f'未查询到任何结果！错误原因是:{i}'
      return helps
  
  def preprocess_input(self,resp):
    pattern = r"TableOracle\[(.*?)\]"
    matches = re.search(pattern, resp)
    # print(matches)
    # 如果找到匹配项，则提取SQL语句
    if matches:
        sql_statement = matches.group(1)
        # print()
        return sql_statement
    else:
        return resp

  def process_task(self,input_text):
    input_text = self.preprocess_input(input_text)
    if self.agent_output:
      print('LLM问询了一次AnswerOracle!问题是:'+input_text+'\n')
    self.update_history({'role': 'user','content': input_text})
    ret = self.generate_answer()
    self.initial_history()
    return ret
  
  def discard_agent(self):
    self.sql_conn.close()
    self.sql_cursor.close()

class TableSQLAgent(SQLAgent):
  def initial_database(self):
    conn = pymysql.connect(
        host=sql_ip,		# 主机名（或IP地址）
        port=3310,				# 端口号，默认为3306
        user='root',			# 用户名
        password='password',	# 密码
        charset='utf8mb4'  		# 设置字符编码
    )
    # 创建游标对象
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

    # 选择数据库
    conn.select_db("yongyou")

    return conn,cursor
  def generate_answer(self):
    table_name = self.history_list[-1]['content']
    sql = 'select * from ' + table_name
    self.sql_cursor.execute(sql)
    result = self.sql_cursor.fetchall()
    ret = set()

    for row in result:
        ret.add(str(row))
    
    return list(ret)

class BrainAgent(BaseAgent):
  def __init__(self,task_prompt,sub_agents_register_list,agent_output=True):
    self.agent_output = agent_output
    self.task_prompt = task_prompt
    self.sub_agents = sub_agents_register_list
    self.history_list = self.initial_history()
    
  def register_sub_agents(self,agents):
    self.sub_agents = agents

  def process_task(self,qustion,multi_agent_qa=None):
    self.update_history({'role': 'system','content': self.task_prompt})
    self.update_history({'role': 'user','content': qustion})
    response = self.generate_answer()
    self.update_history({'role': 'assistant','content': response})
    self.dprint('问题是:' + qustion + ',LLM第一个操作是:'+response+'\n')
    if multi_agent_qa is not None:
      multi_agent_qa.append([qustion,response])
    if 'OutputOracle' in response:
      self.sub_agents['OutputOracle'].process_task(response)
      self.dprint('LLM得到了最终的答案!'+response+'\n')
      if multi_agent_qa is not None:
        multi_agent_qa.append(["OutPut Oracle",response])
      self.dprint('对话历史记录:\n\n\n')
      self.dprint(self.history_list)
      self.discard_agent()

    elif 'TableOracle' in response:
      data = self.sub_agents['TableOracle'].process_task(response)
      table_name = self.sub_agents['TableOracle'].history_list[-1]['content']
      self.dprint('LLM询问TableOracle的查询结果是:'+str(data)+'\n')
      if multi_agent_qa is not None:
        multi_agent_qa.append(["TableOracle",str(data)])
      agent_ans = f"表名:{table_name};数据:{data}"

    elif 'AnswerOracle' in response:
      data = self.sub_agents['AnswerOracle'].process_task(response)
      self.dprint('AnswerOracle的回答是:'+data+'\n')
      if multi_agent_qa is not None:
        multi_agent_qa.append(["AnswerOracle",data])
      agent_ans = data
      
    elif 'TextOracle' in response:
      data = self.sub_agents['TextOracle'].process_task(response)
      self.dprint('AnswerOracle的回答是:'+data+'\n')
      if multi_agent_qa is not None:
        multi_agent_qa.append(["TextOracle",data])
      agent_ans = data

    else:
      self.dprint('LLM似乎到达了疑惑的区间...让我们问一问它是不是要结束了!\n')
      agent_ans = '无信息,请输出EPC!'

      # self.update_history({'role': 'user','content': agent_ans})
      # response = self.generate_answer()
      # self.update_history({'role': 'assistant','content': response})
      # self.dprint('LLM接下来的操作是:'+response+'\n')
    return agent_ans,multi_agent_qa
  
  def all_clear(self):
    for k in self.sub_agents.keys():
      self.sub_agents[k].clear_history()
    
  def discard_agent(self):
    for k in self.sub_agents.keys():
      self.sub_agents[k].discard_agent()

class ChatAgent():
  def __init__(self,sys):
    self.reson_agent = ReasonAgent(prompt = REASONNER)
    self.kb_agent = KnowledgeBaseAgent(local_doc_qa = local_doc_qa,vs_path=vs_path)
    self.sql_agent = TableSQLAgent()
    self.task_divide_agent = TaskDivideAgent(prompt = TASK_DIVIDE)

    agents_register = {'TableOracle':self.sql_agent,'TextOracle':self.kb_agent}
    self.brain_agent = BrainAgent(task_prompt=PATCHER,sub_agents_register_list=agents_register)

    self.divide_agent_history=[{'role': 'system','content': self.task_prompt}]

class AgentSystem():
    def __init__(self,local_doc_qa,vs_path):
        self.single_qa_agent = BaseAgent(prompt=PROMPT_DIVIDE_TASK_QWEN)

        self.step2_agent = BaseAgent(prompt=PROMPT_COT_QW)

        self.reson_agent = ReasonAgent(prompt = REASONNER)
        self.kb_agent = KnowledgeBaseAgent(local_doc_qa = local_doc_qa,vs_path=vs_path)
        self.sql_agent = TableSQLAgent()
        self.task_divide_agent = TaskDivideAgent(prompt = TASK_DIVIDE)

        agents_register = {'TableOracle':self.sql_agent,'TextOracle':self.kb_agent}
        self.brain_agent = BrainAgent(task_prompt=PATCHER,sub_agents_register_list=agents_register)
    
    def process_task(self,qustion,multi_agent_qa=None):
        task_list = self.task_divide_agent.process_task(qustion)
        self.reson_agent.reason_tasks = task_list
        if task_list is not None:
          multi_agent_qa.append(["任务分解Agent输出",'\n'.join(task_list)])

        for simple_q in task_list:
            ret,multi_agent_qa = self.brain_agent.process_task(simple_q,multi_agent_qa)
            self.reson_agent.update_material(ret)
        ans,multi_agent_qa = self.reson_agent.process_task(qustion,multi_agent_qa)
        return ans,multi_agent_qa
    
    def clear_all(self):
      self.brain_agent.all_clear()
      self.reson_agent.clear_history()
      self.single_qa_agent.clear_history()
      self.step2_agent.clear_history()
