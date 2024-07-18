# ComplexDocumentQA

## 功能
本项目在 langchain-ChatGLM 的基础之上进行修改，增加的功能如下：
- 支持活字大模型进行问答
- 实现基于复杂任务分解推理、多智能体推理、以及混合推理的文档推理策略
- 实现 PDF 解析(尤其是表格数据的解析)
- 在 langchain-ChatGLM 的基础之上，改进了基于 Gradio 的问答界面，增加了 3 种推理策略的推理结果，增加了针对 PDF 的实时问答

## 如何运行
- 完成 langchain-ChatGLM 的 requirement 安装
- 如果使用Qwen72b进行问答，安装Qwen72b的相关依赖
- 为了使用Web端，**请确保Gradio版本大于4.0**，并安装目录下的 gradio_kmpdf-0.0.1-py3-none-any.whl 进行PDF渲染问答

## 关键文件
- AgentSystem.py : 多智能体推理框架
- cli_demo.py : 原始的文档问答命令行demo
- cli_demo_multi_agent.py : 基于多智能体文档问答的命令行demo
- webui_multiagent.py : 可视化网页运行

## 如何拓展 PDF 问答模块 ?
参考资料：https://www.gradio.app/guides/pdf-component-example 请根据这个资料完成PDF自定义组件的编写，可以参考 /home/zrchen/chatdoc-master/kmpdf/frontend/Index.svelte 我对里面进行了一些细微修改，能够较清晰的显示pdf细节，但是缺少一种更直观渲染的方法。

为了实现高亮功能，首先确保配置成功 https://www.gradio.app/guides/pdf-component-example ，能够成功运行里面的demo，然后进行修改：(实际上就是pdf.js库的魔改和应用)
- 文字复制参考资料：https://segmentfault.com/a/1190000042089590
- 高亮相关参考资料：https://www.cnblogs.com/Beson/p/16372402.html   https://blog.csdn.net/wang13679201813/article/details/129798858  

## 如何使用各种推理接口
- qwen2 7b模型：可以去 siliconflow 申请免费的API KEY，在CDQA/models/fastchat_qwen2_7b_llm.py填写相应的KEY
- HuoZi：先在web登录https://huozi.8wss.com/，然后用https://huozi.8wss.com/api得到api。
- qwen 72b：先安装好相应的依赖，然后使用3个bash开下面3个命令：
python -m fastchat.serve.controller  
python -m fastchat.serve.vllm_worker --model-path /home/zrchen/yongyou/qwen70b/Qwen-72B-Chat(72b模型地址) --trust-remote-code --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --dtype bfloat16  
python -m fastchat.serve.openai_api_server --host localhost --port 5214
