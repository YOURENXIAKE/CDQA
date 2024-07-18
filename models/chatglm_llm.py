# from abc import ABC
# from langchain.llms.base import LLM
# from typing import Optional, List
# from models.loader import LoaderCheckPoint
# from models.base import (BaseAnswer,
#                          AnswerResult)


# class ChatGLM(BaseAnswer, LLM, ABC):
#     max_token: int = 10000
#     temperature: float = 0.01
#     top_p = 0.9
#     checkPoint: LoaderCheckPoint = None
#     # history = []
#     history_len: int = 10

#     def __init__(self, checkPoint: LoaderCheckPoint = None):
#         super().__init__()
#         self.checkPoint = checkPoint

#     @property
#     def _llm_type(self) -> str:
#         return "ChatGLM"

#     @property
#     def _check_point(self) -> LoaderCheckPoint:
#         return self.checkPoint

#     @property
#     def _history_len(self) -> int:
#         return self.history_len

#     def set_history_len(self, history_len: int = 10) -> None:
#         self.history_len = history_len

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         print(f"__call:{prompt}")
#         response, _ = self.checkPoint.model.chat(
#             self.checkPoint.tokenizer,
#             prompt,
#             history=[],
#             max_length=self.max_token,
#             temperature=self.temperature
#         )
#         print(f"response:{response}")
#         print(f"+++++++++++++++++++++++++++++++++++")
#         return response

#     def generatorAnswer(self, tokenizer,prompt: str,
#                          history: List[List[str]] = [],
#                          streaming: bool = False):

#         if streaming:
#             history += [[]]
#             for inum, (stream_resp, _) in enumerate(self.checkPoint.model.stream_chat(
#                     self.checkPoint.tokenizer,
#                     prompt,
#                     history=history[-self.history_len:-1] if self.history_len > 1 else [],
#                     max_length=self.max_token,
#                     temperature=self.temperature
#             )):
#                 # self.checkPoint.clear_torch_cache()
#                 history[-1] = [prompt, stream_resp]
#                 answer_result = AnswerResult()
#                 answer_result.history = history
#                 answer_result.llm_output = {"answer": stream_resp}
#                 yield answer_result
#         else:
#             response, _ = self.checkPoint.model.chat(
#                 self.checkPoint.tokenizer,
#                 prompt,
#                 history=history[-self.history_len:] if self.history_len > 0 else [],
#                 max_length=self.max_token,
#                 temperature=self.temperature
#             )
#             self.checkPoint.clear_torch_cache()
#             history += [[prompt, response]]
#             answer_result = AnswerResult()
#             answer_result.history = history
#             answer_result.llm_output = {"answer": response}
#             yield answer_result


from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult)


class ChatGLM(BaseAnswer, LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"__call:{prompt}")
        response, _ = self.checkPoint.model.chat(
            self.checkPoint.tokenizer,
            prompt,
            history=[],
            max_length=self.max_token,
            temperature=self.temperature
        )
        print(f"response:{response}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return response

    def generatorAnswer(self,prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):

        if streaming:
            history += [[]]
            for inum, (stream_resp, _) in enumerate(self.checkPoint.model.stream_chat(
                    self.checkPoint.tokenizer,
                    prompt,
                    history=history[-self.history_len:-1] if self.history_len > 1 else [],
                    max_length=self.max_token,
                    temperature=self.temperature
            )):
                # self.checkPoint.clear_torch_cache()
                history[-1] = [prompt, stream_resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}
                yield answer_result
        else:
            print('history:::')
            print(history[-self.history_len:] if self.history_len > 0 else [])
            if len(history) > 0:
                his = '<历史记录:'
                for q in history:
                    his = his + str(q[0]) + str(q[1])
                prompt = his + '历史记录结束>' + prompt
            print('Prompt:::')
            print(prompt)
            response, _ = self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                max_length=self.max_token,
                temperature=self.temperature
            )
            import re
            pattern = re.compile(r'^.+回答问题机器人最终输出：(.+)',re.DOTALL)
            res = pattern.match(response)
            if res is not None:
                response = res.group(1)  
            else:
                pattern = re.compile(r'^.+最终输出：(.+)',re.DOTALL)
                res = pattern.match(response)
                if res is not None:
                    response = res.group(1)  
            self.checkPoint.clear_torch_cache()
            history += [['用户提问:'+prompt, '<你的回答:'+response+'>']]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}
            yield answer_result


