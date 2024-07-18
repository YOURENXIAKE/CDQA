from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult)


class QWenLLM(BaseAnswer, LLM, ABC):
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
        return "QWen"

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
        response, _ = self.checkPoint.model.chat(prompt, history=None, system="请用二次元可爱语气和我说话")
        print(f"response:{response}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return response

    def generatorAnswer(self,prompt: str,system="",
                         history = [],
                         streaming: bool = False):
        # print(prompt)
        response, history = self.checkPoint.model.chat(prompt, history=None, system=system)
        self.checkPoint.clear_torch_cache()
        # response, history = self.checkPoint.model.chat(prompt, history=None, system=system)
        history += [['用户提问:'+prompt, '<你的回答:'+response+'>']]
        answer_result = AnswerResult()
        answer_result.history = history
        answer_result.llm_output = {"answer": response}
        yield answer_result

    def generatorAnswer_task_devide(self,prompt: str,system,
                         history: List[List[str]] = [],
                         streaming: bool = False):
        print(prompt)
        response, history = self.checkPoint.model.chat(prompt, history=None, system=system)
        # response, history = self.checkPoint.model.chat(prompt, history=None, system=system)
        self.checkPoint.clear_torch_cache()
        history += [['用户提问:'+prompt, '<你的回答:'+response+'>']]
        answer_result = AnswerResult()
        answer_result.history = history
        answer_result.llm_output = {"answer": response}
        yield answer_result

