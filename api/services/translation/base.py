from abc import ABC, abstractmethod


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, target_lang: str) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
