from abc import ABC
from dataclasses import dataclass


@dataclass
class Config(ABC):
    name: str = "gpt"
