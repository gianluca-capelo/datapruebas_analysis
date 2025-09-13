from attr import dataclass


@dataclass
class NeuropruebasTarget:
    content: str
    x: int
    y: int
