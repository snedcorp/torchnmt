from dataclasses import dataclass

from torch import Tensor


@dataclass
class BeamSearchResult:
    """
    Represents the result of a single beam in the beam search process.

    Attributes:
        s (str): The decoded string result for this beam.
        ix (list[int]): A list of integer indices representing the tokens contained within the beam.
        val (float): The score or value associated with this beam's result - a length-normalized conditional
        log-probability (with sign flipped).
        att (Tensor | None): The attention weights associated with this beam, if applicable.
    """

    s: str
    ix: list[int]
    val: float
    att: Tensor | None
