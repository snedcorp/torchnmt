from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union

from pydantic import Field, PositiveInt
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(kw_only=True)
class LoadModelConfig:
    """
    Configuration class for loading a pre-trained model.

    Attributes:
        dir_ (str): The directory path where the model files are located.
        experiment (str): The experiment to which the model belongs.
        name (str): The name of the model to be loaded.
        epoch (PositiveInt | None): The specific training epoch to load. If `None`, the latest available epoch is used. Defaults to `None`.
        kind (Literal["load"]): A fixed literal value indicating the configuration type. Defaults to "load".
    """

    dir_: str
    experiment: str
    name: str
    epoch: PositiveInt | None = None
    kind: Literal["load"] = "load"


@pydantic_dataclass(kw_only=True)
class BuildModelConfig:
    """
    Base configuration class for building a new model.

    Attributes:
        n_layers (PositiveInt): The number of layers in the model. Must be a positive integer.
        dropout (float): The dropout rate, for regularization. Must be a value between 0 and 1 (inclusive). Defaults to 0.
    """

    n_layers: PositiveInt
    dropout: float = Field(ge=0, le=1)


@pydantic_dataclass(kw_only=True)
class RNNModelConfig(BuildModelConfig):
    """
    Configuration class that extends `BuildModelConfig` to include additional parameters specific to recurrent neural
    network (RNN) models.

    Attributes:
        n_emb (PositiveInt): The dimensionality of the input embeddings. Must be a positive integer.
        n_hidden (PositiveInt): The dimensionality of the RNN's hidden state. Must be a positive integer.
    """

    n_emb: PositiveInt
    n_hidden: PositiveInt


@pydantic_dataclass(kw_only=True)
class SutskeverConfig(RNNModelConfig):
    """
    Configuration class indicating that the desired RNN variant is one containing the SutskeverRNNDecoder.

    Attributes:
        kind (Literal["sutskever"]): A fixed literal value indicating the model type as "sutskever". Defaults to "sutskever".
    """

    kind: Literal["sutskever"] = "sutskever"


@pydantic_dataclass(kw_only=True)
class RNNAttentionModelConfig(RNNModelConfig):
    """
    Configuration class that extends RNNModelConfig to include an additional parameter specific to building models that
    contain an attention mechanism.

    Attributes:
        use_context (bool): Whether to use the final hidden states from the encoder as the initial hidden states in the
        decoder.
    """

    use_context: bool


@pydantic_dataclass(kw_only=True)
class BahdanauConfig(RNNAttentionModelConfig):
    """
    Configuration class indicating that the desired RNN variant is one containing the BahdanauRNNDecoder.

    Attributes:
        kind (Literal["bahdanau"]): A fixed literal value indicating the model type as "bahdanau". Defaults to "bahdanau".
    """

    kind: Literal["bahdanau"] = "bahdanau"


class ScoreMethod(str, Enum):
    """
    An enumeration of all the supported scoring methods when calculating attention within RNN-based models.
    """

    DOT = "dot"
    GENERAL = "general"
    ADDITIVE = "additive"
    SCALED_DOT = "scaled_dot"


@pydantic_dataclass(kw_only=True)
class LuongConfig(RNNAttentionModelConfig):
    """
    Configuration class indicating that the desired RNN variant is one containing the LuongRNNDecoder.

    Attributes:
        att_score_method (ScoreMethod): Scoring method to use when calculating attention.
        input_feeding (bool): Whether to use the previous time step's attentional hidden state in order to calculate
        the current time step's hidden state.
        kind (Literal["luong"]): A fixed literal value indicating the model type as "luong". Defaults to "luong".
    """

    att_score_method: ScoreMethod
    input_feeding: bool
    kind: Literal["luong"] = "luong"


@pydantic_dataclass(kw_only=True)
class TransformerConfig(BuildModelConfig):
    """
    Configuration class that extends `BuildModelConfig` to include additional parameters specific to transformer models.

    Attributes:
        d_model (PositiveInt): Dimensionality of the model.
        n_heads (PositiveInt): Number of heads within each multi-head attention module.
        d_ff (PositiveInt): Dimensionality of the feedforward network modules.
        sine_pos (bool): Whether to include sinusoidal position encoding, instead of learned positional encoding.
        pre_norm (bool): Whether to perform layer normalization within a sub-layer's residual connection, or after it.
        naive (bool): Whether to compute multi-head attention the naive way (easier to follow, but slower).
        kind (Literal["transformer"]): A fixed literal value indicating the model type as "transformer". Defaults to "transformer".
    """

    d_model: PositiveInt
    n_heads: PositiveInt
    d_ff: PositiveInt
    sine_pos: bool
    pre_norm: bool
    naive: bool
    kind: Literal["transformer"] = "transformer"


ModelConfig = Union[SutskeverConfig, BahdanauConfig, LuongConfig, TransformerConfig, LoadModelConfig]


@dataclass(frozen=True)
class ModelParams:
    """
    Base model configuration class containing the fields needed by every encoder and decoder model.

    Attributes:
        - vocab_size (int): Size of the vocabulary (src if encoder model, tgt if decoder model).
        - n_layers (int): Number of layers within the GRU.
        - dropout (float): The dropout rate, for regularization.
        - pad_ix (int): Index of padding token within the vocabulary.
    """

    vocab_size: int
    n_layers: int
    dropout: float
    pad_ix: int


@dataclass(frozen=True)
class RNNParams(ModelParams):
    """
    Model configuration class extending ModelParams, containing the fields needed by every RNN-based encoder and
    decoder model.

    Attributes:
        n_emb (int): The dimensionality of the input embeddings.
        n_hidden (int): The dimensionality of the RNN's hidden state.
    """

    n_emb: int
    n_hidden: int


class MergeMethod(Enum):
    """
    An enumeration of all the supported ways to merge the forward and backward hidden states from the encoder RNN's
    last layer, in order to obtain the encoder's outputs.
    """

    SUM = 1
    CONCAT = 2


@dataclass(frozen=True)
class RNNEncoderParams(RNNParams):
    """
    Model configuration class extending RNNParams, containing the fields needed by every RNN-based encoder model.

    Attributes:
        use_context (bool): Whether to use the final hidden states from the encoder as the initial hidden states in the
        decoder.
        merge_method (MergeMethod): How to construct the encoder outputs from the hidden states of its last layer.
    """

    use_context: bool
    merge_method: MergeMethod


@dataclass(frozen=True)
class RNNDecoderParams(RNNParams):
    """
    Model configuration class extending RNNParams, containing the fields needed by every RNN-based decoder model.

    Attributes:
        att_score_method (ScoreMethod): Scoring method to use when calculating attention.
        input_feeding (bool): Whether to use the previous time step's attentional hidden state in order to calculate
        the current time step's hidden state.
    """

    att_score_method: ScoreMethod | None = None
    input_feeding: bool | None = None


@dataclass(frozen=True)
class TransformerParams(ModelParams):
    """
    Model configuration class extending ModelParams, containing the fields needed by every transformer encoder or decoder model.

    Attributes:
        max_seq_len (int): Maximum observed size of input sequences, needed to determine dimensions of positional encoding matrix.
        d_model (int): Dimensionality of the model.
        n_heads (int): Number of heads within each multi-head attention module.
        d_ff (int): Dimensionality of the feedforward network modules.
        sine_pos (bool): Whether to include sinusoidal position encoding, instead of learned positional encoding.
        pre_norm (bool): Whether to perform layer normalization within a sub-layer's residual connection, or after it.
        naive (bool): Whether to compute multi-head attention the naive way (easier to follow, but slower).
    """

    max_seq_len: int
    d_model: int
    n_heads: int
    d_ff: int
    sine_pos: bool
    pre_norm: bool
    naive: bool
