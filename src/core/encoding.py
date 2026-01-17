# -*- coding: utf-8 -*-
"""
Encoding utilities for variable-length NAS architectures.
"""
import random
import copy
from typing import List, Tuple, Optional
from configuration.config import config

# Block parameter count for encoding/decoding.
BLOCK_PARAM_COUNT = 10


class BlockParams:
    """Container for block-level hyperparameters."""

    def __init__(self, out_channels: int, groups: int, pool_type: int,
                 pool_stride: int, has_senet: int, activation_type: int = 0,
                 dropout_rate: float = 0.0, skip_type: int = 0, kernel_size: int = 3,
                 expansion: int = 2):
        """
        Initialize block parameters.

        Args:
            out_channels (int): Number of output channels.
            groups (int): Number of groups for convolution.
            pool_type (int): 0 for MaxPool, 1 for AvgPool.
            pool_stride (int): Stride for pooling (1 or 2).
            has_senet (int): 1 if SE block is used, 0 otherwise.
            activation_type (int): 0 for ReLU, 1 for SiLU.
            dropout_rate (float): Dropout probability.
            skip_type (int): 0 for add, 1 for concat, 2 for none.
            kernel_size (int): Convolution kernel size (3 or 5).
            expansion (int): Application of expansion factor on mid_channels.
        """
        self.out_channels = out_channels
        self.groups = groups
        self.pool_type = pool_type
        self.pool_stride = pool_stride
        self.has_senet = has_senet
        self.activation_type = activation_type  # 0=ReLU, 1=SiLU
        self.dropout_rate = dropout_rate        # Dropout rate
        self.skip_type = skip_type              # 0=add, 1=concat, 2=none
        self.expansion = expansion              # Block expansion: 1 or 2
        self.kernel_size = kernel_size          # Kernel size: 3 or 5

    def to_list(self) -> List:
        """
        Serialize block parameters to a list of integers.

        Returns:
            List: Flattened parameters.
        """
        dropout_encoded = int(self.dropout_rate * 100)
        return [self.out_channels, self.groups, self.pool_type,
                self.pool_stride, self.has_senet, self.activation_type,
                dropout_encoded, self.skip_type, self.kernel_size, self.expansion]

    @classmethod
    def from_list(cls, params: List) -> "BlockParams":
        """
        Deserialize block parameters from a list.

        Args:
            params (List): Flattened parameters.

        Returns:
            BlockParams: Reconstructed object.
        """
        # Default expansion for legacy encodings that predate this field.
        dropout_rate = params[6] / 100.0

        return cls(params[0], params[1], params[2], params[3], params[4],
                   params[5], dropout_rate, params[7], params[8], params[9])

    def __repr__(self):
        activation_names = {0: "ReLU", 1: "SiLU"}
        skip_names = {0: "add", 1: "concat", 2: "none"}
        return (f"BlockParams(out_ch={self.out_channels}, groups={self.groups}, "
                f"pool_type={self.pool_type}, pool_stride={self.pool_stride}, "
                f"has_senet={self.has_senet}, activation={activation_names.get(self.activation_type, 'ReLU')}, "
                f"dropout={self.dropout_rate}, skip={skip_names.get(self.skip_type, 'add')}, "
                f"kernel_size={self.kernel_size}, expansion={self.expansion})")


class Individual:
    """Candidate architecture representation."""

    def __init__(self, encoding: Optional[List[int]] = None):
        """
        Initialize an individual.

        Args:
            encoding (List[int], optional): Genetic encoding of the architecture.
        """
        self.id = None
        self.encoding = encoding if encoding is not None else []
        self.quick_score = 0
        self.fitness = None
        self.param_count = None
        self.op_history = []

    def copy(self) -> "Individual":
        """
        Create a deep copy of the individual.

        Returns:
            Individual: A new instance with copied data.
        """
        new_ind = Individual(copy.deepcopy(self.encoding))
        new_ind.id = self.id
        new_ind.fitness = self.fitness
        new_ind.quick_score = self.quick_score
        new_ind.op_history = copy.deepcopy(self.op_history)
        return new_ind

    def __repr__(self):
        return f"Individual(id={self.id}, fitness={self.fitness}, encoding_len={len(self.encoding)})"


class Encoder:
    """Encoder/decoder for variable-length architecture encodings."""

    @staticmethod
    def decode(encoding: List[int]) -> Tuple[int, List[int], List[List[BlockParams]]]:
        """
        Decode the integer encoding into structural components.

        Args:
            encoding (List[int]): The architecture encoding.

        Returns:
            Tuple[int, List[int], List[List[BlockParams]]]:
                - Number of units
                - List of block counts per unit
                - Nested list of BlockParams for each block.
        """
        if not encoding:
            raise ValueError("Encoding is empty")

        unit_num = encoding[0]
        block_nums = encoding[1:1 + unit_num]

        block_params_list = []
        idx = 1 + unit_num

        for block_num in block_nums:
            unit_blocks = []
            for _ in range(block_num):
                params = encoding[idx:idx + BLOCK_PARAM_COUNT]
                if len(params) < BLOCK_PARAM_COUNT:
                    raise ValueError(f"Incomplete block params at index {idx}")
                block_params = BlockParams.from_list(params)
                unit_blocks.append(block_params)
                idx += BLOCK_PARAM_COUNT
            block_params_list.append(unit_blocks)

        return unit_num, block_nums, block_params_list

    @staticmethod
    def encode(unit_num: int, block_nums: List[int],
               block_params_list: List[List[BlockParams]]) -> List[int]:
        """
        Encode structural components into a flat integer list.

        Args:
            unit_num (int): Number of units.
            block_nums (List[int]): Number of blocks per unit.
            block_params_list (List[List[BlockParams]]): Block parameters.

        Returns:
            List[int]: Flattened encoding.
        """
        encoding = [unit_num]
        encoding.extend(block_nums)

        for unit_blocks in block_params_list:
            for block_params in unit_blocks:
                encoding.extend(block_params.to_list())

        return encoding

    @staticmethod
    def validate_encoding(encoding: List[int]) -> bool:
        """
        Check if an encoding is valid within the search space constraints.

        Args:
            encoding (List[int]): The encoding to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            unit_num, block_nums, block_params_list = Encoder.decode(encoding)

            if not (config.MIN_UNIT_NUM <= unit_num <= config.MAX_UNIT_NUM):
                print("WARN: invalid unit count")
                return False

            for block_num in block_nums:
                if not (config.MIN_BLOCK_NUM <= block_num <= config.MAX_BLOCK_NUM):
                    print("WARN: invalid block count")
                    return False

            for i, unit_blocks in enumerate(block_params_list):
                for bp in unit_blocks:
                    if bp.out_channels not in config.CHANNEL_OPTIONS:
                        return False
                    
                    # 64 channels only allowed in the first unit (index 0)
                    if i > 0 and bp.out_channels == 64:
                        return False
                    
                    # 1024 channels only allowed in the last unit
                    if i < unit_num - 1 and bp.out_channels == 1024:
                        return False

                    if bp.groups not in config.GROUP_OPTIONS:
                        return False
                    if bp.pool_type not in config.POOL_TYPE_OPTIONS:
                        return False
                    if bp.pool_stride not in config.POOL_STRIDE_OPTIONS:
                        return False
                    if bp.has_senet not in config.SENET_OPTIONS:
                        return False
                    if bp.activation_type not in config.ACTIVATION_OPTIONS:
                        return False
                    if bp.dropout_rate not in config.DROPOUT_OPTIONS:
                        return False
                    if bp.skip_type not in config.SKIP_TYPE_OPTIONS:
                        return False
                    if bp.kernel_size not in config.KERNEL_SIZE_OPTIONS:
                        return False
                    if bp.expansion not in config.EXPANSION_OPTIONS:
                        return False

            # Concat skips are only allowed at the last block of each unit.
            for unit_blocks in block_params_list:
                last_idx = len(unit_blocks) - 1
                for idx, bp in enumerate(unit_blocks):
                    if bp.skip_type == 1 and idx != last_idx:
                        return False

            if not Encoder.validate_feature_size(encoding):
                print("WARN: feature map too small")
                return False

            if not Encoder.validate_channel_count(encoding):
                print("WARN: channel limit exceeded")
                return False

            return True
        except Exception:
            return False

    @staticmethod
    def validate_channel_count(encoding: List[int], init_channels: int = None) -> bool:
        """
        Validate that channel counts stay within configured limits.
        
        Args:
            encoding (List[int]): The encoding to check.
            init_channels (int, optional): Initial number of channels.

        Returns:
            bool: True if channel counts satisfy limits.
        """
        if init_channels is None:
            init_channels = config.INIT_CONV_OUT_CHANNELS

        try:
            _, _, block_params_list = Encoder.decode(encoding)
            current_channels = init_channels

            for unit_blocks in block_params_list:
                for bp in unit_blocks:
                    out_channels = bp.out_channels * bp.expansion
                    if bp.skip_type == 1:  # concat
                        final_channels = out_channels + current_channels
                    else:
                        final_channels = out_channels

                    if final_channels > config.MAX_CHANNELS:
                        return False
                    current_channels = final_channels

            return True
        except Exception:
            return False

    @staticmethod
    def validate_feature_size(encoding: List[int], input_size: int = None) -> bool:
        """
        Ensure feature map size does not become too small (e.g., < 1x1).

        Args:
            encoding (List[int]): The encoding to check.
            input_size (int, optional): Input image size.

        Returns:
            bool: True if feature size remains valid.
        """
        if input_size is None:
            input_size = config.INPUT_IMAGE_SIZE

        try:
            _, _, block_params_list = Encoder.decode(encoding)
            current_size = input_size

            for unit_blocks in block_params_list:
                for bp in unit_blocks:
                    if bp.pool_stride == 2:
                        current_size = (current_size + 1) // 2
                    if current_size < config.MIN_FEATURE_SIZE:
                        return False
            return current_size >= config.MIN_FEATURE_SIZE
        except Exception:
            return False

    @staticmethod
    def get_max_downsampling(input_size: int = None) -> int:
        """
        Calculate maximum allowable downsampling operations.

        Args:
            input_size (int, optional): Input image size.

        Returns:
            int: Max number of downsample steps.
        """
        if input_size is None:
            input_size = config.INPUT_IMAGE_SIZE
        import math
        return int(math.log2(input_size / config.MIN_FEATURE_SIZE))

    @staticmethod
    def print_architecture(encoding: List[int]):
        """
        Print a human-readable summary of the architecture.

        Args:
            encoding (List[int]): The architecture encoding.
        """
        unit_num, block_nums, block_params_list = Encoder.decode(encoding)

        # Best-effort parameter count for display.
        param_count_str = "N/A"
        try:
            from models.network import NetworkBuilder
            param_count = NetworkBuilder.calculate_param_count(encoding)
            param_count_str = f"{param_count:,}"
        except Exception as e:
            param_count_str = f"Error calculating params: {e}"

        print(
            f"INFO: architecture units={unit_num} blocks_per_unit={block_nums} "
            f"params={param_count_str}"
        )
        for i, unit_blocks in enumerate(block_params_list):
            print(f"INFO: unit {i + 1} blocks={len(unit_blocks)}")
            for j, bp in enumerate(unit_blocks):
                pool_type_str = "MaxPool" if bp.pool_type == 0 else "AvgPool"
                senet_str = "Yes" if bp.has_senet == 1 else "No"
                activation_names = {0: "ReLU", 1: "SiLU"}
                skip_names = {0: "add", 1: "concat", 2: "none"}
                activation_str = activation_names.get(bp.activation_type, "ReLU")
                skip_str = skip_names.get(bp.skip_type, "add")
                print(
                    "INFO: block "
                    f"{i + 1}.{j + 1} out_ch={bp.out_channels} groups={bp.groups} "
                    f"pool={pool_type_str} stride={bp.pool_stride} senet={senet_str} "
                    f"act={activation_str} dropout={bp.dropout_rate} skip={skip_str} "
                    f"kernel={bp.kernel_size} expansion={bp.expansion}"
                )
