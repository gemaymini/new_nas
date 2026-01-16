# -*- coding: utf-8 -*-
"""
Search space definitions and random sampling utilities.
"""
import random
from typing import List, Optional
from configuration.config import config
from core.encoding import Encoder, Individual, BlockParams
from utils.constraints import evaluate_encoding_params


class SearchSpace:
    """
    Defines available hyperparameter options and sampling methods.
    """

    def __init__(self):
        self.min_unit_num = config.MIN_UNIT_NUM
        self.max_unit_num = config.MAX_UNIT_NUM
        self.min_block_num = config.MIN_BLOCK_NUM
        self.max_block_num = config.MAX_BLOCK_NUM
        self.channel_options = config.CHANNEL_OPTIONS
        self.group_options = config.GROUP_OPTIONS
        self.pool_type_options = config.POOL_TYPE_OPTIONS
        self.pool_stride_options = config.POOL_STRIDE_OPTIONS
        self.senet_options = config.SENET_OPTIONS
        self.activation_options = config.ACTIVATION_OPTIONS
        self.dropout_options = config.DROPOUT_OPTIONS
        self.skip_type_options = config.SKIP_TYPE_OPTIONS
        self.kernel_size_options = config.KERNEL_SIZE_OPTIONS
        self.expansion_options = config.EXPANSION_OPTIONS

    def sample_unit_num(self) -> int:
        """Randomly sample the number of units."""
        return random.randint(self.min_unit_num, self.max_unit_num)

    def sample_block_num(self) -> int:
        """Randomly sample the number of blocks within a unit."""
        return random.randint(self.min_block_num, self.max_block_num)

    def sample_channel(self) -> int:
        return random.choice(self.channel_options)

    def sample_groups(self) -> int:
        return random.choice(self.group_options)

    def sample_pool_type(self) -> int:
        return random.choice(self.pool_type_options)

    def sample_pool_stride(self) -> int:
        return random.choice(self.pool_stride_options)

    def sample_senet(self) -> int:
        return random.choice(self.senet_options)

    def sample_activation(self) -> int:
        # Activation types: 0=ReLU, 1=SiLU.
        return random.choice(self.activation_options)

    def sample_dropout(self) -> float:
        return random.choice(self.dropout_options)

    def sample_skip_type(self, allow_concat: bool = True) -> int:
        # Skip types: 0=add, 1=concat, 2=none.
        options = self.skip_type_options if allow_concat else [
            opt for opt in self.skip_type_options if opt != 1
        ]
        return random.choice(options)

    def sample_kernel_size(self) -> int:
        return random.choice(self.kernel_size_options)

    def sample_expansion(self) -> int:
        """Randomly sample the expansion factor."""
        return random.choice(self.expansion_options)


    def sample_block_params(self, allow_concat: bool = True) -> BlockParams:
        """
        Sample a complete set of parameters for a single block.

        Args:
            allow_concat (bool): Whether to allow 'concat' skip connection.

        Returns:
            BlockParams: Sampled block parameters.
        """
        # allow_concat limits concat skips to the last block in a unit.
        out_channels = self.sample_channel()
        groups = self.sample_groups()
        pool_type = self.sample_pool_type()
        pool_stride = self.sample_pool_stride()
        has_senet = self.sample_senet()
        activation_type = self.sample_activation()
        dropout_rate = self.sample_dropout()
        skip_type = self.sample_skip_type(allow_concat=allow_concat)
        kernel_size = self.sample_kernel_size()
        expansion = self.sample_expansion()
        return BlockParams(out_channels, groups, pool_type, pool_stride, has_senet,
                           activation_type, dropout_rate, skip_type, kernel_size, expansion)


class PopulationInitializer:
    """Creates initial valid individuals under constraints."""

    def __init__(self, search_space: SearchSpace):
        """
        Initialize the population initializer.

        Args:
            search_space (SearchSpace): Search space definition to sample from.
        """
        self.search_space = search_space

    def create_valid_individual(self) -> Optional[Individual]:
        """
        Create a single valid individual that satisfies all constraints.

        Returns:
            Individual: A valid, initialized individual.
        """
        while True:
            encoding = self._create_constrained_encoding()
            if not Encoder.validate_encoding(encoding):
                print(f"WARN: invalid individual; resampling encoding={encoding}")
                continue

            ok, reason, param_count = evaluate_encoding_params(encoding)
            if not ok:
                print(f"WARN: param bounds failed ({reason}); resampling")
                continue

            ind = Individual(encoding)
            ind.param_count = param_count
            return ind

    def _create_constrained_encoding(self) -> List[int]:
        """
        Generate a random encoding and attempt to enforce constraints during sampling.

        Returns:
            List[int]: A potential architecture encoding.
        """
        max_downsampling = Encoder.get_max_downsampling()
        unit_num = self.search_space.sample_unit_num()
        encoding = [unit_num]
        block_nums = []

        for _ in range(unit_num):
            block_num = self.search_space.sample_block_num()
            block_nums.append(block_num)
            encoding.append(block_num)

        downsampling_count = 0
        current_channels = config.INIT_CONV_OUT_CHANNELS

        for block_num in block_nums:
            for block_idx in range(block_num):
                # Only the last block in a unit may use concat skips.
                is_last_block = block_idx == block_num - 1
                # Try to generate valid block params, up to 10 attempts.
                accepted = False
                for _ in range(10):
                    block_params = self.search_space.sample_block_params(
                        allow_concat=is_last_block
                    )

                    final_stride = block_params.pool_stride
                    if downsampling_count >= max_downsampling and final_stride == 2:
                        block_params.pool_stride = 1
                        final_stride = 1

                    out_channels = block_params.out_channels * block_params.expansion
                    if block_params.skip_type == 1:  # concat
                        final_channels = out_channels + current_channels
                    else:
                        final_channels = out_channels

                    if final_channels <= config.MAX_CHANNELS:
                        current_channels = final_channels
                        encoding.extend(block_params.to_list())
                        if final_stride == 2:
                            downsampling_count += 1
                        accepted = True
                        break
                if accepted:
                    continue

                # Fall back to add skip if sampling fails repeatedly.
                if block_params.pool_stride == 2 and downsampling_count >= max_downsampling:
                    block_params.pool_stride = 1
                block_params.skip_type = 0

                # Ensure fallback channels do not blow up concat limits.
                max_allowed = max(
                    c for c in self.search_space.channel_options
                    if c * block_params.expansion <= config.MAX_CHANNELS
                )
                if block_params.out_channels * block_params.expansion > config.MAX_CHANNELS:
                    block_params.out_channels = max_allowed

                out_channels = block_params.out_channels * block_params.expansion
                final_channels = out_channels
                current_channels = final_channels
                final_stride = block_params.pool_stride
                if final_stride == 2:
                    downsampling_count += 1
                encoding.extend(block_params.to_list())

        return encoding


# Global instances
search_space = SearchSpace()
population_initializer = PopulationInitializer(search_space)
