# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, generation, training, utils
from torchtune.data import Message, Role
from torchtune.training import FullModelTorchTuneCheckpointer

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)

        training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)

        if self._quantization_mode is not None:
            if not isinstance(checkpointer, FullModelTorchTuneCheckpointer):
                raise ValueError(
                    "Quantization is only supported for models quantized and saved with the "
                    "FullModelTorchTuneCheckpointer - please ensure you have quantized your "
                    "model and are using the quantized weights!"
                )
            if "qat" in self._quantization_mode:
                raise ValueError(
                    "You have specified a quantizer with 'QAT' - "
                    "QAT quantizers should only be used during quantization aware training "
                    "and when quantizing models. Please use the corresponding post-training "
                    "quantizer e.g. Int8DynActInt4WeightQuantizer for Int8DynActInt4WeightQATQuantizer."
                )

        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)
            for k, v in model_state_dict.items():
                model_state_dict[k] = v.to(self._device)
            model.load_state_dict(model_state_dict, assign=True)
        else:
            model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        return model

    def convert_messages_to_tokens(
        self,
        messages: list[Message],
    ) -> list[int]:
        logger.debug(f"convert_messages_to_tokens - Messages: {messages}")
        return self._tokenizer({"messages": messages}, inference=True)["tokens"]

    @torch.inference_mode()
    def agentgen(self, cfg: DictConfig):
        original_prompt_dict = cfg.prompt

        messages = []
        if "system" in original_prompt_dict and original_prompt_dict["system"] is not None:
            messages.append(Message(role="system", content=original_prompt_dict["system"]))

        initial_user_message = Message(role="user", content=original_prompt_dict["user"])
        messages.append(initial_user_message)

        if cfg.enable_kv_cache:
            # 先建立一個假的初始 prompt tensor 來計算最大長度
            # 注意：這裡假設初始 prompt 不會太長。如果需要更精確，可以在第一次生成後再調整。
            # 或者，可以預估一個較大的總長度。
            dummy_tokens = self.convert_messages_to_tokens(messages)
            dummy_prompt_len = len(dummy_tokens)
            estimated_max_total_len = dummy_prompt_len + 4 * cfg.max_new_tokens # 粗略估計

            with self._device:
                 # 設置足夠大的 cache 以容納所有步驟 (或動態調整)
                self._model.setup_caches(
                    batch_size=1,
                    dtype=self._dtype,
                    decoder_max_seq_len=estimated_max_total_len,
                )

        custom_generate_next_token = None

        # if quantization: i dont care

        # --- 核心代理流程 ---
        agent_responses = [] # 用來儲存每個代理的回答
        all_messages_for_summarizer = list(messages) # 為總結者複製一份包含所有歷史的列表

        for agent_idx in range(3):
            current_agent_messages = list(messages)
            for i, resp in enumerate(agent_responses):
                current_agent_messages.append(Message(role="assistant", content=resp))

            current_agent_messages.append(Message(role="assistant", content=""))

            tokens = self.convert_messages_to_tokens(current_agent_messages)
            prompt_tensor = torch.tensor(tokens, dtype=torch.int, device=self._device)
            logger.debug(f"Prompt tensor shape for Agent {agent_idx + 1}: {prompt_tensor.shape}")

            t0 = time.perf_counter()
            generated_tokens, _ = generation.generate(
                model=self._model,
                prompt=prompt_tensor,
                max_generated_tokens=cfg.max_new_tokens,
                pad_id=self._tokenizer.pad_id,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                compiled_generate_next_token=custom_generate_next_token,
            )
            generated_tokens = generated_tokens.tolist()
            t = time.perf_counter() - t0

            response_text = self._tokenizer.decode(generated_tokens[0])
            agent_responses.append(response_text)
            all_messages_for_summarizer.append(Message(role="assistant", content=response_text))

            logger.info(f"\n--- Agent {agent_idx + 1} Response ---")
            logger.info(response_text)
            logger.info(f"Agent {agent_idx + 1} Time: {t:.02f} sec")

        # --- 總結者步驟 ---
        # all_messages_for_summarizer 現在已經包含了系統訊息、初始問題、三個代理的回答
        # 我們可以添加一個特殊的指令給總結者
        summarizer_instruction = "\n\nPlease provide a concise summary of the above discussion, considering the initial question and all three perspectives."
        all_messages_for_summarizer.append(Message(role="user", content=summarizer_instruction))
        # 添加空的 assistant 訊息以啟動總結者的生成
        all_messages_for_summarizer.append(Message(role="assistant", content=""))

        # Tokenize 總結者的 prompt
        summary_tokens = self.convert_messages_to_tokens(all_messages_for_summarizer)
        summary_prompt_tensor = torch.tensor(summary_tokens, dtype=torch.int, device=self._device)

        # --- 執行總結者生成 ---
        t0_summary = time.perf_counter()
        final_generated_tokens, _ = generation.generate(
            model=self._model,
            prompt=summary_prompt_tensor,
            max_generated_tokens=cfg.max_new_tokens, # 可以為總結者設置不同的 max_new_tokens
            pad_id=self._tokenizer.pad_id,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=self._tokenizer.stop_tokens,
            compiled_generate_next_token=custom_generate_next_token,
        )
        final_generated_tokens = final_generated_tokens.tolist()
        t_summary = time.perf_counter() - t0_summary

        # 解碼並輸出最終總結
        final_summary_text = self._tokenizer.decode(final_generated_tokens[0])

        logger.info("\n--- Final Summary ---")
        logger.info(final_summary_text)
        logger.info(f"Summary Time: {t_summary:.02f} sec")

        # --- (可選) 輸出性能指標 ---
        # 注意：這裡的性能指標會比較複雜，因為涉及多次生成。
        # 可以選擇輸出最後一次生成的時間，或者計算總時間。
        # model_size 計算保持不變
        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )
        # 計算最後一次生成的 tokens/sec (僅供參考)
        tokens_generated_summary = len(final_generated_tokens[0]) - summary_prompt_tensor.size(0)
        tokens_sec_summary = tokens_generated_summary / t_summary if t_summary > 0 else 0
        logger.info(
            f"(Summary Generation) Time: {t_summary:.02f} sec, Tokens/sec: {tokens_sec_summary:.02f}"
        )
        logger.info(f"Bandwidth achieved (estimate): {model_size * tokens_sec_summary / 1e9:.02f} GB/s")
        if self._device.type != "cpu":
            torch_device = utils.get_torch_device_namespace()
            logger.info(
                f"Peak Memory used: {torch_device.max_memory_allocated() / 1e9:.02f} GB"
            )


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    logger.info("================================agentgen start================================")
    recipe.agentgen(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
