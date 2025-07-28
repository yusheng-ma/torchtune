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
        # logger.debug(f"convert_messages_to_tokens - Messages: {messages}")
        return self._tokenizer({"messages": messages}, inference=True)["tokens"]

    @torch.inference_mode()
    def _generate_response(self, messages, cfg, custom_generate_next_token, agent_name="Agent"):
        """通用的響應生成函數"""
        tokens = self.convert_messages_to_tokens(messages)
        prompt_tensor = torch.tensor(tokens, dtype=torch.int, device=self._device)
        
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
        
        # 提取新生成的部分
        new_generated_tokens = generated_tokens[:, prompt_tensor.shape[0]:]
        new_generated_tokens = new_generated_tokens.tolist()
        response_text = self._tokenizer.decode(new_generated_tokens[0])
        
        t = time.perf_counter() - t0
        
        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        logger.info(f"[{agent_name}] {response_text}")
        
        tokens_generated = len(generated_tokens[0]) - prompt_tensor.size(0)
        tokens_sec = tokens_generated / t
        logger.info(
            f"[{agent_name}] Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"[{agent_name}] Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        if self._device.type != "cpu":
            torch_device = utils.get_torch_device_namespace()
            logger.info(
                f"[{agent_name}] Memory used: {torch_device.max_memory_allocated() / 1e9:.02f} GB"
            )

        return response_text, tokens_generated, t

    @torch.inference_mode()
    def agent1(self, cfg, custom_generate_next_token, initial_question):
        """第一個代理：回答初始問題"""
        # 使用從 cfg.prompt["user"] 讀取的問題
        messages = [
            Message(role="user", content=initial_question),
            Message(role="assistant", content="")
        ]
        return self._generate_response(messages, cfg, custom_generate_next_token, "Agent 1")

    @torch.inference_mode() # 添加裝飾器
    def agent2(self, cfg, custom_generate_next_token, initial_question, first_response):
        """第二個代理：基於第一個代理的回答進行回應"""
        # 使用從 cfg.prompt["user"] 讀取的問題
        user_content = f"{initial_question} This is the response from other thinkers: {first_response} for your reference."
        messages = [
            Message(role="user", content=user_content),
            Message(role="assistant", content="")
        ]
        return self._generate_response(messages, cfg, custom_generate_next_token, "Agent 2")

    @torch.inference_mode() # 添加裝飾器
    def agent3(self, cfg, custom_generate_next_token, initial_question, first_response, second_response):
        """第三個代理：基於前兩個代理的回答進行回應"""
        # 使用從 cfg.prompt["user"] 讀取的問題
        previous_responses = f"{first_response}; {second_response}"
        user_content = f"{initial_question} These are previous responses for reference: {previous_responses}"
        messages = [
            Message(role="user", content=user_content),
            Message(role="assistant", content="")
        ]
        return self._generate_response(messages, cfg, custom_generate_next_token, "Agent 3")

    @torch.inference_mode() # 添加裝飾器
    def summarizer(self, cfg, custom_generate_next_token, initial_question, first_resp, second_resp, third_resp):
        """總結者：基於三個代理的回答進行總結"""
        # 使用從 cfg.prompt["user"] 讀取的問題
        all_responses = f"{first_resp}; {second_resp}; {third_resp}"
        user_content = f"{initial_question} Please conclude the answer for this question based on the answers from the three thinkers: {all_responses}"
        messages = [
            Message(role="user", content=user_content),
            Message(role="assistant", content="")
        ]
        return self._generate_response(messages, cfg, custom_generate_next_token, "Final Summary")

    @torch.inference_mode()
    def agentgen(self, cfg: DictConfig):
        # 處理初始 prompt (如果有的話，例如系統訊息)
        original_prompt_dict = cfg.prompt
        messages = []
        if "system" in original_prompt_dict and original_prompt_dict["system"] is not None:
            messages.append(Message(role="system", content=original_prompt_dict["system"]))

        # 從 cfg.prompt["user"] 獲取初始問題
        initial_question = original_prompt_dict.get("user", "Please provide an answer.") # 提供默認值以防萬一

        # 設置 KV Cache (如果啟用)
        if cfg.enable_kv_cache:
            # 為了 KV Cache，我們需要一個粗略的最大長度估計
            # 這裡假設初始訊息 + 4次生成（3個代理+1個總結者）都不會超過這個長度
            dummy_tokens = self.convert_messages_to_tokens(messages + [
                Message(role="user", content=initial_question), # 使用動態問題
                Message(role="assistant", content="")
            ])
            dummy_prompt_len = len(dummy_tokens)
            estimated_max_total_len = dummy_prompt_len + 4 * cfg.max_new_tokens
            with self._device:
                self._model.setup_caches(
                    batch_size=1,
                    dtype=self._dtype,
                    decoder_max_seq_len=estimated_max_total_len,
                )

        custom_generate_next_token = None
        # if quantization: i dont care

        # --- 核心代理流程 (使用函數式方法) ---
        logger.info("================================ Multi-Agent Generation Start ================================")

        results = []
        # 第一個代理 - 傳入初始問題
        first_response, tokens_gen, t = self.agent1(cfg, custom_generate_next_token, initial_question)
        results.append({"agent": "Agent 1", "tokens": tokens_gen, "time": t, "response": first_response})
        if cfg.enable_kv_cache: self._model.reset_caches()

        # 第二個代理 - 傳入初始問題和第一個代理的回答
        second_response, tokens_gen, t = self.agent2(cfg, custom_generate_next_token, initial_question, first_response)
        results.append({"agent": "Agent 2", "tokens": tokens_gen, "time": t, "response": second_response})
        if cfg.enable_kv_cache: self._model.reset_caches()

        # 第三個代理 - 傳入初始問題、第一和第二個代理的回答
        third_response, tokens_gen, t = self.agent3(cfg, custom_generate_next_token, initial_question, first_response, second_response)
        results.append({"agent": "Agent 3", "tokens": tokens_gen, "time": t, "response": third_response})
        if cfg.enable_kv_cache: self._model.reset_caches()

        # --- 總結者步驟 ---
        final_summary, tokens_gen, t = self.summarizer(cfg, custom_generate_next_token,
                                       initial_question, # 傳入初始問題
                                       first_response, second_response, third_response)
        results.append({"agent": "Final Summary", "tokens": tokens_gen, "time": t, "response": final_summary})

        total_time = sum(r["time"] for r in results)
        total_tokens = sum(r["tokens"] for r in results)
        avg_time_per_step = total_time / len(results)
        overall_tps = total_tokens / total_time

        logger.info("================================ Multi-Agent Generation Summary ================================")
        for r in results:
            tps = r["tokens"] / r["time"] if r["time"] > 0 else 0
            logger.info(f"[{r['agent']}] Tokens: {r['tokens']}, Time: {r['time']:.02f}s, Speed: {tps:.02f} t/s")

        logger.info(
            f"[Overall] Total tokens: {total_tokens}, Total time: {total_time:.02f}s, "
            f"Overall speed: {overall_tps:.02f} tokens/sec"
        )

@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.agentgen(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
