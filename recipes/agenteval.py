# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time

from typing import Union

import PIL

import torch

import pandas as pd
import json
from typing import Any, Dict, List
from collections import defaultdict
from lm_eval.evaluator_utils import (
    get_sample_size,
    get_task_list,
)

from lm_eval.evaluator import evaluate
from lm_eval.models.hf_vlms import HFMultimodalLM
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict, TaskManager
from lm_eval.utils import make_table
from omegaconf import DictConfig
from torchtune import config, training, utils
from torchtune.data import (
    format_content_with_images,
    left_pad_sequence,
    Message,
    padded_collate_tiled_images_and_mask,
)
from torchtune.generation import generate, sample
from torchtune.modules import TransformerDecoder
from torchtune.modules.common_utils import local_kv_cache
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.modules.transforms import Transform

from torchtune.modules.transforms.tokenizers import (
    HuggingFaceModelTokenizer,
    ModelTokenizer,
)
from torchtune.recipe_interfaces import EvalRecipeInterface
from torchtune.training import FullModelTorchTuneCheckpointer


class _LLMEvalWrapper(HFLM):
    """An EvalWrapper for EleutherAI's eval harness based on gpt-fast's
    EvalWrapper: https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py.

    Note:
        This is for text-only decoder models.

    Args:
        model (TransformerDecoder): The model to evaluate.
        tokenizer (ModelTokenizer): Tokenizer associated with the model being evaluated.
            This should be the same tokenizer used when fine-tuning the model.
        device (torch.device): The device to use.
        max_seq_length (int): The maximum sequence length to use.
        batch_size (int): The batch size per GPU to use.
        dtype (torch.dtype): dtype for the model caches during generation.
        enable_kv_cache (bool): Whether to enable KV cache for generation.
    """

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: ModelTokenizer,
        *,
        device: torch.device,
        max_seq_length: int = 4096,
        batch_size: int = 8,
        dtype: torch.dtype = torch.float32,
        enable_kv_cache: bool = True,
    ):
        # TODO (@joecummings): Remove this init function so we don't load in extraneous stuff
        super().__init__(pretrained="gpt2", device=str(device))
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype
        self._enable_kv_cache = enable_kv_cache

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def enable_kv_cache(self):
        return self._enable_kv_cache

    def tok_encode(self, text: str, **kwargs) -> list[int]:
        print("get in tok_encode") # Yes
        # Note on add_bos flag: setting to False as this gives better results, for example
        # +1% on truthfulqa_mc2 with a LoRA finetune. lit-gpt also sets this to False,
        # see https://github.com/Lightning-AI/lit-gpt/blob/main/eval/lm_eval_harness.py#L66,
        # though notably fast-gpt does the opposite
        # https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py#L123.
        if isinstance(self._tokenizer, HuggingFaceModelTokenizer):
            return self._tokenizer.base_tokenizer.encode(
                text=text, add_bos=False, add_eos=False
            )
        return self._tokenizer.encode(text=text, add_bos=False, add_eos=False)

    def tok_batch_encode(
        self, text: list[str], left_truncate_len: int = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        print("get in tok_batch_encode") # No?
        tokenized_text = [self.tok_encode(x) for x in text]

        # pad left
        x = left_pad_sequence(
            [torch.tensor(x) for x in tokenized_text],
            batch_first=True,
            padding_value=self._tokenizer.pad_id,
        )

        # the harness will use left_truncate_len to indicate that the current batch
        # needs to be truncated to self.max_seq_len - self.max_gen_toks
        if left_truncate_len is not None:
            x = x[:, -left_truncate_len:]

        return x, torch.ones_like(x)  # return 'mask' b/c it's expected by the harness

    def tok_decode(self, tokens: Union[list[int], int], **kwargs) -> str:
        print("get in tok_decode") # No?
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        print("get in _model_call") # Yes
        return self._model(inps)

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        if hasattr(self._tokenizer, "prompt_template"):
            return self._tokenizer.prompt_template(chat_history)
        if isinstance(self.tokenizer, HuggingFaceModelTokenizer):
            return self.tokenizer.render_template(chat_history)
        raise ValueError(
            "You can't use a tokenizer without a prompt template and apply_chat_template: True. "
            "Use HuggingFaceModelTokenizer if you do not require a custom one."
        )

    @torch.inference_mode()
    def _model_generate(
        self, context: torch.Tensor, **generation_kwargs
    ) -> torch.Tensor:
        print("get in _model_generate") # No? I guess its for gen job!
        bsz, seq_len = context.shape

        temperature = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", False)
        if do_sample or temperature != 0.0:
            raise RuntimeError(
                "Any decoding strategy other than greedy is not supported."
            )

        # if we've recieved fewer than self._batch_size samples in the current
        # batch we need to pad the batch out. here we're padding the end of the
        # current batch to the correct length. this is because when we use static
        # KV-caches, the model will expect a fixed batch size for all samples.
        maybe_padded_context = torch.nn.functional.pad(
            context,
            (0, 0, 0, self._batch_size - bsz),
            value=self._tokenizer.eos_id,  # pad with one of the tokenizer's stop tokens so generation can stop early
        )
        with local_kv_cache(
            self.model,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self._dtype,
            decoder_max_seq_len=self.max_length,
        ):
            toks, _ = generate(
                self.model,
                maybe_padded_context,
                max_generated_tokens=self.max_gen_toks,
                temperature=temperature,
                top_k=None,
                pad_id=self._tokenizer.pad_id,
                stop_tokens=self._tokenizer.stop_tokens,
            )
        return toks[:bsz]


class EleutherEvalRecipe(EvalRecipeInterface):
    """
    This recipe runs evaluation on a trained model using EleutherAI's eval harness.
    This assumes the user has the EleutherAI eval harness installed. See
    https://github.com/EleutherAI/lm-evaluation-harness for more details.

    Features:
        - Single GPU evaluation. Multi-GPU evaluation is currently not supported.
        - Quantization (for text-only models) is supported.
        - Any task from the EleutherAI eval harness

    We recommend launching evaluation using the tune CLI::

        tune run eleuther_eval --config eleuther_evaluation \
            tasks=["truthfulqa_mc2","hellaswag"] \
            limit=50 \
    """

    def __init__(self, cfg: DictConfig) -> None:
        # Double check we have the right Eval Harness version
        from importlib.metadata import version

        if version("lm-eval") < "0.4.5" or version("lm-eval") > "0.4.8":
            raise RuntimeError(
                "This recipe requires EleutherAI Eval Harness between v0.4.5 - 0.4.8."
                "Please install with `pip install lm-eval==0.4.8`"
            )

        # General variable initialization
        self.device = utils.get_device(device=cfg.device)
        self.dtype = training.get_dtype(dtype=cfg.dtype, device=self.device)
        self.logger = utils.get_logger(cfg.get("log_level", "info"))
        training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )

        # Eval specific variables
        self.limit = cfg.limit
        self.tasks = list(cfg.tasks)
        self.batch_size = cfg.batch_size
        self.enable_kv_cache = cfg.get("enable_kv_cache", True)
        self.include_path = cfg.get("include_path", None)
        self.apply_chat_template = cfg.get("chat_template", False)

        self.enable_multi_agent_eval = cfg.get("enable_multi_agent_eval", False)
        self.max_new_tokens = cfg.get("max_new_tokens", 512)
        self.temperature = cfg.get("temperature", 0.7)
        self.top_k = cfg.get("top_k", 50)

    def setup(self, cfg: DictConfig) -> None:
        # Initialize quantizer and quantization mode
        quantizer = config.instantiate(cfg.quantizer)
        quantization_mode = training.get_quantizer_mode(quantizer)

        # Load checkpoint
        checkpointer = config.instantiate(cfg.checkpointer)

        # Initialize model
        with training.set_default_dtype(self.dtype), self.device:
            model = config.instantiate(cfg.model)

        # Quantize model if requested
        if quantization_mode is not None:
            if not isinstance(checkpointer, FullModelTorchTuneCheckpointer):
                raise ValueError(
                    "Quantization is only supported for models quantized and saved with the "
                    "FullModelTorchTuneCheckpointer - please ensure you have quantized your "
                    "model and are using the quantized weights!"
                )
            if "qat" in quantization_mode:
                raise ValueError(
                    "You have specified a quantizer with 'QAT' - "
                    "QAT quantizers should only be used during quantization aware training "
                    "and when quantizing models. Please use the corresponding post-training "
                    "quantizer e.g. Int8DynActInt4WeightQuantizer for Int8DynActInt4WeightQATQuantizer."
                )
            model = quantizer.quantize(model)
            model = model.to(device=self.device, dtype=self.dtype)
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)[
                training.MODEL_KEY
            ]
            for k, v in ckpt_dict.items():
                ckpt_dict[k] = v.to(self.device)
            model.load_state_dict(ckpt_dict, assign=True)
        else:
            ckpt_dict = checkpointer.load_checkpoint()[training.MODEL_KEY]
            model.load_state_dict(ckpt_dict)

        # Load model weights into initialized model
        self.logger.info(f"Model is initialized with precision {self.dtype}.")

        # Put model in eval mode.
        # Note: This will not disable the dropout applied in SDPA,
        # see https://github.com/pytorch/pytorch/issues/124464
        model.eval()

        # Initialize tokenizer/transform
        model_transform = config.instantiate(cfg.tokenizer)

        # Finally, we setup the actual EvalWrapper class
        if isinstance(model, DeepFusionModel):
            pass # i dont care about vlm
            # eleuther_model_wrapper = _VLMEvalWrapper
            # if not self.enable_kv_cache:
            #     self.logger.debug(
            #         "Received enable_kv_cache=False, but KV cache is required for running "
            #         "multimodal generation in a timely manner. Setting enable_kv_cache=True."
            #     )
        elif isinstance(model, TransformerDecoder):
            eleuther_model_wrapper = _LLMEvalWrapper
        self.eleuther_model_wrapper = eleuther_model_wrapper(
            model,
            model_transform,
            device=self.device,
            max_seq_length=cfg.max_seq_length,
            batch_size=self.batch_size,
            dtype=self.dtype,
            enable_kv_cache=self.enable_kv_cache,
        )

    def _run_multi_agent_loglikelihood(self, reqs: List, instance_idx_to_task: Dict[int, Any]) -> List[torch.Tensor]:
        """
        對一批 loglikelihood 請求執行多智能體協作評估。
        Args:
            reqs: List of Instance objects with request_type 'loglikelihood'.
            instance_idx_to_task: A dictionary mapping instance.idx to its parent Task.
        Returns:
            List of torch.Tensor, the log likelihoods for each request.
        """
        responses = []
        print(f"get in _run_multi_agent_loglikelihood, # of reqs: {len(reqs)}") # i guess 4
        for req in reqs:
            # 獲取必要的信息
            doc = req.doc
            # === 關鍵修正：使用 req.idx 作為鍵 ===
            task = instance_idx_to_task[req.idx] # 這裡是 idx，不是 index
            choices = task.doc_to_choice(doc) # 現在 task 是有效的

            # --- 關鍵修改：確保 context_tensor 是 tensor ---
            # req.args[0] 可能是 str 或 torch.Tensor
            context_input = req.args[0]
            if isinstance(context_input, str):
                # 如果是字符串，用 wrapper 的 tok_encode 將其轉換為 tensor
                context_tensor = torch.tensor(
                    self.eleuther_model_wrapper.tok_encode(context_input),
                    device=self.device
                ).unsqueeze(0) # 增加 batch 維度
            elif isinstance(context_input, torch.Tensor):
                # 如果已經是 tensor，直接使用
                context_tensor = context_input.to(self.device)
            else:
                raise TypeError(f"Unexpected type for context: {type(context_input)}")

            # 現在 context_tensor 肯定是 tensor 了，可以安全解碼
            context_str = self.eleuther_model_wrapper.tok_decode(context_tensor[0].tolist())

            # --- 智能體 1: Thinker ---
            thinker_prompt = f"{context_str}\n\nPlease act as a thoughtful expert. Analyze the question and each option carefully. Explain your reasoning step by step for each option, and then give your best prediction (only the letter)."
            thinker_tensor = self.eleuther_model_wrapper.tok_encode(thinker_prompt)
            thinker_tensor = torch.tensor([thinker_tensor], device=self.device)
            with local_kv_cache(self.eleuther_model_wrapper.model, batch_size=1, device=self.device, dtype=self.dtype, decoder_max_seq_len=self.eleuther_model_wrapper.max_length):
                thinker_output, _ = generate(
                    self.eleuther_model_wrapper.model,
                    thinker_tensor,
                    max_generated_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    pad_id=self.eleuther_model_wrapper._tokenizer.pad_id,
                    stop_tokens=self.eleuther_model_wrapper._tokenizer.stop_tokens,
                )
            thinker_response = self.eleuther_model_wrapper.tok_decode(thinker_output[0].tolist())
            print("finish agent 1")
            # --- 智能體 2: Critic ---
            critic_prompt = f"{context_str}\n\nHere is an analysis from a fellow expert:\n{thinker_response}\n\nPlease act as a critical reviewer. Identify any flaws, biases, or errors in the above analysis. Do you agree with the final prediction? If not, explain why and provide your own reasoning. Then give your own prediction (only the letter)."
            critic_tensor = self.eleuther_model_wrapper.tok_encode(critic_prompt)
            critic_tensor = torch.tensor([critic_tensor], device=self.device)
            with local_kv_cache(self.eleuther_model_wrapper.model, batch_size=1, device=self.device, dtype=self.dtype, decoder_max_seq_len=self.eleuther_model_wrapper.max_length):
                critic_output, _ = generate(
                    self.eleuther_model_wrapper.model,
                    critic_tensor,
                    max_generated_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    pad_id=self.eleuther_model_wrapper._tokenizer.pad_id,
                    stop_tokens=self.eleuther_model_wrapper._tokenizer.stop_tokens,
                )
            critic_response = self.eleuther_model_wrapper.tok_decode(critic_output[0].tolist())
            print("finish agent 2")
            # --- 智能體 3: Final Judge ---
            judge_prompt = f"{context_str}\n\nHere are two expert analyses:\n1. Thinker: {thinker_response}\n2. Critic: {critic_response}\n\nPlease act as the final judge. Synthesize both viewpoints. Which prediction do you find more convincing? Give a final, definitive prediction (only the letter)."
            judge_tensor = self.eleuther_model_wrapper.tok_encode(judge_prompt)
            judge_tensor = torch.tensor([judge_tensor], device=self.device)
            with local_kv_cache(self.eleuther_model_wrapper.model, batch_size=1, device=self.device, dtype=self.dtype, decoder_max_seq_len=self.eleuther_model_wrapper.max_length):
                judge_output, _ = generate(
                    self.eleuther_model_wrapper.model,
                    judge_tensor,
                    max_generated_tokens=64,
                    temperature=0.0, # Greedy
                    pad_id=self.eleuther_model_wrapper._tokenizer.pad_id,
                    stop_tokens=self.eleuther_model_wrapper._tokenizer.stop_tokens,
                )
            judge_prediction = self.eleuther_model_wrapper.tok_decode(judge_output[0].tolist()).strip()
            print("finish agent 3")
            # --- 解析最終預測並生成 log likelihoods ---
            num_choices = len(choices)
            fake_log_likelihoods = torch.full((1, num_choices), -1000.0, device=self.device)
            import re
            match = re.search(r'\b([A-D])\b', judge_prediction)
            if match:
                chosen_idx = ord(match.group(1)) - ord('A')
                if 0 <= chosen_idx < num_choices:
                    fake_log_likelihoods[0, chosen_idx] = 0.0
            else:
                fake_log_likelihoods[0, 0] = 0.0 # 默認

            # 重複以匹配 req.repeats
            for _ in range(req.repeats):
                responses.append(fake_log_likelihoods)

        return responses

    def agenteval(self) -> None:
        # === 新的多智能體評估流程 ===
        task_manager = TaskManager(include_path=self.include_path)
        task_dict = get_task_dict(self.tasks, task_manager)
        eval_tasks = get_task_list(task_dict) # 從 evaluator.py 來的

        # 2. Build All Requests (關鍵步驟)
        requests = defaultdict(list)
        # === 修正：使用 instance.idx 作為鍵 ===
        instance_idx_to_task = {}
        
        for task_output in eval_tasks:
            task = task_output.task
            limit = get_sample_size(task, self.limit)
            # 這會填充 task.instances
            task.build_all_requests(
                limit=limit,
                rank=0, # 假設單GPU
                world_size=1,
                cache_requests=False,
                rewrite_requests_cache=False,
                system_instruction=None,
                apply_chat_template=self.apply_chat_template,
                fewshot_as_multiturn=False,
                chat_template=self.eleuther_model_wrapper.apply_chat_template if self.apply_chat_template else None,
                tokenizer_name="",
            )
            # 按 reqtype 分類
            for instance in task.instances:
                requests[instance.request_type].append(instance)
                # === 關鍵修正：使用 instance.idx 作為鍵 ===
                instance_idx_to_task[instance.idx] = task # 這裡是 idx，不是 index

        # 3. 執行多智能體評估 (模擬 getattr(lm, reqtype)(cloned_reqs))
        all_responses = {}
        for reqtype, reqs in requests.items():
            self.logger.info(f"Running multi-agent {reqtype} evaluation")
            if reqtype == "loglikelihood":
                responses = self._run_multi_agent_loglikelihood(reqs, instance_idx_to_task)
            else:
                # 對於其他類型（如 generate_until），使用原始模型
                cloned_reqs = [req for req in reqs for _ in range(req.repeats)]
                inps = torch.stack([req.args[0] for req in cloned_reqs]) # 假設 args[0] 是 context_tensor
                outs = self.eleuther_model_wrapper._model_call(inps)
                responses = [outs[i] for i in range(len(cloned_reqs))]
            all_responses[reqtype] = responses

        # 4. 將響應填回 instances
        response_idx = 0
        for reqtype, reqs in requests.items():
            for req in reqs:
                for _ in range(req.repeats):
                    req.resps.append(all_responses[reqtype][response_idx])
                    response_idx += 1

        # 5. 後處理和結果聚合 (直接複用 evaluator.py 的代碼)
        # 這部分可以直接調用 evaluator.py 的 consolidate_results 等函數
        for task_output in eval_tasks:
            task = task_output.task
            task.apply_filters()
            # ... (process_results, calculate_aggregate_metric 等)
            # 由於代碼複雜，我們可以先只實現 loglikelihood 的多智能體流程，並手動構建輸出。

        # 6. 構建最終輸出並調用 log_detailed_results
        # 由於完全複製後處理很複雜，我們可以先手動構建一個簡化的 output 字典
        output = {
            "results": {},
            "configs": {task_output.task_name: task_output.task.config for task_output in eval_tasks},
            "versions": {task_output.task_name: getattr(task_output.task, "VERSION", "N/A") for task_output in eval_tasks},
            "samples": {}
        }
        # 填充 samples
        for task_output in eval_tasks:
            task_name = task_output.task_name
            output["samples"][task_name] = []
            instances_by_doc_id = defaultdict(list)
            for instance in task_output.task.instances:
                instances_by_doc_id[instance.doc_id].append(instance)
            for doc_id, instances in instances_by_doc_id.items():
                instances.sort(key=lambda x: x.idx)
                sample = {
                    "doc": instances[0].doc,
                    "arguments": [req.args for req in instances],
                    "resps": [req.resps for req in instances],
                    "filtered_resps": [req.filtered_resps[list(req.filtered_resps.keys())[0]] for req in instances] if instances[0].filtered_resps else [req.resps for req in instances],
                }
                output["samples"][task_name].append(sample)
        print(output)
        # 調用你已有的日誌函數
        df = log_detailed_results(
            output,
            output_file="agenteval_detailed.json",
            md_output_file="agenteval_report.md",
            mode="loglikelihood"
        )

        # 顯示前 5 筆
        self.logger.info("\n\n📌 First 5 detailed results:")
        self.logger.info("\n" + df[["question", "is_correct"]].head(5).to_string(index=False))

        formatted_output = make_table(output)
        self.logger.info(f"\n\n{formatted_output}\n")


    def evaluate(self) -> None:
        # Initialize tasks for the harness
        task_manager = TaskManager(include_path=self.include_path)
        task_dict = get_task_dict(self.tasks, task_manager)

        # Run evaluation
        t0 = time.time()
        self.logger.info(f"Running evaluation on the following tasks: {self.tasks}")
        self.logger.info(task_dict)
        self.logger.info(self.apply_chat_template)
        output = evaluate(
            self.eleuther_model_wrapper,
            task_dict,
            apply_chat_template=self.apply_chat_template,
            limit=self.limit,
            write_out=True,

        )
        t1 = time.time() - t0

        # Log metrics
        self.logger.info(f"Eval completed in {t1:.02f} seconds.")
        if self.device.type != "cpu" and self.device.type != "mps":
            torch_device = utils.get_torch_device_namespace()
            self.logger.info(
                f"Max memory allocated: {torch_device.max_memory_allocated() / 1e9:.02f} GB"
            )

        # print(output)
        df = log_detailed_results(
            output,
            output_file="eval_detailed.json",
            md_output_file="eval_report.md",
            mode="loglikelihood"
        )

        # 顯示前 5 筆
        self.logger.info("\n\n📌 First 5 detailed results:")
        self.logger.info("\n" + df[["question", "is_correct"]].head(5).to_string(index=False))

        formatted_output = make_table(output)
        self.logger.info(f"\n\n{formatted_output}\n")


def log_detailed_results(
    output: Dict[str, Any],
    output_file: str = "detailed_results.json",
    md_output_file: str = "detailed_report.md",
    mode: str = "loglikelihood",  # 或 "generation"
) -> pd.DataFrame:
    """
    從 EleutherAI eval harness 的 output 中提取每一題的詳細結果，
    包括問題、選項、log likelihood、是否正確，並輸出為 JSON 和美觀 Markdown。
    支援 loglikelihood 和 future generation mode。
    """
    records = []
    prompts_and_responses = []  # 用於 Markdown 的 rich 展示

    for task_name, task_data in output["samples"].items():
        config = output["configs"][task_name]
        version = output["versions"].get(task_name, "N/A")

        for idx, sample in enumerate(task_data):
            doc = sample["doc"]
            question = doc["question"]
            choices = doc.get("mc1_targets", {}).get("choices", doc.get("mc2_targets", {}).get("choices", []))
            labels = doc.get("mc1_targets", {}).get("labels", doc.get("mc2_targets", {}).get("labels", []))
            true_indices = [i for i, lbl in enumerate(labels) if lbl == 1]

            # === 根據 mode 分支處理 ===
            if mode == "loglikelihood":
                try:
                    pred_log_likelihoods = [resp[0][0] for resp in sample["resps"]]
                except ValueError: # with my agenteval... only one element tensors can be converted to Python scalars
                    first_resp = sample["resps"][0]  # 假設所有響應相同
                    # first_resp 應該是像 [tensor([[ll_A, ll_B, ...]])] 這樣的列表
                    # 所以 first_resp[0] 是 tensor([[ll_A, ll_B, ...]])
                    # 我們需要 squeeze 成一維並轉換為 Python list
                    pred_log_likelihoods_tensor = first_resp[0].squeeze(0)  # 形狀: (num_choices,)
                    pred_log_likelihoods = pred_log_likelihoods_tensor.tolist() # 轉換為 Python list
                pred_idx = int(torch.argmax(torch.tensor(pred_log_likelihoods)).item())
                is_correct = pred_idx in true_indices
                probs = torch.softmax(torch.tensor(pred_log_likelihoods), dim=0)
                max_prob = probs[pred_idx].item()

                # 提取 prompt：使用 arguments[0][0]（所有選項共享同一個 prompt）
                prompt = sample["arguments"][0][0].strip()
                generated_text = choices[pred_idx]
                log_likelihoods = {f"choice_{i}": float(ll) for i, ll in enumerate(pred_log_likelihoods)}

            elif mode == "generation":
                # TODO: 未來實現自由生成的分析
                # 假設 output 格式會有: sample["generated_text"]
                raise NotImplementedError("Generation mode not implemented yet. Use mode='loglikelihood'.")

            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # === 準備 JSON 記錄（完整資料）===
            records.append({
                "task": task_name,
                "version": version,
                "question": question,
                "choices": choices,
                "correct_indices": true_indices,
                "model_predicted_index": pred_idx,
                "model_predicted_text": choices[pred_idx],
                "is_correct": is_correct,
                "max_prob": round(max_prob, 4) if mode == "loglikelihood" else None,
                "log_likelihoods": log_likelihoods if mode == "loglikelihood" else None,
                "prompt": prompt if mode == "loglikelihood" else None,
                "generated_text": generated_text if mode == "loglikelihood" else None,
            })

            # === 準備 Markdown 用的 rich 展示 ===
            result = "✅ Correct" if is_correct else "❌ Incorrect"
            short_question = question[:50] + "..." if len(question) > 50 else question

            prompts_and_responses.append({
                "index": idx,
                "task": task_name,
                "short_question": short_question,
                "result": result,
                "prompt": prompt,
                "predicted_text": choices[pred_idx],
                "correct_indices": true_indices,
                "choices": choices,
            })

    # === 保存 JSON ===
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # === 生成 Markdown 報告 ===
    _generate_detailed_markdown_report(prompts_and_responses, md_output_file, records)

    # === 返回 DataFrame ===
    df = pd.DataFrame(records)
    print(f"\n✅ Detailed results saved to:")
    print(f"   - {output_file}")
    print(f"   - {md_output_file}")
    print(f"   Total samples: {len(df)}")

    return df

def _generate_detailed_markdown_report(data: List[Dict], md_file: str, full_records: List[Dict]):
    """生成美觀的 Markdown 報告，包含摘要表格和每題詳細 prompt 分析。"""
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# 📊 Evaluation Detailed Report\n\n")

        # === 摘要統計 ===
        total = len(data)
        correct = sum(1 for d in data if "✅" in d["result"])
        accuracy = correct / total if total > 0 else 0
        f.write(f"**Summary**: {correct}/{total} correct ({accuracy:.1%})\n\n")

        # === 摘要表格（簡潔）===
        f.write("## 📈 Summary Table\n\n")
        f.write("| # | Task | Question | Result |\n")
        f.write("|---|------|----------|--------|\n")
        for d in data:
            f.write(f"| {d['index']} | `{d['task']}` | {d['short_question']} | {d['result']} |\n")
        f.write("\n")

        # === 每題詳細分析 ===
        f.write("## 🧩 Detailed Analysis\n\n")
        for d in data:
            f.write(f"### Question {d['index']} ({d['task']})\n")
            f.write(f"**Result**: {d['result']}\n\n")
            f.write("**Question**: " + d["short_question"] + "\n\n")

            # Correct choices
            correct_choices = [d["choices"][i] for i in d["correct_indices"]]
            f.write("**Correct Answer(s)**:\n")
            for c in correct_choices:
                f.write(f"- ✅ `{c}`\n")
            f.write("\n")

            # Model prediction
            f.write(f"**Model Prediction**: `{d['predicted_text']}`\n\n")

            # Prompt (用 collapsible block 收起來)
            f.write("<details>\n")
            f.write("<summary>🔍 Show Prompt</summary>\n\n")
            f.write("```\n")
            f.write(d["prompt"].replace("`", "\\`") + "\n")
            f.write("```\n")
            f.write("</details>\n\n")

            f.write("---\n\n")

    print(f"✅ Markdown report generated at {md_file}")


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    config.log_config(recipe_name="EleutherEvalRecipe", cfg=cfg)
    recipe = EleutherEvalRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate()
    recipe.agenteval()


if __name__ == "__main__":
    sys.exit(recipe_main())
