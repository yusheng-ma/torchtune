# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

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

from torchtune._my_evaluator import my_evaluate
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

import copy
from typing import List, Tuple
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.models.utils import Collator, handle_stop_sequences
import torch

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
        # print("get in tok_encode") # Yes
        # print(text)
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
        # print("get in tok_batch_encode") # No?
        # print(text)
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
        # print("get in tok_decode") # No?
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        # print("get in _model_call") # Yes
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
        # print("get in _model_generate") # No? I guess its for gen job!
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
    
    @torch.inference_mode()
    def _generate_responses(
        self, contexts: List[str], max_ctx_len: int, max_gen_toks: int, until: List[str], kwargs: Dict[str, Any],
    ) -> List[str]:
        """通用的生成回應方法，適用於任何 context list"""
        # encode, pad, and truncate
        context_enc, attn_masks = self.tok_batch_encode(
            contexts,
            left_truncate_len=max_ctx_len,
            truncation=self.truncation,
        )
        context_enc = context_enc.to(self.device)
        attn_masks = attn_masks.to(self.device)

        if "max_length" not in kwargs:
            kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

        # 生成
        cont = self._model_generate(
            context=context_enc,
            attention_mask=attn_masks,
            stop=until,
            **kwargs,
        )
        cont_toks_list = cont.tolist()

        # 解碼與後處理
        responses = []
        for cont_toks in cont_toks_list:
            if self.backend == "causal":
                cont_toks = cont_toks[context_enc.shape[1]:]
            s = self.tok_decode(cont_toks)
            # 使用 stop sequences 截斷
            for term in until:
                if len(term) > 0:
                    s = s.split(term)[0]
            responses.append(s.strip())
        return responses

    @torch.inference_mode()
    def agent1(
        self, contexts: List[str], max_ctx_len: int, max_gen_toks: int, until: List[str], kwargs: Dict[str, Any],
    ) -> List[str]:
        """第一輪思考：直接對原始問題生成回應"""
        return self._generate_responses(contexts, max_ctx_len, max_gen_toks, until, kwargs)

    @torch.inference_mode()
    def agent2(
        self, contexts: List[str], first_responses: List[str], max_ctx_len: int, max_gen_toks: int, until: List[str], kwargs: Dict[str, Any],
    ) -> List[str]:
        """第二輪思考：基於第一輪回應重新生成"""
        def _build_agent2_context(self, original_context: str, first_response: str) -> str:
            return (
                "Write a solution to the following problem and make sure that it passes the tests. "
                "This is the response from other thinkers: {resp} for your reference.\n"
                "```python\n{ctx}\n```"
            ).format(resp=first_response, ctx=original_context)
        
        second_contexts = [
            _build_agent2_context(ctx, resp)
            for ctx, resp in zip(contexts, first_responses)
        ]
        return self._generate_responses(second_contexts, max_ctx_len, max_gen_toks, until, kwargs)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        # print("get in generate_until here pls")
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                assert max_ctx_len > 0, (
                    f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
                )
            elif self.backend == "seq2seq":
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            first_responses = self.agent1(contexts, max_ctx_len, max_gen_toks, until, kwargs)
            second_responses = self.agent2(contexts, first_responses, max_ctx_len, max_gen_toks, until, kwargs)

            # === final ===
            for resp, context in zip(second_responses, contexts):
                res.append(resp)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), resp)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

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


    def agenteval(self) -> None:
        # Initialize tasks for the harness
        task_manager = TaskManager(include_path=self.include_path)
        task_dict = get_task_dict(self.tasks, task_manager)

        # Run evaluation
        t0 = time.time()
        self.logger.info(f"Running evaluation on the following tasks: {self.tasks}")
        self.logger.info(task_dict)
        self.logger.info(self.apply_chat_template)
        output = my_evaluate(
            self.eleuther_model_wrapper,
            task_dict,
            apply_chat_template=self.apply_chat_template,
            limit=self.limit,
            write_out=True,
            confirm_run_unsafe_code=True, # humaneval
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
            mode="generation"
        )

        # 顯示前 5 筆
        self.logger.info("\n\n📌 First 5 detailed results:")
        summary_df = df[["problem_id", "entry_point", "is_correct"]].copy()
        summary_df["is_correct"] = summary_df["is_correct"].map({True: "✅ Passed", False: "❌ Failed"})
        self.logger.info("\n" + summary_df.head(5).to_string(index=False))

        formatted_output = make_table(output)
        self.logger.info(f"\n\n{formatted_output}\n")


def log_detailed_results(
    output: Dict[str, Any],
    output_file: str = "eval_detailed.json",
    md_output_file: str = "eval_report.md",
    mode: str = "generation"  # 支援 "generation" 模式，特別用於 code generation
) -> pd.DataFrame:
    """
    從 EleutherAI eval harness 的 output 中提取程式碼生成任務（如 humaneval）的詳細結果。
    支援 generation 模式，輸出模型生成的程式碼、測試結果、正確與否等資訊。
    """
    records = []
    prompts_and_responses = []

    for task_name, task_samples in output["samples"].items():
        config = output["configs"][task_name]
        version = output["versions"].get(task_name, "N/A")

        for idx, sample in enumerate(task_samples):
            doc = sample["doc"]
            prompt = doc["prompt"]           # 包含 typing 和函數 signature
            entry_point = doc["entry_point"] # 函數名稱
            test_code = doc["test"]          # 檢查用的測試程式碼
            canonical_solution = doc.get("canonical_solution", "").strip()

            # 模型生成的回應（可能有多個，但 repeats=1 所以通常只有一個）
            generated_code_list = sample["resps"]
            # filtered_resps 是加上 prompt 後的完整程式碼
            filtered_code_list = sample["filtered_resps"]

            # 通常只有一個生成結果（repeats=1）
            generated_code = generated_code_list[0][0].strip() if generated_code_list and generated_code_list[0] else ""
            full_generated_code = filtered_code_list[0][0].strip() if filtered_code_list and filtered_code_list[0] else ""

            # pass@1 結果
            is_correct = bool(sample.get("pass@1", False))

            # 提取 prompt（不含 docstring 後的空白）
            prompt_display = prompt.strip()

            # === 準備 JSON 記錄 ===
            records.append({
                "task": task_name,
                "version": version,
                "problem_id": doc["task_id"],
                "prompt": prompt,
                "entry_point": entry_point,
                "test_code": test_code,
                "canonical_solution": canonical_solution,
                "model_generated_code": generated_code,
                "full_generated_code": full_generated_code,
                "is_correct": is_correct,
                "pass_at_1": is_correct,  # 對應 metrics
            })

            # === 準備 Markdown 用資料 ===
            result = "✅ Passed" if is_correct else "❌ Failed"
            short_prompt = prompt.split("def ")[-1].split("(")[0] + "()"  # 取函數名作為簡短標題

            prompts_and_responses.append({
                "index": idx,
                "task": task_name,
                "short_prompt": short_prompt,
                "result": result,
                "prompt": prompt_display,
                "canonical_solution": canonical_solution,
                "generated_code": generated_code,
                "full_generated_code": full_generated_code,
                "test_code": test_code,
                "is_correct": is_correct,
            })

    # === 保存 JSON 詳細結果 ===
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
    print(f"   Accuracy (pass@1): {df['is_correct'].mean():.2%}")
    return df


def _generate_detailed_markdown_report(data: List[Dict], md_file: str, full_records: List[Dict]):
    """生成美觀的 Markdown 報告，包含摘要與每題詳細分析（程式碼生成專用）"""
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# 🧑‍💻 Code Generation Evaluation Report\n\n")

        # === 摘要統計 ===
        total = len(data)
        correct = sum(1 for d in data if d["is_correct"])
        accuracy = correct / total if total > 0 else 0
        f.write(f"**Overall Accuracy (pass@1)**: {correct}/{total} ({accuracy:.1%})\n\n")

        # === 摘要表格 ===
        f.write("## 📊 Summary Table\n\n")
        f.write("| # | Task | Function | Result |\n")
        f.write("|---|------|----------|--------|\n")
        for d in data:
            f.write(f"| {d['index']} | `{d['task']}` | `{d['short_prompt']}` | {d['result']} |\n")
        f.write("\n")

        # === 詳細分析 ===
        f.write("## 🔍 Detailed Analysis\n\n")
        for d in data:
            f.write(f"### Problem {d['index']} - `{d['short_prompt']}`\n\n")
            f.write(f"**Result**: {d['result']}\n\n")

            # Prompt
            f.write("<details>\n")
            f.write("<summary>📌 Show Problem Prompt</summary>\n\n")
            f.write("```python\n")
            f.write(d["prompt"].replace("```", "\\`\\`\\`") + "\n")
            f.write("```\n")
            f.write("</details>\n\n")

            # Canonical Solution
            if d["canonical_solution"]:
                f.write("<details>\n")
                f.write("<summary>✅ Show Reference Solution</summary>\n\n")
                f.write("```python\n")
                f.write(d["canonical_solution"].replace("```", "\\`\\`\\`") + "\n")
                f.write("```\n")
                f.write("</details>\n\n")

            # Generated Code
            f.write("### 🤖 Model Generated Code\n\n")
            f.write("```python\n")
            f.write(d["full_generated_code"].replace("```", "\\`\\`\\`") + "\n")
            f.write("```\n\n")

            # Test Code
            f.write("<details>\n")
            f.write("<summary>🧪 Show Test Cases</summary>\n\n")
            f.write("```python\n")
            f.write(d["test_code"].replace("```", "\\`\\`\\`") + "\n")
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
    recipe.agenteval()


if __name__ == "__main__":
    sys.exit(recipe_main())
