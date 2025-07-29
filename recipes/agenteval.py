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
        print("get in _model_generate") # No?
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

    def evaluate(self) -> None:
        # Initialize tasks for the harness
        task_manager = TaskManager(include_path=self.include_path)
        task_dict = get_task_dict(self.tasks, task_manager)

        # Run evaluation
        t0 = time.time()
        self.logger.info(f"Running evaluation on the following tasks: {self.tasks}")
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

        print(output)
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
                pred_log_likelihoods = [resp[0][0] for resp in sample["resps"]]
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


if __name__ == "__main__":
    sys.exit(recipe_main())
