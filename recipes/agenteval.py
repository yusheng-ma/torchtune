# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import sys
import time
import json
import os
from typing import Union
import PIL
import torch
import pandas as pd
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
        model_name: str,
    ):
        super().__init__(pretrained="gpt2", device=str(device))
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype
        self._enable_kv_cache = enable_kv_cache
        self.model_name = model_name
        # Add a list to store evaluation traces
        self.eval_traces = []

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
        if isinstance(self._tokenizer, HuggingFaceModelTokenizer):
            return self._tokenizer.base_tokenizer.encode(
                text=text, add_bos=False, add_eos=False
            )
        return self._tokenizer.encode(text=text, add_bos=False, add_eos=False)

    def tok_batch_encode(
        self, text: list[str], left_truncate_len: int = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized_text = [self.tok_encode(x) for x in text]
        x = left_pad_sequence(
            [torch.tensor(x) for x in tokenized_text],
            batch_first=True,
            padding_value=self._tokenizer.pad_id,
        )
        if left_truncate_len is not None:
            x = x[:, -left_truncate_len:]
        return x, torch.ones_like(x)

    def tok_decode(self, tokens: Union[list[int], int], **kwargs) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
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
        bsz, seq_len = context.shape
        temperature = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", False)
        if do_sample or temperature != 0.0:
            raise RuntimeError(
                "Any decoding strategy other than greedy is not supported."
            )
        maybe_padded_context = torch.nn.functional.pad(
            context,
            (0, 0, 0, self._batch_size - bsz),
            value=self._tokenizer.eos_id,
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
    def _generate_responses_with_trace(
        self, contexts: List[str], max_ctx_len: int, max_gen_toks: int, until: List[str], kwargs: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict]]:
        """
        Returns: (responses, traces)
        trace = {
            "input_tokens": int,
            "output_tokens": int,
            "preprocess_time": float,
            "generate_time": float,
            "postprocess_time": float,
            "total_time": float
        }
        """
        import time
        traces = []

        # a. encode
        start_pre = time.time()
        context_enc, attn_masks = self.tok_batch_encode(
            contexts,
            left_truncate_len=max_ctx_len,
            truncation=self.truncation,
        )
        pre_time = time.time() - start_pre

        context_enc = context_enc.to(self.device)
        attn_masks = attn_masks.to(self.device)

        if "max_length" not in kwargs:
            kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

        # b. generate
        start_gen = time.time()
        cont = self._model_generate(
            context=context_enc,
            attention_mask=attn_masks,
            stop=until,
            **kwargs,
        )
        gen_time = time.time() - start_gen

        cont_toks_list = cont.tolist()
        input_lens = [len(x) for x in context_enc]

        # c. decode & postprocess
        start_post = time.time()
        responses = []
        for i, cont_toks in enumerate(cont_toks_list):
            if self.backend == "causal":
                output_toks = cont_toks[input_lens[i]:]
            else:
                output_toks = cont_toks
            s = self.tok_decode(output_toks)
            for term in until:
                if term and term in s:
                    s = s.split(term)[0]
            if self.model_name == "Mistral-7B-Instruct-v0.2":
                print("mistral add space")
                s = " " + s
            responses.append(s)

            total_time = pre_time + gen_time + (time.time() - start_post) / len(contexts)
            trace = {
                "input_tokens": input_lens[i],
                "output_tokens": len(output_toks),
                "preprocess_time": pre_time / len(contexts),  # 均攤
                "generate_time": gen_time / len(contexts),
                "postprocess_time": (time.time() - start_post) / len(contexts),
                "total_time": total_time
            }
            traces.append(trace)

        return responses, traces

    @torch.inference_mode()
    def agent1_with_trace(
        self, contexts: List[str], max_ctx_len: int, max_gen_toks: int, until: List[str], kwargs: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict]]:
        return self._generate_responses_with_trace(contexts, max_ctx_len, max_gen_toks, until, kwargs)

    @torch.inference_mode()
    def agent2_with_trace(
        self, contexts: List[str], first_responses: List[str], max_ctx_len: int, max_gen_toks: int, until: List[str], kwargs: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict]]:
        def _build_agent2_context(original_context: str, first_response: str) -> str:
            return (
                "Write a solution to the following problem and make sure that it passes the tests. "
                "This is the response from other thinkers: ```python\n{resp}\n``` for your reference.\n"
                "This is the problem: ```python\n{ctx}\n```"
            ).format(resp=first_response, ctx=original_context)
        second_contexts = [
            _build_agent2_context(ctx, resp)
            for ctx, resp in zip(contexts, first_responses)
        ]
        return self._generate_responses_with_trace(contexts, max_ctx_len, max_gen_toks, until, kwargs)
        return self._generate_responses_with_trace(second_contexts, max_ctx_len, max_gen_toks, until, kwargs)

    @torch.inference_mode()
    def agent3_with_trace(
        self, contexts: List[str], first_responses: List[str], second_responses: List[str], max_ctx_len: int, max_gen_toks: int, until: List[str], kwargs: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict]]:
        def _build_agent3_context(original_context: str, first_response: str, second_response: str) -> str:
            return (
                "Write a solution to the following problem and make sure that it passes the tests. "
                "These are two previous responses for reference: ```python\n{first_resp}\n```\n ```python\n{second_resp}\n```\n"
                "This is the problem: ```python\n{ctx}\n```"
            ).format(first_resp=first_response, second_resp=second_response, ctx=original_context)
        third_contexts = [
            _build_agent3_context(ctx, first_resp, second_resp)
            for ctx, first_resp, second_resp in zip(contexts, first_responses, second_responses)
        ]
        return self._generate_responses_with_trace(contexts, max_ctx_len, max_gen_toks, until, kwargs)
        return self._generate_responses_with_trace(third_contexts, max_ctx_len, max_gen_toks, until, kwargs)

    @torch.inference_mode()
    def summarizer_with_trace(
        self, contexts: List[str], first_responses: List[str], second_responses: List[str], third_responses: List[str], max_ctx_len: int, max_gen_toks: int, until: List[str], kwargs: Dict[str, Any],
    ) -> Tuple[List[str], List[Dict]]:
        def _build_summarizer_context(original_context: str, first_response: str, second_response: str, third_response: str) -> str:
            return (
                "Conclude a solution to the following problem and make sure that it passes the tests "
                "based the answers from the three thinkers: ```python\n{first_resp}\n``` ```python\n{second_resp}\n``` ```python\n{third_resp}\n```\n"
                "This is the problem: ```python\n{ctx}\n```"
            ).format(first_resp=first_response, second_resp=second_response, third_resp=third_response, ctx=original_context)
        summarize_contexts = [
            _build_summarizer_context(ctx, first_resp, second_resp, third_resp)
            for ctx, first_resp, second_resp, third_resp in zip(contexts, first_responses, second_responses, third_responses)
        ]
        return self._generate_responses_with_trace(contexts, max_ctx_len, max_gen_toks, until, kwargs)
        return self._generate_responses_with_trace(summarize_contexts, max_ctx_len, max_gen_toks, until, kwargs)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []
        traces = []
        def _collate(req: Tuple[str, dict]):
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
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
            gen_kwargs = all_gen_kwargs[0]
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks
            if self.backend == "causal":
                max_ctx_len = self.max_length - max_gen_toks
                assert max_ctx_len > 0, (
                    f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
                )
            elif self.backend == "seq2seq":
                max_ctx_len = self.max_length

            print("================================ Multi-Agent Evaluation Start ================================")

            # Run the multi-agent process and collect traces
            first_responses, first_traces = self.agent1_with_trace(contexts, max_ctx_len, max_gen_toks, until, kwargs)
            # second_responses, second_traces = self.agent2_with_trace(contexts, first_responses, max_ctx_len, max_gen_toks, until, kwargs)
            # third_responses, third_traces = self.agent3_with_trace(contexts, first_responses, second_responses, max_ctx_len, max_gen_toks, until, kwargs)
            # final_summary, summarizer_traces = self.summarizer_with_trace(contexts, first_responses, second_responses, third_responses, max_ctx_len, max_gen_toks, until, kwargs)
            second_responses, second_traces = first_responses, first_traces
            third_responses, third_traces = first_responses, first_traces
            final_summary, summarizer_traces = first_responses, first_traces

            # Build trace for the current batch
            # batch_traces = []
            for i, context in enumerate(contexts):
                trace = {
                    "task": requests[i].task_name if hasattr(requests[i], 'task_name') else "unknown",
                    # "request_index": requests[i].doc_id if hasattr(requests[i], 'doc_id') else i,
                    "prompt": context,
                    "agent1": {
                        "response": first_responses[i],
                        "input_tokens": first_traces[i]["input_tokens"],
                        "output_tokens": first_traces[i]["output_tokens"],
                        "preprocess_time": first_traces[i]["preprocess_time"],
                        "generate_time": first_traces[i]["generate_time"],
                        "postprocess_time": first_traces[i]["postprocess_time"],
                        "total_time": first_traces[i]["total_time"]
                    },
                    "agent2": {
                        "response": second_responses[i],
                        "input_tokens": second_traces[i]["input_tokens"],
                        "output_tokens": second_traces[i]["output_tokens"],
                        "preprocess_time": second_traces[i]["preprocess_time"],
                        "generate_time": second_traces[i]["generate_time"],
                        "postprocess_time": second_traces[i]["postprocess_time"],
                        "total_time": second_traces[i]["total_time"]
                    },
                    "agent3": {
                        "response": third_responses[i],
                        "input_tokens": third_traces[i]["input_tokens"],
                        "output_tokens": third_traces[i]["output_tokens"],
                        "preprocess_time": third_traces[i]["preprocess_time"],
                        "generate_time": third_traces[i]["generate_time"],
                        "postprocess_time": third_traces[i]["postprocess_time"],
                        "total_time": third_traces[i]["total_time"]
                    },
                    "summarizer": {
                        "response": final_summary[i],
                        "input_tokens": summarizer_traces[i]["input_tokens"],
                        "output_tokens": summarizer_traces[i]["output_tokens"],
                        "preprocess_time": summarizer_traces[i]["preprocess_time"],
                        "generate_time": summarizer_traces[i]["generate_time"],
                        "postprocess_time": summarizer_traces[i]["postprocess_time"],
                        "total_time": summarizer_traces[i]["total_time"]
                    },
                    "final_response": final_summary[i]
                }
                # batch_traces.append(trace)
                traces.append(trace)

            # Reorder the batch traces to the original order and add to the global list
            # original_order_traces = re_ords.get_original(batch_traces)
            # self.eval_traces.extend(original_order_traces)

            # Append final responses for the harness
            for s, context in zip(final_summary, contexts):
                res.append(s)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)

        self.eval_traces = re_ords.get_original(traces)

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
        # Extract model name from checkpoint_dir
        self.model_name = self._extract_model_name(cfg.checkpointer.checkpoint_dir)
        self.output_jsonl_name = cfg.get("output_jsonl_name", "results.jsonl")
        self.output_markdown_name = cfg.get("output_markdown_name", "report.md")

    def _extract_model_name(self, checkpoint_dir: str) -> str:
        """Extract model name from the checkpoint directory path."""
        # Example: /tmp/Meta-Llama-3.1-8B-Instruct/ -> Meta-Llama-3.1-8B-Instruct
        import os
        return os.path.basename(os.path.normpath(checkpoint_dir))

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
            model_name = self.model_name,
        )

    def _extract_results_from_output(self, output: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract the results table data from the output dictionary of evaluate().
        This replicates the logic in lm_eval.utils.make_table.
        """
        results = []
        keys = output["results"].keys()
        for k in keys:
            dic = output["results"][k]
            version = output["versions"].get(k, "N/A")
            n_shot = str(output.get("n-shot", {}).get(k, " "))
            higher_is_better = output.get("higher_is_better", {}).get(k, {})

            metric_items = sorted(dic.items())
            for (mf), v in metric_items:
                m, _, f = mf.partition(",")
                if m.endswith("_stderr"):
                    continue

                hib = ""  # We'll use emoji later, not symbols
                v = f"{v:.4f}" if isinstance(v, float) else str(v)

                stderr_key = m + "_stderr" + "," + f
                se = dic[stderr_key] if stderr_key in dic else "N/A"
                se = f"{se:.4f}" if isinstance(se, float) else str(se)

                results.append({
                    "Tasks": k,
                    "Version": version,
                    "Filter": f,
                    "n-shot": n_shot,
                    "Metric": m,
                    "Value": v,
                    "Stderr": se
                })
        return results

    def dump_detailed_results(self, output, output_dir="eval_output"):
        """
        Dump detailed evaluation results to JSONL and a structured Markdown report.
        Args:
            output (dict): The output dictionary from lm_eval.evaluator.evaluate().
            output_dir (str): The directory to save the output files.
        """
        os.makedirs(output_dir, exist_ok=True)
        traces = self.eleuther_model_wrapper.eval_traces

        # ==================== Step 1: Prepare Data ====================
        # Extract results from the 'output' dictionary
        result_rows = self._extract_results_from_output(output)

        # Create a mapping from doc_id to pass@1 result for each task
        correctness_map = {}
        if "samples" in output:
            for task_name, task_samples in output["samples"].items():
                for sample in task_samples:
                    doc_id = sample["doc_id"]
                    # Use the 'pass@1' metric to determine correctness
                    is_correct = bool(sample.get("pass@1", 0.0))
                    correctness_map[(task_name, doc_id)] = is_correct

        # Create a DataFrame for the overview table
        overview_data = []
        for i, trace in enumerate(traces):
            task_name = trace.get('task', 'unknown')
            doc_id = i
            task_id = f"{task_name}_{doc_id}"
            function_name = self._extract_function_name(trace['prompt']) or "unknown"
            # Look up the correctness from the map
            is_correct = correctness_map.get((task_name, doc_id), False)
            emoji = "✅" if is_correct else "❌"

            agents = ["agent1", "agent2", "agent3", "summarizer"]
            total_tokens = [trace[agent]["input_tokens"] + trace[agent]["output_tokens"] for agent in agents]
            times = [trace[agent]["total_time"] for agent in agents]
            overall_tokens = sum(total_tokens)
            overall_time = sum(times)

            overview_data.append({
                "Task ID": task_id,
                "Function": function_name,
                "Correct": emoji,
                "Tokens (A1/A2/A3/S)": f"{total_tokens[0]}/{total_tokens[1]}/{total_tokens[2]}/{total_tokens[3]}",
                "Time (A1/A2/A3/S)": f"{times[0]:.2f}/{times[1]:.2f}/{times[2]:.2f}/{times[3]:.2f}",
                "Overall Tokens": overall_tokens,
                "Overall Time": f"{overall_time:.2f}s"
            })
        df_overview = pd.DataFrame(overview_data)

        # ==================== Step 2: Write JSONL ====================
        # Output 1: JSONL (for analysis)
        jsonl_path = os.path.join(output_dir, self.output_jsonl_name)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for trace in traces:
                # Optionally, add the 'correct' field to the JSON trace for completeness
                task_name = trace.get('task', 'unknown')
                doc_id = trace.get('request_index', i)
                is_correct = correctness_map.get((task_name, doc_id), False)
                trace_with_correct = trace.copy()
                trace_with_correct["correct"] = is_correct
                f.write(json.dumps(trace_with_correct, ensure_ascii=False) + "\n")
        self.logger.info(f"Saved detailed results to {jsonl_path}")

        # ==================== Step 3: Write Markdown ====================
        md_path = os.path.join(output_dir, self.output_markdown_name)
        with open(md_path, "w", encoding="utf-8") as f:
            # --- Section 1: Header and Model Info ---
            f.write("# 📊 Multi-Agent Evaluation Report\n\n")
            f.write(f"**Model**: `{self.model_name}`\n")
            f.write(f"**Tasks**: `{', '.join(self.tasks)}`\n")
            f.write(f"**Batch Size**: `{self.batch_size}`\n")
            f.write(f"**Sample Limit**: `{self.limit}`\n\n")

            # --- Section 2: Results Table ---
            f.write("## ✅ Evaluation Results\n")
            f.write("Results from the evaluation harness.\n\n")
            # Create a Markdown table from result_rows
            header = "| Tasks | Version | Filter | n-shot | Metric | Value | Stderr |\n"
            separator = "|-------|--------:|--------|-------:|--------|------:|-------:|\n"
            f.write(header)
            f.write(separator)
            for row in result_rows:
                f.write(f"| {row['Tasks']} | {row['Version']} | {row['Filter']} | {row['n-shot']} | {row['Metric']} | {row['Value']} | {row['Stderr']} |\n")
            f.write("\n\n")

            # --- Section 3: Performance Summary ---
            f.write("## 📈 Performance Summary\n")
            f.write(df_overview.to_markdown(index=False, tablefmt="pipe"))
            f.write("\n\n")

            # --- Section 4: Detailed Case Analysis ---
            f.write("## 🔍 Detailed Case Analysis\n")
            for i, trace in enumerate(traces):
                task_name = trace.get('task', 'unknown')
                doc_id = trace.get('request_index', i)
                task_id = f"{task_name}_{doc_id}"
                function_name = self._extract_function_name(trace['prompt']) or "unknown"
                # Look up the correctness from the map
                is_correct = correctness_map.get((task_name, doc_id), False)
                emoji = "✅" if is_correct else "❌"

                # Write the main header for the test case
                f.write(f"### {task_id} - `{function_name}` {emoji}\n")
                # Write the stats for all agents on the same line
                agents = ["agent1", "agent2", "agent3", "summarizer"]
                stats_parts = []
                for agent in agents:
                    data = trace[agent]
                    total_tokens = data['input_tokens'] + data['output_tokens']
                    stats_parts.append(f"**{agent.upper()}** {total_tokens}/{data['total_time']:.2f}s")
                f.write(" ".join(stats_parts) + "\n\n")

                # Write the prompt
                f.write(f"**Prompt**:\n")
                f.write(f"```python\n{trace['prompt']}\n```\n\n")
                f.write(f"**Final Response (Summarizer)**:\n")
                f.write(f"```python\n{trace['summarizer']['response']}\n```\n\n")

                # Write individual agent responses as collapsible sections
                for agent in agents:
                    data = trace[agent]
                    f.write(f"<details>\n")
                    f.write(f"<summary>{agent.upper()} -- Input Tokens: {data['input_tokens']}, Output Tokens: {data['output_tokens']}, Time: {data['total_time']:.2f}s</summary>\n\n")
                    f.write(f"```python\n{data['response']}\n```\n")
                    f.write(f"</details>\n\n")

        self.logger.info(f"Saved detailed report to {md_path}")

    def _extract_function_name(self, prompt: str) -> str:
        """Helper function to extract the function name from the prompt."""
        import re
        match = re.search(r"def\s+(\w+)\(", prompt)
        return match.group(1) if match else "unknown"

    def agenteval(self) -> None:
        """Entry point for the evaluation process."""
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
            confirm_run_unsafe_code=True,
        )
        t1 = time.time() - t0

        # Log metrics
        self.logger.info(f"Eval completed in {t1:.02f} seconds.")
        if self.device.type != "cpu" and self.device.type != "mps":
            torch_device = utils.get_torch_device_namespace()
            self.logger.info(
                f"Max memory allocated: {torch_device.max_memory_allocated() / 1e9:.02f} GB"
            )
        # Log the table for reference
        # print(output)
        formatted_output = make_table(output)
        self.logger.info(f"\n{formatted_output}\n")

        # Pass the 'output' from evaluate() to dump_detailed_results
        self.dump_detailed_results(output=output)


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    config.log_config(recipe_name="EleutherEvalRecipe", cfg=cfg)
    recipe = EleutherEvalRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.agenteval()

if __name__ == "__main__":
    sys.exit(recipe_main())