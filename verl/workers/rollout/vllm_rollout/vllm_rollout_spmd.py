# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union
import re
import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

import re, nltk
from collections import Counter
from typing import Dict, Optional, List
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords





# Preset NLTK resources and define globals
for _pkg, _src in (("punkt", "tokenizers/punkt"),
                   ("stopwords", "corpora/stopwords")):
    try:
        nltk.data.find(_src)
    except LookupError:
        nltk.download(_pkg, quiet=True)

# 1.  Globals (lazy‑loaded model)
_DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL: Optional[SentenceTransformer] = None          # will be loaded once
_THRESHOLD    = 0.7
_STOP_WORDS   = set(stopwords.words("english"))
_PLACEHOLDER  = "[KW]"
_UNIQUENESS_THRESHOLD = 0.78     


def _uniq_bucket(rate: float) -> float:
    """Map uniqueness rate ∈ [0,1] to a score."""
    if rate == 1.0:
        return 1.0
    elif rate > 0.8:
        return 0.75
    elif rate > 0.6:
        return 0.4
    else:
        return 0.0


def _uniqueness_rate(sentences: List[str],
                     mask_kws: Optional[set[str]] = None) -> float:
    """Compute uniqueness rate = unique sentences / total sentences."""
    n = len(sentences)
    if n <= 1:
        return 1.0

    # Optional masking
    proc_sents = (
        [_mask_sentence(s, mask_kws) for s in sentences] if mask_kws else sentences
    )

    # Encode & similarity
    model = _get_model()
    em  = model.encode(proc_sents, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(em, em).cpu().numpy()

    # Mark redundant sentences
    redundant = set()
    for i in range(n):
        if i in redundant:
            continue
        for j in range(i + 1, n):
            if sim[i, j] >= _UNIQUENESS_THRESHOLD:
                redundant.add(j)

    uniq_cnt = n - len(redundant)
    return uniq_cnt / n


_PERSPECTIVE  = re.compile(
    r"(?:^|\s)(?:From|In) the perspective of ([^,.\n]+)",
    re.IGNORECASE,
)

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print(" LOADING SentenceTransformer model ...")
        _MODEL = SentenceTransformer(
            "/p/project1/westai0052/self_rewarding/verl/MODEL/st-scale70",
            device=_DEVICE,
        )
    return _MODEL

# OP-GRPO Helper utilities ----------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    text += "\n"
    pat = re.compile(r"(?:^|\n)(?=(?:From|In) the perspective of )",
                     re.IGNORECASE)
    return [seg.strip() for seg in pat.split(text) if seg.strip()]

def _extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and t.isalnum()}

def _mask_sentence(sent: str, kws: set[str]) -> str:
    if not kws:
        return sent
    toks = re.findall(r"\b\w+\b", sent)
    return " ".join((_PLACEHOLDER if t.lower() in kws else t) for t in toks)

def _mutual_best_pairs(sim: np.ndarray, thr: float):
    M, pairs = sim.copy(), []
    while True:
        mx = np.nanmax(M)
        if np.isnan(mx) or mx < thr:
            break
        r, c = np.unravel_index(np.nanargmax(M), M.shape)
        if mx == np.nanmax(M[r]) and mx == np.nanmax(M[:, c]):
            pairs.append((r, c, mx)); M[r] = np.nan; M[:, c] = np.nan
        else:
            M[r, c] = np.nan
    return pairs

def _sentence_match_rate(ref_s: List[str], cand_s: List[str],
                         mask_kws: Optional[set[str]] = None) -> float:
    if not ref_s or not cand_s:
        return 0.0
    cand_proc = (_mask_sentence(s, mask_kws) for s in cand_s) if mask_kws else cand_s
    model = _get_model()
    em_r = model.encode(ref_s,  convert_to_tensor=True, normalize_embeddings=True)
    em_c = model.encode(list(cand_proc), convert_to_tensor=True, normalize_embeddings=True)
    sim  = util.cos_sim(em_r, em_c).cpu().numpy()
    return len(_mutual_best_pairs(sim, _THRESHOLD)) / len(ref_s)

def _bucket_reward(rate: float) -> int:
    if rate == 0: return 0
    if rate < .2: return 1
    if rate < .4: return 2
    if rate < .6: return 3
    if rate < .8: return 4
    return 5

# OP-GRPO Helper utilities FINISH----------------------------------------------------FINISH

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        lora_kwargs = kwargs.pop('lora_kwargs', {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        

        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        
        decoded_prompts = [self.tokenizer.decode(idx[i], skip_special_tokens=True) for i in range(batch_size)]
        
        

        non_tensor_batch = prompts.non_tensor_batch
        meta_info = prompts.meta_info
        batch = prompts.batch
        # print(" non_tensor_batch keys:", list(non_tensor_batch.keys()))
        # print(" meta_info keys:", list(meta_info.keys()))
        # print(" batch keys:", list(batch.keys())
        # )

     


        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id=lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}",lora_int_id=lora_int_id,lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        
        
        print("$$$$$$$$$$$$$$$$$$$$ Start rewarding… $$$$$$$$$$$$$$$$$$$$")
        print("batch size:", batch_size)
        reward_model_batch = non_tensor_batch.get("reward_model")
        if reward_model_batch is None:
            raise ValueError("Missing 'reward_model' in non_tensor_batch.")

       
        # ground‑truth 
        ground_truths = [x.get("ground_truth", "") for x in reward_model_batch]   # len = B_prompt


        # decode all rollouts
        total_B   = response.size(0)            
        resp_len  = response.size(1)                # Lresp
        decoded_prompts = [self.tokenizer.decode(idx[i], skip_special_tokens=True) for i in range(batch_size)]
        decoded_responses_raw = [
            self.tokenizer.decode(response[i], skip_special_tokens=True)
            for i in range(total_B)                 # ← 全部样本
        ]

    
        # decoded_responses = extracted_core_blocks  
        STRICT_FORMAT_RE = re.compile(
            r"""(?is)^\s*                                 
                <core\s+perspectives>\s*
                (?P<core>.+?)                            
                \s*</core\s+perspectives>\s*              
                (?P<gap>\s*)                             
                <summary>\s*
                (?P<sum>.+?)                              
                \s*</summary>\s*$                        
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

        format_scores: list[float] = []
        extracted_core_blocks: list[str] = []
        invalid_indices: list[int] = []

        for i, txt in enumerate(decoded_responses_raw):
            if not isinstance(txt, str):
                # Non-string -> invalid
                format_scores.append(-0.2)
                invalid_indices.append(i)
                extracted_core_blocks.append(str(txt).strip())
                continue

            m = STRICT_FORMAT_RE.match(txt)
            if m:
                core_content = m.group("core").strip()
                sum_content  = m.group("sum").strip()

                # ensure both have at least one non-whitespace "word" character
                if core_content and sum_content:
                    format_scores.append(0.0)                     # good
                    extracted_core_blocks.append(core_content)    # use core block downstream
                else:
                    format_scores.append(-0.2)
                    invalid_indices.append(i)
                    extracted_core_blocks.append(txt.strip())     # fallback to full text for downstream
            else:
                format_scores.append(-0.2)
                invalid_indices.append(i)
                extracted_core_blocks.append(txt.strip())         # fallback to full text for downstream

        # set the text used for downstream similarity
        decoded_responses = extracted_core_blocks

        # Print a concise report (avoid spamming the log)
        total_bad = len(invalid_indices)
        if total_bad > 0:
            sample_ids = invalid_indices[:20]  
            print(f"[FORMAT] Invalid tag layout: {total_bad}/{len(decoded_responses_raw)}. "
                f"First indices: {sample_ids}")
        else:
            print("[FORMAT] All responses have valid <core perspectives> -> <summary> format.")


        keywords_list = [_extract_keywords(p) for p in decoded_prompts]  # len = B_prompt
        # ── if total_B > B_prompt，copy the key wrods to response ──
        if total_B != batch_size:
            assert total_B % batch_size == 0, "total_B must be multiple of batch_size"
            keywords_list = keywords_list * (total_B // batch_size)      # len = total_B


        # if total_B > len(ground_truths), copy the ground_truth to response
        if len(ground_truths) != total_B:
            assert total_B % len(ground_truths) == 0,\
                f"ground_truth len {len(ground_truths)} cannot broadcast to {total_B}"
            repeat_n = total_B // len(ground_truths)
            ground_truths = [gt for gt in ground_truths for _ in range(repeat_n)]

    
        # similarity score per rollout
        similarity_scores: list[int] = []
        uniqueness_scores_raw:  List[float] = []  
        print_count = 0
        for ref_txt, cand_txt, kws in zip(ground_truths, decoded_responses, keywords_list):
            ref_sents  = _split_sentences(ref_txt)
            cand_sents = _split_sentences(cand_txt)


            # Debug print for the first 2 samples
            # if print_count < 2:
            #     print("────────────────────────────────────────────")
            #     print(f"[SIM DEBUG] Sample {print_count + 1}")
            #     print("[REF SENTS]:", ref_sents)
            #     print("[CAND SENTS]:", cand_sents)
            #     print("────────────────────────────────────────────")
            #     print_count += 1


            sim_rate   = _sentence_match_rate(ref_sents, cand_sents, mask_kws=kws)
            similarity_scores.append(_bucket_reward(sim_rate))

            # grounded uniqueness rate
            uniq_rate  = _uniqueness_rate(cand_sents, mask_kws=kws)
            uniqueness_scores_raw.append(_uniq_bucket(uniq_rate))

    
        # (total_B, Lresp) reward tensor
        reward_tensor  = (
            torch.tensor(similarity_scores, dtype=torch.float32, device=response.device)
                .unsqueeze(1)
                .expand(-1, resp_len)          # (total_B, Lresp)
        )

        tensor_uniq = (
            torch.tensor(uniqueness_scores_raw, dtype=torch.float32, device=response.device)
                .unsqueeze(1)
                .expand(-1, resp_len)          # (total_B, Lresp)
        )

        tensor_tag_format = (
            torch.tensor(format_scores, dtype=torch.float32, device=response.device)
                .unsqueeze(1).expand(-1, resp_len)         
        )

        # print("response shape :", response.shape)      # should be (2560, 400)
        # print("sim_rewards shape:", tensor_sim.shape)  # should match (2560, 400)

      
        # keep the reward model info for each sample
        rm_obj = reward_model_batch
        if isinstance(rm_obj, dict):
            rep_list = [rm_obj] * total_B
        else:
            rep_list = list(rm_obj)
            if len(rep_list) != total_B:
                assert total_B % len(rep_list) == 0,\
                    f"Cannot broadcast reward_model: {len(rep_list)} -> {total_B}"
                rep_list *= total_B // len(rep_list)

        non_tensor_batch["reward_model"] = np.asarray(rep_list, dtype=object)

        

        # update batch
        batch = TensorDict(
            {
                "prompts":            idx,
                "responses":          response,
                "input_ids":          seq,
                "rollout_log_probs":  rollout_log_probs,
                "attention_mask":     attention_mask,
                "position_ids":       position_ids,
                "sim_rewards":        reward_tensor,     #new
                "uniq_rewards":        tensor_uniq,       #new
                "tag_format_rewards":     tensor_tag_format, #new
            },
            batch_size=total_B,
        )


        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
