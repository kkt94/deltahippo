
import os
import logging
import torch
from adapters import AutoAdapterModel
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import (
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)

from utils.prompt import get_auto_prompt_tuning_init_text

# -----------------------------------------------------------------------------
#  Attention kernel: flash_attention_2 when the package is available,
#  otherwise PyTorch SDPA (same attention math, different kernel).
# -----------------------------------------------------------------------------
try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
except Exception:
    ATTN_IMPL = "sdpa"
print(f"[ATTN] attn_implementation={ATTN_IMPL}")

# -----------------------------------------------------------------------------
#  Supported backbone name → type mapping
# -----------------------------------------------------------------------------
BACKBONE2TYPE = {
    # ── Generative ───────────────────────────────────────────────
    "gpt2": "generative",
    "gpt2-large": "generative",
    "EleutherAI/pythia-70m-deduped": "generative",
    "EleutherAI/pythia-160m-deduped": "generative",
    "EleutherAI/pythia-410m-deduped": "generative",
    "EleutherAI/pythia-1b-deduped": "generative",
    "EleutherAI/pythia-1.4b-deduped": "generative",
    "EleutherAI/pythia-2.8b-deduped": "generative",
    "baffo32/decapoda-research-llama-7B-hf": "generative",
    "meta-llama/Llama-3.2-1B": "generative",
    "lmsys/vicuna-7b-v1.1": "generative",
    "llama2-13b-orca-8k-3319": "generative",
    # Qwen local checkpoints
    "Qwen3-0.6B": "generative",
    "Qwen2-0.5B": "generative",
    "Qwen2.5-0.5B": "generative",
    "gpt-oss-20b": "generative",
    # ── Discriminative ────────────────────────────────────────────
    "roberta-base": "discriminative",
    "roberta-large": "discriminative",
    "bert-base-cased": "discriminative",
    "bert-base-uncased": "discriminative",
    "bert-large-cased": "discriminative",
    "bert-large-uncased": "discriminative",
}

# -----------------------------------------------------------------------------
#  Common utilities: tokenizer & embedding synchronization
# -----------------------------------------------------------------------------

def _sync_tokenizer_and_embeddings(tokenizer, model, params, num_task: int = 1):
    """Ensure pad/special tokens exist and resize model embeddings if needed."""
    import torch
    import logging
    
    # 1) Ensure a pad token ---------------------------------------------------------
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id

    # 2) Define additional/special tokens -----------------------------------------------------
    add_tokens = ["__ans__"]
    if getattr(params, "LAMOL_use_task_specific_gen_token", True):
        add_tokens += [f"__{i}__" for i in range(num_task)]
    else:
        add_tokens.append("__gen__")

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})

    # 3) Resize embeddings ---------------------------------------------------------
    if num_added > 0:
        try:
            # Disable mean_resizing on meta device to avoid linalg_eig
            embed_tokens = model.get_input_embeddings()
            lm_head = model.get_output_embeddings()
            
            # Check for meta device
            if embed_tokens.weight.device.type == 'meta':
                logging.warning("[Backbone] Meta device detected. Using safe resize method.")
                # Set mean_resizing=False to avoid the eigendecomposition operation
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            else:
                model.resize_token_embeddings(len(tokenizer))
                
            model.config.vocab_size = len(tokenizer)  # ensure generate safety
            logging.info(
                f"[Backbone] Added {num_added} tokens → new vocab_size = {len(tokenizer)}"
            )
        except Exception as e:
            logging.error(f"[Backbone] Failed to resize embeddings: {e}")
            
            # On meta device, skip the embedding resize and update only the config
            if embed_tokens.weight.device.type == 'meta':
                logging.warning("[Backbone] Skipping embedding resize for Meta device. Config updated only.")
                model.config.vocab_size = len(tokenizer)
                # the tokenizer already contains the new tokens, so use it as is
                logging.info(f"[Backbone] Config updated: vocab_size = {len(tokenizer)}")
            else:
                # On a real device, resize the embeddings manually
                old_vocab_size = embed_tokens.weight.shape[0]
                new_vocab_size = len(tokenizer)
                
                # Create new embedding weights
                embed_dim = embed_tokens.weight.shape[1]
                
                # Create the new weights on CPU
                new_embed_weight = torch.zeros(new_vocab_size, embed_dim)
                new_embed_weight[:old_vocab_size] = embed_tokens.weight.to('cpu')
                
                # Initialize new tokens with a normal distribution
                if new_vocab_size > old_vocab_size:
                    std = embed_tokens.weight.std().item()
                    new_embed_weight[old_vocab_size:].normal_(mean=0.0, std=std)
                
                # Replace the embeddings
                embed_tokens.weight = torch.nn.Parameter(new_embed_weight)
                
                # Handle the case where lm_head is separate
                if lm_head is not embed_tokens:
                    new_lm_weight = torch.zeros(new_vocab_size, embed_dim)
                    new_lm_weight[:old_vocab_size] = lm_head.weight.to('cpu')
                    if new_vocab_size > old_vocab_size:
                        new_lm_weight[old_vocab_size:].normal_(mean=0.0, std=std)
                    lm_head.weight = torch.nn.Parameter(new_lm_weight)
                
                model.config.vocab_size = new_vocab_size
                logging.info(f"[Backbone] Manual resize completed: {old_vocab_size} → {new_vocab_size}")


# -----------------------------------------------------------------------------
#  Main builder function
# -----------------------------------------------------------------------------

def get_backbone(params, num_task: int = 1):
    """load backbone model & tokenizer and apply PEFT if needed"""

    # ───────────────────── Determine backbone type ────────────────────────────
    if params.backbone_type == "auto":
        assert (
            params.backbone in BACKBONE2TYPE
        ), f"Not implemented for backbone {params.backbone}"
        setattr(params, "backbone_type", BACKBONE2TYPE[params.backbone])
    else:
        assert params.backbone_type in (
            "generative",
            "discriminative",
        ), f"Invalid backbone_type {params.backbone_type}"

    # ======================================================================
    #  1) Llama local ckpt (Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B)
    # ======================================================================
    if params.backbone in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "Llama-3.1-8B", "Llama-3.2-1B", "Llama-3.2-3B"]:
        model_path = f"./{params.backbone.split('/')[-1]}"
        print(model_path)

        # Load the tokenizer first to add special tokens
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side="left" if params.backbone_type == "generative" else "right",
        )
        
        # Add special tokens in advance
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        add_tokens = ["__ans__"]
        if getattr(params, "LAMOL_use_task_specific_gen_token", True):
            add_tokens += [f"__{i}__" for i in range(num_task)]
        else:
            add_tokens.append("__gen__")
        
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
        
        # The checkpoint's own config (including its llama3 RoPE scaling) is used unchanged.
        
        # All Llama sizes are loaded in full bf16 (the quantized branch below is not used).
        is_1b = True
        if is_1b:
            try:
                if bool(getattr(params, "pdr_no_liger", False)):
                    raise RuntimeError("disabled by pdr_no_liger (PDR needs the lm_head forward)")
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama()
                print("[FAST] Applied Liger kernel patches to Llama")
            except Exception as e:
                print(f"[FAST] Liger patch skipped: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
                attn_implementation=str(getattr(params, "pdr_attn", ATTN_IMPL)),
            ).to("cuda")
            print(f"[FAST] Loaded Llama-1B in bf16 + {getattr(params, 'pdr_attn', ATTN_IMPL)}")
            if bool(getattr(params, "backbone_grad_ckpt", False)):
                # Optional gradient checkpointing: recompute activations in the backward pass
                # instead of storing them (numerically neutral). Off in all released configurations.
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
                print("[FAST] gradient checkpointing enabled (backbone_grad_ckpt)")
        else:
            try:
                # Try 8bit quantization (3B/8B for memory savings)
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=False,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                print("Loaded with 8bit quantization")
            except Exception as e:
                print(f"8bit quantization failed: {e}")
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=False,
                        quantization_config=quantization_config,
                        device_map="auto",
                    )
                    print("Loaded with 4bit quantization")
                except Exception as e2:
                    print(f"4bit quantization also failed: {e2}")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32,
                        trust_remote_code=False,
                    )
                    print("Loaded on CPU with float32")
        
        # Resize embeddings (using a safe method)
        if num_added > 0:
            try:
                # Disable mean resizing to avoid the linalg_eig operation
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Resized embeddings: +{num_added} tokens → {len(tokenizer)}")
            except Exception as e:
                logging.warning(f"[Backbone] Embedding resize failed: {e}")
                # On meta device, update only the config
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Config updated: vocab_size = {len(tokenizer)}")

        # ───────────────────── PEFT (optional) ─────────────────────
        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    # ======================================================================
    #  2) GPT-OSS-20B local checkpoint
    # ======================================================================
    elif params.backbone == "gpt-oss-20b":
        print(f"Loading gpt-oss-20b from local path: {params.backbone}")
        model_path = "./gpt-oss-20b"
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens
        add_tokens = ["__ans__"]
        if getattr(params, "LAMOL_use_task_specific_gen_token", True):
            add_tokens += [f"__{i}__" for i in range(num_task)]
        else:
            add_tokens.append("__gen__")
        
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
        
        # Enable gradient checkpointing (to save memory)
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
        # Additional memory optimization settings
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False  # disable KV cache to save memory
            
        # Additional optimizations for MoE models
        if hasattr(model.config, 'num_local_experts'):
            # Disable expert parallelism to save memory
            if hasattr(model.config, 'output_router_logits'):
                model.config.output_router_logits = False
        
        # Resize embeddings (using a safe method)
        if num_added > 0:
            try:
                # Disable mean resizing to avoid the linalg_eig operation
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Resized embeddings: +{num_added} tokens → {len(tokenizer)}")
            except Exception as e:
                logging.warning(f"[Backbone] Embedding resize failed: {e}")
                # On meta device, update only the config
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Config updated: vocab_size = {len(tokenizer)}")
        
        # ───────────────────── PEFT (optional) ─────────────────────
        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    # ======================================================================
    #  3) Qwen local checkpoints (0.5B / 0.6B)
    # ======================================================================
    elif params.backbone in ["Qwen3-0.6B", "Qwen2-0.5B", "Qwen2.5-0.5B", "Qwen3-4B", "Qwen3-8B"]:
        model_path = f"./{params.backbone}"
        print(model_path)

        if params.backbone in ["Qwen3-4B", "Qwen3-8B"]:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, device_map="auto",
                torch_dtype="auto", attn_implementation=ATTN_IMPL,
            )
        else:
            # Load in bf16 like the other backbones (without a dtype, transformers loads fp32).
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, device_map="auto", torch_dtype=torch.bfloat16
            )
            print("[FAST] Loaded %s in %s" % (params.backbone, next(model.parameters()).dtype))
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            padding_side="left" if params.backbone_type == "generative" else "right",
        )

        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    # ======================================================================
    #  4) Other HuggingFace Hub models (online/cached)
    # ======================================================================
    else:
        try:
            config = AutoConfig.from_pretrained(params.backbone)
            config.return_dict = True
        except Exception:
            config = AutoConfig.from_pretrained(
                os.path.join(params.backbone_cache_path, params.backbone)
            )
            config.return_dict = True

        if params.method == "AdapterCL":
            model = AutoAdapterModel.from_pretrained(params.backbone, config=config)

        if params.method == "CPFD":
            config.output_attentions = True

        # ── Load model ---------------------------------------------------------
        def _load_model_from_pretrained(path, **kwargs):
            # torch_dtype="auto": follow the checkpoint dtype (bf16 for
            # modern releases) instead of the transformers default fp32.
            kwargs.setdefault("torch_dtype", "auto")
            return AutoModelForCausalLM.from_pretrained(
                path, low_cpu_mem_usage=True, device_map="auto", **kwargs
            )

        if params.backbone_revision:
            # A specific revision is requested
            cache_dir = os.path.join(
                params.backbone_cache_path,
                os.path.join(os.path.basename(params.backbone), params.backbone_revision),
            )
            try:
                model = _load_model_from_pretrained(
                    params.backbone,
                    config=config,
                    revision=params.backbone_revision,
                    cache_dir=cache_dir,
                )
            except Exception:
                model = _load_model_from_pretrained(
                    os.path.join(cache_dir), config=config
                )
        else:
            # Use the latest revision by default
            try:
                model = _load_model_from_pretrained(params.backbone, config=config)
            except Exception:
                model = _load_model_from_pretrained(
                    os.path.join(params.backbone_cache_path, params.backbone),
                    config=config,
                )

        # Multimodal configs (e.g. Gemma3 4B+) keep hidden_size etc. only under text_config;
        # expose them at the top level for code that reads model.config.hidden_size.
        _tc = getattr(model.config, "text_config", None)
        if _tc is not None and not hasattr(model.config, "hidden_size"):
            for _k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                       "intermediate_size", "head_dim", "vocab_size"):
                if hasattr(_tc, _k):
                    setattr(model.config, _k, getattr(_tc, _k))

        # Random initialization option
        if getattr(params, "backbone_random_init", False):
            model.apply(model._init_weights)

        # ── PEFT --------------------------------------------------------------
        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # ── Load tokenizer ----------------------------------------------------
        def _load_tokenizer(path, **kwargs):
            return AutoTokenizer.from_pretrained(
                path,
                padding_side="left" if params.backbone_type == "generative" else "right",
                **kwargs,
            )

        if params.backbone_revision:
            cache_dir = os.path.join(
                params.backbone_cache_path,
                os.path.join(os.path.basename(params.backbone), params.backbone_revision),
            )
            try:
                tokenizer = _load_tokenizer(
                    params.backbone, revision=params.backbone_revision, cache_dir=cache_dir
                )
            except Exception:
                tokenizer = _load_tokenizer(os.path.join(cache_dir))
        else:
            try:
                tokenizer = _load_tokenizer(params.backbone)
            except Exception:
                tokenizer = _load_tokenizer(
                    os.path.join(params.backbone_cache_path, params.backbone)
                )

    # ======================================================================
    #  Common post-processing (pad_token / special_token / embedding sync)
    # ======================================================================
    # Llama and GPT-OSS-20B models were already handled above, so skip them
    if params.backbone_type == "generative" and params.backbone not in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "gpt-oss-20b"]:
        _sync_tokenizer_and_embeddings(tokenizer, model, params, num_task)

    # pad_token safeguard (including discriminative)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if hasattr(model, "config") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if bool(getattr(params, "backbone_grad_ckpt", False)):
        # Optional gradient checkpointing, applied here so that it covers every backbone
        # family (numerically neutral). Off in all released configurations.
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("[CKPT] gradient checkpointing enabled (backbone_grad_ckpt)")

    print(model)
    return model, tokenizer

# -----------------------------------------------------------------------------
#  obtain_features / obtain_generate_ids
# -----------------------------------------------------------------------------

def _validate_lm_input_for_features(lm_input, model, context="obtain_features"):
    """Safety validation for obtain_features"""
    logger = logging.getLogger()
    validated_input = {}
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if "input_ids" in lm_input:
        input_ids = lm_input["input_ids"]
        if input_ids.max() >= vocab_size or input_ids.min() < 0:
            logger.warning(
                f"[{context}] input_ids out of range (0–{vocab_size-1}). Clamping to unk_token."
            )
            unk_id = getattr(model.config, "unk_token_id", 0) or 0
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        validated_input["input_ids"] = input_ids
    if "attention_mask" in lm_input:
        validated_input["attention_mask"] = lm_input["attention_mask"]
    for k in lm_input:
        if k not in validated_input:
            validated_input[k] = lm_input[k]
    return validated_input


def obtain_features(params, model, lm_input, tokenizer):
    """Extract last hidden state features for classification"""
    safe_input = _validate_lm_input_for_features(lm_input, model)
    if params.backbone_type == "generative":
        assert params.classification_type == "sentence-level"
        all_hs = model(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        ).hidden_states
        assert params.backbone_extract_token == "last_token"
        feat = all_hs[-1][:, -1, :].contiguous()
    elif params.backbone_type == "discriminative":
        all_hs = model(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            output_hidden_states=True,
        ).hidden_states
        if params.classification_type == "sentence-level":
            if params.backbone_extract_token == "last_token":
                idx = safe_input["attention_mask"].sum(dim=-1) - 1
                last = all_hs[-1]
                batch, seq, dim = last.size()
                idx = idx.view(-1, 1, 1).expand(-1, 1, dim)
                feat = last.gather(1, idx).squeeze(1).contiguous()
            elif params.backbone_extract_token == "cls_token":
                feat = all_hs[-1][:, 0, :].contiguous()
            else:
                raise NotImplementedError()
        elif params.classification_type == "word-level":
            feat = all_hs[-1]
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return feat


def obtain_generate_ids(params, model, lm_input, tokenizer):
    """Generate continuation ids"""
    safe_input = _validate_lm_input_for_features(lm_input, model, "obtain_generate_ids")
    in_len = safe_input["input_ids"].shape[1]
    _gen_kw = {}
    _lp = getattr(params, "_ri_logproc", None)
    if _lp is not None:
        from transformers import LogitsProcessorList
        _gen_kw["logits_processor"] = LogitsProcessorList([_lp])
    import os as _os
    if _os.environ.get("PDR_LAND") or _os.environ.get("PDR_DRIFT"):
        # Diagnostic (env PDR_LAND / PDR_DRIFT, evaluation only): record the last-layer state at the last prompt
        # position; the generated sequence is unchanged.
        _g = model.generate(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            max_new_tokens=params.backbone_max_new_token,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
            return_dict_in_generate=True, output_hidden_states=True, **_gen_kw,
        )
        try:
            _h0 = _g.hidden_states[0][-1][:, -1, :].float()       # last layer, last prompt position
            globals().setdefault("_PDR_LAND", []).append(_h0.detach().cpu())
        except Exception:
            pass
        return _g.sequences[:, in_len:].contiguous()
    if _os.environ.get("PDR_MOVE"):
        # Diagnostic (env PDR_MOVE, evaluation only): record per-class mean answer states; the generated
        # sequence is unchanged.
        _g = model.generate(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            max_new_tokens=params.backbone_max_new_token,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
            return_dict_in_generate=True, output_hidden_states=True, **_gen_kw,
        )
        seq = _g.sequences[:, in_len:].contiguous()
        try:
            _tg = lm_input.get("target", None)
            _acc = globals().setdefault("_PDR_MOVE", {})
            for _b in range(seq.shape[0]):
                if _tg is None or _b >= len(_tg):
                    continue
                _cl = str(_tg[_b]).strip()
                _gt = tokenizer(" " + _cl, add_special_tokens=False)["input_ids"]
                for _o in range(min(len(_gt), seq.shape[1], len(_g.hidden_states))):
                    _h = _g.hidden_states[_o][-1][_b, -1, :].float().cpu()
                    _k = (_cl, _o)
                    _e = _acc.get(_k)
                    if _e is None:
                        _acc[_k] = [_h.clone(), 1, int(_gt[_o])]
                    else:
                        _e[0] += _h; _e[1] += 1
        except Exception:
            pass
        return seq
    if _os.environ.get("PDR_PART"):
        # Diagnostic (env PDR_PART, evaluation only): at the first position where the generated answer departs
        # from the gold one, record the score gap and the state's similarity to the stored record; the generated
        # sequence is unchanged.
        _g = model.generate(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            max_new_tokens=params.backbone_max_new_token,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
            return_dict_in_generate=True, output_scores=True, output_hidden_states=True, **_gen_kw,
        )
        seq = _g.sequences[:, in_len:].contiguous()
        try:
            _tg = lm_input.get("target", None)
            _rec = globals().get("_PDR_REC", None)
            _st = torch.stack(list(_g.scores), dim=1).float()             # (B, T, V)
            _acc = globals().setdefault("_PDR_PART", [])
            for _b in range(seq.shape[0]):
                if _tg is None or _b >= len(_tg):
                    continue
                _gt = tokenizer(" " + str(_tg[_b]).strip(), add_special_tokens=False)["input_ids"]
                _o = None
                for _i in range(min(len(_gt), seq.shape[1])):
                    if int(seq[_b, _i]) != int(_gt[_i]):
                        _o = _i
                        break
                if _o is None:
                    continue                                              # answer matches gold
                _row = _st[_b, _o]
                _win = int(_row.argmax())
                _gap = float(_row[_win] - _row[int(_gt[_o])])
                _cs = -1.0
                if _rec is not None:
                    _k = (str(_tg[_b]).strip(), _o)
                    _r = _rec.get(_k)
                    if _r is not None:
                        try:
                            _hv = _g.hidden_states[_o][-1][_b, -1, :].float().cpu()
                            _cs = float(torch.nn.functional.cosine_similarity(
                                _hv.unsqueeze(0), _r.unsqueeze(0), dim=1))
                        except Exception:
                            _cs = -1.0
                _acc.append((_o, _gap, _cs, str(_tg[_b]).strip(), tokenizer.decode([_win]).strip()))
        except Exception:
            pass
        return seq
    if _os.environ.get("PDR_MARGIN"):
        # Diagnostic (env PDR_MARGIN, evaluation only): record the smallest top-1/top-2 score gap of each
        # generated answer; the generated sequence is unchanged.
        _g = model.generate(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            max_new_tokens=params.backbone_max_new_token,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
            return_dict_in_generate=True, output_scores=True, **_gen_kw,
        )
        seq = _g.sequences[:, in_len:].contiguous()
        try:
            _eos = tokenizer.eos_token_id
            _st = torch.stack(list(_g.scores), dim=1).float()            # (B, T, V)
            _t2 = _st.topk(2, dim=-1).values
            _gap = _t2[..., 0] - _t2[..., 1]                             # (B, T) how close each step was
            _live = (seq != _eos)
            _live = _live & _live.cumprod(dim=1).bool()                  # only up to the first stop
            _gap = _gap.masked_fill(~_live, float("inf"))
            _w = _gap.min(dim=1).values                                  # the weakest step of each answer
            _w = _w[torch.isfinite(_w)]
            if _w.numel():
                _acc = globals().setdefault("_PDR_MARG", [])
                _acc.extend([float(x) for x in _w.detach().cpu()])
        except Exception:
            pass
        return seq
    gen_all = model.generate(
        input_ids=safe_input["input_ids"],
        attention_mask=safe_input["attention_mask"],
        max_new_tokens=params.backbone_max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        **_gen_kw,
    )
    # Diagnostic (evaluation only): count how often the first generated token is end-of-text (empty answers);
    # the generated sequence is unchanged.
    _seq = gen_all[:, in_len:].contiguous()
    try:
        if _seq.shape[1] > 0:
            _first = _seq[:, 0]
            _e = tokenizer.eos_token_id
            _acc = globals().setdefault("_PDR_EOS", [0, 0])
            _acc[0] += int((_first == _e).sum())
            _acc[1] += int(_first.numel())
    except Exception:
        pass
    return _seq