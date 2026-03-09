"""
╔══════════════════════════════════════════════════════════════════╗
║   🎓 UPSC AI — Qwen2.5-VL-7B  |  Local GPU Version             ║
║   Optimized for: RTX 3060/3070/3080/3090/4070/4080/4090         ║
║   Run: python upsc_train_local_gpu.py                            ║
╠══════════════════════════════════════════════════════════════════╣
║  SETUP (run once in terminal before this script):                ║
║    pip install torch torchvision --index-url                     ║
║         https://download.pytorch.org/whl/cu121                  ║
║    pip install transformers==4.47.0 peft==0.13.0 trl==0.12.0    ║
║    pip install bitsandbytes==0.44.1 accelerate==0.34.0           ║
║    pip install datasets qwen-vl-utils pillow                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# ①  CONFIG — Edit these to match your setup
# ══════════════════════════════════════════════
class CFG:
    # ── Paths ──────────────────────────────────
    DATA_PATH  = "upsc_dataset.jsonl"      # ← put your .jsonl here
    OUTPUT_DIR = "./upsc-checkpoints"
    FINAL_DIR  = "./upsc-qwen-final"
    LOG_DIR    = "./logs"

    # ── Model ──────────────────────────────────
    MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"

    # ── QLoRA ──────────────────────────────────
    LORA_R       = 64       # Rank — how much new UPSC knowledge fits
    LORA_ALPHA   = 128      # Always 2x LORA_R
    LORA_DROPOUT = 0.05

    # ── Training ───────────────────────────────
    EPOCHS       = 3
    MAX_SEQ_LEN  = 2048

    # ── GPU PRESETS — uncomment your GPU ───────
    # RTX 4090 / 3090 (24 GB) — fastest
    BATCH_SIZE  = 2
    GRAD_ACCUM  = 4

    # RTX 4080 / 3080 (16 GB) — use these instead:
    # BATCH_SIZE  = 1
    # GRAD_ACCUM  = 8

    # RTX 3060 / 4060 (8–12 GB) — use these instead:
    # BATCH_SIZE  = 1
    # GRAD_ACCUM  = 16

    LR           = 2e-4
    WARMUP_RATIO = 0.03
    MAX_GRAD     = 0.3
    SAVE_STEPS   = 100
    LOG_STEPS    = 10

    # ── Precision ──────────────────────────────
    # bf16=True  → RTX 3000+ series (Ampere & newer), best choice
    # fp16=True  → older GPUs (RTX 2000 series)
    USE_BF16 = True     # ← change to False if your GPU is older
    USE_FP16 = False    # ← change to True if USE_BF16 causes errors


# ══════════════════════════════════════════════
# ②  GPU CHECK
# ══════════════════════════════════════════════
def check_gpu():
    if not torch.cuda.is_available():
        log.error("❌ No CUDA GPU found!")
        log.error("   Make sure NVIDIA drivers + CUDA toolkit are installed.")
        log.error("   Check: nvidia-smi")
        sys.exit(1)

    gpu_name  = torch.cuda.get_device_name(0)
    gpu_mem   = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"✅ GPU : {gpu_name}")
    log.info(f"   VRAM: {gpu_mem:.1f} GB")

    # Auto-switch to fp16 for older GPUs
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 8:   # Ampere = compute cap 8.x
        log.warning("⚠️  GPU is pre-Ampere → switching to fp16 (bf16 not supported)")
        CFG.USE_BF16 = False
        CFG.USE_FP16 = True

    log.info(f"   Precision: {'bf16' if CFG.USE_BF16 else 'fp16'}")
    return gpu_mem


# ══════════════════════════════════════════════
# ③  DATASET LOADER  (fixes ArrowInvalid)
# ══════════════════════════════════════════════
def normalize_content(content) -> str:
    """
    Converts ANY content type → plain string.
    Needed because PyArrow requires uniform column types.

    "plain text"                        → "plain text"
    [{"type":"text","text":"hi"}]       → "hi"
    [{"type":"image","image":"x.jpg"},  → "[IMAGE:x.jpg] explain"
     {"type":"text","text":"explain"}]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            t = item.get("type", "")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "image":
                src = item.get("image", item.get("url", ""))
                parts.append(f"[IMAGE:{src}]")
            elif t == "video":
                parts.append(f"[VIDEO:{item.get('video','')}]")
        return " ".join(parts)
    return str(content)


def load_dataset_from_jsonl(path: str) -> Dataset:
    path = Path(path)
    if not path.exists():
        log.error(f"❌ Dataset not found: {path.resolve()}")
        log.error("   Put upsc_dataset.jsonl in the same folder as this script.")
        sys.exit(1)

    records, skipped = [], 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Line {line_num} bad JSON → skipped: {e}")
                skipped += 1
                continue

            messages = obj.get("messages", [])
            if not messages:
                skipped += 1
                continue

            # Normalized version — uniform types, PyArrow-safe
            normalized = [
                {"role": m.get("role", "user"),
                 "content": normalize_content(m.get("content", ""))}
                for m in messages
            ]

            records.append({
                "messages":     normalized,
                "raw_messages": json.dumps(messages, ensure_ascii=False),
            })

    log.info(f"✅ Loaded  : {len(records):,} examples  |  skipped: {skipped}")
    dataset = Dataset.from_list(records)
    log.info(f"   Features: {list(dataset.features.keys())}")

    # Show a sample
    s = dataset[0]
    for m in s["messages"]:
        preview = m["content"][:80].replace("\n", " ")
        log.info(f"   [{m['role']:>9}]: {preview}")

    return dataset


# ══════════════════════════════════════════════
# ④  MODEL + PROCESSOR LOADER
# ══════════════════════════════════════════════
def load_model_and_processor():
    log.info(f"⏳ Loading {CFG.MODEL_ID} ...")
    log.info("   (First run downloads ~15 GB — this is normal)")

    compute_dtype = torch.bfloat16 if CFG.USE_BF16 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # Best accuracy at 4-bit
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,        # Extra RAM saving
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",                     # Auto-maps to your GPU
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()         # Required for QLoRA
    model.config.use_cache = False             # Must be off during training

    processor = AutoProcessor.from_pretrained(
        CFG.MODEL_ID,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    # Fix padding token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    trainable = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"✅ Model loaded  ({trainable:.1f}B total params)")
    return model, processor


# ══════════════════════════════════════════════
# ⑤  LORA SETUP
# ══════════════════════════════════════════════
def apply_lora(model):
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=CFG.LORA_R,
        lora_alpha=CFG.LORA_ALPHA,
        lora_dropout=CFG.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
            "gate_proj", "up_proj", "down_proj",        # FFN
        ],
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"✅ LoRA applied")
    log.info(f"   Trainable: {trainable/1e6:.1f}M / {total/1e9:.1f}B  "
             f"({100*trainable/total:.2f}%)")
    return model, lora_config


# ══════════════════════════════════════════════
# ⑥  DATA COLLATOR  (vision-aware)
# ══════════════════════════════════════════════
@dataclass
class UPSCDataCollator:
    """
    Handles both text-only and image+text examples.
    Reads raw_messages to reconstruct image content at training time.
    """
    processor: Any
    max_length: int = 2048

    def _load_image(self, src: str):
        from PIL import Image
        try:
            if os.path.exists(src):
                return Image.open(src).convert("RGB")
            if src.startswith("http"):
                import requests
                from io import BytesIO
                r = requests.get(src, timeout=8)
                return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            log.debug(f"Image load failed ({src}): {e}")
        return None

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        texts, all_images = [], []

        for ex in examples:
            try:
                raw_msgs = json.loads(ex["raw_messages"])
            except Exception:
                raw_msgs = ex["messages"]

            imgs_this_ex = []
            rebuilt = []

            for msg in raw_msgs:
                content = msg.get("content", "")
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if item.get("type") == "image":
                            img = self._load_image(item.get("image", ""))
                            if img:
                                imgs_this_ex.append(img)
                                new_content.append({"type": "image"})
                        else:
                            new_content.append(item)
                    rebuilt.append({"role": msg["role"], "content": new_content})
                else:
                    rebuilt.append(msg)

            text = self.processor.apply_chat_template(
                rebuilt, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            all_images.extend(imgs_this_ex)

        # Tokenize
        if all_images:
            batch = self.processor(
                text=texts, images=all_images,
                return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            )
        else:
            batch = self.processor.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


# ══════════════════════════════════════════════
# ⑦  TRAIN
# ══════════════════════════════════════════════
def train(model, dataset, data_collator, lora_config):
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(CFG.LOG_DIR,    exist_ok=True)

    steps_per_epoch = max(1, len(dataset) // (CFG.BATCH_SIZE * CFG.GRAD_ACCUM))
    total_steps     = steps_per_epoch * CFG.EPOCHS

    log.info("🚀 Starting training")
    log.info(f"   Examples      : {len(dataset):,}")
    log.info(f"   Epochs        : {CFG.EPOCHS}")
    log.info(f"   Steps/epoch   : {steps_per_epoch}")
    log.info(f"   Total steps   : {total_steps}")
    log.info(f"   Effective batch: {CFG.BATCH_SIZE * CFG.GRAD_ACCUM}")
    log.info(f"   Learning rate : {CFG.LR}")

    training_args = TrainingArguments(
        # ── Output ──────────────────────────
        output_dir=CFG.OUTPUT_DIR,
        logging_dir=CFG.LOG_DIR,

        # ── Epochs & Batch ──────────────────
        num_train_epochs=CFG.EPOCHS,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        gradient_accumulation_steps=CFG.GRAD_ACCUM,

        # ── LR & Schedule ───────────────────
        learning_rate=CFG.LR,
        warmup_ratio=CFG.WARMUP_RATIO,
        lr_scheduler_type="cosine",

        # ── Precision ───────────────────────
        bf16=CFG.USE_BF16,
        fp16=CFG.USE_FP16,

        # ── Memory ──────────────────────────
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",       # 8-bit optimizer saves more VRAM locally
        max_grad_norm=CFG.MAX_GRAD,
        dataloader_pin_memory=True,     # True is faster on local (unlike Colab)
        dataloader_num_workers=2,       # Parallel data loading — faster locally

        # ── Saving ──────────────────────────
        save_strategy="steps",
        save_steps=CFG.SAVE_STEPS,
        save_total_limit=3,

        # ── Logging ─────────────────────────
        logging_steps=CFG.LOG_STEPS,
        logging_first_step=True,
        report_to="tensorboard",        # run: tensorboard --logdir ./logs

        # ── Other ───────────────────────────
        remove_unused_columns=False,    # CRITICAL for vision models
        group_by_length=True,           # Batches similar-length seqs → faster
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        peft_config=lora_config,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    log.info(f"🎓 Training complete in {h}h {m}m {s}s")
    return trainer


# ══════════════════════════════════════════════
# ⑧  SAVE + MERGE
# ══════════════════════════════════════════════
def save_and_merge(trainer, processor):
    adapter_dir = "./upsc-lora-adapter"
    log.info(f"💾 Saving LoRA adapter → {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    log.info("⏳ Merging LoRA into base model (takes ~5 min)...")
    compute_dtype = torch.bfloat16 if CFG.USE_BF16 else torch.float16

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.MODEL_ID,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()
    merged.save_pretrained(CFG.FINAL_DIR)
    processor.save_pretrained(CFG.FINAL_DIR)

    log.info(f"✅ Final model saved → {CFG.FINAL_DIR}/")
    log.info("🎁 Your UPSC AI is ready!")


# ══════════════════════════════════════════════
# ⑨  QUICK TEST
# ══════════════════════════════════════════════
def quick_test(processor):
    log.info("🧪 Running quick test on final model...")
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=CFG.FINAL_DIR,
        tokenizer=processor.tokenizer,
        torch_dtype=torch.bfloat16 if CFG.USE_BF16 else torch.float16,
        device_map="auto",
    )
    messages = [
        {"role": "system",  "content": "You are an expert UPSC mentor. Give structured, concise answers."},
        {"role": "user",    "content": "Explain cooperative federalism with Indian examples. (GS2 Polity)"},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out    = pipe(prompt, max_new_tokens=300, temperature=0.7, do_sample=True)
    answer = out[0]["generated_text"].split("<|im_start|>assistant")[-1].strip()

    print("\n" + "═" * 60)
    print("🧪  UPSC AI TEST OUTPUT:")
    print("═" * 60)
    print(answer)
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════
# ⑩  MAIN
# ══════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════╗")
    print("║   🎓 UPSC AI Fine-tuning — Local GPU     ║")
    print("╚══════════════════════════════════════════╝\n")

    # 1. GPU check
    gpu_vram = check_gpu()

    # Auto adjust batch size based on VRAM
    if gpu_vram < 12:
        log.warning("⚠️  Low VRAM detected — forcing BATCH=1, GRAD_ACCUM=16")
        CFG.BATCH_SIZE = 1
        CFG.GRAD_ACCUM = 16
    elif gpu_vram < 20:
        log.info("ℹ️  Medium VRAM — using BATCH=1, GRAD_ACCUM=8")
        CFG.BATCH_SIZE = 1
        CFG.GRAD_ACCUM = 8
    else:
        log.info(f"🔥 High VRAM ({gpu_vram:.0f}GB) — using BATCH=2, GRAD_ACCUM=4")
        CFG.BATCH_SIZE = 2
        CFG.GRAD_ACCUM = 4

    # 2. Load dataset
    log.info("\n── STEP 1: Load Dataset ─────────────────")
    dataset = load_dataset_from_jsonl(CFG.DATA_PATH)

    # 3. Load model
    log.info("\n── STEP 2: Load Model ───────────────────")
    model, processor = load_model_and_processor()

    # 4. Apply LoRA
    log.info("\n── STEP 3: Apply QLoRA ──────────────────")
    model, lora_config = apply_lora(model)

    # 5. Data collator
    log.info("\n── STEP 4: Setup Data Collator ──────────")
    data_collator = UPSCDataCollator(processor=processor, max_length=CFG.MAX_SEQ_LEN)
    log.info("✅ Data collator ready")

    # 6. Train
    log.info("\n── STEP 5: Train ────────────────────────")
    trainer = train(model, dataset, data_collator, lora_config)

    # 7. Save + merge
    log.info("\n── STEP 6: Save & Merge ─────────────────")
    save_and_merge(trainer, processor)

    # 8. Test
    log.info("\n── STEP 7: Quick Test ───────────────────")
    quick_test(processor)

    log.info("✅ All done! Share the ./upsc-qwen-final folder with your friend.")


if __name__ == "__main__":
    main()
