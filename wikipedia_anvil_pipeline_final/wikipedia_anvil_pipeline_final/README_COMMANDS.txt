WIKIPEDIA GPT-2 CONTROL PIPELINE ON ANVIL
========================================

Repository location assumed by the .sub files:

  /anvil/projects/x-cis261275/training/wikipedia-pipeline

Required repository contents:

  prepare_wikipedia.py
  tokenize_wikipedia.py
  train_gpt2_wikipedia.py
  prepare_wikipedia.sub
  tokenize_wikipedia.sub
  train_gpt2_wikipedia.sub
  tokenizer_bracket/
      tokenizer.json
      tokenizer_config.json

The tokenizer_bracket directory must be the exact tokenizer used by the bracket
model. The control text itself contains no bracket annotations, but the shared
vocabulary keeps tokenization, embedding size, LM-head size, and parameter
count matched.

0. ONE-TIME SETUP
-----------------

  export PROJECT=/anvil/projects/x-cis261275
  export REPO_DIR="$PROJECT/training/wikipedia-pipeline"

  mkdir -p "$REPO_DIR" "$PROJECT/training/wikipedia"/{raw,hf_cache} "$PROJECT/training/gpt2-wikipedia"
  cd "$REPO_DIR"

Verify the environment:

  module purge
  module load modtree/gpu
  module load conda
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/x-jhouck1/.conda/envs/pipeline

  python -c "import datasets, transformers, accelerate, tensorboard, pyarrow, huggingface_hub; print('all imports succeeded')"

Verify the exact tokenizer before processing:

  python - <<'PY'
  from pathlib import Path
  from transformers import AutoTokenizer

  path = Path('tokenizer_bracket').resolve()
  tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
  tok.pad_token = tok.eos_token
  print('path:', path)
  print('base vocab:', tok.vocab_size)
  print('total vocab:', len(tok))
  print('EOS/PAD:', tok.eos_token_id)
  print('added vocab:', tok.get_added_vocab())
  PY

1. PREPARE CANONICAL UNTOKENIZED WIKIPEDIA
------------------------------------------

  cd "$REPO_DIR"
  PREP_JOB=$(sbatch --parsable prepare_wikipedia.sub)
  echo "PREP_JOB=$PREP_JOB"

Monitor:

  squeue -j "$PREP_JOB"
  tail -f "$PROJECT/training/wiki-prepare-${PREP_JOB}.out"
  tail -f "$PROJECT/training/wiki-prepare-${PREP_JOB}.err"

Output:

  $PROJECT/training/wikipedia/cleaned

This stage does not use a tokenizer. It is safe to rerun; completed source
shards are reused.

2. TOKENIZE WITH tokenizer_bracket
----------------------------------

After preparation succeeds:

  cd "$REPO_DIR"
  TOKEN_JOB=$(sbatch --parsable tokenize_wikipedia.sub)
  echo "TOKEN_JOB=$TOKEN_JOB"

Monitor:

  squeue -j "$TOKEN_JOB"
  tail -f "$PROJECT/training/wiki-tokenize-${TOKEN_JOB}.out"
  tail -f "$PROJECT/training/wiki-tokenize-${TOKEN_JOB}.err"

Outputs:

  $PROJECT/training/wikipedia/comparison_tokenizer
  $PROJECT/training/wikipedia/tokenized_gpt2_1024

The script saves a semantic SHA-256 fingerprint of the token-to-ID mapping. It
will refuse to reuse a completed dataset made with a different tokenizer.

If tokenized_gpt2_1024 was previously produced with the stock GPT-2 tokenizer,
force a rebuild without deleting the cleaned corpus:

  TOKEN_JOB=$(sbatch --parsable --export=ALL,TOKENIZE_FORCE=1 tokenize_wikipedia.sub)

3. VERIFY THE SAVED DATA
------------------------

  python - <<'PY'
  import json
  import os
  from datasets import load_from_disk
  from transformers import AutoTokenizer

  root = os.path.join(os.environ['PROJECT'], 'training', 'wikipedia')
  clean = load_from_disk(os.path.join(root, 'cleaned'))
  tokens = load_from_disk(os.path.join(root, 'tokenized_gpt2_1024'))
  tok = AutoTokenizer.from_pretrained(os.path.join(root, 'comparison_tokenizer'), local_files_only=True)
  info = json.load(open(os.path.join(root, 'tokenized_gpt2_1024', 'TOKENIZATION_INFO.json')))

  print(clean)
  print('clean columns:', clean['train'].column_names)
  print('article sample:', clean['train'][0])
  print(tokens)
  print('first block length:', len(tokens['train'][0]['input_ids']))
  print('tokenizer length:', len(tok))
  print('fingerprint:', info['tokenizer_fingerprint'])
  print('train tokens:', info['train_tokens'])
  PY

4. TRAIN THE RANDOMLY INITIALIZED CONTROL
-----------------------------------------

  cd "$REPO_DIR"
  TRAIN_JOB=$(sbatch --parsable train_gpt2_wikipedia.sub)
  echo "TRAIN_JOB=$TRAIN_JOB"

Monitor:

  squeue -j "$TRAIN_JOB"
  tail -f "$PROJECT/training/gpt2-wikipedia-${TRAIN_JOB}.out"
  tail -f "$PROJECT/training/gpt2-wikipedia-${TRAIN_JOB}.err"

Output for seed 42:

  $PROJECT/training/gpt2-wikipedia/control-brackettok-seed-42/

The directory final-best-validation/ is a copy of the checkpoint with the
lowest observed validation loss. BEST_CHECKPOINT.json records its original
checkpoint path and loss.

The model is GPT-2 small constructed with GPT2LMHeadModel(config) after
set_seed(42). It does not load /storage/HF_control/checkpoint-34350 or any other
old OpenWebText checkpoint.

5. SUBMIT WITH DEPENDENCIES
---------------------------

You may queue the stages in order:

  cd "$REPO_DIR"
  PREP_JOB=$(sbatch --parsable prepare_wikipedia.sub)
  TOKEN_JOB=$(sbatch --parsable --dependency=afterok:"$PREP_JOB" tokenize_wikipedia.sub)
  TRAIN_JOB=$(sbatch --parsable --dependency=afterok:"$TOKEN_JOB" train_gpt2_wikipedia.sub)
  printf 'prepare=%s\ntokenize=%s\ntrain=%s\n' "$PREP_JOB" "$TOKEN_JOB" "$TRAIN_JOB"

For the first run, inspecting each stage before submitting the next is safer.

6. BATCH-SIZE OPTIONS
---------------------

Default training batch:

  MICRO_BATCH_SIZE=32
  GRAD_ACCUM_STEPS=2
  effective batch = 64 sequences = 65,536 tokens per optimizer update

This matches the old effective batch of 8 x 8 while using fewer accumulation
steps. After a short throughput test, an H100 may support 64 x 1:

  sbatch --export=ALL,MICRO_BATCH_SIZE=64,GRAD_ACCUM_STEPS=1,EVAL_BATCH_SIZE=128 train_gpt2_wikipedia.sub

Use the same selected values for every control/bracket pair. Do not use
auto_find_batch_size, because it can silently make the experiments differ.

7. IMPORTANT EXPERIMENT NOTES
-----------------------------

- The training script saves the best validation checkpoint, as requested.
- Keep EVAL_STEPS and SAVE_STEPS identical and make SAVE_STEPS a multiple of
  EVAL_STEPS.
- The default learning rate, cosine scheduler, warmup, weight decay, BF16, and
  effective batch match the previous control setup.
- NUM_TRAIN_EPOCHS defaults to 1 for a fresh Wikipedia run. Change it only if
  the matched bracket experiment uses the same total schedule.
- The custom collator preserves real EOS article-boundary labels. The ordinary
  DataCollatorForLanguageModeling can mask those EOS labels when EOS is also PAD.
- Reusing an existing output directory is allowed only when the saved run
  settings match. Otherwise the script stops rather than resuming the wrong run.
- Preprocessing jobs use cis261275-ai / ai without requesting a GPU because no
  CPU allocation is available. If Slurm rejects that account/resource
  combination, the allocation administrator or Anvil support must specify the
  permitted submission form.
