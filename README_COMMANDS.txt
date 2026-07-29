ANVIL COMMANDS
==============

1. Put the three Python files and two .sub files in:

   /anvil/projects/x-cis261275/training

2. Activate the environment and verify the required packages:

   module purge
   module load modtree/gpu
   module load conda
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate /home/x-jhouck1/.conda/envs/pipeline

   python -c "import datasets, transformers, accelerate, tensorboard, pyarrow, huggingface_hub; print('all required imports succeeded')"
   python -m pip freeze > /anvil/projects/x-cis261275/training/wikipedia/python_environment-before-run.txt

   Only when an import is missing, install it without blindly upgrading the
   entire working environment:

   python -m pip install datasets transformers accelerate tensorboard pyarrow huggingface_hub

3. Create all required directories:

   export PROJECT=/anvil/projects/x-cis261275
   mkdir -p \
     "$PROJECT/training/wikipedia/raw" \
     "$PROJECT/training/wikipedia/hf_cache" \
     "$PROJECT/training/wikipedia/cleaned" \
     "$PROJECT/training/wikipedia/tokenized_gpt2_1024" \
     "$PROJECT/training/gpt2-wikipedia"

4. Submit preparation + tokenization:

   cd "$PROJECT/training"
   PREP_JOB=$(sbatch --parsable prepare_wikipedia.sub)
   echo "$PREP_JOB"

   squeue -j "$PREP_JOB"
   tail -f "$PROJECT/training/wiki-prepare-${PREP_JOB}.out"
   tail -f "$PROJECT/training/wiki-prepare-${PREP_JOB}.err"

   The scripts resume at completed source/tokenization shards. If the 48-hour
   wall time expires, submit the same file again:

   sbatch prepare_wikipedia.sub

5. Verify the completed datasets interactively (small metadata/sample reads only):

   module purge
   module load modtree/gpu
   module load conda
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate /home/x-jhouck1/.conda/envs/pipeline
   export PROJECT=/anvil/projects/x-cis261275

   python - <<'PY'
   import os
   from datasets import load_from_disk

   root = os.path.join(os.environ["PROJECT"], "training", "wikipedia")
   clean = load_from_disk(os.path.join(root, "cleaned"))
   tokens = load_from_disk(os.path.join(root, "tokenized_gpt2_1024"))

   print(clean)
   print(clean["train"].column_names)
   print(clean["train"][0])
   print(tokens)
   print(len(tokens["train"][0]["input_ids"]))
   PY

6. Check storage:

   du -sh "$PROJECT/training/wikipedia"/* "$PROJECT/training/gpt2-wikipedia" 2>/dev/null

7. Submit GPU training after tokenization succeeds:

   cd "$PROJECT/training"
   TRAIN_JOB=$(sbatch --parsable train_gpt2_wikipedia.sub)
   echo "$TRAIN_JOB"

   squeue -j "$TRAIN_JOB"
   tail -f "$PROJECT/training/gpt2-wikipedia-${TRAIN_JOB}.out"
   tail -f "$PROJECT/training/gpt2-wikipedia-${TRAIN_JOB}.err"

8. Completed-job diagnostics:

   seff "$PREP_JOB"
   seff "$TRAIN_JOB"
   sacct -j "$TRAIN_JOB" --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode

9. TensorBoard:

   tensorboard --logdir "$PROJECT/training/gpt2-wikipedia/seed-42/runs" --port 6006

IMPORTANT PARTITION NOTE
========================

prepare_wikipedia.sub uses account cis261275-ai and partition ai but requests
no GPU, so the Python work itself is CPU-only. Anvil separates CPU, GPU, and AI
allocations. If the scheduler rejects a no-GPU AI job, you need the name of a
CPU allocation and should change only these two lines in prepare_wikipedia.sub:

   #SBATCH -A <your CPU allocation>
   #SBATCH -p shared

Check available allocations and partitions with:

   mybalance
   showpartitions

Do not add --gpus-per-node=1 merely to make preprocessing run unless Anvil
support explicitly tells you that this is required for your allocation.


RESEARCH-COMPARABILITY NOTE
===========================

This control tokenization is pinned to the ordinary GPT-2 tokenizer and keeps
the canonical cleaned dataset untokenized. If the later metadata model adds
new bracket tokens, its embedding/output vocabulary will be larger. For an
exact parameter-matched control, create a second control tokenization/model
using the same augmented tokenizer; ordinary Wikipedia text will still encode
the same when the added bracket tokens never occur. Keep that output in a new
directory rather than overwriting tokenized_gpt2_1024.
