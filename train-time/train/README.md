# Training 

We follow the implementation from the official Verl Recipe [https://github.com/verl-project/verl-recipe/tree/main/retool](https://github.com/verl-project/verl-recipe/tree/main/retool). Please begin by preparing the original dataset. There are two training stages i) Supervised Fine-Tuning (SFT) and ii) Reinforcement Learning (RL).

## Stage 1: Supervised Fine-Tuning (SFT)
The SFT training script template is organized in this template
```sh
nnodes=1
nproc_per_node=4
master_addr=127.0.0.1
master_port=29500
node_rank=${ARNOLD_ID:-0}

project_name=retool
experiment_name=multiturn-{APPROACH}-{MODEL_NAME}

DATA_ROOT=${DATA_ROOT:-$PWD}

TRAIN_DATA=$DATA_ROOT/data/{TRAIN_DATA_PATH}
EVAL_DATA=$DATA_ROOT/data/{EVAL_DATA_PATH}
MODEL_PATH={MODEL_PATH}
SAVE_PATH=$DATA_ROOT/checkpoint/$experiment_name

torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     --master-addr=$master_addr \
     --master-port=$master_port \
     --node-rank=$node_rank \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    trainer.resume_mode="auto" \
    data.max_length=20000 \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=6 \
    trainer.save_freq=62 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true
```

## Stage 2: Reinforcement Learning (RL)
To begin with the stage 2, you need to run an isolated sandbox environment for the tool execution.
```sh
bash start_ci.sh
```

The GRPO training script template is organized in below example where the data and yaml for tool configuration is referenced from the original verl-recipe repository.
```sh
DATA_ROOT=${DATA_ROOT:-$PWD/data}

dapo_math_17k=$DATA_ROOT/BytedTsinghua-SIA/DAPO-Math-17k
aime_2024=$DATA_ROOT/Maxwell-Jia/AIME_2024
aime_2025=$DATA_ROOT/yentinglin/aime_2025
model_path={SFT_MODEL_PATH}

train_files="['$dapo_math_17k']"
test_files="['$aime_2025', '$aime_2024']"

# tool
tool_config_path={RETOOL_MODULE_PATH}/sandbox_fusion_tool_config.yaml

# wandb
project_name={PROJECT_NAME}
experiment_name={EXPERIMENT_NAME}
default_local_dir={SAVE_PATH}

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=16
max_prompt_length=2048
max_response_length=16384
actor_lr=1e-6

train_batch_size=64
ppo_mini_batch_size=16
ppo_micro_batch_size=8
n_resp_per_prompt=16
n_resp_per_prompt_val=30

# ================= perfomance =================
infer_tp=4 # vllm
train_sp=4 # train
offload=True # original is True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 1 ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ppo_micro_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.resume_mode="auto" \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 $@
```

The individual scripts are located in [jobs folder.](./jobs/)

## (Optional) Apptainer

In our setup, we adopt Apptainer, a container framework for slurm cluster where we build a container using
```
cd container
bash build.sh
```

Then we submit a job using slurm template
```
#SBATCH --job-name={JOB_NAME}
#SBATCH --error=log/task.out.%j  # STDOUT output is written in slurm.out.JOBID
#SBATCH --output=log/task.out.%j # STDOUT error is written in slurm.err.JOBID
#SBATCH --nodes=1
#SBATCH --partition={PARTITION}
#SBATCH --qos={QOS}
#SBATCH --gres=gpu:4
#SBATCH --mem=80GB
#SBATCH --time=4-00:00:00

ulimit -u

export RAY_TMPDIR=/tmp/ray  # Ensure ray has a clean place for sockets
export OMP_NUM_THREADS=1

apptainer exec --nv --bind /dev/shm:/dev/shm --writable-tmpfs ./container/custom_img.sif /bin/bash -c \
    "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:\${LD_LIBRARY_PATH} \
    && nvidia-smi && pip install numpy==2.2.6 \
    && ./jobs/{JOB_SCRIPT}"
```

## Evaluation

Please refer to `train-time/evaluation` directory.