#!/bin/bash
set -x


# MEM opt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

ENV="dlc"

MEGATRON_PATH="/cpfs/shared/h800-queue-1/donny/Megatron-LM"
CODE_ROOT=${MEGATRON_PATH}

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
if false; then
    GPUS_PER_NODE=4  # GLOBAL_WORLD_SIZE=300
else
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"
#--override-opt_param-scheduler \
#--fp8-format hybrid \
#--fp8-amax-compute-algo max \
#--fp8-amax-history-len 1024 \
#--transformer-impl transformer_engine \
custom_options="--disable-bias-linear \
                --swiglu \
                --untie-embeddings-and-output-weights \
                --position-embedding-type rope \
                --init-method-std 0.02 \
                --disable-scaled-init-method \
                --normalization LayerNorm \
                --apply-layernorm-1p \
                --norm-epsilon 1e-5 \
                --adam-eps 1e-8 \
                --num-query-groups 8 \
                --clone-scatter-output-in-embedding \
                --num-layers-per-virtual-pipeline-stage 3 \
                --make-vocab-size-divisible-by 32 \
                --rotary-base 500000 \
                "
NUM_LAYERS=48
HIDDEN_SIZE=8192
INTERMIDIATE_SIZE=22016
NUM_ATTN_HEADS=64
# NUM_KEY_VALUE_HEADS=8
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2048
LR=5e-5
MIN_LR=5e-6
SEQ_LEN=4096
PR=bf16
TP=2
PP=4
AC="none"
DO=true
FL=true
SP=true
SAVE_INTERVAL=1000

DATA_ROOT="/cpfs/shared/h800-queue-1/liushan/data"
TOKENIZER_NAME_OR_PATH="${DATA_ROOT}/tokenizer_v3/tokenizer.model"

DATASET_PATH=" \
85  ${DATA_ROOT}/data_0522/inf_github_code_0522   \
3  ${DATA_ROOT}/data_0522/algebraic_stack  \
13  ${DATA_ROOT}/data_0522/open_web_math_0522  \
2  ${DATA_ROOT}/data_0522/math-like-v2_0701  \
2.3  ${DATA_ROOT}/data_0522/sft_mix  \
1  ${DATA_ROOT}/data_0522/stackmathqa  \
3.2  ${DATA_ROOT}/data_0522/instruct_323  \
0.72  ${DATA_ROOT}/data_0108/webtext-homework_study_0108  \
1.15  ${DATA_ROOT}/data_0522/math_instructions_0615  \
1  ${DATA_ROOT}/data_0522/code_instructions_0701   \
1.08  ${DATA_ROOT}/data_0522/openhermes  \
3  ${DATA_ROOT}/data_0522/nl_instructions_0522  \
0.48  ${DATA_ROOT}/data_0522/flanop2  \
3  ${DATA_ROOT}/data_0522/exam \
0.27  ${DATA_ROOT}/data_0522/jinnai_code_excercise_0615  \
0.55  ${DATA_ROOT}/data_0522/kp_refine_0701  \
2.88  ${DATA_ROOT}/data_0522/tiku_cc_0701  \
2.25  ${DATA_ROOT}/data_0522/mammoth10m    \
26.72  ${DATA_ROOT}/med_fp/med_pretrain_0809  \
1.04  ${DATA_ROOT}/med_fp/med_sft_0809       \
12.5  ${DATA_ROOT}/data_0522/text_book_select  \
10  ${DATA_ROOT}/data_0522/ultra_textbooks   \
0.96  ${DATA_ROOT}/data_0522/tiku_book  \
2.34  ${DATA_ROOT}/data_0522/paper-redpajama_arxiv_0108  \
13  ${DATA_ROOT}/data_0522/paper-cnki_0108  \
15  ${DATA_ROOT}/data_0522/paper-nssd_0108  \
9.04  ${DATA_ROOT}/data_0522/baidu_baike_0522  \
2.476  ${DATA_ROOT}/data_0522/wiki-wiki-20230401_en_0108  \
0.436  ${DATA_ROOT}/data_0522/wiki-wiki_zh_0108  \
0.535  ${DATA_ROOT}/data_0522/wiki-wikipedia_20220301_en_0108  \
4.86  ${DATA_ROOT}/data_0522/qa_mix   \
21.8  ${DATA_ROOT}/data_0522/cosmos_0108  \
2.81  ${DATA_ROOT}/data_0522/webtext-zhihu_article_kl_0522  \
10  ${DATA_ROOT}/data_0410_diff_2/webtext-cc_wet_zh_0108  \
20  ${DATA_ROOT}/data_0108/webtext-weixin_page_0108  \
3.21  ${DATA_ROOT}/data_0522/wiki-like_0701     \
5.17  ${DATA_ROOT}/data_0522/wiki-like-url_0701   \
75  ${DATA_ROOT}/data_0522/fineweb_edu_0701  \
10  ${DATA_ROOT}/data_0108/webtext-skypile-150b_0108  \
"

PRETRAIN_CHECKPOINT_PATH="none"
TRAIN_TOKENS=372940000000
# 3322204619268  # 3322.204619268B tokens
LR_DECAY_TOKENS=${TRAIN_TOKENS} # 3400000000000 # 3.4T tokens
WARMUP_TOKENS=$(( 1000 * ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} ))

OUTPUT_BASEPATH="/cpfs/shared/h800-queue-1/donny/34b/med0809"
if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
        --load $PRETRAIN_CHECKPOINT_PATH"
fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
        --recompute-method block \
        --recompute-num-layers 4 \
        --recompute-select-layers qkv"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
        --fp16 \
        --initial-loss-scale 65536"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
fi

if [ $DO = true ]; then
    do_options=" \
        --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
        --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
        --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${LR_DECAY_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME="xx"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --load /cpfs/shared/h800-queue-1/donny/checkpoints/34b_v0411_8w \
        --load-from-non-pipeline-model \
        --finetune \
        --split 100,0,0 \
        --data-path ${DATASET_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 0.5 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMIDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --log-interval 1 \
        --eval-interval ${SAVE_INTERVAL} \
        --eval-iters 50 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-workers 4 \
        --seed 42 \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_NAME_OR_PATH} \
        "

cd ${MEGATRON_PATH}

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${CODE_ROOT}/pretrain_gpt.py \
         ${megatron_options} \
         ${activation_checkpoint_options} \
         ${do_options} \
         ${pr_options} \
         ${sp_options} \
         ${flash_options} \
         ${load_options} \
         ${custom_options} \
         "

echo ${run_cmd}
eval ${run_cmd}
