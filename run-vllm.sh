# bash run_vllm.sh google/gemma-3-27b-it logs_run1
model=$1
log=$2
## lowercase the model name
log_dir="${model,,}"
# replace / with _
log_dir="${log_dir//\//_}"

log_dir="${log}/r3_${log_dir}.log"


export VLLM_WORKER_MULTIPROC_METHOD=spawn


lighteval vllm \
    model_name=$model,tensor_parallel_size=2,gpu_memory_utilization=0.85,cache_dir=.cache_vllm \
     'examples/tasks/Qimma_tasks_QA.txt' \
    --output-dir evals \
    --load-tasks-multilingual \
    --save-details \
    > "$log_dir" 2>&1

    # --max-samples 1 \
    
