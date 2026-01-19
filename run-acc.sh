# bash run_acc.sh google/gemma-3-27b-it logs_run1 4
model=$1
log=$2
proc=$3
## lowercase the model name
log_dir="${model,,}"
# replace / with _
log_dir="${log_dir//\//_}"

log_dir="${log}/${log_dir}.log"

accelerate launch --multi_gpu --num_processes=$proc -m lighteval \
    accelerate \
    model_name=$model,batch_size=8 \
    'examples/tasks/Qimma_tasks.txt' \
    --output-dir evals \
    --load-tasks-multilingual \
    --save-details \
    > "$log_dir" 2>&1
    # --max-samples 1 \
