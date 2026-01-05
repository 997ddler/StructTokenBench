tokenizer=WrappedESM3Tokenizer
tokenizername=esm3
d_model=128
lr=0.001
EXTRA_MODEL_ARGS=""


SHARED_ARGS="tokenizer=$tokenizer model.d_model=$d_model trainer.devices=[0] optimization.optimizer.lr=$lr data.target_field=$target_field experiment_name=${experiment_prefix}_${tokenizername}_lr${lr} run_name=tryout_test default_data_dir=$DIR/struct_token_bench_release_data/ data.pdb_data_dir=$DIR/pdb_data/mmcif_files/ trainer.default_root_dir=$DIR/struct_token_bench_logs/ ${EXTRA_TASK_ARGS} ${EXTRA_MODEL_ARGS}"


# CASP14

target_field=null task_goal="codebook_utilization" experiment_prefix="${task_goal}_casp14"
EXTRA_TASK_ARGS="test_only=true model.task_goal=${task_goal} experiment_name=${experiment_prefix}_${tokenizername} optimization.micro_batch_size=8"

CUDA_VISIBLE_DEVICES=0 python ./src/script/run_supervised_task.py --config-name=casp14.yaml  $SHARED_ARGS

# CAMEO
target_field=null task_goal="codebook_utilization" experiment_prefix="${task_goal}_cameo"
EXTRA_TASK_ARGS="test_only=true model.task_goal=${task_goal} experiment_name=${experiment_prefix}_${tokenizername} optimization.micro_batch_size=8"

CUDA_VISIBLE_DEVICES=0 python ./src/script/run_supervised_task.py --config-name=cameo.yaml  $SHARED_ARGS