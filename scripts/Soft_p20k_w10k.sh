# AminoAseed
use_linear_project=false
freeze_codebook=false
model_name="Soft_p20k_w10k"
  

# shared command
warmup_step=5426
total_step=108530
lr=0.0001
fast_dev=false # enable to debug with 500 samples

CUDA_VISIBLE_DEVICES=4,5,6,7 python ./src/script/run_pretraining_vqvae.py --config-name=soft_p20k_w10k.yaml \
    tokenizer=WrappedESM3Tokenizer \
    trainer.devices=[0,1,2,3] \
    optimization.micro_batch_size=8 \
    optimization.scheduler.num_warmup_steps=${warmup_step} \
    max_steps=${total_step} \
    optimization.optimizer.lr=$lr \
    optimization.scheduler.plateau_ratio=0.0 \
    lightning.callbacks.checkpoint.monitor="validation_bb_rmsd" \
    lightning.callbacks.checkpoint.mode="min" \
    lightning.callbacks.checkpoint.save_top_k=1 \
    trainer.log_every_n_steps=512 \
    data.fast_dev_run=${fast_dev} \
    data.data_version=mmcif_files_filtered_subsample10 \
    experiment_name=vqvae-pretrain-subsample10_${model_name}_fastdev${fast_dev} \
    run_name=test \
    model.quantizer.use_linear_project=${use_linear_project} \
    model.quantizer.freeze_codebook=${freeze_codebook} \
    model.ckpt_path=null \
    default_data_dir=$DIR/struct_token_bench_release_data/ \
    data.pdb_data_dir=$DIR/pdb_data/mmcif_files/ \
    trainer.default_root_dir=$DIR/struct_token_bench_logs/
