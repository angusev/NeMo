# TODO(oktai15): remove +model.text_tokenizer.add_blank_at=true when we update FastPitch checkpoint
cat ./9017_5_mins/manifest.json | tail -n 10 > ./9017_manifest_dev_ns_all_local.json
cat ./9017_5_mins/manifest.json | head -n -10 > ./9017_manifest_train_dur_5_mins_local.json
ln -sf ./9017_5_mins/audio audio

python fastpitch_finetune.py --config-name=fastpitch_align_v1.05.yaml \
  train_dataset=./9017_manifest_train_dur_5_mins_local.json \
  validation_datasets=./9017_manifest_dev_ns_all_local.json \
  sup_data_path=./fastpitch_sup_data \
  phoneme_dict_path=tts_dataset_files/cmudict-0.7b_nv22.10 \
  heteronyms_path=tts_dataset_files/heteronyms-052722 \
  exp_manager.exp_dir=./studio_pete_exp \
  +init_from_nemo_model=./tts_en_fastpitch_align.nemo \
  +trainer.max_steps=10000000 ~trainer.max_epochs \
  trainer.check_val_every_n_epoch=100 \
  model.train_ds.dataloader_params.batch_size=12 model.validation_ds.dataloader_params.batch_size=24 \
  model.n_speakers=1 model.pitch_mean=116.7689 model.pitch_std=21.1965 \
  model.pitch_fmin=65.4063 model.pitch_fmax=694.4336 model.optim.lr=2e-4 \
  ~model.optim.sched model.optim.name=adam trainer.devices=1 trainer.strategy=null \
  +model.text_tokenizer.add_blank_at=true