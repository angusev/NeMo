python examples/tts/hifigan_finetune.py \
    --config-name=hifigan.yaml \
    model.train_ds.dataloader_params.batch_size=32 \
    model.max_steps=1000000 \
    model.optim.lr=0.00001 \
    ~model.optim.sched \
    train_dataset=./hifigan_train_ft.json \
    validation_datasets=./hifigan_val_ft.json \
    exp_manager.exp_dir=hifigan_ft \
    +init_from_pretrained_model=tts_en_hifigan \
    trainer.check_val_every_n_epoch=100 \
    model/train_ds=train_ds_finetune \
    model/validation_ds=val_ds_finetune
