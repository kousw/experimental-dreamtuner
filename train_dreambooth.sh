# prepare with `accelerate config`

accelerate launch train_dreambooth.py --pretrained_model_name_or_path=./models/dreamtuner \
  --train_data_dir=./datasets/dreambooth \
  --config=./models/dreamtuner/v1-inference.yaml \
  --reference_image datasets/sample/00008_rgb.png \
  --prompt "1girl" \
  --subject_prompt "1girl of {}" \
  --placeholder_token "<sks-girl>" \
  --initializer_token "girl" \
  --num_ti_token_vectors 1 \
  --validation_steps=100 \
  --checkpointing_steps=100 \
  --num_train_epochs=200 \
  --learning_rate=1e-6 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --skip_generation \
  --use_8bit_adam \
  --mixed_precision=fp16 \
