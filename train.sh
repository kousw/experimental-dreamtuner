# prepare with `accelerate config`

accelerate launch train.py --pretrained_model_name_or_path=./models/stable-diffusion-v1-5 \
  --controlnet_model_name_or_path=lllyasviel/sd-controlnet-depth \
  --train_data_dir=./datasets/large \
  --config=./models/stable-diffusion-v1-5/v1-inference.yaml \
  --validation_image datasets/sample/00008_depth.png \
  --validation_prompt "1girl" \
  --validation_reference_image datasets/sample/00008_rgb.png \
  --validation_steps=100 \
  --checkpointing_steps=1000 \
  --num_train_epochs=20 \
  --learning_rate=5e-5 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=6 \
  --enable_xformers_memory_efficient_attention

# if you want to use the pretrained subject encoder and unet, use this settings
#  --pretrained_subejct_encoder_model_name_or_path=./output/backup/0004/checkpoint-80epoch

# if you want to initialize the subject encoder by ip-adapter, download ip-adapter_sd15.bin and use this settings
# --pretrained_ip_adapter_model_name_or_path=./models/IP-Adapter/models/ip-adapter_sd15.bin 