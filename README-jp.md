## About

DreamTuner(https://arxiv.org/abs/2312.13691) の試験的な実装を行いました。
**これは非公式な実装**であり、実装が正しいかは不明ですが、一定の効果は確認できました。

この実装では２ステップのトレーニングを行っています。
1. Subject Encoder Training
2. Subject Driven Fine-tuning

1.Subject Encoder Trainigでは 、リファレンス画像の特徴抽出を行う Subject EncoderとUnetのTransformer BlockのSelf-Attention層とCross-Attention層の間に Self-Encoder-Attentionを追加し、トレーニングを行います。(train.py)
Subject Encoderのみ推論は、inference.py を使用して行うことができます。

2.Subject Driven Fine-tuningは基本的には、Dreambooth + Textual Inversionによる微調整です。
この実装の前に、Self-Subject AttentionとReference用のunetおよび、CFGに代わるReference Gaidanceの実装が大きな効果を持つため、その効果を確認する必要があります。
Self-Subject Drivenな推論は、inference_ss.py を使用して行うことができます。

生成結果が問題なければ、Subject Driven Fine-tuningを行います。（train_dreambooth.py）
出力されるSubject Encoder, Unet, Text Inversionのモデルを使用し、inference_ss.py で推論を行うことができます。

今のところControlNetを使用した Subject Driven Inferenceには対応していません。

【注意事項】
いくつか論文の内容からは分からなかった部分があるため、独自の実装を行っています。
特に Subject Encoderは顕著な差が見られなかったので、SSR-Encoder(https://arxiv.org/abs/2312.16272)の実装に置き換えています

トレーニングはアニメモデルをベースに、5万枚ほどの画像で行いました。
論文で説明されていた通りのある程度の効果は見込まれたものの、データセットの質や量、GPUの制限による高めのラーニングレートを選択したこともあり、プロンプトでの制御の効きがよくない結果となりました。
実験する場合には、Subject Encoder betaの値 や Reference Guidance Scaleなどの値を上手く調整する必要があります。

GPU処理時間やデータセットの質や量の問題から、個人でのこれ以上のトレーニングは難しいと判断しました。
開発を継続するつもりはありませんが、公開しておきます。

訓練済みSubject Encoderは[こちら](https://huggingface.co/kousw/subject-encoder-sd15)からダウンロードすることができます。
models/dreamtuner配下のPretrainedなモデルを置き換えて使用してください。

## Environment

*Python* 3.10.9
*CUDA* 12.2

## Installation

```
python -m venv venv
source venv/bin/activate
```

```
pip install -r requirements.txt
```


### ベースモデルダウンロード

```
mkdir datasets
mkdir models

cd ./models
git clone some_diffusers_sd_15_model_from_huggin_face
# ex) git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 or some_anime_model

# copy base model for finetuning
cp -r some_diffusers_sd_15_model_from_huggin_face/ dreamtuner
```

### データセットの用意

```
# your datasets into ./datasets 
```

背景切り抜き、マスク、深度の画像を作成
```
python process.py path
```

タグを作成
```
python process_tags.py path
```


## Training

with accelerater
```
# configulation
accelerate config

# training 
bash ./train.sh
```

トレーニングが完了したら、生成したファイル（subject_encoder, unet）をdreamtunerフォルダにコピーしてください。

読み込み時にsubject encoder と unetのモジュールを置き換えるために、model_index.jsonを以下のように上書きする必要があります。
また、DDIM以外のスケジューラで動作しないため、スケジューラが異なる場合には修正してください。

```
{
    "_class_name": "DreamTunerPipelineSelfSubject",
    "_diffusers_version": "0.8.0.dev0",
    "feature_extractor": [
      "transformers",
      "CLIPImageProcessor"
    ],
    "safety_checker": [
      "stable_diffusion",
      "StableDiffusionSafetyChecker"
    ],
    "scheduler": [
      "diffusers",
      "DDIMScheduler"
    ],
    "text_encoder": [
      "transformers",
      "CLIPTextModel"
    ],
    "tokenizer": [
      "transformers",
      "CLIPTokenizer"
    ],
    "subject_encoder": [
      "dreamtuner.models.subject_encoder",
      "SubjectEncoder"
    ],
    "unet": [
      "dreamtuner.models.unet",
      "SDUNet2DConditionModel"
    ],
    "vae": [
      "diffusers",
      "AutoencoderKL"
    ],
    "unet_reference": [
      "dreamtuner.models.unet",
      "SDUNet2DConditionModel"
    ]
}
```

### train dreambooth

```
bash ./train_dreambooth.sh
```

## Inference

### Subject Encoder inference

```
python main.py --model_name_or_path ./models/dreamtuner --subject_encoder_beta 0.5 --num_samples 1 --num_inference_steps 50 --reference_image ./datasets/sample/00006_rgb.png --negative_prompt "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting" --prompt "best quality,1girl,outdoor"
```

### Subject Driven inference

witout mask, without reference guidance (self-subject attention only)

```
python inference_ss.py --model_name_or_path ./models/dreamtuner --subject_encoder_beta 0.2 --num_samples 1 --num_inference_steps 50 --reference_image ./datasets/sample/00006_rgb.png --negative_prompt "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting" --prompt "best quality,1girl,outdoor" --enable_reference_guidance --reference_guidance_scale 2  --dtype float16
```

with mask, with reference guidance
```
python inference_ss.py --model_name_or_path ./models/dreamtuner --subject_encoder_beta 0.2 --num_samples 1 --num_inference_steps 50 --reference_image ./datasets/sample/00006_rgb.png --mask_image ./datasets/sample/00006_mask.png --negative_prompt "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting" --prompt "best quality,1girl,outdoor" --enable_reference_guidance --reference_guidance_scale 2  --dtype float16
```


with textual invertion
```
python inference_ss.py --model_name_or_path ./models/dreamtuner_tuned --text_embeds_name_or_path ./models/dreamtuner_tuned/textual_inversion --subject_encoder_beta 0.2 --num_samples 1 --num_inference_steps 50 --reference_image ./datasets/sample/00009_rgb.png --mask_image ./datasets/sample/00009_mask.png --negative_prompt "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting" --prompt "best quality,1girl,outdoor" --enable_reference_guidance --reference_guidance_scale 2  --dtype float16
```