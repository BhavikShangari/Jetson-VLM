# Jetson-VLM: An Open-Source VLM for Edge Applications
<hr style="border: 2px solid gray;"></hr>

This Repository Implements [Prsimatic VLM](https://arxiv.org/abs/2402.07865) that could be deployed onto Low Powered Devices such as Jetson Nano using Optimized Models.

<hr style="border: 2px solid gray;"></hr>

# Model Architecture
Model comprises of DINO V2 Base (224px), SIGLIP Base(224px) as Image Encoders and Llama 3.2:1B as Language Model.

<p align="center">
  <img src="extras/Model.png" alt="Description of image" width="600">
</p>

<hr style="border: 2px solid gray;"></hr>

# Installation

```
git clone https://github.com/BhavikShangari/Jetson-VLM.git
cd Jetson-VLM
conda env create --name jetson_vlm --file environment.yml
```

<hr style="border: 2px solid gray;"></hr>

# Dataset Downloading

### Llava v1.5 595K mixture
For loading dataset to train Your Model, we have Modified Llava v1.5 595K Mixture Dataset and performed text formatting over it, and created a csv file to make it easy for Loading.

Either Download CSV Manually from this [Link](https://drive.google.com/file/d/1yZagkp2xFmPd53Zo0FDPU-CNy8GmAyII/view?usp=sharing) or use

```
pip install gdown
gdown 1yZagkp2xFmPd53Zo0FDPU-CNy8GmAyII
```

Also Download Images.zip Manually [here]() or
```
gdown 1MsjR_tfk2YHRwLTX1tLOzGc7r8JQdOfi
unzip Images.zip
```
<hr style="border: 2px solid gray;"></hr>

# Training

If starting from a Checkpoint


```
python3 train.py --model_path path/to/checkpoint.pt --per_device_batch_size 32 --learning_rate 2e-5 --output_dir ./results --epochs 10 --torch_compile True --save_strategy no --report_to wandb --lr_scheduler cosine --warmup_ratio 0.10 --logging_steps 100 --dataset_path data.csv --save_file_name path/to/model.pt
```
else

```
python3 train.py --per_device_batch_size 32 --learning_rate 2e-5 --output_dir ./results --epochs 10 --torch_compile True --save_strategy no --report_to wandb --lr_scheduler cosine --warmup_ratio 0.10 --logging_steps 100 --dataset_path data.csv --save_file_name path/to/model.pt
```
<hr style="border: 2px solid gray;"></hr>

# Checkpoints

Pre Trained Checkpoints are available:
### Vision Language Aligned Models
<table>
<tr>
<th>Checkpoint Name</th>
<th>Model Checkpoint</th>
</tr>

<tr>
<td>Pretrained Llama 3.2:1B + DINOV2 BASE (224px) + SIGLIP BASE (224px) (2 Epochs)</td>
<td><center><a href="https://drive.google.com/file/d/1l9zjOr37XtJiTeWs-sJa6dBHj_4ZbD-K/view?usp=sharing">Link</a></center></td>
</tr>

<tr>
<td>Instruct Llama 3.2:1B + DINOV2 BASE (224px) + SIGLIP BASE (224px) (2 Epochs)</td>
<td><center><a href="https://drive.google.com/file/d/1qGPdM_Bq_rB9f6dwbLph6tEvoAsSSg1j/view?usp=sharing">Link</a></center></td>
</tr>

<tr>
<td>Instruct Llama 3.2:1B + DINOV2 BASE (224px) + SIGLIP BASE (224px) (6 Epochs)</td>
<td><center><a href="https://drive.google.com/file/d/14KM7lqMrAB02yZB_kF2-hwoeZhuTKtHr/view?usp=sharing">Link</a></center></td>
</tr>
</table>

### Multimodal Instruction Tuned Models
Coming soon
### 
<hr style="border: 2px solid gray;"></hr>

# Generation
For Generation Download the Checkpoints and place in the Checkpoints Folder

```
cd Checkpoints
gdown {Checkpoint}
cd ..
python3 generate.py --model_path Checkpoints/{MODEL}.pt --image_path Path/to/image.png --prompt 'Explain what this image depicts' --device cuda:0
```
<hr style="border: 2px solid gray;"></hr>

# Deployment on Jetson Nano
Coming Soon...