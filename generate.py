from Models.model import VQAModel, VQAProcessor
import torch
import argparse
from PIL import Image
from transformers import GenerationConfig

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--image_path', type=str, required=True)
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--device', type=str, required=True, default='cpu')

args = parser.parse_args()

model = VQAModel()
model.load_state_dict(torch.load(args.model_path))
model.to(args.device)

processor = VQAProcessor()

image = Image.open(args.image_path)
image.show()

base_prompt = '''<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant assisting in task specified by user based on the Image provided in context<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBased on Image answer: {user_prompt}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''

gen_config = GenerationConfig(max_new_tokens=30, use_cache=False)
processed = processor(images=image, text = base_prompt.format(user_prompt=args.prompt)).to(args.device)
initial_length = len(processed['input_ids'][0, :])
generation = model.generate(input_ids=processed['input_ids'], images=processed['image_features'].unsqueeze(0), gen_config=gen_config, tokenizer=processor.tokenizer)

print(processor.tokenizer.batch_decode(generation[:, initial_length:][0], skip_special_tokens=True))




