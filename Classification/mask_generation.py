
import subprocess
import json
def json_read(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

unlearn_path = "unlearned_models"
model_path = "0model_SA_best.pth.tar"
mask_path = "masks"
##create folder masks
#subprocess.run(f"mkdir {mask_path}", shell=True, check=True)
#subprocess.run(f"mkdir {unlearn_path}", shell=True, check=True)
json_file_path = 'indexes_to_replace42.json'
tenXCifar10 = 4500
forgetting_data_amount = [tenXCifar10]

indexes = json_read(json_file_path)
indexes = json.dumps(indexes)
mask_path = "masks"
command = f"python generate_mask.py --model_path {model_path} --save_dir {mask_path} --indexes_to_replace '{indexes}' --random_prune"
subprocess.run(command, shell=True, check=True)