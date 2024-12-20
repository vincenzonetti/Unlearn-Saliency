
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
json_file_path = 'indexes_to_replace.json'
tenXCifar10 = 4500
forgetting_data_amount = [tenXCifar10] 

#for i in range (0,10):
#    print("class:",i)
#    command = f"python main_forget.py --unlearn GA --unlearn_epochs 10 --unlearn_lr 0.00013 --class_to_replace {i} --model_path {model_path} --save_dir {unlearn_path} --mask_path masks/cifar10_class{i}_mask_0.8.pt --random_prune"
#    subprocess.run(command, shell=True, check=True)
#
#for i in range (0,10):
#    print("class:",i)
#    command = f"python main_forget.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.0013 --class_to_replace {i} --model_path {model_path} --save_dir {unlearn_path} --mask_path masks/cifar10_class{i}_mask_0.8.pt --random_prune"
#    subprocess.run(command, shell=True, check=True)


for amount in forgetting_data_amount:
    indexes = json_read(json_file_path)
    #print(f"Replacing indexes {indexes}")
    indexes = json.dumps(indexes)
    #command = f"python main_forget.py --unlearn GA --unlearn_epochs 10 --unlearn_lr 0.00013 --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune"
    #subprocess.run(command, shell=True, check=True)
    command = f"python main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.13 --num_indexes_to_replace {amount} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune --mask_path masks/cifar10_amount_forget{amount}_mask_0.5.pt"
    subprocess.run(command, shell=True, check=True)
    
    command = f"python main_forget.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.13 --num_indexes_to_replace {amount} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune"
    subprocess.run(command, shell=True, check=True)
    #command = f"python main_random.py --unlearn GA --unlearn_epochs 10 --unlearn_lr 0.00013 --num_indexes_to_replace {amount} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune --mask_path masks/cifar10_amount_forget{amount}_mask_0.5.pt"
    #subprocess.run(command, shell=True, check=True)
    #command = f"python main_random.py --unlearn GA --unlearn_epochs 10 --unlearn_lr 0.0013 --num_indexes_to_replace {amount} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune --mask_path masks/cifar10_amount_forget{amount}_mask_0.5.pt"
    #subprocess.run(command, shell=True, check=True)
    
    #command = f"python main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr 0.013 --num_indexes_to_replace {amount} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune --mask_path masks/cifar10_amount_forget{amount}_mask_0.5.pt"
    #subprocess.run(command, shell=True, check=True)
    #command = f"python main_random.py --unlearn retrain --unlearn_epochs 10 --unlearn_lr 0.0013 --num_indexes_to_replace {amount} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune --mask_path masks/cifar10_amount_forget{amount}_mask_0.5.pt"
    #subprocess.run(command, shell=True, check=True)
        



        