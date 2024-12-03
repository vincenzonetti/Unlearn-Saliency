import subprocess

unlearn_path = "unlearned_models"
model_path = "0model_SA_best.pth.tar"
mask_path = "masks"

json_file_path = 'indexes_to_replace42.json'
tenXCifar10 = 4500
forgetting_data_amount = [tenXCifar10, 2*tenXCifar10, 5*tenXCifar10]
mask_path = "masks"
import numpy as np

def generate_random_indexes(n_samples=4500, max_val=45000, seed=42):
    """
    Generate random unique integers and save to JSON.
    
    Args:
        n_samples: Number of samples to generate
        max_val: Maximum value (exclusive)
        seed: Random seed for reproducibility
    """
    # Set random seed
    rng = np.random.RandomState(seed)
    
    # Generate unique random numbers
    numbers = rng.choice(max_val, size=n_samples, replace=False)
    numbers = sorted(numbers.tolist())  # Convert to sorted list
    print(f"Are all numbers unique? {len(set(numbers)) == len(numbers)}")
    


for amount in forgetting_data_amount:
    indexes = generate_random_indexes(n_samples=amount, max_val=45000, seed=int(amount/4500))
    command = f"python generate_mask.py --model_path {model_path} --save_dir {mask_path} --indexes_to_replace '{indexes}' --random_prune"
    for lr in [0.0001, 0.001, 0.01, 0.1]:
        command = f"python main_forget.py --unlearn GA --unlearn_epochs 10 --unlearn_lr {lr} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune"
        subprocess.run(command, shell=True, check=True)
        command = f"python main_random.py --unlearn RL --unlearn_epochs 10 --unlearn_lr {lr} --num_indexes_to_replace {amount} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune --mask_path masks/cifar10_amount_forget{amount}_mask_0.5.pt"
        subprocess.run(command, shell=True, check=True)

        command = f"python main_forget.py --unlearn RL --unlearn_epochs 10 --unlearn_lr {lr} --num_indexes_to_replace {amount} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune"
        subprocess.run(command, shell=True, check=True)
        command = f"python main_random.py --unlearn GA --unlearn_epochs 10 --unlearn_lr {lr} --num_indexes_to_replace {amount} --model_path {model_path} --save_dir {unlearn_path} --indexes_to_replace '{indexes}' --random_prune --mask_path masks/cifar10_amount_forget{amount}_mask_0.5.pt"
        subprocess.run(command, shell=True, check=True)

        



        