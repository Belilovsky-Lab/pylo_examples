import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download, login
from tqdm import tqdm

def download_dataset(output_dir, test_mode=False, repo_id=None):
    """
    Download the edufineweb-tokenized dataset from Hugging Face Hub
    
    Args:
        output_dir: Directory to save the dataset files
        test_mode: If True, only download a few files for testing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Login to Hugging Face (will prompt for token if not already logged in)
    login()
    
    
    if test_mode:
        # In test mode, download only one file from each split
        print("Test mode: Downloading only one file per split")
        files_to_download = [
            "edufineweb_train_0.npy",
            "edufineweb_val_0.npy"
        ]
        
        for filename in tqdm(files_to_download, desc="Downloading files", unit="file"):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=output_dir
                )
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
    else:
        # Download the entire dataset
        print(f"Downloading the entire dataset from {repo_id}")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded the entire dataset to {output_dir}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download edufineweb-tokenized dataset from Hugging Face Hub')
    parser.add_argument('--output_dir', type=str, default="/home/mila/b/benjamin.therien/scratch/fw_upload",
                        help='Directory to save the dataset files')
    parser.add_argument('--repo_id', type=str, default="btherien/edufineweb-tokenized",
                        help='dataset to download')
    parser.add_argument('--test', action='store_true', help='Download only one file per split for testing')
    args = parser.parse_args()
    
    download_dataset(args.output_dir, args.test, args.repo_id)
