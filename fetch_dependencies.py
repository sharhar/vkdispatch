import subprocess
import os
import urllib.request
import tarfile

def clone_and_checkout(repo_url, commit_hash, output_dir):
    """
    Clones the given git repository into the specified output directory (creates it if not existent),
    and checks out the specified commit.

    Args:
    repo_url (str): URL of the git repository.
    commit_hash (str): The commit hash to check out.
    output_dir (str): Local path for the repository to be cloned into.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if the directory is empty (to decide whether to clone or not)
    if not os.listdir(output_dir):
        # Clone the repository into the output directory
        print(f"Cloning {repo_url} into {output_dir}")
        subprocess.run(["git", "clone", repo_url, "."], cwd=output_dir, check=True)
    else:
        print(f"Directory {output_dir} already exists and is not empty.")

        subprocess.run(["git", "fetch", "--all"], cwd=output_dir, check=True)

    # Checkout the specified commit
    print(f"Checking out commit {commit_hash}")
    subprocess.run(["git", "checkout", commit_hash], cwd=output_dir, check=True)

dependencies = [
    ("https://github.com/DTolm/VkFFT.git", "066a17c17068c0f11c9298d848c2976c71fad1c1", "deps/VkFFT"),
    ("https://github.com/sharhar/VKL.git", "cb367ff79845d0866c753f330440394ae7392405", "deps/VKL"),
    ("https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git", "5677097bafb8477097c6e3354ce68b7a44fd01a4", "deps/VMA"),
    ("https://github.com/KhronosGroup/Vulkan-Headers.git", "eaa319dade959cb61ed2229c8ea42e307cc8f8b3", "deps/Vulkan-Headers"),
    ("https://github.com/KhronosGroup/Vulkan-Utility-Libraries.git", "ad7f699a7b2b5deb66eb3de19f24aa33597ed65b", "deps/Vulkan-Utility-Libraries"),
    ("https://github.com/KhronosGroup/glslang.git", "e8dd0b6903b34f1879520b444634c75ea2deedf5", "deps/glslang"),
    ("https://github.com/zeux/volk.git", "3a8068a57417940cf2bf9d837a7bb60d015ca2f1", "deps/volk/volk")
]

for dep in dependencies:
    clone_and_checkout(*dep)

os.makedirs("deps/MoltenVK", exist_ok=True)

molten_vk_url = "https://github.com/KhronosGroup/MoltenVK/releases/download/v1.2.8/MoltenVK-macos.tar"
molten_vk_path = "deps/MoltenVK"
molten_vk_filename = "MoltenVK-macos.tar"
molten_vk_full_file_path = os.path.join(molten_vk_path, molten_vk_filename)

try:
    print("Downloading MoltenVK...")
    urllib.request.urlretrieve(molten_vk_url, molten_vk_full_file_path)
    print(f"File downloaded: {molten_vk_full_file_path}")

    with tarfile.open(molten_vk_full_file_path) as tar:
            tar.extractall(path=molten_vk_path)
            print(f"Files extracted to: {molten_vk_path}")
except Exception as e:
    print(f"An error occurred: {e}")

