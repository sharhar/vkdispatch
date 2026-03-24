import subprocess
import os
import urllib.request
import tarfile
import sys

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
        try:
            subprocess.run(["git", "clone", repo_url, "."], cwd=output_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during git clone: {e}")
            sys.exit(1)
    else:
        print(f"Directory {output_dir} already exists and is not empty.")
        try:
            subprocess.run(["git", "fetch", "--all"], cwd=output_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during git fetch: {e}")
            sys.exit(1)

    # Checkout the specified commit
    print(f"Checking out commit {commit_hash}")
    try:
        subprocess.run(["git", "checkout", commit_hash], cwd=output_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during git checkout: {e}")
        sys.exit(1)

dependencies = [
    ("https://github.com/sharhar/VkFFT.git", "9cb0da8ab98f0f3a10debd1466f13cab65ce9bc3", "deps/VkFFT"), # my fork of VkFFT, will change to official repo once PR is merged
    ("https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git", "5677097bafb8477097c6e3354ce68b7a44fd01a4", "deps/VMA"),
    ("https://github.com/KhronosGroup/Vulkan-Headers.git", "eaa319dade959cb61ed2229c8ea42e307cc8f8b3", "deps/Vulkan-Headers"),
    ("https://github.com/KhronosGroup/Vulkan-Utility-Libraries.git", "ad7f699a7b2b5deb66eb3de19f24aa33597ed65b", "deps/Vulkan-Utility-Libraries"),
    ("https://github.com/KhronosGroup/glslang.git", "e8dd0b6903b34f1879520b444634c75ea2deedf5", "deps/glslang"),
    ("https://github.com/zeux/volk.git", "3a8068a57417940cf2bf9d837a7bb60d015ca2f1", "deps/volk/volk")
]

for dep in dependencies:
    clone_and_checkout(*dep)

if len(sys.argv) > 1 and sys.argv[1] == "--no-molten-vk":
    print("Skipping MoltenVK download.")
    sys.exit(0)

os.makedirs("deps/MoltenVK", exist_ok=True)

molten_vk_url = "https://release-assets.githubusercontent.com/github-production-release-asset/1189959592/f4463a14-40e8-45cc-8554-51e9605c43e5?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-03-24T05%3A30%3A25Z&rscd=attachment%3B+filename%3DMoltenVK-macos.tar&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-03-24T04%3A29%3A42Z&ske=2026-03-24T05%3A30%3A25Z&sks=b&skv=2018-11-09&sig=V1Yia4bQv95%2F%2FIi0mmWjLypLovSdfKx7FfxfsPh3n0w%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc3NDMyOTI4MCwibmJmIjoxNzc0MzI3NDgwLCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.LHUq45ielOiCY5ctrCMmX3w1GKsgL1vH6plB30MNYdo&response-content-disposition=attachment%3B%20filename%3DMoltenVK-macos.tar&response-content-type=application%2Foctet-stream"
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
