import subprocess

for model in ["mobilenet", "resnet", "efficientnet"]:
    print(f"🔁 Trening: {model}")
    subprocess.run(["python3", "train_model.py", "--model", model])