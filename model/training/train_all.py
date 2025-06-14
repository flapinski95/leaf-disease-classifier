import subprocess

models = ["mobilenet", "resnet", "efficientnet"]
processes = []

for model in models:
    print(f"Trening: {model}")
    p = subprocess.Popen(
        ["python", "train_model.py", "--model", model]
    )
    processes.append(p)

for p in processes:
    p.wait()

print("✅ Wszystkie modele wytrenowane!")