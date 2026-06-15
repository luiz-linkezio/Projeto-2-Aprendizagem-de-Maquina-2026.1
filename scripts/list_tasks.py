import openml

suite = openml.study.get_suite("tabarena-v0.1")

for task_id in suite.tasks:
    task = openml.tasks.get_task(task_id)
    ds = task.get_dataset()
    n = int(ds.qualities["NumberOfInstances"])
    n_feat = int(ds.qualities["NumberOfFeatures"])
    n_classes = int(ds.qualities["NumberOfClasses"])
    regime = "pequeno" if n < 1000 else ("medio" if n <= 10000 else "grande")
    tipo = "binario" if n_classes == 2 else "multiclasse"
    print(f"{task_id} | {ds.name} | n={n} | feat={n_feat} | classes={n_classes} | {regime} | {tipo}")
    print(f"Task type: {task.task_type}")
