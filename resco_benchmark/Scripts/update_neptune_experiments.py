import neptune.new as neptune
import re
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass


ap = argparse.ArgumentParser()
ap.add_argument("--experiment_id",
                type=str,
                default="MAL-1179",
                required=True)
args = ap.parse_args()

project = neptune.init_project(
    mode="read-only",
)
neptune_experiments_names = project.fetch_runs_table(
    columns=["sys/id"], state="inactive").to_pandas()
list_of_neptune_experiments_names = neptune_experiments_names["sys/id"].to_list()
# Descending sorting of experiments
list_of_neptune_experiments_names = sorted(list_of_neptune_experiments_names, 
                                           key=lambda s: int(s.split('-')[-1]),
                                           reverse=True)
list_of_neptune_experiments_names = list_of_neptune_experiments_names[
    :list_of_neptune_experiments_names.index(args.experiment_id)
]


pattern = r"Reward:\s*(\w+),"

for experiment_id in list_of_neptune_experiments_names:
    run = neptune.init_run(with_id=experiment_id)

    text = run["sys/tags"].fetch()
    text = ", ".join(text)
    match = re.search(pattern, text)
    result = match.group(1)
    run["parameters/reward"] = result
