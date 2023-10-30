import itertools as it
import uuid
import numpy as np
import pandas as pd
import json
from jinja2 import Template
import os

name = "stability_experiment_bgnmf"

# Create and write the experiment json array file
K_list = np.arange(2,21)

modelseed_list = [413, 617, 781]
data_path = ["data/methylation/sarcoma/data_top5k.npz"]
# model_list = ['ggg',]
# lam_list = [275,300,400,500,600,750,1000]

params = it.product(K_list, modelseed_list, data_path)
# params = it.filterfalse(lambda x: x[0] > x[1], params )
params_df = pd.DataFrame(params, columns=['K','seed', 'data_path'])

# params_df['data_path'] = 'data/methylation/sarcoma/data_top5k.npz'
# params_df['output_path'] = '/work/tmp'
params_df['output_path'] = '/work/data/tmp'


params_df['uuid'] = [uuid.uuid4().hex for _ in range(len(params_df.index))]

os.makedirs(f"results/{name}", exist_ok=True)
experiment_config_json_path = f"results/{name}/{name}_config.json"
params_df.to_json(experiment_config_json_path, orient='records')


# Create a write the submission script for the experiment
with open('src/scripts/submit_stability_experiment_bgnmf.j2', 'r') as f:
    template_string = f.read()

# Create a Jinja2 template object
template = Template(template_string)

# Render the template with the provided values
num_nodes = 1
num_tasks_per_node = 10
array_range = len(params_df)

rendered_template = template.render(
    experiment_name=name,
    num_nodes=num_nodes,
    num_tasks_per_node=num_tasks_per_node,
    array_range=array_range,
    experiment_config_json=experiment_config_json_path
)

with open(f"results/{name}/submit_{name}.sh",'w') as f:
    f.write(rendered_template)

