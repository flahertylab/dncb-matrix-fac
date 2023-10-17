import itertools as it
import uuid
import pandas as pd
import json
from jinja2 import Template

name = "experiment_1"


# Create and write the experiment json array file
C_list = [6,8,10,12,14]
K_list = [8,10,14,20,30]

maskseed_list = [413, 617, 781]
modelseed_list = [413, 617, 781]
model_list = ['ggg',]
lam_list = [275,300,400,500,600,750,1000]

params = it.product(C_list, K_list, maskseed_list, modelseed_list, model_list, lam_list)
params = it.filterfalse(lambda x: x[0] > x[1], params )
params_df = pd.DataFrame(params, columns=['C','K','mask_seed', 'model_seed', 'model', 'lam'])

params_df['data_path'] = '/data/projects/dncbtd/methylation/sarcoma/data_top5k.npz'
params_df['output_path'] = '/work/tmp'


params_df['uuid'] = [uuid.uuid4().hex for _ in range(len(params_df.index))]


experiment_config_json_path = f"results/{name}/{name}_config.json"
params_df.to_json(experiment_config_json_path, orient='records')


# Create a write the submission script for the experiment
with open('src/scripts/submit_experiment.j2', 'r') as f:
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

