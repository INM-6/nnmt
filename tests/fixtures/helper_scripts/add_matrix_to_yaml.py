"""
Helper script that writes a given 2d matrix to a yaml file.

You need to specify the key and the unit in the script below.

Usage:

```
python add_matrix_to_yaml.py <matrix_file>.npy <yaml_file>.yaml
```
"""

import numpy as np
import sys
import yaml
import re


def insert_matrix_before_end(yaml_content, key, matrix_str):
    key_str = '{}:\n'.format(key)
    matrix_with_key = '{}{}'.format(key_str, matrix_str)
    return re.sub(r'(\s*\.\.\.\s*)$', '\n\n' + matrix_with_key + '\n...\n', yaml_content)


if __name__ == "__main__":

    key = 'J_eff'
    unit = 'mV'

    matrix_file = sys.argv[1]
    yaml_file = sys.argv[2]

    # Load connectivity matrix
    W = np.load(matrix_file)
    # W = np.load(matrix_file, allow_pickle=True)['network_params'].tolist()['W']
    new_value = W.tolist()

    # Load the existing YAML content from file into a string
    with open(yaml_file, "r") as file:
        yaml_content = file.read()

    # Convert the matrix to a YAML-compatible string
    matrix_str = yaml.dump(new_value, default_flow_style=False).strip()

    # formatting
    matrix_str = matrix_str.replace('\n- -', '\n    - -')
    matrix_str = matrix_str.replace('\n  -', '\n      -')
    matrix_str = matrix_str.replace('\n  -', '\n      -')
    matrix_str = '  val:\n    ' + matrix_str + f'\n  unit: {unit}'

    # Insert the matrix before the end of the YAML content
    updated_yaml_content = insert_matrix_before_end(
        yaml_content, key, matrix_str)

    # Overwrite the existing YAML file with the updated content
    with open(yaml_file, "w") as file:
        file.write(updated_yaml_content)
