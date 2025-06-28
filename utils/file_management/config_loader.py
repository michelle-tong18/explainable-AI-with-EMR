import os
import yaml
from typing import Union
import warnings
warnings.filterwarnings('ignore')
# -------------------- Basic Tools --------------------

def load_yaml(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file) or {}
    return config

def flatten_context(context, parent_key='', sep='.'):
    items = []
    for k, v in context.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_context(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def process_config_values(config):

    # Flatten the configuration context for dot notation support
    flat_config = flatten_context(config)
    
    # Function to substitute placeholders with actual values
    def substitute(current_config):
        if isinstance(current_config, dict):
            for key, value in current_config.items():
                current_config[key] = substitute(value)
        elif isinstance(current_config, list):
            return [substitute(element) for element in current_config]
        elif isinstance(current_config, str):
            # Detect placeholder and substitute with the value from flat_config
            while '${' in current_config:
                start_index = current_config.find('${')
                end_index = current_config.find('}', start_index)
                if start_index == -1 or end_index == -1:
                    break  # No substitution needed
                placeholder = current_config[start_index + 2:end_index]
                if placeholder in flat_config:
                    # Convert the value to a string if it's not already a string
                    substitute_value = str(flat_config[placeholder])
                    # Replace placeholder with actual value
                    current_config = current_config.replace(
                        f'${{{placeholder}}}', substitute_value)
                else:
                    break  # Placeholder not found in flat_config
        return current_config
    
    # Substitute placeholders in the original nested config
    return substitute(config)