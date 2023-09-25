import yaml

def check_config_keys(cfg, required_keys):
    """
    Checks the config file and raise a value if there is a problem
    Args:
        cfg (omegaconf.dictconfig.DictConfig): A config file that is shared through 'hydra'
        required_keys (list): A list of required keys to be checked

    Raises:
        ValueError: if a key is missing
        ValueError: if a key is none
    """
    for key in required_keys:
        value = cfg
        for subkey in key.split("."):
            if subkey not in value:
                raise ValueError(f"Key '{key}' is missing in the configuration.")
            value = value[subkey]

        if value is None:
            raise ValueError(f"Value for key '{key}' is None in the configuration.")


def check_config_file(config_path):
    """
    checks if the config file exists and can be properly loaded
    Args:
        config_path (str): the path of the config file

    Raises:
        ValueError: if the config file is empty or invalid
        FileNotFoundError: if the config file was not found
        FileNotFoundError: if there was a problem in parsing data
        ValueError: if there was a problem in loading data

    Returns:
        dict: a dictionary containing the config file
    """
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        if config is None:
            raise ValueError("The YAML file is empty or invalid.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file 'config.yaml' was not found.")
    except yaml.YAMLError as e:
        raise FileNotFoundError(f"Error parsing the YAML configuration file:")
    except ValueError as e:
        raise ValueError(f"Error loading the configuration data:")
    else:
        # Configuration loaded successfully, you can access settings here
        print("Configuration loaded successfully.")
    return config
