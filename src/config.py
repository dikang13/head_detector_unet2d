# config.py
import yaml
import os

def load_config(path_config, path_root):
    """Load and process configuration file."""
    with open(path_config, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

    exp_name = config_data["exp_name"]
    path_exp_base = os.path.join(path_root, exp_name)
    path_log_tb = os.path.join(path_exp_base, "runs", exp_name)
    path_checkpoint = os.path.join(path_exp_base, exp_name + ".tar")
    path_log_csv = os.path.join(path_exp_base, exp_name + ".csv")

    if not os.path.isdir(path_exp_base):
        os.makedirs(path_exp_base, exist_ok=True)

    # Gather your main training parameters
    args = {
        "start_epoch": config_data["start_epoch"],
        "print_freq": config_data["print_freq"],
        "checkpoint_freq": config_data["checkpoint_freq"],
        "n_epochs": config_data["n_epochs"],
        "n_features": config_data["n_features"],
        "lr_scheduler_patience": config_data["lr_scheduler_patience"],
        "lr_scheduler_factor": config_data["lr_scheduler_factor"],
        "loss_weight": config_data["loss_weight"],
        "learning_rate": config_data["lr"],
        "batch_size": config_data["batch_size"],
        "scaling_factor": config_data["scaling_factor"],
        "n_test": config_data["n_test"],
        "n_val": config_data["n_val"],

        # save paths
        "path_checkpoint": path_checkpoint,
        "path_log_tb": path_log_tb,
        "path_log_csv": path_log_csv,
    }

    # If your config has an "augmentation" section, add it to args
    if "augmentation" in config_data:
        args["augmentation"] = config_data["augmentation"]

    return args, path_exp_base, exp_name