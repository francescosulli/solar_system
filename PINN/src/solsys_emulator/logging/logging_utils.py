def flatten_config(cfg, parent_key='', sep='/'):
    """
    Recursively flatten a nested config/dict structure into scalar key-value pairs.
    
    Args:
        cfg: Config object, dict, or dataclass to flatten
        parent_key: Prefix for nested keys
        sep: Separator between nested levels
    
    Returns:
        dict: Flattened dictionary with only scalar values (int, float, str, bool)
    """
    items = []
    
    # Handle dataclasses, Namespace, or custom config objects
    if hasattr(cfg, '__dict__'):
        cfg_dict = cfg.__dict__
    elif hasattr(cfg, '__dataclass_fields__'):
        import dataclasses
        cfg_dict = dataclasses.asdict(cfg)
    else:
        cfg_dict = cfg
    
    for k, v in cfg_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # Recursively flatten nested dicts/objects
        if isinstance(v, dict) or hasattr(v, '__dict__'):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        
        # Handle lists/tuples - convert to string or skip if too complex
        elif isinstance(v, (list, tuple)):
            if len(v) == 0:
                items.append((new_key, 'empty'))
            elif all(isinstance(x, (int, float, str, bool)) for x in v):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, f'list[{len(v)}]'))
        
        # Handle None
        elif v is None:
            items.append((new_key, 'None'))
        
        # Keep scalars as-is
        elif isinstance(v, (int, float, str, bool)):
            items.append((new_key, v))
        
        # Convert everything else to string
        else:
            items.append((new_key, str(v)))
    
    return dict(items)


def log_config_to_tensorboard(writer, cfg, model_cfg=None, **extra_params):
    """
    Log all hyperparameters to TensorBoard in a safe, flattened format.
    
    Args:
        writer: TensorBoard SummaryWriter
        cfg: Training config
        model_cfg: Model config (optional)
        **extra_params: Additional scalar parameters (world_size, rank, etc.)
    """
    hparams = {}
    
    # Flatten training config
    hparams.update(flatten_config(cfg, parent_key='train'))
    
    # Flatten model config if provided
    if model_cfg is not None:
        hparams.update(flatten_config(model_cfg, parent_key='model'))
    
    # Add extra parameters
    hparams.update(extra_params)
    
    # Log to TensorBoard
    writer.add_hparams(hparams, {})
    
    return hparams


def log_config_to_wandb(cfg, model_cfg=None, **extra_params):
    """
    Log all hyperparameters to wandb (wandb handles nested configs better).
    
    Args:
        cfg: Training config
        model_cfg: Model config (optional)
        **extra_params: Additional parameters
    """
    import wandb
    
    config_dict = {
        'train': flatten_config(cfg),
        'model': flatten_config(model_cfg) if model_cfg else {},
        **extra_params
    }
    
    # Update wandb config (preserves nesting)
    wandb.config.update(config_dict)
    
    return config_dict
