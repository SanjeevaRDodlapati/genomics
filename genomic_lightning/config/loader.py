"""Configuration utilities for loading and parsing YAML configs."""

import os
import yaml
from typing import Dict, Any, Optional, Union, List, TextIO


class ConfigLoader:
    """Utility class for loading and parsing configuration files."""
    
    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If the file does not exist
            yaml.YAMLError: If the file cannot be parsed as YAML
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                return config
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file {path}: {str(e)}")
    
    @staticmethod
    def load_yaml_with_obj_tags(path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file with support for !obj tags.
        
        This is for compatibility with UAVarPrior/FuGEP style configs.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If the file does not exist
            yaml.YAMLError: If the file cannot be parsed as YAML
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        # Define a custom YAML constructor for !obj tags
        def obj_constructor(loader, node):
            # Get the class path and arguments
            if isinstance(node, yaml.ScalarNode):
                # Simple case: !obj:class.path
                class_path = node.value
                args = {}
            elif isinstance(node, yaml.MappingNode):
                # Complex case: !obj:class.path {arg1: value1, arg2: value2}
                items = loader.construct_mapping(node)
                class_path = list(items.keys())[0]
                args = list(items.values())[0]
            else:
                class_path = ""
                args = {}
            
            # Return a dictionary with the class path and arguments
            return {'!obj:' + class_path: args}
        
        # Register the custom constructor
        yaml.add_constructor('!obj', obj_constructor)
        
        with open(path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                return config
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file {path}: {str(e)}")
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = base_config.copy()
        
        for k, v in override_config.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                # Recursively merge nested dictionaries
                result[k] = ConfigLoader.merge_configs(result[k], v)
            else:
                # Override or add key
                result[k] = v
        
        return result


def load_config(path: str, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file with optional base config.
    
    Args:
        path: Path to YAML file
        base_config: Optional base configuration to merge with
        
    Returns:
        Configuration dictionary
    """
    # Load config from file
    config = ConfigLoader.load_yaml_with_obj_tags(path)
    
    # Merge with base config if provided
    if base_config is not None:
        config = ConfigLoader.merge_configs(base_config, config)
    
    return config
