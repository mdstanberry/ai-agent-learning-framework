"""
Configuration Management Module

This module handles loading and accessing configuration settings from:
1. Environment variables (.env file)
2. YAML configuration file (config.yaml)

Environment variables take precedence over YAML settings.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """
    Configuration manager that loads settings from .env and config.yaml files.
    
    Usage:
        config = Config()
        api_key = config.get("OPENAI_API_KEY")
        model = config.get_nested("llm.openai.default_model")
    """
    
    def __init__(self, env_file: Optional[str] = None, config_file: Optional[str] = None):
        """
        Initialize configuration by loading .env and config.yaml files.
        
        Args:
            env_file: Path to .env file (defaults to .env in project root)
            config_file: Path to config.yaml file (defaults to config.yaml in project root)
        """
        # Determine project root (parent of utils directory)
        self.project_root = Path(__file__).parent.parent
        
        # Load environment variables from .env file
        env_path = env_file or self.project_root / ".env"
        load_dotenv(dotenv_path=env_path)
        
        # Load YAML configuration
        config_path = config_file or self.project_root / "config.yaml"
        self.yaml_config = self._load_yaml(config_path)
        
        # Store combined configuration (env vars take precedence)
        self._config = self._merge_config()
    
    def _load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing YAML configuration
        """
        if not config_path.exists():
            print(f"Warning: Config file not found at {config_path}. Using defaults.")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config or {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            return {}
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}
    
    def _merge_config(self) -> Dict[str, Any]:
        """
        Merge environment variables with YAML config.
        Environment variables take precedence.
        
        Returns:
            Merged configuration dictionary
        """
        config = self.yaml_config.copy()
        
        # Override with environment variables where they exist
        # Map common environment variables to config structure
        env_mappings = {
            'OPENAI_API_KEY': ('api_keys', 'openai'),
            'ANTHROPIC_API_KEY': ('api_keys', 'anthropic'),
            'LLM_PROVIDER': ('llm', 'provider'),
            'DEFAULT_MODEL': ('llm', 'default_model'),
            'TEMPERATURE': ('llm', 'temperature'),
            'MAX_TOKENS': ('llm', 'max_tokens'),
            'LOG_LEVEL': ('logging', 'level'),
            'VECTOR_DB_PATH': ('memory', 'episodic', 'db_path'),
            'MAX_ITERATIONS': ('agent', 'max_iterations'),
            'TRACK_TOKENS': ('cost_tracking', 'enabled'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested(config, config_path, self._convert_type(value))
        
        # Add API keys section if not present
        if 'api_keys' not in config:
            config['api_keys'] = {}
        
        # Ensure API keys are set from environment
        config['api_keys']['openai'] = os.getenv('OPENAI_API_KEY', '')
        config['api_keys']['anthropic'] = os.getenv('ANTHROPIC_API_KEY', '')
        
        return config
    
    def _set_nested(self, config: Dict, path: tuple, value: Any) -> None:
        """
        Set a value in a nested dictionary using a path tuple.
        
        Args:
            config: The configuration dictionary to modify
            path: Tuple of keys representing the path (e.g., ('llm', 'openai', 'model'))
            value: The value to set
        """
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _convert_type(self, value: str) -> Any:
        """
        Convert string values to appropriate types.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value (bool, int, float, or str)
        """
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get("llm.provider")
            config.get("OPENAI_API_KEY")
        """
        # Try direct environment variable first
        env_value = os.getenv(key)
        if env_value is not None:
            return self._convert_type(env_value)
        
        # Try nested key in config
        if '.' in key:
            return self.get_nested(key, default)
        
        # Try direct key in config
        return self._config.get(key, default)
    
    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation.
        
        Args:
            path: Dot-separated path to the value (e.g., "llm.openai.default_model")
            default: Default value if path is not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get_nested("llm.openai.default_model")
        """
        keys = path.split('.')
        current = self._config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def get_api_key(self, provider: str) -> str:
        """
        Get API key for a specific provider.
        
        Args:
            provider: Provider name ("openai" or "anthropic")
            
        Returns:
            API key string
            
        Raises:
            ValueError: If API key is not found or empty
        """
        api_key = self.get_nested(f"api_keys.{provider}")
        
        if not api_key or api_key == f"your_{provider}_api_key_here":
            raise ValueError(
                f"{provider.upper()} API key not found. "
                f"Please set {provider.upper()}_API_KEY in your .env file. "
                f"Copy .env.example to .env and add your API key."
            )
        
        return api_key
    
    def get_model_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model configuration for a specific provider.
        
        Args:
            provider: Provider name ("openai" or "anthropic"). 
                     If None, uses the default provider from config.
                     
        Returns:
            Dictionary containing model configuration
        """
        if provider is None:
            provider = self.get_nested("llm.provider", "openai")
        
        return self.get_nested(f"llm.{provider}", {})
    
    def is_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature: Feature path (e.g., "memory.episodic.enabled")
            
        Returns:
            True if enabled, False otherwise
        """
        return bool(self.get_nested(feature, False))
    
    def __repr__(self) -> str:
        """String representation of Config object."""
        return f"Config(provider={self.get('llm.provider')}, model={self.get_nested('llm.openai.default_model')})"


# Global configuration instance
# Import this in other modules: from utils.config import config
config = Config()


if __name__ == "__main__":
    # Test configuration loading
    print("Configuration loaded successfully!")
    print(f"LLM Provider: {config.get('llm.provider')}")
    print(f"OpenAI Model: {config.get_nested('llm.openai.default_model')}")
    print(f"Anthropic Model: {config.get_nested('llm.anthropic.default_model')}")
    print(f"Max Iterations: {config.get_nested('agent.max_iterations')}")
    print(f"Cost Tracking: {config.get_nested('cost_tracking.enabled')}")

