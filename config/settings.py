"""Configuration settings for the algorithmic trading system."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="Algorithmic Trading ML", description="Application name")
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=True, description="Debug mode")
    
    # Database
    database_url: str = Field(
        default="postgresql://trading_user:trading_password@localhost:5432/trading_db",
        description="PostgreSQL connection URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    
    # API Keys (using SecretStr to prevent logging)
    alpha_vantage_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Alpha Vantage API key"
    )
    
    newsapi_key: Optional[SecretStr] = Field(
        default=None,
        description="NewsAPI key for sentiment analysis"
    )
    
    # Trading Parameters
    initial_capital: float = Field(
        default=100000.0,
        description="Initial capital for backtesting",
        gt=0
    )
    
    max_position_size: float = Field(
        default=0.05,
        description="Maximum position size as fraction of portfolio",
        gt=0,
        le=1.0
    )
    
    max_sector_exposure: float = Field(
        default=0.20,
        description="Maximum sector exposure as fraction of portfolio",
        gt=0,
        le=1.0
    )
    
    max_drawdown_threshold: float = Field(
        default=0.10,
        description="Maximum allowed drawdown before stopping trading",
        gt=0,
        le=1.0
    )
    
    # Risk Management
    stop_loss_pct: float = Field(
        default=0.05,
        description="Stop loss percentage",
        gt=0,
        le=1.0
    )
    
    take_profit_pct: float = Field(
        default=0.10,
        description="Take profit percentage",
        gt=0
    )
    
    # Data Collection
    data_collection_interval: int = Field(
        default=3600,
        description="Data collection interval in seconds",
        gt=0
    )
    
    api_rate_limit_per_minute: int = Field(
        default=5,
        description="API rate limit per minute",
        gt=0
    )
    
    # Model Training
    model_retrain_interval_days: int = Field(
        default=7,
        description="Model retraining interval in days",
        gt=0
    )
    
    cross_validation_folds: int = Field(
        default=3,
        description="Number of cross-validation folds for time series",
        ge=2
    )
    
    # Feature Engineering
    technical_indicators_window: int = Field(
        default=20,
        description="Default window size for technical indicators",
        gt=0
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    log_file: str = Field(
        default="logs/trading.log",
        description="Log file path"
    )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()
