from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECOMMENDER_")

    model_dir: Path = _REPO_ROOT / "out" / "final_best"
    db_path: Path = _REPO_ROOT / "data" / "recommender_history.db"

    n_candidates: int = 200
    default_top_k: int = 3

    # Optional key for public endpoints: invalid non-empty key is rejected.
    api_key: str | None = None
    # Admin endpoints always require a concrete key. If this is not set,
    # api_key is used as a fallback; if neither is set, admin access is closed.
    admin_key: str | None = None


settings = Settings()
