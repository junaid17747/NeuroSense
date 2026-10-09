"""Local authentication client used by the Streamlit dashboard."""

from backend.services import auth_service


def register(email: str, password: str, display_name: str) -> dict:
    auth_service.register_user(email, password, display_name)
    return auth_service.login_user(email, password)


def login(email: str, password: str) -> dict:
    return auth_service.login_user(email, password)


def current_user(token: str | None) -> dict | None:
    return auth_service.current_user(token)


def logout(token: str | None) -> None:
    auth_service.logout_user(token)
