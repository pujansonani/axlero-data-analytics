from __future__ import annotations

from fastapi import Request

from app.services.store import Store


def get_store(request: Request) -> Store:
    return request.app.state.store
