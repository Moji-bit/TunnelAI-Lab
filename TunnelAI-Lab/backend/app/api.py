from __future__ import annotations

from fastapi import APIRouter

from .scenario_store import ScenarioStore


def build_api_router(store: ScenarioStore) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/scenarios")
    def list_scenarios() -> list[dict]:
        return [scenario.model_dump() for scenario in store.list_scenarios()]

    @router.get("/scenarios/{scenario_id}/meta")
    def scenario_meta(scenario_id: str) -> dict:
        return store.load_meta(scenario_id)

    return router
