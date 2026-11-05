"""Shared service objects, created once per process in the app lifespan."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from api.broker import AlertBroker
from api.jobs import JobManager
from config.settings import Settings
from data_engine.ingestion import IngestionEngine
from models.pipeline import EWSPipeline


@dataclass
class Services:
    settings: Settings
    pipeline: EWSPipeline
    ingestion: IngestionEngine
    broker: AlertBroker
    jobs: JobManager


def services(request: Request) -> Services:
    return request.app.state.services
