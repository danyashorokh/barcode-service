import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.containers import AppContainer
from src.routes import barcode as barcode_routes
from src.routes.routers import router as app_router

PORT = 1024


def create_app() -> FastAPI:
    container = AppContainer()
    cfg = OmegaConf.load('config/config.yml')
    container.config.from_dict(cfg)
    container.wire([barcode_routes])

    app = FastAPI()
    set_routers(app)
    return app


def set_routers(app: FastAPI):
    app.include_router(app_router, prefix='/barcodes', tags=['barcodes'])


if __name__ == '__main__':
    app = create_app()
    uvicorn.run(app, port=PORT, host='0.0.0.0')
