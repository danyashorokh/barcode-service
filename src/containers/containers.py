from dependency_injector import containers, providers

from src.services.barcode_recognition import BarcodeRecognition


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    barcode_recognition = providers.Singleton(
        BarcodeRecognition,
        config=config.services.barcode_recognition,
    )
