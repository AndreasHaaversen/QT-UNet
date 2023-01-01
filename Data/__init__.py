from Data.CT_ssl import CTSSLDataModule
from Data.CityScapes_ssl import CityScapesSSLDataModule
from Data.MSD_ssl import MSDSSLDataModule
from Data.NTNU import NTNUDatamodule
from .MSD import MSDDataModule
from .BTCV import BTCVDataModule
from .BTCV_ssl import BTCVSSSLDataModule
from .CityScapes import CustomCityScapesDataModule
from .Pannuke import PannukeDataModule
from .Brats import BratsDataModule
from .Brats_ssl import BraTSSSLDataModule

data_modules = {
    "CityScapes": CustomCityScapesDataModule,
    "CityScapesCat": CustomCityScapesDataModule,
    "CityScapes-SSL": CityScapesSSLDataModule,
    "Pannuke": PannukeDataModule,
    "BraTS": BratsDataModule,
    "BraTS-SSL": BraTSSSLDataModule,
    "BTCV": BTCVDataModule,
    "BTCV-SSL": BTCVSSSLDataModule,
    "MSD": MSDDataModule,
    "MSD-SSL": MSDSSLDataModule,
    "CT-SSL": CTSSLDataModule,
    "NTNU": NTNUDatamodule,
    "NTNUCat": NTNUDatamodule,
}
