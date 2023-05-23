from Data.CT_ssl import CTSSLDataModule
from Data.MSD_ssl import MSDSSLDataModule
from .MSD import MSDDataModule
from .BTCV import BTCVDataModule
from .BTCV_ssl import BTCVSSSLDataModule
from .Brats import BratsDataModule
from .Brats_ssl import BraTSSSLDataModule

data_modules = {
    "BraTS": BratsDataModule,
    "BraTS-SSL": BraTSSSLDataModule,
    "BTCV": BTCVDataModule,
    "BTCV-SSL": BTCVSSSLDataModule,
    "MSD": MSDDataModule,
    "MSD-SSL": MSDSSLDataModule,
    "CT-SSL": CTSSLDataModule,
}
