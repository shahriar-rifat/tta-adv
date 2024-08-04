from tta_algo.rotta import RoTTA
from tta_algo.tent import TENT
from tta_algo.note import NOTE
from tta_algo.eata import EATA

def build_tta_adapter(cfg):

    if cfg.TTA.NAME == 'rotta':
        return RoTTA
    elif cfg.TTA.NAME == "tent":
        return TENT
    elif cfg.TTA.NAME == 'note':
        return NOTE
    elif cfg.TTA.NAME == 'eata':
        return EATA
    else:
        raise NotImplementedError("The tta_adapter is not Implemented")
