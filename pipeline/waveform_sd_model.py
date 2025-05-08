# -*- coding: utf-8 -*-

from pipeline.constants import Constants as const

from sd.decoder import VAE_Decoder
from sd.diffusion import Diffusion
from sd.encoder import VAE_Encoder
from sd.siglip import SiglipTimeseriesModel, SiglipTimeseriesConfig

class WaveformSDModel:

    @classmethod
    def get_untrained_model(cls, device):
        encoder = VAE_Encoder().to(device)
        decoder = VAE_Decoder().to(device)
        diffusion = Diffusion().to(device)
        siglip = SiglipTimeseriesModel(SiglipTimeseriesConfig()).to(device)

        return {const.SIGLIP: siglip,
                const.ENCODER: encoder,
                const.DECODER: decoder,
                const.DIFFUSION: diffusion}
