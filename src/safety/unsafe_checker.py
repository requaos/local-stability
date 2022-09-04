from transformers import CLIPConfig, PreTrainedModel

class StableDiffusionUnSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

    @torch.no_grad()
    def forward(self, clip_input, images):
        return images, []