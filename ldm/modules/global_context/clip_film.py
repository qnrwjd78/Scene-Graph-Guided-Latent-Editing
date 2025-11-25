import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipImageEncoder(nn.Module):
    """
    Wraps an OpenAI CLIP image encoder to provide a global embedding for FiLM.
    CLIP 가중치는 외부에서 제공해야 하며, 모델 파라미터는 기본적으로 freeze 됩니다.
    """

    def __init__(self, clip_name: str = "ViT-B/32", ckpt_path: str = None):
        super().__init__()
        try:
            import clip  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "clip 패키지가 없습니다. `pip install git+https://github.com/openai/CLIP.git` 로 설치하거나 "
                "로컬에 설치된 패키지를 사용하세요."
            ) from exc

        try:
            model, _ = clip.load(clip_name, device="cpu", download=False)
        except TypeError:
            # older clip versions do not accept download flag
            model, _ = clip.load(clip_name, device="cpu")
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)

        # freeze
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.eval()
        # CLIP ViT는 고정 해상도 입력을 기대함
        self.input_res = getattr(model.visual, "input_resolution", 224)

        # CLIP 기본 normalize
        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: [-1,1] 범위의 BCHW 텐서 (Stage2의 입력 이미지를 그대로 사용)
        return: [B, D] global embedding
        """
        device = image.device
        # ensure encoder is on the same device as input
        if next(self.model.parameters()).device != device:
            self.model = self.model.to(device)
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

        x = (image + 1) * 0.5
        x = (x - self.mean) / self.std
        if x.shape[-1] != self.input_res or x.shape[-2] != self.input_res:
            x = F.interpolate(x, size=(self.input_res, self.input_res), mode="bicubic", align_corners=False)
        # CLIP expects float32
        x = x.to(dtype=torch.float32)
        return self.model.encode_image(x)
