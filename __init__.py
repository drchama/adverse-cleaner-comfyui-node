class AdverseCleaner:
    import torch

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "steps1": ("INT", {"default": 64, "min": 0, "max": 1000, "step": 1}),
                "d": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
                "sigma_color": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 1000.0, "step": 1.0},
                ),
                "sigma_space": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 1000.0, "step": 1.0},
                ),
                "steps2": ("INT", {"default": 4, "min": 0, "max": 1000, "step": 1}),
                "radius": ("INT", {"default": 4, "min": 0, "max": 1000, "step": 1}),
                "eps": (
                    "FLOAT",
                    {"default": 16.0, "min": 0.0, "max": 1000.0, "step": 1.0},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "remove_noise"

    CATEGORY = "image"

    def remove_noise(
        self,
        image: torch.Tensor,
        steps1: int,
        d: int,
        sigma_color: float,
        sigma_space: float,
        steps2: int,
        radius: int,
        eps: float,
    ):
        import numpy as np
        import cv2
        import torch

        def sub(image: torch.Tensor):
            guide = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
            dst = guide.copy()

            for _ in range(steps1):
                dst = cv2.bilateralFilter(dst, d, sigma_color, sigma_space)

            for _ in range(steps2):
                dst = cv2.ximgproc.guidedFilter(guide, dst, radius, eps)

            return torch.from_numpy(dst.astype(np.float32) / 255.0).unsqueeze(0)

        if len(image) > 1:
            tensors = []

            for child in image:
                tensor = sub(child)
                tensors.append(tensor)

            return (torch.cat(tensors, dim=0),)

        else:
            tensor = sub(image)
            return (tensor,)


NODE_CLASS_MAPPINGS = {"AdverseCleaner": AdverseCleaner}

NODE_DISPLAY_NAME_MAPPINGS = {"AdverseCleaner": "Remove Adversarial Noise"}
