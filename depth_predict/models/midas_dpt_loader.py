from transformers import DPTForDepthEstimation

__all__ = ["load_midas_model"]

def load_midas_model(model_type: str = "dpt-hybrid-midas"):
    """Hugging Face MiDaS 모델 로드"""
    return DPTForDepthEstimation.from_pretrained(f"Intel/{model_type}")
