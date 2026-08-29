import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import settings
from embeddings import load_embedder


class RagModel:
    """Loads the generative model + embedder and exposes the core generate pipeline.

    Task-specific logic (HyDE, text-to-SQL, etc.) lives in separate modules
    that take a RagModel instance as a dependency — this class stays focused
    on loading weights and running generation.
    """

    def __init__(self):
        self._dtype = getattr(torch, settings.dtype)  # resolve string -> torch dtype here

        self.tok = AutoTokenizer.from_pretrained(settings.gen_model_id)
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            settings.gen_model_id, torch_dtype=self._dtype, device_map=settings.device_map
        )
        # The same builder the index was written with: SPECTER2 needs CLS pooling,
        # and query vectors must come from the identical encoder as the documents.
        self.embedder = load_embedder()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = settings.default_max_new_tokens,
        temperature: float = settings.default_temperature,
    ) -> str:
        # return_dict is the transformers default now, so ask for it explicitly and
        # unpack — passing the BatchEncoding itself lands it in `inputs_tensor`.
        inputs = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(self.gen_model.device)

        # temperature alongside do_sample=False is a warning, and sql_temperature is 0.0
        sampling = (
            {"do_sample": True, "temperature": temperature}
            if temperature > 0
            else {"do_sample": False}
        )
        out = self.gen_model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        return self.tok.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
