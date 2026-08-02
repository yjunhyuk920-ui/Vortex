from types import SimpleNamespace

import torch
from torch import nn

from vortex_runtime.adjoint_profiler import profile_exact_target_margin_tiles
from vortex_runtime.decision_tile_repair import (
    DecisionResidualTileAtlasLinearModule,
)


class FakeTokenizer:
    def __call__(self, _prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


class TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.o_proj = DecisionResidualTileAtlasLinearModule(
            nn.Linear(8, 8, bias=False),
            max_rank=2,
        )
        self.lm_head = nn.Linear(8, 16, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert attention_mask.shape[:2] == inputs_embeds.shape[:2]
        assert use_cache is False
        assert return_dict is True
        hidden = torch.tanh(self.o_proj(inputs_embeds))
        return SimpleNamespace(logits=self.lm_head(hidden))


def encode_prompt(
    tokenizer: FakeTokenizer,
    prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in tokenizer(prompt, return_tensors="pt").items()
    }


def test_direct_autograd_profiler_collects_projection_gradient() -> None:
    torch.manual_seed(71)
    model = TinyCausalModel()
    tokenizer = FakeTokenizer()
    module = model.o_proj

    module.set_mode("learn_exact")
    build_ids = torch.tensor([[6, 7]], dtype=torch.long)
    model(
        inputs_embeds=model.embedding(build_ids),
        attention_mask=torch.ones_like(build_ids),
        use_cache=False,
        return_dict=True,
    )

    profile = profile_exact_target_margin_tiles(
        model=model,
        tokenizer=tokenizer,
        eval_prompt="ignored",
        exact_sequence=[1, 2, 3, 4, 5],
        replacements={"o_proj": module},
        device=torch.device("cpu"),
        row_tile=4,
        col_tile=4,
        encode_prompt=encode_prompt,
    )

    assert profile.generated_targets == 2
    assert profile.missing_gradient_modules == []
    assert profile.non_differentiable_modules == []
    assert len(profile.candidates) == 4
    assert any(
        float(item["absolute_margin_contribution"]) > 0
        for item in profile.candidates
    )
