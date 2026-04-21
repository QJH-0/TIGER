# Binary TIGER Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the TIGER codebase with the documented binary-training and standalone-distillation design while preserving separate training flows.

**Architecture:** Extend the existing binary layer primitives and conversion policy so they match the documented protection strategy, then update binary and distillation training systems independently. Keep the binary phase focused on binary optimization only, and keep distillation as a separate teacher-student fine-tuning path.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, pytest, YAML config

---

### Task 1: Tighten Binary Layer Primitives

**Files:**
- Modify: `look2hear/layers/binary_layers.py`
- Test: `tests/test_binary_layers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_rsign_uses_learnable_alpha_threshold():
    layer = RSign(4)
    layer.alpha.data.fill_(0.25)
    x = torch.tensor([[[0.2], [0.3], [0.24], [0.26]]])
    out = layer(x)
    assert out.tolist() == [[[-1.0], [1.0], [-1.0], [1.0]]]


def test_rprelu_exposes_beta_gamma_zeta_parameters():
    layer = RPReLU(3)
    assert tuple(layer.beta.shape) == (1, 3, 1)
    assert tuple(layer.gamma.shape) == (1, 3, 1)
    assert tuple(layer.zeta.shape) == (1, 3, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_binary_layers.py -v`
Expected: FAIL because `RSign` does not expose `alpha` and `RPReLU` does not expose `beta/gamma/zeta`

- [ ] **Step 3: Write minimal implementation**

```python
class RSign(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (torch.where(x >= self.alpha, torch.ones_like(x), -torch.ones_like(x)) - x).detach()


class RPReLU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1, channels, 1) * 0.5)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 0.1)
        self.zeta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.maximum(x, torch.zeros_like(x))
        neg = torch.minimum(x, torch.zeros_like(x))
        return self.beta * pos + self.gamma * neg + self.zeta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_binary_layers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_binary_layers.py look2hear/layers/binary_layers.py
git commit -m "feat(binary): align binary layer primitives with design"
```

### Task 2: Update Converter Protection Policy

**Files:**
- Modify: `look2hear/utils/model_converter.py`
- Test: `tests/test_model_converter.py`

- [ ] **Step 1: Write the failing test**

```python
def test_converter_supports_named_protect_patterns():
    converter = TIGERBinaryConverter(protect_patterns=["bandsplit.proj", "mask"])
    assert converter._matches_protect_pattern("bandsplit.proj.0.conv")
    assert converter._matches_protect_pattern("separator.mask_gen")


def test_converter_protects_qkv_modules_even_when_1x1_is_disabled():
    converter = TIGERBinaryConverter(
        protect_attention=True,
        protect_1x1_conv=False,
        protect_patterns=["q_proj", "k_proj", "v_proj"],
    )
    module = nn.Conv1d(8, 8, 1)
    assert converter._should_protect_conv(module, "separator.block.q_proj", {"seen_conv": True})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_converter.py -v`
Expected: FAIL because `protect_patterns` matching is not implemented

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass
class TIGERBinaryConverter:
    protect_patterns: list[str] = field(default_factory=list)

    def _matches_protect_pattern(self, full_name: str) -> bool:
        tokens = full_name.lower()
        return any(pattern.lower() in tokens for pattern in self.protect_patterns)

    def _should_protect_conv(...):
        if self._matches_protect_pattern(full_name):
            return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_converter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_converter.py look2hear/utils/model_converter.py
git commit -m "feat(binary): add named protection patterns"
```

### Task 3: Extend Binary Training Stage Control

**Files:**
- Modify: `look2hear/system/binary_audio_litmodule.py`
- Test: `tests/test_binary_audio_litmodule.py`

- [ ] **Step 1: Write the failing test**

```python
def test_binary_module_supports_three_stage_schedule():
    module = BinaryAudioLightningModule(...)
    module.binary_stage_epochs = {"activation_warmup": 1, "weight_binarize": 3}
    module.current_epoch = 0
    assert module._resolve_stage() == "activation_warmup"
    module.current_epoch = 1
    assert module._resolve_stage() == "weight_binarize"
    module.current_epoch = 3
    assert module._resolve_stage() == "finetune"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_binary_audio_litmodule.py -v`
Expected: FAIL because only `warmup` and `binary` stages exist

- [ ] **Step 3: Write minimal implementation**

```python
def _resolve_stage(self) -> str:
    activation_warmup = int(self.binary_stage_epochs.get("activation_warmup", 0))
    weight_binarize = int(self.binary_stage_epochs.get("weight_binarize", 0))
    if self.current_epoch < activation_warmup:
        return "activation_warmup"
    if self.current_epoch < weight_binarize:
        return "weight_binarize"
    return "finetune"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_binary_audio_litmodule.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_binary_audio_litmodule.py look2hear/system/binary_audio_litmodule.py
git commit -m "feat(training): add three-stage binary schedule"
```

### Task 4: Keep Distillation Standalone and Explicit

**Files:**
- Modify: `look2hear/system/distill_audio_litmodule.py`
- Modify: `audio_train.py`
- Test: `tests/test_distill_audio_litmodule.py`

- [ ] **Step 1: Write the failing test**

```python
def test_distill_module_uses_standalone_distillation_loss_mix():
    module = DistillAudioLightningModule(...)
    assert module.distillation_enabled is True
    assert module.kd_lambda == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distill_audio_litmodule.py -v`
Expected: FAIL if standalone configuration parsing or logging is incomplete

- [ ] **Step 3: Write minimal implementation**

```python
self.log("train/task_loss", task_loss, ...)
self.log("train/kd_loss", kd_loss, ...)
self.log("train/loss", loss, ...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distill_audio_litmodule.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_distill_audio_litmodule.py look2hear/system/distill_audio_litmodule.py audio_train.py
git commit -m "feat(distill): keep standalone distillation explicit"
```

### Task 5: Align Config Defaults With the Design

**Files:**
- Modify: `configs/tiger-small-binary.yml`
- Modify: `configs/tiger-small-distill.yml`
- Modify: `configs/tiger-small-kaggle-t4x2-binary.yml`
- Modify: `configs/tiger-small-kaggle-t4x2-distill.yml`

- [ ] **Step 1: Write the failing test**

```python
def test_binary_config_exposes_design_stage_keys():
    config = yaml.safe_load(Path("configs/tiger-small-binary.yml").read_text(encoding="utf-8"))
    assert "activation_warmup" in config["training"]["binary_stage_epochs"]
    assert "weight_binarize" in config["training"]["binary_stage_epochs"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_contracts.py -v`
Expected: FAIL because the config keys do not exist yet

- [ ] **Step 3: Write minimal implementation**

```yaml
training:
  binary_stage_epochs:
    activation_warmup: 5
    weight_binarize: 100
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_contracts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/tiger-small-binary.yml configs/tiger-small-distill.yml configs/tiger-small-kaggle-t4x2-binary.yml configs/tiger-small-kaggle-t4x2-distill.yml tests/test_config_contracts.py
git commit -m "chore(config): align binary and distill configs with design"
```
