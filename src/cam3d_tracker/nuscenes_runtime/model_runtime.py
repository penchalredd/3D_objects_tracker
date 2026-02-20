from __future__ import annotations

from typing import Any

from .dynamic_import import import_symbol


class DetectorRuntime:
    def __init__(
        self,
        model_class_path: str,
        checkpoint_path: str,
        model_kwargs: dict[str, Any] | None,
        device: str,
        input_adapter_path: str,
        output_adapter_path: str,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for nuscenes_runtime. Install PyTorch first.") from exc

        self._torch = torch
        model_class = import_symbol(model_class_path)
        input_adapter = import_symbol(input_adapter_path)
        output_adapter = import_symbol(output_adapter_path)

        self.model = model_class(**(model_kwargs or {}))
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a valid state_dict")

        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state[k[7:]] = v
            else:
                clean_state[k] = v

        missing, unexpected = self.model.load_state_dict(clean_state, strict=False)
        if missing:
            print(f"[nuscenes_runtime] warning: missing keys: {len(missing)}")
        if unexpected:
            print(f"[nuscenes_runtime] warning: unexpected keys: {len(unexpected)}")

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

    def infer(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        with self._torch.no_grad():
            model_input = self.input_adapter(frame=frame, device=self.device, torch=self._torch)
            if hasattr(self.model, "predict"):
                raw = self.model.predict(model_input)
            else:
                raw = self.model(model_input)
        return self.output_adapter(raw_output=raw, frame=frame)
