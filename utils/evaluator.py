import torch


class Evaluator:
    def __init__(self, model, config):
        self.model  = model
        self.config = config
        self._cached = {}

    def _get_tensors(self, dataset):
        key = id(dataset)
        if key not in self._cached:
            self._cached[key] = (
                torch.from_numpy(dataset.face_feats).float(),
                torch.from_numpy(dataset.audio_feats).float(),
                torch.from_numpy(dataset.labels).long(),
            )
        return self._cached[key]

    def accuracy_from_tensors(self, face, audio, labels, head="fusion", mask_face=1):
        """
        head     : "fusion" | "face" | "audio"  (used for FOP / MultiBranchFOP)
        mask_face: 1 = face available, 0 = audio-only  (used for MaskedFOP)
        """
        self.model.eval()
        model_type = self.config.model_type.lower()

        with torch.no_grad():
            if model_type == "masked_fop":
                out    = self.model(face, audio, mask_face=mask_face)
                logits = out["logits"]

            elif model_type == "multibranch":
                out = self.model(face, audio)
                logits = {
                    "fusion": out["fusion_logits"],
                    "face":   out["face_logits"],
                    "audio":  out["audio_logits"],
                }[head]

            else:   # baseline FOP
                _, logits, _, _ = self.model(face, audio)

            preds   = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()

        return 100.0 * correct / labels.size(0)

    def accuracy(self, dataset, head="fusion", mask_face=1):
        """
        Vectorized accuracy over a full dataset.

        mask_face=1  → multimodal (P3/P5 proxy)
        mask_face=0  → audio-only  (P4/P6 proxy)
        """
        face, audio, labels = self._get_tensors(dataset)
        face   = face.to(self.config.device,   non_blocking=True)
        audio  = audio.to(self.config.device,  non_blocking=True)
        labels = labels.to(self.config.device, non_blocking=True)

        return self.accuracy_from_tensors(face, audio, labels, head=head, mask_face=mask_face)
