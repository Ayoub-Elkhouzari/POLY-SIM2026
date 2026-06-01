import torch
import torch.nn.functional as F
from tqdm import tqdm
from .losses import OrthogonalProjectionLoss, SupConLoss, CrossLingualTripletLoss
from .m3 import M3


class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config

        self.ce      = torch.nn.CrossEntropyLoss(label_smoothing=getattr(config, "label_smoothing", 0.0))
        self.opl     = OrthogonalProjectionLoss()
        self.supcon  = SupConLoss(temperature=getattr(config, "supcon_temp", 0.1))
        self.triplet = CrossLingualTripletLoss(margin=getattr(config, "triplet_margin", 0.3))

        if getattr(config, "use_cl_align", False):
            C = config.resolved_num_classes
            D = config.embedding_dim
            self.en_proto      = torch.zeros(C, D)
            self.ur_proto      = torch.zeros(C, D)
            self.en_proto_init = torch.zeros(C, dtype=torch.bool)
            self.ur_proto_init = torch.zeros(C, dtype=torch.bool)

        if getattr(config, "use_m3", False):
            self.opt = M3(
                self.model.parameters(),
                lr=config.lr,
                slow_chunk=getattr(config, "m3_slow_chunk", 500),
                weight_decay=getattr(config, "weight_decay", 0.0),
            )
        else:
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=getattr(config, "weight_decay", 0.0),
            )

        if getattr(config, "lr_schedule", False):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=config.max_epochs, eta_min=1e-5
            )
        else:
            self.scheduler = None

    def train_epoch(self, loader, alpha, unseen_loader=None, logger=None, epoch=None):
        self.model.train()
        total_loss = 0.0
        model_type = self.config.model_type.lower()

        unseen_iter = iter(unseen_loader) if unseen_loader is not None else None

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}",
            disable=not self.config.debug,
            leave=False,
        )

        use_supcon = getattr(self.config, "use_supcon", False)

        for audio, face, labels in pbar:
            audio  = audio.to(self.config.device, non_blocking=True)
            face   = face.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)

            # --------------------------------------------------
            # MaskedFOP — per-sample modality dropout
            # --------------------------------------------------
            if model_type == "masked_fop":
                if use_supcon and unseen_iter is not None:
                    # Combined EN+UR step: single backward with SupCon on merged audio_e
                    try:
                        au, fa, la = next(unseen_iter)
                    except StopIteration:
                        unseen_iter = iter(unseen_loader)
                        au, fa, la = next(unseen_iter)
                    au = au.to(self.config.device, non_blocking=True)
                    fa = fa.to(self.config.device, non_blocking=True)
                    la = la.to(self.config.device, non_blocking=True)
                    loss = self._combined_supcon_step(face, audio, labels, fa, au, la, alpha)
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    continue

                # domain_label=0 → English (seen language)
                loss = self._masked_fop_loss(face, audio, labels, alpha, per_sample=True, domain_label=0)

            # --------------------------------------------------
            # MultiBranchFOP
            # --------------------------------------------------
            elif model_type == "multibranch":
                out = self.model(face, audio)
                loss = (
                    self.config.loss_face    * self.ce(out["face_logits"],    labels)
                    + self.config.loss_audio * self.ce(out["audio_logits"],   labels)
                    + self.config.loss_fusion * self.ce(out["fusion_logits"], labels)
                )
                if alpha > 0:
                    loss = loss + alpha * self.opl(out["fusion_embed"], labels)

            # --------------------------------------------------
            # Baseline FOP
            # --------------------------------------------------
            else:
                fused, logits, _, _ = self.model(face, audio)
                loss = self.ce(logits, labels)
                if alpha > 0:
                    loss = loss + alpha * self.opl(fused, labels)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            total_loss += loss.item()

            # --------------------------------------------------
            # Unseen-language (Urdu) batch — forced audio-only
            # --------------------------------------------------
            if unseen_iter is not None and model_type == "masked_fop":
                try:
                    au, fa, la = next(unseen_iter)
                except StopIteration:
                    unseen_iter = iter(unseen_loader)
                    au, fa, la = next(unseen_iter)

                au = au.to(self.config.device, non_blocking=True)
                fa = fa.to(self.config.device, non_blocking=True)
                la = la.to(self.config.device, non_blocking=True)

                # Urdu: forced audio-only (no distillation — preserve audio-only robustness for P6)
                loss_u = self._masked_fop_loss(fa, au, la, alpha, per_sample=False, force_mask=0, domain_label=1)

                self.opt.zero_grad(set_to_none=True)
                loss_u.backward()
                self.opt.step()
                total_loss += loss_u.item()

                # Cross-lingual step: triplet + centroid alignment (shared forward passes)
                use_triplet  = getattr(self.config, "use_triplet",  False)
                use_cl_align = getattr(self.config, "use_cl_align", False)
                if use_triplet or use_cl_align:
                    en_ae = F.normalize(self.model(face, audio, mask_face=0)["audio_e"], dim=1)
                    ur_ae = F.normalize(self.model(fa,   au,    mask_face=0)["audio_e"], dim=1)
                    xl_loss = torch.tensor(0.0, device=audio.device)

                    if use_triplet:
                        xl_loss = xl_loss + getattr(self.config, "triplet_weight", 0.1) * \
                                  self.triplet(en_ae, ur_ae, labels, la)

                    if use_cl_align:
                        cl_w      = getattr(self.config, "cl_align_weight", 0.3)
                        momentum  = getattr(self.config, "cl_ema_momentum", 0.99)
                        dev       = audio.device
                        en_lbl    = labels.cpu().tolist()
                        ur_lbl    = la.cpu().tolist()

                        with torch.no_grad():
                            en_det = en_ae.detach().cpu()
                            ur_det = ur_ae.detach().cpu()
                            for i, c in enumerate(en_lbl):
                                if self.en_proto_init[c]:
                                    self.en_proto[c] = momentum * self.en_proto[c] + (1-momentum) * en_det[i]
                                else:
                                    self.en_proto[c]      = en_det[i].clone()
                                    self.en_proto_init[c] = True
                            for i, c in enumerate(ur_lbl):
                                if self.ur_proto_init[c]:
                                    self.ur_proto[c] = momentum * self.ur_proto[c] + (1-momentum) * ur_det[i]
                                else:
                                    self.ur_proto[c]      = ur_det[i].clone()
                                    self.ur_proto_init[c] = True

                        ur_tgt   = F.normalize(self.ur_proto[labels.cpu()], dim=1).to(dev)
                        en_tgt   = F.normalize(self.en_proto[la.cpu()],     dim=1).to(dev)
                        en_valid = torch.tensor([self.ur_proto_init[c].item() for c in en_lbl], dtype=torch.bool)
                        ur_valid = torch.tensor([self.en_proto_init[c].item() for c in ur_lbl], dtype=torch.bool)

                        cl_parts = []
                        if en_valid.any():
                            cl_parts.append((1.0 - (en_ae[en_valid] * ur_tgt[en_valid]).sum(dim=1)).mean())
                        if ur_valid.any():
                            cl_parts.append((1.0 - (ur_ae[ur_valid] * en_tgt[ur_valid]).sum(dim=1)).mean())
                        if cl_parts:
                            xl_loss = xl_loss + cl_w * torch.stack(cl_parts).mean()

                    self.opt.zero_grad(set_to_none=True)
                    xl_loss.backward()
                    self.opt.step()
                    total_loss += xl_loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(loader)

    def _masked_fop_loss(self, face, audio, labels, alpha, per_sample=True, force_mask=None, domain_label=None):
        """
        Compute masked-FOP loss.

        per_sample=True  → each sample independently draws mask from Bernoulli(dropout_p)
        per_sample=False → use force_mask for the whole batch (1=AV, 0=audio-only)
        """
        noise_std = getattr(self.config, 'noise_std', 0.0)
        if noise_std > 0.0 and self.model.training:
            audio = audio + torch.randn_like(audio) * noise_std

        if force_mask is not None:
            idx_av = torch.arange(audio.shape[0]) if force_mask == 1 else torch.tensor([], dtype=torch.long)
            idx_a  = torch.arange(audio.shape[0]) if force_mask == 0 else torch.tensor([], dtype=torch.long)
        else:
            masks  = torch.rand(audio.shape[0]) >= self.config.modality_dropout_prob
            idx_av = masks.nonzero(as_tuple=True)[0]
            idx_a  = (~masks).nonzero(as_tuple=True)[0]

        loss = None
        domain_logits_all = []

        if len(idx_av) > 1:
            out = self.model(face[idx_av], audio[idx_av], mask_face=1, labels=labels[idx_av])
            l = (self.config.loss_fusion * self.ce(out["logits"],       labels[idx_av])
               + self.config.loss_audio  * self.ce(out["audio_logits"], labels[idx_av]))
            if alpha > 0:
                l = l + alpha * self.opl(out["fused"], labels[idx_av])
            # Hybrid distillation: when face is present, pull audio_e toward fused.detach()
            if getattr(self.config, 'use_crossmodal_distill', False):
                distill_w = getattr(self.config, 'crossmodal_distill_weight', 1.0)
                audio_n = F.normalize(out["audio_e"], dim=1)
                fused_n = F.normalize(out["fused"].detach(), dim=1)
                l = l + distill_w * (1.0 - (audio_n * fused_n).sum(dim=1)).mean()
            # AV consistency: penalise fused head diverging from audio-only head
            if getattr(self.config, 'use_av_consistency', False):
                cons_w = getattr(self.config, 'av_consistency_weight', 0.5)
                log_p_av = F.log_softmax(out["logits"], dim=1)
                p_audio  = F.softmax(out["audio_logits"].detach(), dim=1)
                l = l + cons_w * F.kl_div(log_p_av, p_audio, reduction='batchmean')
            loss = l
            if out["domain_logits"] is not None:
                domain_logits_all.append(out["domain_logits"])

        if len(idx_a) > 1:
            out = self.model(face[idx_a], audio[idx_a], mask_face=0, labels=labels[idx_a])
            l = self.config.loss_audio * self.ce(out["audio_logits"], labels[idx_a])
            if alpha > 0:
                l = l + alpha * self.opl(out["audio_e"], labels[idx_a])
            loss = l if loss is None else loss + l
            if out["domain_logits"] is not None:
                domain_logits_all.append(out["domain_logits"])

        # Domain adversarial loss — pushes audio_e to be language-agnostic
        if domain_label is not None and domain_logits_all:
            d_logits = torch.cat(domain_logits_all, dim=0)
            d_labels = torch.full((d_logits.shape[0],), domain_label,
                                  dtype=torch.long, device=audio.device)
            adv_loss = getattr(self.config, "domain_adv_weight", 0.1) * F.cross_entropy(d_logits, d_labels)
            loss = adv_loss if loss is None else loss + adv_loss

        if loss is None:
            loss = torch.tensor(0.0, device=audio.device, requires_grad=True)

        return loss

    def _crossmodal_distill_loss(self, face, audio, labels, alpha, domain_label=None):
        """
        Cross-modal distillation: no masking. Forward all samples with face available.
        Audio branch is trained to mimic the fused (face+audio) embedding via cosine loss.
        Works for both English and Urdu batches — face acts as a teacher for audio_e.
        """
        out = self.model(face, audio, mask_face=1)
        audio_e = out["audio_e"]
        fused   = out["fused"]

        loss = (self.config.loss_fusion * self.ce(out["logits"],       labels)
              + self.config.loss_audio  * self.ce(out["audio_logits"], labels))
        if alpha > 0:
            loss = loss + alpha * self.opl(fused, labels)

        # Distillation: pull audio_e toward fused.detach() in cosine space
        distill_w = getattr(self.config, 'crossmodal_distill_weight', 1.0)
        audio_n   = F.normalize(audio_e, dim=1)
        fused_n   = F.normalize(fused.detach(), dim=1)
        loss = loss + distill_w * (1.0 - (audio_n * fused_n).sum(dim=1)).mean()

        if domain_label is not None and out["domain_logits"] is not None:
            d_labels = torch.full((audio_e.shape[0],), domain_label, dtype=torch.long, device=audio.device)
            loss = loss + getattr(self.config, "domain_adv_weight", 0.1) * F.cross_entropy(out["domain_logits"], d_labels)

        return loss

    def _combined_supcon_step(self, face_en, audio_en, labels_en, face_ur, audio_ur, labels_ur, alpha):
        """
        Single forward pass over EN + UR batches.
        Computes CE + OPL + domain_adv for both, plus SupCon on merged audio_e.
        Cross-lingual same-identity pairs are natural positives for SupCon.
        """
        audio_e_list, label_list = [], []
        loss = torch.tensor(0.0, device=audio_en.device)

        # --- English: per-sample modality dropout ---
        masks  = torch.rand(audio_en.shape[0]) >= self.config.modality_dropout_prob
        idx_av = masks.nonzero(as_tuple=True)[0]
        idx_a  = (~masks).nonzero(as_tuple=True)[0]

        if len(idx_av) > 0:
            out = self.model(face_en[idx_av], audio_en[idx_av], mask_face=1)
            loss = loss + self.config.loss_fusion * self.ce(out["logits"],       labels_en[idx_av])
            loss = loss + self.config.loss_audio  * self.ce(out["audio_logits"], labels_en[idx_av])
            if alpha > 0:
                loss = loss + alpha * self.opl(out["fused"], labels_en[idx_av])
            if out["domain_logits"] is not None:
                d = torch.zeros(len(idx_av), dtype=torch.long, device=audio_en.device)
                loss = loss + self.config.domain_adv_weight * F.cross_entropy(out["domain_logits"], d)
            audio_e_list.append(F.normalize(out["audio_e"], dim=1))
            label_list.append(labels_en[idx_av])

        if len(idx_a) > 0:
            out = self.model(face_en[idx_a], audio_en[idx_a], mask_face=0)
            loss = loss + self.config.loss_audio * self.ce(out["audio_logits"], labels_en[idx_a])
            if alpha > 0:
                loss = loss + alpha * self.opl(out["audio_e"], labels_en[idx_a])
            if out["domain_logits"] is not None:
                d = torch.zeros(len(idx_a), dtype=torch.long, device=audio_en.device)
                loss = loss + self.config.domain_adv_weight * F.cross_entropy(out["domain_logits"], d)
            audio_e_list.append(F.normalize(out["audio_e"], dim=1))
            label_list.append(labels_en[idx_a])

        # --- Urdu: forced audio-only ---
        out_ur = self.model(face_ur, audio_ur, mask_face=0)
        loss = loss + self.config.loss_audio * self.ce(out_ur["audio_logits"], labels_ur)
        if alpha > 0:
            loss = loss + alpha * self.opl(out_ur["audio_e"], labels_ur)
        if out_ur["domain_logits"] is not None:
            d = torch.ones(audio_ur.shape[0], dtype=torch.long, device=audio_ur.device)
            loss = loss + self.config.domain_adv_weight * F.cross_entropy(out_ur["domain_logits"], d)
        audio_e_list.append(F.normalize(out_ur["audio_e"], dim=1))
        label_list.append(labels_ur)

        # --- SupCon on merged EN + UR audio_e ---
        all_audio_e = torch.cat(audio_e_list, dim=0)
        all_labels  = torch.cat(label_list,  dim=0)
        supcon_w    = getattr(self.config, "supcon_weight", 0.1)
        loss = loss + supcon_w * self.supcon(all_audio_e, all_labels)

        return loss
