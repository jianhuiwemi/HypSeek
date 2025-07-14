# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F
from unicore.losses import UnicoreLoss, register_loss
from unicore import metrics
from . import lorentz as L
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import random


def calculate_bedroc(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """

    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    return bedroc


@register_loss("three_hybrid_loss")
class ThreeHybridLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        args = task.args
        self.eta        = float(args.aperture_eta)
        self.gamma_ce   = float(args.gamma_ce)
        self.alpha_poc  = float(getattr(args, "alpha_poc", 1.0))
        self.alpha_prot = float(getattr(args, "alpha_prot", 1.0))
        self.chl_r0       = float(args.chl_r0)          
        self.chl_dr       = float(args.chl_dr)          
        self.chl_eta0     = float(args.chl_eta0)         
        self.chl_deta     = float(args.chl_deta)       
        self.lambda_rad   = float(args.lambda_rad)       
        self.lambda_ang   = float(args.lambda_ang)      
        self.gamma_chl    = float(args.gamma_chl)       
        self.bounds       = torch.tensor(args.hbce_bounds, dtype=torch.float32)
        self.lambda_ham = float(getattr(args, "lambda_ham", 0.10))
        self.lambda_her = float(getattr(args, "lambda_her", 0.10))

    def forward(self, model, sample, reduce=True, fix_encoder=False):

        h_prot, h_poc, h_mol = model(
            **sample["pocket"]["net_input"],
            **sample["lig"]["net_input"],
            protein_sequences=sample["protein"],
            features_only=True,
            fix_encoder=fix_encoder,
            is_train=self.training,
        )
        B           = h_mol.size(0)
        κ           = model.curv.exp().detach()
        logit_scale = model.logit_scale.exp().detach()


        # ====== H² Cone-Hierarchy Loss ======
        poc_space = h_poc[:, 1:]     
        lig_space = h_mol[:, 1:]

        poc_idx = []
        for i, (s, e) in enumerate(sample["batch_list"]):
            poc_idx += [i] * (e - s)
        poc_idx = torch.tensor(poc_idx, device=h_poc.device)

        poc_sel = poc_space[poc_idx]         

        dist_mat = L.pairwise_dist(poc_sel, lig_space, curv=κ) 
        dist     = dist_mat.diagonal()                        
        phi   = L.oxy_angle(lig_space, poc_space[poc_idx], curv=κ)      
        omega = L.half_aperture(poc_space[poc_idx], curv=κ)          

        act_flat = torch.tensor([x for sub in sample["act_list"] for x in sub],
                                device=h_poc.device, dtype=torch.float32)
        bounds  = self.bounds.to(h_poc.device)
        bucket  = torch.bucketize(act_flat, bounds)                      
        r_k     = self.chl_r0 + bucket.float() * self.chl_dr
        eta_k   = self.chl_eta0 - bucket.float() * self.chl_deta

        Nl    = dist.size(0)
        L_rad = F.relu(dist - r_k).sum() / math.sqrt(Nl)
        L_ang = F.relu(phi  - eta_k * omega).sum() / math.sqrt(Nl)
        loss_chl = self.lambda_rad * L_rad + self.lambda_ang * L_ang

        # =========  Hyperbolic Regularizers ========= #

        m_margin = 0.15                
        loss_ham = F.relu(phi - eta_k*omega + m_margin).sum() / math.sqrt(Nl)


        loss_her = torch.zeros(1, device=device)
        cnt_her  = 0
        β = 80.5
        offset = 0
        for i_poc, (s, e) in enumerate(sample["batch_list"]):
            L_i = e - s
            if L_i < 1:
                continue

            d_i  = dist[offset:offset + L_i].detach()  
            rank = (d_i.unsqueeze(0) < d_i.unsqueeze(1)).float().sum(1) + 1
            w    = torch.exp(-β * (rank - 1) / L_i)     

            logits_row = torch.matmul(poc_space[i_poc:i_poc+1], lig_space.T) * logit_scale
            row_probs  = F.softmax(logits_row[0, s:e], dim=-1)  

            pos_mask   = act_flat[offset:offset + L_i] < 5    
            if pos_mask.any():
                loss_her += -(w[pos_mask] * row_probs[pos_mask].log()).sum() / (w[pos_mask].sum()+1e-9)
                cnt_her  += 1
            offset += L_i
        loss_her = loss_her / max(cnt_her, 1)

        loss_reg = self.lambda_ham * loss_ham + self.lambda_her * loss_her

        loss_dict_poc = self.compute_hcc_pair_official_style(
            emb_poc=h_poc,
            emb_mol=h_mol,
            batch_list=sample["batch_list"],
            act_list=sample["act_list"],
            uniprot_poc=sample.get("uniprot_list"),
            uniprot_mol=sample.get("lig_uniprot_list"),
            pocket_lig_smiles=sample.get("pocket_lig_smiles"),
            lig_smiles=sample["lig"]["smi_name"],
            logit_scale=logit_scale,
        )

        loss_dict_prot = self.compute_hcc_pair_official_style(
            emb_poc=h_prot,             
            emb_mol=h_mol,
            batch_list=sample["batch_list"],
            act_list=sample["act_list"],
            uniprot_poc=sample.get("uniprot_list"),    
            uniprot_mol=sample.get("lig_uniprot_list"),
            pocket_lig_smiles=sample.get("pocket_lig_smiles"), 
            lig_smiles=sample["lig"]["smi_name"],
            logit_scale=logit_scale,
        )

        loss_hcc     = self.alpha_poc * loss_dict_poc["loss"] + self.alpha_prot * loss_dict_prot["loss"]
        total_loss = loss_hcc + self.gamma_chl * loss_chl + loss_reg

        if self.training:
            logging_output = {
                "loss":             total_loss.data,
                "sample_size":      B,
                "loss_ham":  loss_ham.item(),
                "loss_her":  loss_her.item(),
                "loss_chl":         loss_chl.item(),
                "loss_hcc_poc":     loss_dict_poc["loss"].item(),
                "loss_poc_pocket":  loss_dict_poc["loss_pocket"].item(),
                "loss_poc_mol":     loss_dict_poc["loss_mol"].item(),
                "loss_poc_rank":    loss_dict_poc["loss_rank"].item(),
                "loss_hcc_prot":    loss_dict_prot["loss"].item(),
                "loss_prot_pocket": loss_dict_prot["loss_pocket"].item(),
                "loss_prot_mol":    loss_dict_prot["loss_mol"].item(),
                "loss_prot_rank":   loss_dict_prot["loss_rank"].item(),
            }
        else:
            sim_masked = loss_dict_poc["sim_masked"].detach()  
            sample_size = B
            targets     = torch.arange(sample_size, dtype=torch.long, device=sim_masked.device)
            probs       = F.softmax(sim_masked[:, :sample_size].float(), dim=-1)
            logging_output = {
                "loss":        torch.tensor(0., device=sim_masked.device),
                "prob":        probs.data,
                "target":      targets,
                "smi_name":    sample["lig"]["smi_name"],
                "sample_size": sample_size,
            }

        return total_loss, B, logging_output

    def compute_hcc_pair_official_style(
        self,
        emb_poc, emb_mol,
        batch_list, act_list,
        uniprot_poc, uniprot_mol,
        pocket_lig_smiles, lig_smiles,
        logit_scale,
    ):

        B = emb_poc.size(0)

        logits = torch.matmul(emb_poc[:, 1:], emb_mol[:, 1:].T) * logit_scale 

        mask = torch.zeros_like(logits, dtype=torch.bool)
        if uniprot_poc is not None and uniprot_mol is not None:
            for i in range(B):
                for j in range(B):
                    if uniprot_poc[i] == uniprot_mol[j]:
                        mask[i, j] = True
        if pocket_lig_smiles is not None:
            for i in range(B):
                bad = pocket_lig_smiles[i]
                for j in range(B):
                    if lig_smiles[j] in bad:
                        mask[i, j] = True

        minus_inf  = torch.finfo(logits.dtype).min
        sim_masked = logits.masked_fill(mask, minus_inf)  

        loss_mol_list, loss_rank_list = [], []
        for i in range(B):
            s, e   = batch_list[i]
            acts   = act_list[i]
            L_i    = e - s
            out_i  = sim_masked[i, s:e]  

            for k in range(s, e):
                row_mask = torch.full_like(sim_masked[i], minus_inf)
                row_mask[k] = 0
                lprobs   = F.log_softmax(row_mask + sim_masked[i], dim=-1)
                if L_i > 1 and acts[k - s] < 5:
                    continue
                loss_mol_list.append(-lprobs[k] / math.sqrt(L_i))


            if L_i > 2:
                for k_rel in range(L_i - 1):
                    m = torch.zeros_like(out_i)
                    for idx in range(L_i):
                        if idx == k_rel: continue
                        if acts[k_rel] - math.log10(3) <= acts[idx]:
                            m[idx] = minus_inf
                    lprobs_rank = F.log_softmax(m + out_i, dim=-1)
                    loss_rank_list.append(-lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i)))

        loss_mol  = torch.stack(loss_mol_list).sum()   if loss_mol_list  else torch.tensor(0., device=logits.device)
        loss_rank = torch.stack(loss_rank_list).sum()  if loss_rank_list else torch.tensor(0., device=logits.device)

        net_T            = sim_masked.T               
        lprobs_pocket_all= F.log_softmax(net_T, dim=-1) 

        idx2poc = []
        for i, (s, e) in enumerate(batch_list):
            idx2poc += [i] * (e - s)
        targets = torch.tensor(idx2poc, dtype=torch.long, device=logits.device) 

        loss_pocket_list = []
        for i, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i == 0:
                continue
            rows      = list(range(s, e))
            lprobs_sub = lprobs_pocket_all[rows] 
            targ_sub   = targets[rows]           
            loss_tmp   = F.nll_loss(lprobs_sub, targ_sub, reduction="none") 
            loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))

        loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0., device=logits.device)

        total = loss_pocket + loss_mol + loss_rank

        return {
            "loss":        total,
            "loss_pocket": loss_pocket,
            "loss_mol":    loss_mol,
            "loss_rank":   loss_rank,
            "sim_masked":  sim_masked,
        }

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid", args=None):
        loss_sum    = sum(log.get("loss", 0)        for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if sample_size == 0:
            return

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)

        if "train" in split:

            val_hcl = sum(log.get("loss_chl", 0) for log in logging_outputs)
            if val_hcl != 0:
                metrics.log_scalar("loss_chl", val_hcl / sample_size, sample_size, round=3)
            
            for key in ["loss_ham", "loss_her"]:
                val = sum(log.get(key, 0) for log in logging_outputs)
                if val != 0:
                    metrics.log_scalar(key, val / sample_size, sample_size, round=3)

            val_hcc_poc = sum(log.get("loss_hcc_poc", 0) for log in logging_outputs)
            if val_hcc_poc != 0:
                metrics.log_scalar("loss_hcc_poc", val_hcc_poc / sample_size, sample_size, round=3)

            for key in ["loss_poc_pocket", "loss_poc_mol", "loss_poc_rank"]:
                val = sum(log.get(key, 0) for log in logging_outputs)
                if val != 0:
                    metrics.log_scalar(key, val / sample_size, sample_size, round=3)

            val_hcc_prot = sum(log.get("loss_hcc_prot", 0) for log in logging_outputs)
            if val_hcc_prot != 0:
                metrics.log_scalar("loss_hcc_prot", val_hcc_prot / sample_size, sample_size, round=3)

            for key in ["loss_prot_pocket", "loss_prot_mol", "loss_prot_rank"]:
                val = sum(log.get(key, 0) for log in logging_outputs)
                if val != 0:
                    metrics.log_scalar(key, val / sample_size, sample_size, round=3)

            return

        valid_set    = args.valid_set
        split_method = getattr(args, "split_method", "")

        if valid_set in ["FEP", "TIME", "TYK2", "OOD", "DEMO"]:
            return


        acc_sum = 0
        prob_list, tgt_list = [], []
        for log in logging_outputs:
            prob = log.get("prob"); tgt = log.get("target")
            if prob is None or tgt is None: continue
            acc_sum += (prob.argmax(dim=-1) == tgt).sum().item()
            prob_list.append(prob); tgt_list.append(tgt)

        if len(prob_list) == 0:
            return
        probs   = torch.cat(prob_list, dim=0)
        targets = torch.cat(tgt_list, dim=0)

        metrics.log_scalar(f"{split}_acc", acc_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("valid_neg_loss", - (loss_sum / sample_size) / math.log(2), sample_size, round=3)


        split_method = args.split_method
        if "train" in split:
            for key in ["loss_mol", "loss_pocket", "loss_rank"]:
                loss_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key, loss_sum / sample_size, sample_size, round=3
                )
        elif valid_set in ["FEP", "TIME", "TYK2", "OOD", "DEMO"]:
            corrs = []
            pearsons = []
            r2s = []
            res_dict = {}
            info_dict = {}
            for log in logging_outputs:
                logit_output = log["logit_output"].detach().cpu().numpy()
                true_act = log["act_list"]
                lig_smi = log["smi_name"]
                for i, (assay_id, span, acts) in enumerate(zip(log["assay_id_list"], log["batch_list"], true_act)):
                    acts = np.array(acts)
                    if len(acts) >= 3:
                        pred_score = logit_output[i, span[0]:span[1]]
                        corr = stats.spearmanr(acts, pred_score).statistic
                        pearson = stats.pearsonr(acts, pred_score).statistic
                        if math.isnan(corr):
                            corr = 0.
                        if math.isnan(pearson):
                            pearson = 0.
                        assay_smi = lig_smi[span[0]:span[1]]
                        res_dict[assay_id] = {
                            "assay_id": assay_id,
                            "pred": [round(x, 3) for x in pred_score.tolist()],
                            "exp": [round(x, 3) for x in acts.tolist()],
                            "spearmanr": corr,
                            "pearson": pearson
                        }
                        info_dict[assay_id] = {
                            "assay_id": assay_id,
                            "smiles": assay_smi,
                        }
                        corrs.append(corr)
                        pearsons.append(pearson)
                        r2s.append(max(pearson, 0) ** 2)
                        # print(pearson, len(acts))

            metrics.log_scalar(f"{split}_mean_corr", np.mean(corrs), sample_size, round=3)
            metrics.log_scalar(f"{split}_mean_pearson", np.mean(pearsons), sample_size, round=3)
            metrics.log_scalar(f"{split}_mean_r2", np.mean(r2s), sample_size, round=3)
            sup_num = float(args.sup_num)
            if args.sup_num > 1:
                sup_num = int(args.sup_num)

            import os
            rank = int(os.environ["LOCAL_RANK"])
            if rank == 0 and args.few_shot:
                if args.results_path.endswith(".jsonl"):
                    write_file = args.results_path
                else:
                    write_file = f"{args.results_path}/{split_method}_{args.seed}_sup{sup_num}.jsonl"
                    if args.active_learning_resfile != "":
                        write_file = f"{args.results_path}/{args.active_learning_resfile}"
                import os
                if not os.path.exists(write_file):
                    with open(write_file, "a") as f:
                        f.write(json.dumps(info_dict) + "\n")
                with open(write_file, "a") as f:
                    f.write(json.dumps(res_dict) + "\n")
                print(f"saving to {write_file}")

        else:
            acc_sum = sum(sum(log.get("prob").argmax(dim=-1) == log.get("target")) for log in logging_outputs)

            prob_list = []
            if len(logging_outputs) == 1:
                prob_list.append(logging_outputs[0].get("prob"))
            else:
                for i in range(len(logging_outputs) - 1):
                    prob_list.append(logging_outputs[i].get("prob"))
            probs = torch.cat(prob_list, dim=0)

            metrics.log_scalar(f"{split}_acc", acc_sum / sample_size, sample_size, round=3)
            metrics.log_scalar("valid_neg_loss", -loss_sum / sample_size / math.log(2), sample_size, round=3)
            targets = torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
            # print(targets.shape, probs.shape)

            targets = targets[:len(probs)]
            bedroc_list = []
            auc_list = []
            for i in range(len(probs)):
                prob = probs[i]
                target = targets[i]
                label = torch.zeros_like(prob)
                label[target] = 1.0
                cur_auc = roc_auc_score(label.cpu(), prob.cpu())
                auc_list.append(cur_auc)
                bedroc = calculate_bedroc(label.cpu(), prob.cpu(), 80.5)
                bedroc_list.append(bedroc)
            bedroc = np.mean(bedroc_list)
            auc = np.mean(auc_list)

            top_k_acc = top_k_accuracy_score(targets.cpu(), probs.cpu(), k=3, normalize=True)
            metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
            metrics.log_scalar(f"{split}_bedroc", bedroc, sample_size, round=3)
            metrics.log_scalar(f"{split}_top3_acc", top_k_acc, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train):
        return is_train
