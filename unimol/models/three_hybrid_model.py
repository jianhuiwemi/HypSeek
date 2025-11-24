import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.data import Dictionary
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .lorentz import exp_map0
from . import distributed as dist_utils
from .unimol import NonLinearHead, UniMolModel, base_architecture
import numpy as np
import math


@register_model("three_hybrid_model")
class ThreeHybridModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            help="pocket encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )
        parser.add_argument("--aperture-eta", type=float, default=1.2)
        parser.add_argument("--curv-init", type=float, default=1.0, help="initial curv")
        parser.add_argument("--learn-curv", action="store_true", help="learnable")

    def __init__(self, args, mol_dictionary: Dictionary, pocket_dictionary: Dictionary):
        super().__init__()
        three_hybrid_architecture(args)

        self.args = args

        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D", use_fast=False)
        self.protein_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")

        self.logit_scale = nn.Parameter(torch.ones([1], device="cuda") * np.log(13))
        self.mol_project = NonLinearHead(
            args.mol.encoder_embed_dim, 128, "relu"
        )
        self.pocket_project = NonLinearHead(
            args.pocket.encoder_embed_dim, 128, "relu"
        )
        self.protein_project = NonLinearHead(
            self.protein_model.config.hidden_size, 128, "relu"
        )
        self.curv = nn.Parameter(torch.tensor([args.curv_init]).log(), requires_grad=args.learn_curv)
        self._curv_minmax = {
            "max": math.log(args.curv_init * 10),
            "min": math.log(args.curv_init / 10),
        }

        self.mol_alpha     = nn.Parameter(torch.tensor([128**-0.5]).log(), requires_grad=True)
        self.pocket_alpha  = nn.Parameter(torch.tensor([128**-0.5]).log(), requires_grad=True)
        self.protein_alpha = nn.Parameter(torch.tensor([128**-0.5]).log(), requires_grad=True)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary)

    def get_dist_features(self, dist, et, flag):
        if flag == "mol":
            n_node = dist.size(-1)
            gbf_feature = self.mol_model.gbf(dist, et)
            gbf_result = self.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        else:
            n_node = dist.size(-1)
            gbf_feature = self.pocket_model.gbf(dist, et)
            gbf_result = self.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias


    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        protein_sequences,
        encode=False,
        masked_tokens=None,
        features_only=True,
        is_train=True,
        **kwargs
    ):

        self.mol_alpha.data     = torch.clamp(self.mol_alpha.data, max=0.0)
        self.pocket_alpha.data  = torch.clamp(self.pocket_alpha.data, max=0.0)
        self.protein_alpha.data = torch.clamp(self.protein_alpha.data, max=0.0)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        κ = self.curv.exp()
        # ——— Mol ———
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias)
        mol_rep_eu = mol_outputs[0][:, 0, :]
        u_mol = self.mol_project(mol_rep_eu)  # [B_m, 128]
        u_mol = u_mol * self.mol_alpha.exp()  # [B_m, 128]

        # ——— Pocket ———
        poc_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        poc_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        poc_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        poc_outputs = self.pocket_model.encoder(poc_x, padding_mask=poc_padding_mask, attn_mask=poc_graph_attn_bias)
        poc_rep_eu = poc_outputs[0][:, 0, :]
        u_poc = self.pocket_project(poc_rep_eu)  # [B_c, 128]
        u_poc = u_poc * self.pocket_alpha.exp()  # [B_c, 128]

        # ——— Protein ———
        inputs = self.tokenizer(
            protein_sequences, return_tensors="pt", padding="longest", truncation=True, max_length=512
        )
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        prot_outputs = self.protein_model(**inputs, output_hidden_states=True)
        prot_hidden_states = prot_outputs.hidden_states[-1]
        prot_rep_eu = prot_hidden_states[:, 0, :]
        u_prot = self.protein_project(prot_rep_eu)  # [B_pr, 128]
        u_prot = u_prot * self.protein_alpha.exp()  # [B_pr, 128]

        with torch.autocast(u_mol.device.type, dtype=torch.float32):
            h_mol = exp_map0(u_mol, κ)   
            h_poc = exp_map0(u_poc, κ)  
            h_prot = exp_map0(u_prot, κ)
        return h_prot, h_poc, h_mol

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates

    def get_num_updates(self):
        return getattr(self, "_num_updates", 0)

    def mol_forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        **kwargs
    ):
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = self.get_dist_features(mol_src_distance, mol_src_edge_type, "mol")
        mol_outputs = self.mol_model.encoder(
            mol_x,
            padding_mask=mol_padding_mask,
            attn_mask=mol_graph_attn_bias
        )
        mol_rep_eu = mol_outputs[0][:, 0, :]  # [B_m, hidden]
        u_mol = self.mol_project(mol_rep_eu)           # [B_m, 128]
        u_mol = u_mol * self.mol_alpha.exp()           # exp(alpha)
        with torch.autocast(u_mol.device.type, dtype=torch.float32):
            h_mol = exp_map0(u_mol, self.curv.exp())   # [B_m, hyp_dim+1]
        return h_mol

    def pocket_forward(
        self,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        **kwargs
    ):
        poc_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        poc_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        poc_graph_attn_bias = self.get_dist_features(pocket_src_distance, pocket_src_edge_type, "pocket")
        poc_outputs = self.pocket_model.encoder(
            poc_x,
            padding_mask=poc_padding_mask,
            attn_mask=poc_graph_attn_bias
        )
        poc_rep_eu = poc_outputs[0][:, 0, :]  # [B_c, hidden]

        u_poc = self.pocket_project(poc_rep_eu)           # [B_c, 128]
        u_poc = u_poc * self.pocket_alpha.exp()           # exp(alpha)
        with torch.autocast(u_poc.device.type, dtype=torch.float32):
            h_poc = exp_map0(u_poc, self.curv.exp())      # [B_c, hyp_dim+1]
        return h_poc


    def protein_forward(
        self,
        protein_sequences,
        **kwargs
    ):
        inputs = self.tokenizer(
            protein_sequences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        )
        device = self.curv.device
        self.protein_model.to(device)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        prot_outputs = self.protein_model(**inputs, output_hidden_states=True)
        prot_hidden_states = prot_outputs.hidden_states[-1]
        prot_rep_eu = prot_hidden_states[:, 0, :]  
        u_prot = self.protein_project(prot_rep_eu)          
        u_prot = u_prot * self.protein_alpha.exp()          
        with torch.autocast(u_prot.device.type, dtype=torch.float32):
            h_prot = exp_map0(u_prot, self.curv.exp())    
        return h_prot


@register_model_architecture("three_hybrid_model", "three_hybrid_model")
def three_hybrid_architecture(args):

    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(args, "pocket_encoder_ffn_embed_dim", 2048)
    args.pocket.encoder_attention_heads = getattr(
        args, "pocket_encoder_attention_heads", 64
    )
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(
        args, "pocket_pooler_activation_fn", "tanh"
    )
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0
    args.curv_init = getattr(args, "curv_init", 1.0)
    args.learn_curv = getattr(args, "learn_curv", False)
    args.hbce_bounds= getattr(args, "hbce_bounds",[5.0, 7.0, 9.0])
    args.chl_r0     = getattr(args, "chl_r0",   0.5)
    args.chl_dr     = getattr(args, "chl_dr",   0.5)
    args.chl_eta0   = getattr(args, "chl_eta0", 0.7)
    args.chl_deta   = getattr(args, "chl_deta", 0.2)
    args.lambda_rad = getattr(args, "lambda_rad", 0.5)
    args.lambda_ang = getattr(args, "lambda_ang", 0.5) 
    args.gamma_chl  = getattr(args, "gamma_chl",  0.1) 
    base_architecture(args)
