# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from src.utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from src.utils import clip_parameters
from src.evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary

logger = getLogger()

class SupervisedBertTrainer(object):

    def __init__(self, bert_model, mapping, args, bert_model1=None,
                trans_types=['self_attention','attention','linear_self_attention','nonlinear_self_attention']):
        """
        Initialize trainer script.
        """
        self.transformer_types = trans_types
        self.args = args
        self.bert_model = bert_model
        self.bert_model1 = bert_model1
        self.mapping = mapping

        if self.args.local_rank == -1 or self.args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        else:
            self.device = torch.device("cuda", self.args.local_rank)

        # optimizers
        if hasattr(args, 'map_optimizer'):
            optim_fn, optim_args = get_optimizer(args.map_optimizer)
            if self.args.map_type == 'fine_tune':
                self.map_optimizer = optim_fn(bert_model.parameters(), **optim_args)
            else:
                self.map_optimizer = optim_fn(mapping.parameters(), **optim_args)

        # best validation score
        self.best_valid_metric = -1e12
        self.decrease_lr = False

    def supervised_mapping_step(self, src_emb, tgt_emb, margin=1, eval_only=False):
        """
        Calculate the loss and backward
        Inputs:
            src_emb [unmasked_len, output_dim] mapped source embeddings
            tgt_emb [unmasked_len, output_dim] target embeddings
        Outputs:
            avg_cos_sim/loss
        """

        # normalization
        if self.args.normalize_embed:
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        # (n, n)
        #scores = src_emb.mm(tgt_emb.transpose(0, 1))
        #rang = torch.arange(scores.shape[0], out=torch.LongTensor())
        #gold_scores = scores[rang, rang]
        
        if self.args.loss == 'cos_sim':
            # (n)
            gold_scores = (src_emb * tgt_emb).sum(1)
            # maximize cosine similarities
            loss = - gold_scores.mean()
        elif self.args.loss == 'l2_dist':
            # (n, d)
            sub = src_emb - tgt_emb
            # (n, d) => (n) => ()
            loss = (sub * sub).sum(1).mean()
        elif self.args.loss.startswith('max_margin_top'):
            # (n)
            gold_scores = (src_emb * tgt_emb).sum(1)
            # (n, n)
            scores = src_emb.mm(tgt_emb.transpose(0, 1))
            # max margin with top k elements
            k = int(self.args.loss.split('-')[1])
            # (n, k)
            top_vals, top_ids = scores.topk(k, 1, True)
            # (n) => (n, k)
            gold_vals = gold_scores.unsqueeze(1).expand_as(top_vals)
            # (n, k)
            margins = torch.ones_like(top_vals) * margin
            # (n, k)
            losses = margins - gold_vals + top_vals
            # mask out less than 0
            losses = torch.where(losses>0, losses, torch.zeros_like(losses))
            # ()
            loss = losses.mean()

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (supervised learning)")
            exit()

        # calculating average cosine similarity
        if not self.args.normalize_embed:
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        # (n)
        avg_cos_sim = (src_emb * tgt_emb).sum(1).mean()

        if eval_only:
            return avg_cos_sim, loss
        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()

        return avg_cos_sim, loss

    def get_unmasked_bert(self, input_ids, input_mask, bert_layer=-1, model_id=0):
        """
        Get BERT
        """
        with torch.no_grad():
            if model_id == 0 or self.bert_model1 is None:
                self.bert_model.eval()
                all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask)
            else:
                self.bert_model1.eval()
                all_encoder_layers, _ = self.bert_model1(input_ids, token_type_ids=None, attention_mask=input_mask)
            encoder_layer = all_encoder_layers[bert_layer]
        # [batch_size, seq_len, output_dim]
        return encoder_layer

    def get_trainable_unmasked_bert(self, input_ids, input_mask, bert_layer=-1, model_id=0):
        """
        Get BERT
        """
        if model_id == 0 or self.bert_model1 is None:
            all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask)
        else:
            all_encoder_layers, _ = self.bert_model1(input_ids, token_type_ids=None, attention_mask=input_mask)
        encoder_layer = all_encoder_layers[bert_layer]
        # [batch_size, seq_len, output_dim]
        return encoder_layer

    def select(self, layer, mask):
        """
        Select all unmasked embed in this batch
        Inputs:
            layer [batch_size, seq_len, output_dim]
            mask [batch_size, seq_len] of 0/1
        Outputs:
            masked_layer [unmasked_len, output_dim]
        """
        batch_size, seq_len, output_dim = list(layer.size())
        # [batch_size, seq_len, output_dim] => [unmasked_len, output_dim]
        return layer.masked_select(mask.byte().view(batch_size, seq_len, 1).expand(-1, -1, output_dim)).view(-1,output_dim)

    def rearange(self, layer, index):
        """
        Rearange layer by index
        Inputs: 
            layer [batch_size, seq_len, output_dim]
            index [batch_size, max_align]
        Outputs:
            rearanged_layer [batch_size, max_align, output_dim]
        """
        batch_size, seq_len, output_dim = list(layer.size())
        batch_size, max_align = list(index.size())
        #[batch_size, max_align]=>[batch_size, max_align, output_dim]
        expanded_index = index.view(batch_size, max_align, 1).expand(-1,-1,output_dim)
        #[batch_size, max_align, output_dim]
        rearanged_layer = layer.gather(1, expanded_index)
        return rearanged_layer

    def get_indexed_mapped_bert(self, input_ids, input_mask, index, align_mask, bert_layer=-1, model_id=0):
        """
        Get bert according to index and align_mask
        """
        if self.args.map_type == 'fine_tune':
            unmasked_bert = self.get_trainable_unmasked_bert(input_ids, input_mask, bert_layer, model_id)
        else:
            unmasked_bert = self.get_unmasked_bert(input_ids, input_mask, bert_layer, model_id)
        if self.args.map_type in self.transformer_types:
            mapped_bert = self.mapping(unmasked_bert, input_mask)
        elif self.args.map_type == 'fine_tune':
            mapped_bert = unmasked_bert
        else:
            mapped_bert = self.mapping(unmasked_bert)
        rearanged_bert = self.rearange(mapped_bert, index)
        indexed_bert = self.select(rearanged_bert, align_mask)
        #print (unmasked_bert, '\n', mapped_bert, '\n', rearanged_bert, '\n', indexed_bert)
        return indexed_bert

    def get_indexed_mapped_bert_from_bert(self, unmasked_bert, input_mask, index, align_mask, bert_layer=-1):
        """
        Get bert according to index and align_mask
        """
        if self.args.map_type in self.transformer_types:
            mapped_bert = self.mapping(unmasked_bert, input_mask)
        elif self.args.map_type == 'fine_tune':
            mapped_bert = unmasked_bert
        else:
            if str(self.device) == 'cpu':
              print ("Transfering to gpu")
              unmasked_bert = unmasked_bert.cuda()
              self.mapping = self.mapping.cuda()
            mapped_bert = self.mapping(unmasked_bert)
            if str(self.device) == 'cpu':
              print ("Trnasfering back to cpu")
              mapped_bert = mapped_bert.cpu()
        rearanged_bert = self.rearange(mapped_bert, index)
        indexed_bert = self.select(rearanged_bert, align_mask)
        #print (unmasked_bert, '\n', mapped_bert, '\n', rearanged_bert, '\n', indexed_bert)
        return indexed_bert

    def get_indexed_bert(self, input_ids, input_mask, index, align_mask, bert_layer=-1, model_id=1):
        """
        Get bert according to index and align_mask
        """
        unmasked_bert = self.get_unmasked_bert(input_ids, input_mask, bert_layer, model_id)
        rearanged_bert = self.rearange(unmasked_bert, index)
        indexed_bert = self.select(rearanged_bert, align_mask)
        return indexed_bert

    def get_indexed_bert_from_bert(self, unmasked_bert, index, align_mask, bert_layer=-1):
        """
        Get bert according to index and align_mask
        """
        rearanged_bert = self.rearange(unmasked_bert, index)
        indexed_bert = self.select(rearanged_bert, align_mask)
        return indexed_bert

    def procrustes(self, src_embs, tgt_embs):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        Input:
            src_embs/tgt_embs: [vocab_size, emb_dim]
        """
        A = src_embs
        B = tgt_embs
        if isinstance(self.mapping, torch.nn.DataParallel):
            W = self.mapping.module.weight.data
        else:
            W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))
        logger.info("Finished Procrustes.")

    def decay_map_lr(self):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.args.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.args.min_lr, old_lr * self.args.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing Mapping learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            path = os.path.join(self.args.model_path, 'best_mapping.pkl')
            self.save_model(path)

    def save_model(self, path):
        """
        Save model to path.
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if self.args.map_type == 'fine_tune':
            logger.info('* Saving the src BERT model to %s ...' % path)
            if isinstance(self.bert_model, torch.nn.DataParallel):
                torch.save(self.bert_model.module.state_dict(), path)
            else:
                torch.save(self.bert_model.state_dict(), path)
            return

        logger.info('* Saving the mapping to %s ...' % path)
        if isinstance(self.mapping, torch.nn.DataParallel):
            torch.save(self.mapping.module.state_dict(), path)
        else:
            torch.save(self.mapping.state_dict(), path)

    def load_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.args.model_path, 'best_mapping.pkl')
        logger.info('* Loading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        self.load_model(path)

    def load_model(self, path):
        """
        load model from path
        """
        if isinstance(self.mapping, torch.nn.DataParallel):
            self.mapping.module.load_state_dict(
                torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.mapping.load_state_dict(
                torch.load(path, map_location=lambda storage, loc: storage))
        return True
