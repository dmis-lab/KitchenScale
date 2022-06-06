import os
import json
import logging
from unicodedata import bidirectional
from pytorch_lightning.utilities.distributed import sync_ddp_if_available

import torch
from torch.nn.modules import dropout
import transformers
import torchmetrics
import pytorch_lightning as pl

import food_utils
from rich import print as rprint
import food_utils

from math_utils import fman, fexp, fexp_embed
from math_utils import log_normal, log_truncate, truncated_normal

from rich.traceback import install as rtinstall
rtinstall(show_locals=False)
log = logging.getLogger('food_v3:model')


def _print_value_log(log_func, head, emoji, value=None):
    if value is not None:
        log_func(f'{emoji} {head:30s}:{value:20.3f}', extra={'markup':True})
    else:
        log_func(f'{emoji} {head:30s}', extra={'markup':True})

class Encoder(torch.nn.Module):
    def __init__(self, unit_dim, 
        exp_ver,
        is_include_serving,
        is_serving_concat,
        is_include_ing_phrase,
        is_include_dimension,
        is_include_title,
        is_include_other_ing,
        is_include_tags,
        semantic_encoder_model,
        lm_tokenizer, 
        last_hidden_dim=768, max_ing_size = 13,
        is_serving_multiply=False,
        data_processing_ver='lm', # lm, lm-embed,  w2v
        gru_bidirectional=False,

    ):
        super().__init__()
        self.data_processing_ver = data_processing_ver
        self.tags_dim = len(food_utils.tags)

        # for weight_amount
        self.unit_dim_dim = 2 # [amount, weight]
        self.exp_ver=exp_ver
        self.is_serving_multiply=is_serving_multiply

        self.is_include_ing_phrase  = is_include_ing_phrase
        self.is_include_serving = is_include_serving
        self.is_serving_concat = is_serving_concat
        self.is_include_dimension = is_include_dimension
        self.is_include_title = is_include_title
        self.is_include_other_ing= is_include_other_ing
        self.is_include_tags= is_include_tags
        self.semantic_encoder_model=semantic_encoder_model
        self.gru_bidirectional=gru_bidirectional
        
        self.lm_tokenizer = lm_tokenizer

        self.temp_lm_encoder_output = None
        self.temp_lm_input = None
        if self.data_processing_ver == 'lm' :
            if semantic_encoder_model == 'bert' or semantic_encoder_model == 'bert-freeze':
                self.lm_encoder = transformers.AutoModel.from_pretrained('bert-base-uncased')
            elif semantic_encoder_model == 'bert-no-pre':
                bert_config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
                self.lm_encoder = transformers.AutoModel.from_config(bert_config)
            self.lm_encoder.resize_token_embeddings(len(self.lm_tokenizer))
        elif self.data_processing_ver == 'lm-embed':
            raise NotImplementedError
        elif self.data_processing_ver == 'w2v':
            self.w2v_encoder = torch.nn.GRU(input_size=50, hidden_size=20, num_layers=3, bidirectional=self.gru_bidirectional)  # todo : 50 should be from args
        else:
            raise NotImplementedError
            # w2v shape - batch * tokens_len * encode_dim 
            # todo : bigru
            # todo : transformers


        self.tags_encoder = torch.nn.Embedding(self.tags_dim, last_hidden_dim)
        self.last_hidden_dim = last_hidden_dim

        if self.is_include_serving and self.is_serving_concat:
            self.last_hidden_dim += 1 
        # if self.is_include_dimension:
            # self.last_hidden_dim += 1 


        self.max_ing_size = max_ing_size 

        self.special_tokens = food_utils.get_additonal_tokens()
        if self.data_processing_ver == 'lm' or self.data_processing_ver == 'lm-embed':
            self.special_token_dict = {
                k:v for k, v in zip(self.special_tokens, self.lm_tokenizer.convert_tokens_to_ids(self.special_tokens))
            }
        self.temp_encoded = None
        
        

    def forward(self, x):
        if 'lm' in self.data_processing_ver:
            _lm_input = {
                'input_ids' : x['input_ids'],
                'token_type_ids' : x['token_type_ids'],
                'attention_mask' : x['attention_mask'],
            }
            self.temp_lm_input = _lm_input
            if self.semantic_encoder_model == 'bert-freeze':
                with torch.no_grad():
                    encoded_x = self.lm_encoder(**_lm_input)
            else:
                encoded_x = self.lm_encoder(**_lm_input)
                # from IPython import embed; embed();exit(1)
            self.temp_lm_encoder_output = encoded_x
            
            encoded = encoded_x.last_hidden_state[:,0,:]


            inputs = [encoded]

            if self.is_include_serving and self.is_serving_concat:
                inputs.append(x['servings'].float().unsqueeze(1))

            # if self.is_include_dimension:
                # inputs.append(x['target_dim_num'].float().unsqueeze(1))
                
            # if self.is_include_unit:
                # inputs.append(x['target_unit_num'].float().unsqueeze(1))

            encoded = torch.cat(inputs, dim=1)
            self.temp_encoded = encoded
        elif self.data_processing_ver == 'w2v':
            encoded = self.w2v_encoder(x['tensors'])[0]            # output, hidden
            encoded = encoded[:,-1,:]
        else:
            raise ValueError(f'{self.data_processing_ver} is not a proper value')

        if self.is_serving_multiply:
            encoded = torch.mul(encoded, x['servings'].float().unsqueeze(1))
        return encoded 

class ScaleLayer(torch.nn.Module):
   def __init__(self):
       super().__init__()
       scale = torch.tensor([.9])
       shift = torch.tensor([.1])
       self.register_buffer('scale', scale)
       self.register_buffer('shift', shift)

   def forward(self, input):
       return (input * self.scale)+self.shift


class BackBone(torch.nn.Module):
    def __init__(self, 
        is_e_predict, is_q_predict, 
        is_u_predict,
        n_exponent, 
        u_num,unit_dim,  regression_layer,
        exp_ver,
        is_include_ing_phrase,
        is_include_serving,
        is_serving_concat,
        is_include_dimension,
        is_include_title,
        is_include_other_ing,
        is_include_tags,
        semantic_encoder_model,
        lm_tokenizer,
        last_hidden_dim=768, min_e=None, d_num=2,
        q_normalize='none',
        is_serving_multiply=False,
        data_processing_ver='lm',
        gru_bidirectional=False,
    ):
        super().__init__()
        self.data_processing_ver = data_processing_ver
        self.exp_ver = exp_ver
        self.is_include_ing_phrase = is_include_ing_phrase
        self.is_include_serving = is_include_serving
        self.is_serving_concat = is_serving_concat
        self.is_include_dimension = is_include_dimension
        self.is_include_title = is_include_title
        self.is_include_other_ing = is_include_other_ing
        self.is_include_tags=is_include_tags
        self.semantic_encoder_model = semantic_encoder_model

        self.min_e = min_e
        self.is_e_predict = is_e_predict
        self.is_q_predict = is_q_predict
        self.is_u_predict = is_u_predict
        self.n_exponent = n_exponent
        self.max_e = n_exponent + min_e - 1 

        self.is_serving_multiply=is_serving_multiply

        self.u_num = u_num
        self.d_num = d_num
        self.unit_dim = unit_dim
        self.regression_layer = regression_layer
        self.lm_tokenizer = lm_tokenizer
        self.q_normalize=q_normalize
        self.gru_bidirectional=gru_bidirectional
        self.encoder = Encoder(
            exp_ver=self.exp_ver,
            is_include_ing_phrase=self.is_include_ing_phrase,
            is_include_serving=self.is_include_serving,
            is_serving_concat = self.is_serving_concat,
            is_include_dimension=self.is_include_dimension,
            is_include_title=self.is_include_title,
            is_include_other_ing=self.is_include_other_ing,
            is_include_tags=self.is_include_tags,
            semantic_encoder_model = self.semantic_encoder_model,
            lm_tokenizer=lm_tokenizer,
            unit_dim=unit_dim,
            last_hidden_dim=last_hidden_dim,
            is_serving_multiply=self.is_serving_multiply,
            data_processing_ver=self.data_processing_ver,
            gru_bidirectional=self.gru_bidirectional,
        )
        encoder_last_hidden_dim = self.encoder.last_hidden_dim
        if is_q_predict and is_e_predict :
            # comes from mnm exponent_bert
            self.exponent_logsoftmax = torch.nn.LogSoftmax()
            self.mlp_exponent = torch.nn.Linear(encoder_last_hidden_dim, self.n_exponent) 
            self.scale = ScaleLayer()
            self.mlp_mean = torch.nn.Sequential(
                torch.nn.Linear(encoder_last_hidden_dim, 64), 
                torch.nn.Sigmoid(),
                torch.nn.Linear(64, self.n_exponent),
                torch.nn.Sigmoid(), 
                self.scale
            )
            self.mlp_logvar = torch.nn.Sequential(
                torch.nn.Linear(encoder_last_hidden_dim, self.n_exponent),
                torch.nn.Sigmoid()
            )
            self.logvar_scale = -5.0
            self.do_logvar = False
            self.output_embed_exp = False

            if self.output_embed_exp:
                exp_hidden_size = 128
                self.mlp_hid_combine_exp = torch.nn.Sequential(
                    torch.nn.Linear(encoder_last_hidden_dim+exp_hidden_size,last_hidden_dim), 
                    torch.nn.ELU()
                )
                self.mlp_output_exponent_embeddings = torch.nn.Embedding(self.n_exponent, exp_hidden_size)
                if self.zero_init:
                    self.mlp_output_exponent_embeddings.weight.data.zero_()

            self.zero_init = False

            
            # self.mlp_mean = torch.nn.Sequential(
            # torch.nn.Linear(last_hidden_dim, 64), torch.nn.Sigmoid(),
            #     torch.nn.Linear(64, self.n_exponent),torch.nn.Sigmoid(), self.scale)
            # self.e_classifier = torch.nn.Linear(last_hidden_dim, self.n_exponent)
            # self.regressor = torch.nn.Linear(last_hidden_dim + self.n_exponent, 1)
        elif is_q_predict:
            # self.regressor = torch.nn.Sequential(
            #     torch.nn.Linear(last_hidden_dim , 256),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(256, 128),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(128, 1),
            #     torch.nn.ReLU(),
            # )
            if self.regression_layer.startswith('single'):
                self.regressor = torch.nn.Linear(encoder_last_hidden_dim , 1)
            elif self.regression_layer.startswith('3mlp'):
                self.regressor = torch.nn.Sequential(
                    torch.nn.Linear(encoder_last_hidden_dim , 256),
                    # torch.nn.ReLU(),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(256, 128),
                    torch.nn.Sigmoid(),
                    # torch.nn.ReLU(),
                    torch.nn.Linear(128, 1),
                )
            if self.regression_layer.endswith('-sigmoid'):
                self.last_activation_layer=torch.nn.Sigmoid()
            elif self.regression_layer.endswith('-relu'):
                self.last_activation_layer=torch.nn.ReLU()
            # self.regressor = torch.nn.Linear(self.unit_dim, 1)
        elif is_e_predict:
            self.e_classifier = torch.nn.Linear(encoder_last_hidden_dim, self.n_exponent)
        elif is_u_predict :
            self.u_classifier = torch.nn.Linear(encoder_last_hidden_dim, self.u_num )
        elif exp_ver == 'dimension':
            self.d_classifier = torch.nn.Linear(encoder_last_hidden_dim, self.d_num )
        else:
            raise ValueError('Invalid Config')
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        # target = self.encoder.lm_encoder.encoder.layer[11]
        # target.attention.forward = target.attention.forward_return
        x = self.encoder(x)
        # from IPython import embed; embed(); exit(1)
        
        if self.is_e_predict and self.is_q_predict:
            if self.output_embed_exp:
                raise NotImplementedError
            exponent_prediction = self.mlp_exponent(x)
            exponent_logprobs = self.exponent_logsoftmax(exponent_prediction)
            mean_prediction = self.mlp_mean(x)
            if self.do_logvar:
                logvar_prediction = self.mlp_logvar(x) * self.logvar_scale
            else:
                self.logvar = torch.ones(exponent_prediction.size(), dtype=torch.float32).to(device=exponent_prediction.device) * -3.0
                logvar_prediction = self.logvar
            # exponent bert - from mnm
            # e_logits = self.softmax(self.activation(self.e_classifier(x)))
            # m = self.activation(self.regressor(torch.cat([x ,e_logits],dim=1))) + 0.000001 # minimum val - todo : think again!
            # torch.einsum -> why? exponent prob multiply regression result
            # m -> dimension 
            # e -> prediction / regressor
            # 

            # todo : mantissa truncation
            # e_max = e_logits.max(dim=1).indices + self.min_e
            # pred_val = torch.pow(10.,e_max) * m.view(-1)
            # return e_logits, m, e_max, pred_val
            return mean_prediction, logvar_prediction, exponent_prediction, exponent_logprobs

        elif self.is_e_predict:
            # x = self.softmax(self.activation(self.e_classifier(x)))
            x = self.e_classifier(x)

        elif self.is_q_predict:
            x = self.regressor(x)
            if self.regression_layer.endswith('-sigmoid') or self.regression_layer.endswith('-relu'):
                x = self.last_activation_layer(x)
            elif self.regression_layer.endswith('-clamp'):
                if self.q_normalize == 'exponent_max':
                    x = torch.clamp(x, min=0., max=1.)
                elif self.q_normalize == 'none':
                    _max = pow(10.,self.max_e)
                    _min = pow(10., self.min_e)
                    x = torch.clamp(x, min=_min, max=_max)
            # mu_pred = x

            # x = self.activation(x)
            # from IPython import embed; embed(colors="Linux")
            # x = self.regressor(x)
            # x = x + 0.000001
            # to get more than 0
            # x = self.activation(x) + 0.000001 # to avioid -inf
        elif self.is_u_predict :
            x = self.u_classifier(x)
        elif self.exp_ver == 'dimension':
            x = self.d_classifier(x)
        else:
            raise ValueError("Invalid configuration for prediction")
        return x

class FoodNumericModel(pl.LightningModule):
    def __init__(self, min_e, n_exponent, 
        is_include_ing_phrase,
        is_include_dimension,
        is_include_title,
        is_include_other_ing,
        is_include_tags,
        semantic_encoder_model,
        exp_ver,
        prediction_model,
        lm_tokenizer,
        is_include_serving=False,
        is_serving_concat=False,
        data_ver='weight_amount',
        name='default_name', 
        learning_rate=1e-3, 
        is_e_predict=True, is_q_predict=False, 
        is_u_predict=False,
        regression_layer='single',
        drop_rate = 0.0,
        is_mape_modified_loss=False,
        q_normalize='none',
        q_loss='l1',
        is_serving_multiply=False,
        data_processing_ver='lm', # lm, lm-embed,  w2v
        gru_bidirectional=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.name=name
        self.data_processing_ver = data_processing_ver

        self.learning_rate = learning_rate
        self.is_e_predict = is_e_predict
        self.is_q_predict = is_q_predict
        self.is_u_predict = is_u_predict
        self.n_exponent = n_exponent
        self.min_e = min_e
        self.drop_rate = drop_rate

        self.data_ver = data_ver
        self.exp_ver = exp_ver
        self.prediction_model = prediction_model

        self.is_include_ing_phrase = is_include_ing_phrase
        self.is_include_serving = is_include_serving
        self.is_serving_concat = is_serving_concat
        self.is_include_dimension = is_include_dimension
        self.is_include_title = is_include_title
        self.is_include_other_ing = is_include_other_ing
        self.is_include_tags = is_include_tags

        self.semantic_encoder_model = semantic_encoder_model
        self.is_mape_modified_loss = is_mape_modified_loss
        self.q_normalize= q_normalize
        self.gru_bidirectional=gru_bidirectional
        
        if 'lm' in self.data_processing_ver:
            self.last_hidden_dim=768
        elif self.data_processing_ver == 'w2v':
            if self.gru_bidirectional:
                self.last_hidden_dim = 40 # todo : should be modified to dynamically
            else:
                self.last_hidden_dim = 20 # todo : should be modified to dynamically
        else:
            raise ValueError(f'{data_processing_ver} ')
        # self.quantity_loss = torch.nn.L1Loss()
        self.units = food_utils.get_units()
        self.unit_to_category_num_dict, self.category_to_num_dict, self.unit_to_category_dict, self.cat_list \
        = food_utils.get_unit_cat_dict()

        self.regression_layer=regression_layer

        if self.data_ver == 'weight_amount':
            unit_dim = len(self.cat_list)
        else:
            raise NotImplementedError

        self.lm_tokenizer = lm_tokenizer
        self.is_serving_multiply = is_serving_multiply

        self.backbone = BackBone(
            last_hidden_dim=self.last_hidden_dim, 
            is_e_predict=self.is_e_predict, 
            is_q_predict=self.is_q_predict,
            is_u_predict=self.is_u_predict,
            n_exponent=self.n_exponent,
            min_e=self.min_e,
            u_num=len(self.units),
            unit_dim=unit_dim,
            regression_layer = self.regression_layer,
            exp_ver=self.exp_ver,
            is_include_ing_phrase=self.is_include_ing_phrase,
            is_include_serving=self.is_include_serving,
            is_serving_concat=self.is_serving_concat,
            is_include_dimension=self.is_include_dimension,
            is_include_title=self.is_include_title,
            is_include_other_ing=self.is_include_other_ing,
            is_include_tags=self.is_include_tags,
            semantic_encoder_model = self.semantic_encoder_model,
            lm_tokenizer=self.lm_tokenizer,
            q_normalize=self.q_normalize,
            is_serving_multiply=self.is_serving_multiply,
            data_processing_ver=self.data_processing_ver,
            gru_bidirectional=self.gru_bidirectional,
        )

        self.set_dropout(self.backbone, self.drop_rate)

        # if is_e_predict:
        self.e_loss = torch.nn.CrossEntropyLoss()

        self.train_acc =  torchmetrics.Accuracy()
        self.val_acc =  torchmetrics.Accuracy()
        self.val_acc_list =  []
        self.test_acc =  torchmetrics.Accuracy()
        self.train_f1 =  torchmetrics.F1()
        self.val_f1 =  torchmetrics.F1()
        self.val_f1_list = [] 
        self.test_f1 =  torchmetrics.F1()

    # if is_q_predict:
        self.q_loss = q_loss
        if self.q_loss == 'l1':
            self.quantity_loss = torch.nn.L1Loss()
        elif self.q_loss=='mse':
            self.quantity_loss = torch.nn.MSELoss()

        # self.quantity_loss = torch.nn.MSELoss()
        # self.quantity_loss_no_reduction = torch.nn.L1Loss(reduction='none')
        self.unit_loss = torch.nn.CrossEntropyLoss()

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae_list = [] 
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.train_lmae = torchmetrics.MeanAbsoluteError()
        self.val_lmae = torchmetrics.MeanAbsoluteError()
        self.val_lmae_list = []
        self.test_lmae = torchmetrics.MeanAbsoluteError()

        self.train_mape = torchmetrics.MeanAbsolutePercentageError()
        self.val_mape = torchmetrics.MeanAbsolutePercentageError()
        self.val_mape_list = []
        self.test_mape = torchmetrics.MeanAbsolutePercentageError()

        self.units = food_utils.get_units()
        if is_e_predict and is_q_predict:
            self.set_func_e()
            self.log_gaussian = log_truncate
            # self.log_gaussian = log_normal
        # self.dims = ['weight', 'amount']
        self.dims = ['amount', 'weight']
        # todo : dangerous?
        self.max_exponent = 5 

    def set_dropout(self, model, drop_rate):
        for name, child in model.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = drop_rate
            self.set_dropout(child, drop_rate=drop_rate)

    def forward(self, x, tokenizer=None, target_ing_text=None, detail_outputs=False):
        if self.is_q_predict and self.is_e_predict:
            if target_ing_text is not None:
                x['target_ing_text'][0] = target_ing_text
                tokenized_q_p = tokenizer([x['target_ing_text'][0]], padding='max_length', return_tensors='pt', max_length=512)
                x['target_input_ids'][0] = tokenized_q_p['input_ids'][0]
                x['target_token_type_ids'][0] = tokenized_q_p['token_type_ids'][0]
                x['target_attention_mask'][0] = tokenized_q_p['attention_mask'][0]

            mean_prediction, logvar_prediction, exponent_prediction, exponent_logprobs = self.backbone(x)
            exp_idx = torch.argmax(exponent_prediction,dim=1)
            f_e = torch.take(self.f_e, exp_idx)
            _mean_pred_max = torch.gather(mean_prediction,1,exp_idx.unsqueeze(dim=1)).squeeze()
            # for sequential squeeze need dim
            pred_values = _mean_pred_max * f_e
            if detail_outputs:
                return  pred_values, (mean_prediction, logvar_prediction, exponent_prediction, exponent_logprobs)
            else:
                return pred_values
        elif self.is_q_predict:
            # from IPython import embed; embed(colors="Linux")
            rprint(f'self.backbone : {self.backbone(x)}')
            rprint(f'self.backbone exp : {self.backbone(x).exp()}')
            rprint(f'self.backbone exp multi exp: {self.backbone(x).exp() * ( 10.0 ** self.max_exponent)}')
            return (self.backbone(x).exp() * ( 10.0 ** self.max_exponent))[0]

        elif self.exp_ver == 'unit':
            preds = self.backbone(x)
            return self.units[torch.argmax(preds, dim=1).item()], preds
        elif self.exp_ver == 'dimension':
            preds = self.backbone(x)
            return self.dims[torch.argmax(preds, dim=1).item()], preds

        else:
            raise NotImplementedError


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        rprint('>> predict_step')
        return self(batch)

    def _convert_quantity_to_e_mantissa(self, value, base_n=10):
        if base_n == 10:
            _log_val = torch.log10(value)
            _exp_val = _log_val.floor()
            return _exp_val, _log_val - _exp_val  
        else:
            raise NotImplementedError

    def set_func_e(self):
        #the exponent table abstracted
        # k = self.n_exponent
        self.max_exponent = 5 
        # self.max_exponent = self.n_exponent + self.min_e + 1 

        # denominator = 10.0**(1.0+torch.arange(k))
        denominator = 10.0**(1.0 + torch.arange(self.min_e, self.max_exponent))
        f_e = denominator
        self.register_buffer('f_e', f_e)

    def _batch_forward(self, batch):
        predictions = self.backbone(batch)
        output_dict = {}
        # if self.is_include_other_ing or self.is_include_title:
        output_dict['text_input'] = batch['text_input']
            # output_dict['info_text'] =  batch['info_text'],

        loss1, loss2 = None, None

        output_dict['original_target_quantity'] = batch['target_quantity']
        if self.q_normalize == 'max_exponent':
            self.max_exponent = 5
            self.denominator = 10.0 ** self.max_exponent
            batch['target_quantity'] = batch['target_quantity'] / self.denominator

        loss = None
        if self.exp_ver == 'ing_q' or self.exp_ver =='calories':
            y = batch['target_quantity'].float()
            # output_dict['y'] = y
            output_dict['y'] = output_dict['original_target_quantity'] 
            # e, mantissa = self._convert_quantity_to_e_mantissa(y)
            e, mantissa = self._convert_quantity_to_e_mantissa(output_dict['original_target_quantity'])

            if self.is_q_predict:
                # todo : remove upper if
                if self.is_e_predict and self.prediction_model == 'expbert':
                    mean_prediction, logvar_prediction, exponent_prediction, exponent_logprobs = predictions

                    denominator = 1.0 / self.f_e
                    y_normalized = torch.einsum('b,k->bk', y, denominator)

                    log_p = self.log_gaussian(y_normalized, mean_prediction, logvar_prediction)
                    log_likelihood = log_p + exponent_logprobs
                    log_likelihood = torch.logsumexp(log_likelihood, dim=1)
                    # log_likelihood = log_likelihood * (output_mask.float()) # original is sequential
                    neg_log_likelihood = -1*torch.sum(log_likelihood)
                    loss = neg_log_likelihood
                    
                    # prediction summary

                    exp_idx = torch.argmax(exponent_prediction,dim=1)
                    f_e = torch.take(self.f_e, exp_idx)
                    _mean_pred_max = torch.gather(mean_prediction,1,exp_idx.unsqueeze(dim=1)).squeeze()
                    # for sequential squeeze need dim
                    pred_values = _mean_pred_max * f_e
                    pred_exponent = fexp(pred_values, ignore=True)
                    pred_mantissa = fman(pred_values, ignore=True)
                    
                    # e, mantissa = self._convert_quantity_to_e_mantissa(y)
                    pred_e_max = exp_idx
                    # pred_e_logits = exponent_prediction
                    pred_e_logits = pred_exponent 
                    pred_m = pred_mantissa
                    predictions = pred_values
     
                else:
                    if self.prediction_model == 'log-laplace':
                        logged_output = torch.log(y)
                        if self.is_mape_modified_loss:
                            _pred = predictions.squeeze()
                            _mape_modified = torch.abs(_pred - logged_output) / (torch.abs( logged_output ) + 1. )
                            loss = torch.mean(_mape_modified)
                        else:
                            # from IPython import embed; embed(); exit(1);
                            distance = self.quantity_loss(logged_output, predictions.squeeze() )
                            # constant = torch.log(torch.abs(1/y))
                            constant = torch.abs(torch.log(1/y))
                            log_likelihood = -distance + constant
                            # log_likelihood = torch.einsum('bs,bs->bs', output_mask.float(), log_likelihood) # for multi output
                            loss = torch.sum(-log_likelihood)
                        predictions = predictions.exp()
                    elif self.prediction_model == 'mlp-mean' or self.prediction_model == 'mlp-mean-sigmoid':
                        if self.is_mape_modified_loss:
                            _pred = predictions.squeeze()
                            _mape_modified = torch.abs(_pred - y) / ( y + 1 )
                            loss = torch.mean(_mape_modified)
                        else:
                            loss = self.quantity_loss(y, predictions.squeeze())
                            
                    else:
                        from IPython import embed; embed(colors="Linux")
                        raise ValueError('Invalid Configuration')

                    # _clamped_prediction = torch.clamp(predictions, 
                    #         min=torch.pow(torch.tensor(10.), self.min_e -1), ).flatten()

                    _clamped_prediction = torch.clamp(predictions, 
                            min=torch.pow(torch.tensor(10.), self.min_e ) + 0.000001, ).flatten()

                    pred_e_max = torch.clamp(torch.floor(torch.log10(_clamped_prediction)), min=self.min_e, max=self.n_exponent + self.min_e) 
                    pred_m = _clamped_prediction - pred_e_max
                    # CAUTION : pred_m is dummy  value
                    pred_e_max = pred_e_max.long() - self.min_e

                if self.exp_ver == 'ing_q' or self.exp_ver == 'dimension':
                    output_dict['target_ing_text'] = batch['target_ing_text']
                    output_dict['target_unit_str'] = batch['target_unit_str']
                    output_dict['target_unit_num'] = batch['target_unit_num']
                    output_dict['target_dim_str'] = batch['target_dim_str']
                    output_dict['target_dim_num'] = batch['target_dim_num']

                if self.q_normalize == 'max_exponent':
                    pred_e_max = pred_e_max + self.max_exponent
                    e = e  + self.max_exponent
                   
                output_dict['y_e'] = e
                output_dict['y_m'] = mantissa
                

            elif self.is_e_predict:
                # todo : need to check
                pred_e_max = predictions.max(dim=1).indices 
                # e, mantissa = self._convert_quantity_to_e_mantissa(y)
                # loss1 = self.quantity_loss(predictions.view(-1), mantissa)
                # logging.info(f'predictions = {predictions}')
                # logging.info(f'y = {y.long()} / e = {e.long().item()} / mantissa = {mantissa.item():.3f}')
                loss = self.e_loss(predictions, e.long() - self.min_e)
                # loss = self.e_loss(predictions, e.long() )

        elif self.is_u_predict:
            if self.exp_ver == 'unit':
                y = batch['target_unit_num']
                output_dict['y'] = y
                loss = self.unit_loss(predictions, y )
                output_dict['target_ing_text'] = batch['target_ing_text']
                output_dict['target_unit_str'] = batch['target_unit_str']
                output_dict['target_unit_num'] = batch['target_unit_num']
                output_dict['target_dim_str'] = batch['target_dim_str']
                output_dict['target_dim_num'] = batch['target_dim_num']
            else:
                raise ValueError
        elif self.exp_ver =='dimension':
            y = batch['target_dim_num']
            output_dict['y'] = y
            loss = self.unit_loss(predictions, y)
            output_dict['target_ing_text'] = batch['target_ing_text']
            output_dict['target_unit_str'] = batch['target_unit_str']
            output_dict['target_unit_num'] = batch['target_unit_num']
            output_dict['target_dim_str'] = batch['target_dim_str']
            output_dict['target_dim_num'] = batch['target_dim_num']
        else:
            raise ValueError("e or q predict needed")

        output_dict['loss'] = loss 

        if self.q_normalize == 'max_exponent':
            output_dict['predictions'] = predictions * self.denominator             
        else:
            output_dict['predictions'] = predictions                 

                   
        if self.is_q_predict:
            output_dict['e_max']= pred_e_max
            output_dict['pred_m'] = pred_m
            if self.is_e_predict:
                output_dict['pred_e_logits'] = pred_e_logits
        # logging.error(output_dict)
        # rprint(output_dict)

        return  loss, output_dict

    def training_step(self, batch, batch_idx):
        loss, output_dict = self._batch_forward(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        if self.exp_ver == 'ing_q' or self.exp_ver == 'calories':
            if self.is_q_predict:
                if self.is_e_predict:
                    self.train_mae(output_dict['predictions'].view(-1), output_dict['y'])
                    self.train_lmae(output_dict['predictions'].view(-1).log10(), output_dict['y'].log10())
                    self.train_mape(output_dict['predictions'].view(-1), output_dict['y'])
                else:
                    prediction = output_dict['predictions'].view(-1)
                    # prediction = torch.pow(prediction, 10.)
                    self.train_mae(prediction, output_dict['y'])
                    self.train_lmae(prediction.log10(), output_dict['y'].log10())
                    self.train_mape(prediction, output_dict['y'])
                self.train_acc(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e )
                self.train_f1(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e )
        elif self.exp_ver == 'unit' or self.exp_ver=='dimension':
            self.train_acc(output_dict['predictions'].argmax(dim=1), output_dict['y'])
            self.train_f1(output_dict['predictions'].argmax(dim=1), output_dict['y'])

        else:
            raise NotImplementedError



        return output_dict

    def validation_step(self, batch, batch_idx):
        loss, output_dict = self._batch_forward(batch)
        self.log('val_loss', loss,  on_epoch=True, sync_dist=True)
        if self.exp_ver == 'ing_q' or self.exp_ver == 'calories':
            if self.is_q_predict:
                if self.is_e_predict:
                    self.val_mae(output_dict['predictions'].view(-1), output_dict['y'])
                    self.val_lmae(output_dict['predictions'].view(-1).log10(), output_dict['y'].log10())
                    self.val_mape(output_dict['predictions'].view(-1), output_dict['y'])
                else:
                    prediction = output_dict['predictions'].view(-1)
                    prediction = torch.pow(prediction, 10.)
                    self.val_mae(prediction, output_dict['y'])
                    self.val_lmae(prediction.log10(), output_dict['y'].log10())
                    self.val_mape(prediction, output_dict['y'])
                # from IPython import embed; embed(); exit(1);
                self.val_acc(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e)
                self.val_f1(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e)
            elif self.is_e_predict:
                self.val_acc(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e)
                self.val_f1(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e)
        elif self.exp_ver == 'unit' or self.exp_ver == 'dimension':
            self.val_acc(output_dict['predictions'].argmax(dim=1), output_dict['y'])
            self.val_f1(output_dict['predictions'].argmax(dim=1), output_dict['y'])
            # todo : f1?
        return output_dict

    def training_epoch_end(self, outputs) :
        if self.exp_ver == 'ing_q' or self.exp_ver == 'calories':
            self.log('train_mae_epoch',  self.train_mae, sync_dist=True)
            self.log('train_lmae_epoch', self.train_lmae, sync_dist=True)
            self.log('train_mape_epoch', self.train_mape, sync_dist=True)
        self.log('train_acc_epoch',  self.train_acc, sync_dist=True)
        self.log('train_f1_epoch',  self.train_f1, sync_dist=True)

    def validation_step_end(self, outputs) :
        if self.exp_ver == 'ing_q' or self.exp_ver == 'calories':
            self.log('val_mae_epoch',  self.val_mae, sync_dist=True)
            self.log('val_lmae_epoch', self.val_lmae, sync_dist=True)
            self.log('val_mape_epoch', self.val_mape, sync_dist=True)
        self.log('val_acc_epoch',  self.val_acc, sync_dist=True)
        self.log('val_f1_epoch',  self.val_f1, sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        #optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.hparams.learning_rate)
        # rprint('learning rate = ', self.hparams.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        return optimizer


    def test_step(self, batch, batch_idx):
        loss, output_dict = self._batch_forward(batch)
        self.log('test_loss', loss, on_step=True, sync_dist=True)
        if self.exp_ver == 'ing_q' or self.exp_ver == 'calories':
            if self.is_q_predict:
                if self.is_e_predict:
                    self.test_mae(output_dict['predictions'].view(-1), output_dict['y'])
                    self.test_lmae(output_dict['predictions'].view(-1).log10(), output_dict['y'].log10())
                    self.test_mape(output_dict['predictions'].view(-1), output_dict['y'])
                else:
                    prediction = output_dict['predictions'].view(-1)
                    prediction = torch.pow(prediction, 10.)
                    self.test_mae(prediction, output_dict['y'])
                    self.test_lmae(prediction.log10(), output_dict['y'].log10())
                    self.test_mape(prediction, output_dict['y'])
                # from IPython import embed; embed(); exit(1);
                self.test_acc(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e)
                self.test_f1(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e)
            elif self.is_e_predict:
                self.test_acc(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e)
                self.test_f1(output_dict['e_max'] , output_dict['y_e'].int() - self.min_e)
        elif self.exp_ver == 'unit' or self.exp_ver == 'dimension':
            self.test_acc(output_dict['predictions'].argmax(dim=1), output_dict['y'])
            self.test_f1(output_dict['predictions'].argmax(dim=1), output_dict['y'])
        else:
            raise NotImplementedError
            # todo : f1?

        return output_dict

    def test_step_end(self, outputs):
        if self.exp_ver == 'ing_q' or self.exp_ver == 'calories':
            self.log('test_mae_epoch',  self.test_mae, sync_dist=True)
            self.log('test_lmae_epoch', self.test_lmae, sync_dist=True)
            self.log('test_mape_epoch', self.test_mape, sync_dist=True)
        self.log('test_acc_epoch',  self.test_acc, sync_dist=True)
        self.log('test_f1_epoch',  self.test_f1, sync_dist=True)

      
