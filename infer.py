import sys
import os

import json
from food_data import FoodNumericDataset, FoodNumericDataModule
from food_model import FoodNumericModel

# from rich import print as rprint
import torch
from torch import tensor
import food_utils
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def get_list(inputs):
    converted_list = []
    for idx, elem in enumerate(inputs):
        _input_str = elem['text_input']
        str_splitted = _input_str.split(' [SEP] ')
        elem['splitted'] = str_splitted
        # for i, v in enumerate(str_splitted):
            # print(i, v)
        elem['splitted_no_dim'] = str_splitted[:3] + str_splitted[4:]
        elem['text_input_no_dim'] = ' [SEP] '.join(elem['splitted_no_dim'])
        a,b,c,d,e = elem['text_input_no_dim'].split(' [SEP] ')
        _converted_val = float(elem['target_quantity']) / food_utils.unit_2_normalize_factor_dict[elem['target_unit_str']]
        converted_list.append({
            'idx': idx,
            'target_text': a,
            'other_ings': '_'.join(b.split(' [SEP2] ')),
            'title': c,
            'tags':'_'.join(d.split(' [SEP2] ')) ,
            'servings':e,
            'file_name': elem['file_name'],
            'recipe_db_id': elem['recipe_db_id'],
            'target_quantity': float(elem['target_quantity']),
            'target_unit': elem['target_unit_str'] ,
            'target_dim': elem['target_dim_str'],
            'converted_quantity': _converted_val ,
        })
    return converted_list


class PredModel:
    def __init__(self, path, ver):
        # path = '../checkpoints/ing_q/latest/ing_q_epoch=37-lmae_val_lmae_epoch=0.33-mae_val_mae_epoch=109.5320816040039.ckpt'
        if ver not in ['ing_q', 'dimension', 'unit']:
            raise ValueError(f'ver [{ver}] canoot be used')
        self.ver = ver

        self.model = FoodNumericModel.load_from_checkpoint(path)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def parse_q_text(self, target_text='',ings=[], title='',dimension='',  tags=[], servings=4):
        input_dict = {
            'target' : f'{target_text}',
            'ings' : ings, 
            'title' : title,
            'dimension': dimension,
            'tags' : tags,
            'servings' : servings,
        }
        if dimension not in ['weight', 'amount']:
            raise ValueError

        text = f"{input_dict['target']} [SEP] {' [SEP2] '.join(input_dict['ings'])}"+ \
            f" [SEP] {input_dict['title']} [SEP] {input_dict['dimension']} [SEP] "+ \
            f"{' [SEP2] '.join(input_dict['tags'])} [SEP] {input_dict['servings']}" 
        return text
    
    def get_pred_ing_q(self, target_text='',ings=[], title='',dimension='',  tags=[], servings=4):
        text = self.parse_q_text(target_text, ings, title, dimension, tags, servings)
        return self.get_pred_ing_q_text(text)

    def get_pred_ing_q_text(self, text):
        # rprint(text)
        _elem =  self.model.lm_tokenizer(text, padding='max_length', return_tensors='pt', max_length=512)
        _elem.to(self.device)

        self.model.eval()
        with torch.no_grad():
            res = self.model.backbone({
                'input_ids' : _elem['input_ids'],
                'token_type_ids' : _elem['token_type_ids'],
                'attention_mask' : _elem['attention_mask'],
            }), self.model({
                'input_ids' : _elem['input_ids'],
                'token_type_ids' : _elem['token_type_ids'],
                'attention_mask' : _elem['attention_mask'],
            })
        # print('res ??? ')
        # print(res)
        # from IPython import embed; embed(colors="Linux")
        if self.model.is_q_predict and self.model.is_e_predict:
            converted_res = {
                'mean_pred': res[0][0][0].tolist(),
                'logvar_pred':res[0][1][0].tolist(),
                'exp_prob': torch.exp(res[0][3][0]).tolist(),
                'pred_val': res[1][0].tolist(),
                'input_txt': text,
            }
        elif self.model.is_q_predict:
            converted_res = {
                'mean_pred': 0.,
                'logvar_pred': 0., # default
                'exp_prob': 0., # default
                'pred_val': res[1][0].tolist() , # default
                'input_txt': text,
            }

        return converted_res


    # target / ing / title / dimension / tags / serving
    def parse_u_text(self, target_text='',ings=[], title='',dimension='',  tags=[], servings=4):
        if dimension not in ['weight', 'amount']:
            raise ValueError
    # target / ing / title / dimension / tags / serving
        input_dict = {
            'target' : f'{target_text}',
            'ings' : ings, 
            'title' : title,
            'dimension': dimension,
            'tags' : tags,
            'servings' : servings,
        }
        text = f"{input_dict['target']} [SEP] {' [SEP2] '.join(input_dict['ings'])}"+ \
            f" [SEP] {input_dict['title']} [SEP] {input_dict['dimension']} [SEP] "+ \
            f"{' [SEP2] '.join(input_dict['tags'])} [SEP] {input_dict['servings']}" 
        return text
        
    def get_pred_unit(self,target_text='',ings=[], title='',dimension='',  tags=[], servings=4):
        text = self.parse_u_text(target_text, ings, title, dimension, tags, servings)
        # rprint(text)
        return self.get_pred_unit_text(text)
        
    def get_pred_unit_text(self, text):
        _elem =  self.model.lm_tokenizer(text, padding='max_length', return_tensors='pt', max_length=512)
        _elem.to(self.device)

        self.model.eval()
        with torch.no_grad():
            res = self.model({
                'input_ids' : _elem['input_ids'],
                'token_type_ids' : _elem['token_type_ids'],
                'attention_mask' : _elem['attention_mask'],
            })
        converted_res = {
            'pred_unit':res[0],
            'prob': res[1][0].tolist(),
            'text' : text,
        }
        return converted_res

    def parse_d_text(self,target_text='',ings=[], title='',  tags=[], servings=4):
        input_dict = {
            'target' : f'{target_text}',
            'ings' : ings, 
            'title' : title,
            'tags' : tags,
            'servings' : servings,
        }
        text = f"{input_dict['target']} [SEP] {' [SEP2] '.join(input_dict['ings'])}"+ \
            f" [SEP] {input_dict['title']} [SEP] "+ \
            f"{' [SEP2] '.join(input_dict['tags'])} [SEP] {input_dict['servings']}" 
        return text
        
    # target / ing / title / dimension / tags / serving
    def get_pred_dimension(self,target_text='',ings=[], title='',  tags=[], servings=4):
    # target / ing / title / dimension / tags / serving
        text = self.parse_d_text(target_text, ings, title, tags, servings)
        
        # rprint(text)
        return self.get_pred_unit_text(text)
        
    def get_pred_dimension_text(self, text):
        _elem =  self.model.lm_tokenizer(text, padding='max_length', return_tensors='pt', max_length=512)
        _elem.to(self.device)

        self.model.eval()
        with torch.no_grad():
            res = self.model({
                'input_ids' : _elem['input_ids'],
                'token_type_ids' : _elem['token_type_ids'],
                'attention_mask' : _elem['attention_mask'],
            })
        converted_res = {
            'pred_dimension':res[0],
            'prob': res[1][0].tolist(),
            'text' : text,
        }
        return converted_res


# def main():
if __name__ == '__main__':
    # main()
    food_data_path=os.environ.get('food_data_path')
    dm = FoodNumericDataModule(
        batch_size=1,
        food_data_path=food_data_path,
        min_e=-2,
        n_exponent = 7,  
        size='all', 
        exp_ver='ing_q', 
        q_ing_phrase_ver='ing_name_q_u_mask',  # ['ing_name', 'ing_name_q_u_mask', 'ing_phrase_q_mask']
        # other_ing_phrase_ver='ing_phrase',
        other_ing_phrase_ver='ing_name', # ['ing_name', 'ing_phrase', ing_phrase_q_u_mask', 'ing_phrase_q_mask']
        data_ver='weight_amount',
        is_include_ing_phrase=True,
        is_include_title=True,
        is_include_tags=True,
        is_include_other_ing=True,
        is_include_dimension=True,
        is_include_serving=True,
    )

    dm.setup('test')
    test_loader = dm.test_dataloader()
    inputs = []
    for elem in iter(test_loader):
        converted = {}
        for k, v in elem.items():
            converted[k] = v[0]
        inputs.append(converted)


    testset_list = get_list(inputs)

    df = pd.DataFrame(testset_list)

    ver = 'ing_q'
    q_path = './checkpoints/ing_q.ckpt'
    _model_q = PredModel(q_path,ver)

    for i in tqdm(range(len(df))):
        ti = df.iloc[i]
        # print(ti)
        pred_q = _model_q.get_pred_ing_q(
                        target_text=ti['target_text'],
                        ings=ti['other_ings'].split('_'),
                        title=ti['title'],
                        dimension=ti['pred_dim'],
                        tags=ti['tags'].split('_'),
                        servings=ti['servings'],
                    )


from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return {
        'Hello' : 'World'
    }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
