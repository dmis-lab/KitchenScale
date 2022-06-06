import os
import json
import torch
import logging
import transformers
import pytorch_lightning as pl
import math

import food_utils
from nltk.tokenize import TreebankWordTokenizer
import torchtext

class FoodNumericDataset(torch.utils.data.Dataset):
    def __init__(self, recipe_list, tokenizer, 
        data_processing_ver='lm', # lm, lm-embed,  w2v
        glove=None,
        exp_ver='ing_q', q_ing_phrase_ver='ing_name', 
        other_ing_phrase_ver='ing_name',
        is_include_ing_phrase=True,
        is_title=False,
        is_include_title=True,
        is_include_tags=True,
        is_include_other_ing=True,
        is_include_dimension=True,
        is_include_serving=True,
        is_serving_concat=False, # last hidden concat
        data_order='target_ing,other_ing,title,dim,tags,servings',
    ):
        super().__init__()
        self.data_processing_ver=data_processing_ver
        self.exp_ver = exp_ver
        self.is_title = is_title
        self.recipe_list = recipe_list
        self.max_tags_len = 0
        self.max_other_ings_len = 0
        self.tokenizer = tokenizer
        self.q_ing_phrase_ver = q_ing_phrase_ver # ['ing_name', 'ing_phrase_q_u_mask', 'ing_phrase_q_mask']
        self.other_ing_phrase_ver = other_ing_phrase_ver # ['ing_name', 'ing_phrase', ing_phrase_q_u_mask', 'ing_phrase_q_mask']
        self.is_include_ing_phrase = is_include_ing_phrase
        self.is_include_title = is_include_title
        self.is_include_tags=is_include_tags
        self.is_include_other_ing = is_include_other_ing
        self.is_include_dimension = is_include_dimension
        self.is_include_serving = is_include_serving
        self.is_serving_concat = is_serving_concat
        self.data_order = data_order
        self.data_order_list = data_order.split(',')
        self.glove = glove

        self.unit_dict = food_utils.get_unit_dict(
            is_include_none=False,
            is_include_others=False,
        )
        self.task_data = self._parse_recipe_list(recipe_list) 


    def _parse_recipe_list(self, recipe_list):
        res = []
        max_over_dict = {}
        for recipe in recipe_list:
            recipe_parsed = {
                'file_name': recipe['file_name'],
                'split': recipe['split'],
                'recipe_db_id': recipe['id'],
            }

            _input_dict = {}
            if (self.exp_ver == 'ing_q' or self.exp_ver == 'unit' or self.exp_ver =='dimension') :
                _target_ing = recipe['target_ing']

                _target_ing_quantity = _target_ing['quantity_converted']
                recipe_parsed['target_quantity'] = _target_ing_quantity
                recipe_parsed['target_unit_num'] = recipe['target_unit_num']
                recipe_parsed['target_unit_str'] = recipe['target_unit_str']
                recipe_parsed['target_dim_num'] = recipe['target_dim_num']
                recipe_parsed['target_dim_str'] = recipe['target_dim_str']

                if self.is_include_ing_phrase:
                    if self.q_ing_phrase_ver == 'ing_name':
                        recipe_parsed['target_ing_text'] = recipe['target_ing']['ing_name']
                    elif self.q_ing_phrase_ver == 'ing_name_q_u_mask':
                        recipe_parsed['target_ing_text'] = recipe['target_ing']['quantity_and_unit_masked_phrase']
                    elif self.q_ing_phrase_ver == 'ing_name_q_mask':
                        recipe_parsed['target_ing_text'] = recipe['target_ing']['quantity_masked_phrase']
                    else:
                        raise ValueError(f'{self.q_ing_phrase_ver} // is not properly defined')
                    # target_text_inputs.append(recipe_parsed['target_ing_text'])
                    _input_dict['target_ing'] = recipe_parsed['target_ing_text']

            elif self.exp_ver == 'calories':
                recipe_parsed['target_quantity'] = recipe['calories']
            else:
                raise NotImplementedError

            recipe_parsed['servings'] = recipe['servings']


            _other_ing_key = None
            if self.other_ing_phrase_ver == 'ing_name':
                _other_ing_key = 'ing_name'
            elif self.other_ing_phrase_ver == 'ing_phrase':
                _other_ing_key = 'phrase'
            elif self.other_ing_phrase_ver == 'ing_name_q_u_mask':
                _other_ing_key = 'quantity_and_unit_masked_phrase'
            elif self.other_ing_phrase_ver == 'ing_name_q_mask':
                _other_ing_key = 'quantity_masked_phrase'
            ing_str_list = [ing[_other_ing_key] for ing in recipe['ingredients']]
            ing_str = ' [SEP2] '.join(ing_str_list)
            if self.is_include_other_ing :
                _input_dict['other_ing'] = ing_str


            if self.is_include_title:
                _input_dict['title'] = recipe['title']

            if self.is_include_dimension:
                _input_dict['dim'] = recipe['target_dim_str']

            if self.is_include_tags:
                _input_dict['tags'] = ' [SEP2] '.join(recipe['tags'])
            
            if self.is_include_serving and not self.is_serving_concat:
                _input_dict['servings'] = str(recipe['servings'])

            target_text_inputs = []
            for k in self.data_order_list:
                target_text_inputs.append(_input_dict[k])
            
            recipe_parsed['text_input'] = ' [SEP] '.join(target_text_inputs)
            if self.data_processing_ver== 'lm':
                tokenized_input = self.tokenizer(recipe_parsed['text_input'], padding='max_length', return_tensors='pt', max_length=512, truncation=False)
                for k, v in tokenized_input.items():
                    recipe_parsed[k] =  v[0]
                if len(tokenized_input['input_ids'][0]) > 512:
                    max_over_dict[recipe['file_name']] = {
                        'recipe':recipe,
                        'recipe_parsed': recipe_parsed,
                    }
                else:
                    res.append(recipe_parsed)

            elif self.data_processing_ver == 'lm-embed':
                raise NotImplementedError()
            elif self.data_processing_ver == 'w2v':
                tokens = self.tokenizer.tokenize(recipe_parsed['text_input'])
                
                _token_tensors = [] 
                for token in tokens:
                    _vec = self.glove[token]
                    _token_tensors.append(_vec)
                _max_length = 512
                if len(_token_tensors) > _max_length:
                    _token_tensors = _token_tensors[:_max_length]
                else:
                    for _ in range(512-len(_token_tensors)):
                        _token_tensors.append(torch.tensor([0.]*50))

                _token_tensors = torch.stack(_token_tensors)
                # todo : tokenize
                recipe_parsed['tensors'] = _token_tensors
                res.append(recipe_parsed)
        
        if 'lm' in self.data_processing_ver:
            logging.info(f'max over = len : {len(max_over_dict)} ')
        return res


    def __getitem__(self, index):
        return self.task_data[index]

    def __len__(self):
        return len(self.task_data)

class FoodNumericDataModule(pl.LightningDataModule):
    def __init__(self, food_data_path, batch_size, min_e, 
        n_exponent,  
        data_processing_ver='lm', # lm, lm-embed,  w2v
        size='sample_trivial', 
        exp_ver='ing_q', 
        q_ing_phrase_ver='ing_name', 
        other_ing_phrase_ver='ing_name',
        is_include_ing_phrase=True,
        is_include_title=True,
        is_include_tags=True,
        is_include_other_ing=True,
        is_include_dimension=True,
        is_include_serving=True,
        is_serving_concat=False,
        data_order='target_ing,other_ing,title,dim,tags,servings',
        ):
        super().__init__()
        self.data_processing_ver=data_processing_ver
        self.exp_ver = exp_ver
        self.q_ing_phrase_ver = q_ing_phrase_ver
        self.other_ing_phrase_ver = other_ing_phrase_ver
        self.batch_size = batch_size
        self.size = size
        self.min_e = min_e
        self.n_exponent = n_exponent
        self.is_include_ing_phrase = is_include_ing_phrase
        self.is_include_title = is_include_title
        self.is_include_tags=is_include_tags
        self.is_include_other_ing = is_include_other_ing
        self.is_include_dimension = is_include_dimension
        self.is_include_serving = is_include_serving
        self.is_serving_concat = is_serving_concat
        if self.data_processing_ver == 'lm' or self.data_processing_ver == 'lm-embed':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': food_utils._special_tokens
            })
            self.glove=None
        elif self.data_processing_ver == 'w2v':
            self.tokenizer = TreebankWordTokenizer()
            self.glove = torchtext.vocab.GloVe(name="6B", # trained on Wikipedia 2014 corpus
                                dim=50)
        else:
            raise ValueError(f'data_processing_ver : {self.data_processing_ver}')

        self.food_data_path = food_data_path

        self.unit_dict = food_utils.get_unit_dict(is_include_none=False, is_include_others=False)
        self.unit_to_category_num_dict, self.category_to_num_dict, self.unit_to_category_dict, self.cat_list = food_utils.get_unit_cat_dict()
        self.data_order = data_order

    def prepare_data(self):
        logging.info('>> prepare data')

    def setup(self, stage):
        # perform on every GPU
        logging.info(f'>> [SETUP] stage = {stage}')
        if stage == "fit" or stage is None:
            # stage fit 
            self.train_raw_data = self._read_data('train')
            self.val_raw_data = self._read_data('val')
            logging.info('[SETUP] reading end')

            self.train_data = []
            for r in self.train_raw_data:
                pr = self._parse_recipe(r)
                if pr is not None:
                    self.train_data.append(pr)

            self.val_data = []
            for r in self.val_raw_data:
                pr = self._parse_recipe(r)
                if pr is not None:
                    self.val_data.append(pr)

            logging.info('[SETUP] parsing end')
            logging.info('[SETUP] Train Dataset Start')
            self.train_ds = FoodNumericDataset(self.train_data, self.tokenizer, 
                data_processing_ver=self.data_processing_ver,
                glove=self.glove,
                q_ing_phrase_ver=self.q_ing_phrase_ver,
                other_ing_phrase_ver=self.other_ing_phrase_ver,
                is_include_ing_phrase=self.is_include_ing_phrase,
                is_include_title=self.is_include_title,
                is_include_tags=self.is_include_tags,
                is_include_other_ing=self.is_include_other_ing,
                is_include_dimension=self.is_include_dimension,
                is_include_serving=self.is_include_serving,
                is_serving_concat = self.is_serving_concat,
                exp_ver = self.exp_ver,
                data_order=self.data_order,
            )
            logging.info('[SETUP] Train Dataset End')
            logging.info('[SETUP] Val Dataset End')
            self.val_ds = FoodNumericDataset(self.val_data, self.tokenizer, 
                data_processing_ver=self.data_processing_ver,
                glove=self.glove,
                q_ing_phrase_ver=self.q_ing_phrase_ver,
                other_ing_phrase_ver=self.other_ing_phrase_ver,
                is_include_ing_phrase=self.is_include_ing_phrase,
                is_include_title=self.is_include_title,
                is_include_tags=self.is_include_tags,
                is_include_other_ing=self.is_include_other_ing,
                is_include_dimension=self.is_include_dimension,
                is_include_serving=self.is_include_serving,
                is_serving_concat = self.is_serving_concat,
                exp_ver = self.exp_ver,
                data_order=self.data_order,
            )
            logging.info('[SETUP] Val Dataset End')

        if stage == "test" or stage is None:
            # staget test
            self.test_raw_data = self._read_data('test')
            self.test_data = []
            for r in self.test_raw_data:
                pr = self._parse_recipe(r)
                if pr is not None:
                    self.test_data.append(pr)
            # self.test_data = [ self._parse_recipe(r) for r in self.test_raw_data]
            self.test_ds = FoodNumericDataset(self.test_data, self.tokenizer, 
                data_processing_ver=self.data_processing_ver,
                glove=self.glove,
                q_ing_phrase_ver=self.q_ing_phrase_ver,
                other_ing_phrase_ver=self.other_ing_phrase_ver,
                is_include_ing_phrase=self.is_include_ing_phrase,
                is_include_title=self.is_include_title,
                is_include_tags=self.is_include_tags,
                is_include_other_ing=self.is_include_other_ing,
                is_include_dimension=self.is_include_dimension,
                is_include_serving=self.is_include_serving,
                is_serving_concat = self.is_serving_concat,
                exp_ver = self.exp_ver,
                data_order=self.data_order,
            )

    def _parse_recipe(self, recipe):
        _ings = recipe['ingredients']
        for ing in _ings:
            ing['unit_category'] = food_utils.unit_2_unit_cat_dict[ing['unit_parsed']] 
            ing['unit_num'] = self.unit_to_category_num_dict[ing['unit_parsed']]

        # only for quantity scenario

        ## quantity convert use
        if (self.exp_ver == 'ing_q'  or self.exp_ver == 'unit' or self.exp_ver=='dimension') :
            try:
                _target_ing = _ings.pop(recipe['quantity_one_mask_num'])
            except BaseException:
                logging.error('? ')
                from IPython import embed; embed()

            try: 
                recipe['target_quantity'] = _target_ing['quantity_converted']
                recipe['target_unit_str'] = _target_ing['unit_parsed']
                recipe['target_unit_num'] = self.unit_dict[_target_ing['unit_parsed']]
                recipe['target_dim_str'] =self.unit_to_category_dict[_target_ing['unit_parsed']]
                recipe['target_dim_num'] = self.unit_to_category_num_dict[_target_ing['unit_parsed']]
                recipe['target_ing'] = _target_ing
                
            except BaseException:
                logging.error('cannnot get recipe quantity')
                from IPython import embed; embed(colors="Linux")

            _log_val = math.log10(recipe['target_quantity'])
            _exp_val = math.floor(_log_val)
            recipe['target_exponent_val'] = _exp_val
            recipe['target_residual_val'] = _log_val - _exp_val  
        elif self.exp_ver == 'serving':
            raise NotImplementedError
        elif self.exp_ver == 'calories':
            # calories 
            if recipe['calories'] <= 0.01 :
                return None # error case 
            recipe['target_quantity'] = recipe['calories']
            _log_val = math.log10(recipe['calories'])
            _exp_val = math.floor(_log_val)
            recipe['target_exponent_val'] = _exp_val
            recipe['target_residual_val'] = _log_val - _exp_val  
        else:
            raise ValueError('parsing ver should be ing, serving, calories, unit')
        return recipe


    def _read_data(self, split):
        logging.info(f"[{split}] read_data Start")
        _base_path = f"{self.food_data_path}/{split}/"
        fnames = os.listdir(_base_path)

        res = []
        import tqdm
        for _, fname in tqdm.tqdm(enumerate(fnames)):
            with open(f"{_base_path}/{fname}") as rf:
                content = json.load(rf)

                if content is not None :
                    res.append(content)
                    if self.size == 'sample_trivial':
                        if len(res) > 40:
                            break
                    elif self.size == 'sample_medium':
                        if len(res) > 1000 and split == 'train':
                            break
                        if len(res) > 100 and (split == 'val' or split == 'test'):
                            break
                    elif self.size == 'all':
                        continue
                    else:
                        raise NotImplementedError
        return res


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size  )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size  )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size  )

    def teardown(self, stage):
        logging.info('>> DataModule >> tear down ')
        pass
