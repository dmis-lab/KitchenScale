
import argparse

import logging
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from food_data import FoodNumericDataset, FoodNumericDataModule
from food_model import FoodNumericModel


        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', help='log_level', default='INFO', type=str)
    parser.add_argument('--name', help='test', default='test', type=str)
    parser.add_argument('--proj_name', help='test', default='test', type=str)
    parser.add_argument('--checkpoint_path', default='./checkpoints', type=str)
    parser.add_argument('--food_data_path', default='./data', type=str)
    parser.add_argument('--data_size', help='sample_trivial', default='sample_trivial', type=str)

    # checkpoint args 
    parser.add_argument('--save_every_n_epoch', help=' ', default=1, type=int)
    parser.add_argument('--save_top_k', help='1 : all model savel , 0 : no model save', default=1, type=int)
    parser.add_argument('--distribution_model', help='LogBert, ', type=str, default='LogBert')
    parser.add_argument('--regression_layer',  type=str, default='single', help='single, 3mlp, single-sigmoid, 3mlp-sigmoid, 3mlp-relu, 3mlp-clamp')

    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--is_debug', default=False, action='store_true')
    parser.add_argument('--is_u_predict', default=False, action='store_true')
    parser.add_argument('--is_e_predict', default=False, action='store_true')
    parser.add_argument('--is_q_predict', default=False, action='store_true')
    parser.add_argument('--n_exponent', default=7, type=int)
    parser.add_argument('--min_e', default=-2, type=int, help="Always minus")
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--drop_rate', default=0.1, type=float)

    parser.add_argument('--exp_ver', default='ing_q', type=str, help='[ing_q, unit, dimension]')
    parser.add_argument('--semantic_encoder_model', default='bert', type=str, help='[bert, bert-no-pre, bert-freeze, bi-gru-bertembedding]')
    parser.add_argument('--prediction_model', default='expbert', type=str, help='[expbert, log-laplace, mlp-mean]')

    parser.add_argument('--is_include_ing_phrase', default=False, action='store_true')
    parser.add_argument('--q_ing_phrase_ver', default='ing_name_q_u_mask', type=str, 
        help="exp_ver == ing_q -> in ['ing_name', 'ing_name_q_u_mask', 'ing_name_q_mask' ]")

    parser.add_argument('--data_order', type=str, 
        default='target_ing,other_ing,title,dim,tags,servings' )
    parser.add_argument('--is_include_serving', default=False, action='store_true')
    parser.add_argument('--is_serving_concat', default=False, action='store_true')

    parser.add_argument('--is_include_dimension', default=False, action='store_true')
    parser.add_argument('--is_include_title', default=False, action='store_true')
    parser.add_argument('--is_include_other_ing', default=False, action='store_true')
    parser.add_argument('--is_include_tags', default=False, action='store_true')

    parser.add_argument('--other_ing_phrase_ver', default='ing_name', type=str, 
        help='ing_name, ing_phrase')
    parser.add_argument('--patience', default=2, type=int, help="EarlyStopping patience")
    parser.add_argument('--early_stopping_metric', default=None, type=str, help="Early stopping criterion")
    parser.add_argument('--is_mape_modified_loss', default=False, action='store_true', help="Early stopping criterion")
    parser.add_argument('--q_normalize', default='none', type=str, help='[ none, exponent_max ]')
    # parser.add_argument('--s_normalize', default='none', type=str, help='[ none, 1, 4 ]')
    parser.add_argument('--q_loss', default='l1', type=str, help='l1, mse')

    parser.add_argument('--is_serving_multiply', default=False, action='store_true')
    parser.add_argument('--is_gru_bidirectional', default=False, action='store_true')
    parser.add_argument('--regression_layer_init', default='none', type=str, help='none, he, xavier, zero')

    parser.add_argument('--data_processing_ver', default='lm', type=str, help='lm, lm-embed, w2v')

    # other_ing_phrase_ver : ['ing_name', 'phrase', 'ing_name_q_mask', 'ing_name_q_u_mask']
    # q_ing_phrase_ver : ['ing_name', 'ing_name_q_u_mask', 'ing_name_q_mask'] 


    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)
    
    if args.is_debug:
        args.log_level = logging.DEBUG
        

    if args.exp_ver == 'ing_q':
        if args.early_stopping_metric is not None:
            _monitor_metric = args.early_stopping_metric
        else:
            _monitor_metric = 'val_lmae_epoch'
        if args.is_q_predict:
            _monitor=_monitor_metric
            _fname = args.exp_ver
            _fname+='_{epoch:02d}-lmae_{val_lmae_epoch:.2f}-mae_{val_mae_epoch}'
        elif args.is_q_predict and args.is_u_predict and not args.is_e_predict:
            raise NotImplementedError
        elif args.is_q_predict and args.is_u_predict and args.is_e_predict:
            raise NotImplementedError
    elif args.exp_ver == 'unit' or args.exp_ver == 'dimension':
        if args.early_stopping_metric is not None:
            _monitor_metric = args.early_stopping_metric
        else:
            _monitor_metric = 'val_loss'
        if args.is_u_predict or args.exp_ver == 'dimension':
            _monitor='val_acc_epoch'
            _fname = args.exp_ver
            _fname+='_{epoch:02d}-acc_{val_acc_epoch:.2f}'
        else:
            raise ValueError
    else:
        raise ValueError(f'exp_ver={args.exp_ver} / e predict =  {args.is_e_predict} / q predict = {args.is_q_predict} / u predict = {args.is_u_predict} problem ')


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=_monitor,
        dirpath=f'{args.checkpoint_path}/{args.exp_ver}/{args.name}/{args.data_size}',
        filename=_fname,
        # every_n_epochs=args.save_every_n_epoch,
        save_last=True,
        save_top_k=args.save_top_k,
    )
    callbacks= []
    # if not args.is_debug: 
    callbacks.append(checkpoint_callback)
    callbacks.append(
       EarlyStopping(monitor=_monitor_metric, patience=args.patience) 
    )

    # sanity check
    if args.min_e > 0:
        raise ValueError('min_e should be minus')
    
    _logger1 = pl_loggers.WandbLogger(
        project=f'{args.proj_name}_{args.exp_ver}_{args.data_size}',
        name=args.name,
    )
    # _logger2 = pl_loggers.CSVLogger('logs', name='output_result' )
    trainer = pl.Trainer.from_argparse_args(
        args, 
        callbacks=callbacks,
        # logger= [_logger1, _logger2],
        logger= _logger1,
    )

    dm = FoodNumericDataModule(
        size=args.data_size,
        batch_size=args.batch_size,
        n_exponent=args.n_exponent,
        food_data_path=args.food_data_path,
        min_e = args.min_e,
        is_include_ing_phrase=args.is_include_ing_phrase,
        is_include_title=args.is_include_title,
        is_include_other_ing=args.is_include_other_ing,
        is_include_dimension=args.is_include_dimension,
        is_include_tags=args.is_include_tags,
        q_ing_phrase_ver=args.q_ing_phrase_ver,
        other_ing_phrase_ver=args.other_ing_phrase_ver,
        is_include_serving=args.is_include_serving,
        is_serving_concat=args.is_serving_concat,
        exp_ver=args.exp_ver,
        data_order=args.data_order,
        data_processing_ver=args.data_processing_ver,
    )

    model = FoodNumericModel(
        learning_rate=args.learning_rate,
        min_e = args.min_e,
        is_e_predict=args.is_e_predict,
        is_q_predict=args.is_q_predict,
        is_u_predict=args.is_u_predict,
        n_exponent=args.n_exponent,
        name=args.name,
        regression_layer=args.regression_layer,
        drop_rate=args.drop_rate,
        exp_ver=args.exp_ver,
        prediction_model=args.prediction_model,
        is_include_ing_phrase=args.is_include_ing_phrase,
        is_include_serving=args.is_include_serving,
        is_serving_concat=args.is_serving_concat,
        is_include_dimension=args.is_include_dimension,
        is_include_title=args.is_include_title,
        is_include_other_ing=args.is_include_other_ing,
        is_include_tags=args.is_include_tags,
        semantic_encoder_model=args.semantic_encoder_model,
        lm_tokenizer=dm.tokenizer,
        is_mape_modified_loss=args.is_mape_modified_loss, 
        q_normalize=args.q_normalize,
        q_loss=args.q_loss,
        is_serving_multiply=args.is_serving_multiply,
        data_processing_ver=args.data_processing_ver,
        gru_bidirectional=args.is_gru_bidirectional,
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == '__main__':
    main()
