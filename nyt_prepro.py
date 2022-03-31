from prepro import Processor
from relations.preprocess import create_nyt_tokens_format # requires nlp-utils/src in path
from relations.translate import create_translated_train_test
from pathlib import Path

class NYTProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        
    def label2Id(self,label, data):
        return data['relation_id']
    
    def get_labels(self):
        return self.rm.get_classes()
    
    def read_all(self,train_file,dev_file,test_file):
        
        class ArgsClass:
            pass
            
        nyt_args = ArgsClass()
        # train_file = r'../Datasets/New York Times Relation Extraction/train.json'
        # dev_file = r'../Datasets/New York Times Relation Extraction/valid.json'
        nyt_args.train_data = train_file
        nyt_args.test_data = dev_file
        nyt_args.filter_nyt = 1000
        
        if getattr(self.args,'translated',False):
            pth_train = Path(nyt_args.train_data)
            train_translated_xlsx_path = pth_train.parent / 'translated' / (pth_train.stem + '.xlsx')
            pth_test = Path(nyt_args.test_data)
            test_translated_xlsx_path = pth_test.parent / 'translated' / (pth_test.stem + '.xlsx')
            
            df_train, df_test, rm = create_translated_train_test(nyt_args, train_translated_xlsx_path, test_translated_xlsx_path)
        else:
            df_train, df_test, rm = create_nyt_tokens_format(nyt_args)
            
        self.rm = rm
        train_features = self.features_from_data(df_train['data'])
        dev_features = self.features_from_data(df_test['data'])
        
        return train_features,dev_features,None

