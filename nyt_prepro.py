from prepro import Processor
from relations.preprocess import create_nyt_tokens_format # requires nlp-utils/src in path

class NYTProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        
    def label2Id(self,label, data):
        return data['relation_id']
    
    def read_all(self,train_file,dev_file,test_file):
        
        class ArgsClass:
            pass
            
        nyt_args = ArgsClass()
        # train_file = r'../Datasets/New York Times Relation Extraction/train.json'
        # dev_file = r'../Datasets/New York Times Relation Extraction/valid.json'
        nyt_args.train_data = train_file
        nyt_args.test_data = dev_file
        nyt_args.filter_nyt = 1000
        
        df_train, df_test = create_nyt_tokens_format(nyt_args)
        train_features = self.features_from_data(df_train['data'])
        dev_features = self.features_from_data(df_test['data'])
                
        return train_features,dev_features,None

