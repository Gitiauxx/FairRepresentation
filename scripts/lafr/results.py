import pandas as pd
import yaml
import argparse
import numpy as np
import os

def combine_results(dname, config):
        config_file = "config/{}".format(config) 

        # read config file (yaml)
        with open(config_file, 'r') as stream:
            config_data = yaml.load(stream, Loader=yaml.SafeLoader)
    
        # set number of iteration by config
        if 'nboot' in config_data:
            nboot = config_data['nboot']
        else: 
            nboot = 1
        
        del config_data['nboot'] # this is hackish....

        # get config name
        name = config_data['name']
        del config_data['name']

        # create tags for each configuration
        for kw, kword in config_data.items():
            if 'min_value' in kword:
                value_range = np.linspace(kword['min_value'], kword['max_value'], kword['steps'])
                tag_list = ['{}_{:.2f}_{}'.format(kw, value, iteration) for value in value_range for iteration in range(nboot) ]

        # collect all results with the same tag
        table_list = []
        for tag in tag_list:
            table = pd.read_csv(os.path.join(dname, 'test_metrics_{}.csv'.format(tag)), names=['metrics', 'value'])
            table['run'] = tag
            table_list.append(table)

        results = pd.concat(table_list)
        results.to_csv(os.path.join(dname, 'test_metrics_{}.csv'.format(name)))

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname')
    parser.add_argument('--config')
    args = parser.parse_args()

    args_dict = {'dirname': args.dirname, 'config':args.config}
    
    combine_results(args.dirname, args.config)

