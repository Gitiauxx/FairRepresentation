import yaml
import numpy as np


def main(scriptname, config, dname):
    command_filename = 'commands.sh'
    config_file = "config/{}".format(config)
    preamble = 'python'

    # read config file (yaml)
    with open(config_file, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)
    
    # set number of iteration by config
    if 'nboot' in config_data:
        nboot = config_data['nboot']
    else: 
        nboot = 1
    
    del config_data['nboot']

    # get config name
    name = config_data['name']
    del config_data['name']

    # keep of list of experiments run in this sweep
    tags_list  = []
    with open(command_filename, 'w') as command_file:
        for kw, kword in config_data.items():
    
            if 'min_value' in kword:
                value_range = np.linspace(kword['min_value'], kword['max_value'], kword['steps'])
                
                for value in value_range:
                    for iteration in range(nboot):
                        tag = '{}_{}_{}'.format(kw, value, iteration)
                        tags_list.append(tag)
                        command_file.write('{} {} --dirname {} --{} {} --tag {}\n'.format(preamble, scriptname, dname, kw, value, tag))
        
        command_file.write('{} {} --dirname {} --config {}\n'.format(preamble, '..\\\\..\\\\lafr\\\\results.py', dname, config))
        

if __name__ == '__main__':
    dirname = 'C:\\\\Users\\\\Xavier\\\\fair_representation\\\\FairRepresentation\\\\data\\\\tests'
    main('../test_latfr.py', 'test1.yml', dirname)