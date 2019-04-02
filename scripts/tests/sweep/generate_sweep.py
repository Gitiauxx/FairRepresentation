import yaml
import numpy as np

def main(scriptname, config):
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

    with open(command_filename, 'w') as command_file:
        for kw, kword in config_data.items():
    
            if 'min_value' in kword:
                value_range = np.linspace(kword['min_value'], kword['max_value'], kword['steps'])
                
                for value in value_range:
                    for iteration in range(nboot):
                        tag = '{}_{}_{}'.format(kw, value, iteration)
                        command_file.write('{} {} --{} {} --tag {}\n'.format(preamble, scriptname, kw, value, tag))
        

if __name__ == '__main__':
    main('../test_latfr.py', 'test1.yml')