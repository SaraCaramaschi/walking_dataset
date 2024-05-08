import os
from settings import running_settings, loaders_functions

if __name__ == '__main__':
    main_path = running_settings.main_path
    project_path = main_path + os.sep + 'walking_dataset_for_6MWT'
    # Loading data
    selection, info_all_tests = loaders_functions.loader(project_path + os.sep + 'data',
                                                         blacklist=running_settings.run_settings['blacklist'])
    print("Successfully loaded data")
