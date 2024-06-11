import os
from settings import running_settings, loaders_functions, analysis_functions

if __name__ == '__main__':
    main_path = running_settings.main_path
    # Loading data
    selection, info_all_tests = loaders_functions.loader(main_path + os.sep + 'data',
                                                         blacklist=running_settings.run_settings['blacklist'])
    print("Successfully loaded data")
    # Visualizing data and baseline analysis
    analysis_functions.analysis_QSS_algorithm(info_all_tests)

    print("Successfully visualized analysis data")


