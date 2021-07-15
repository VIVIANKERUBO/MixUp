import random
UNIVARIATE_DATASET_NAMES = ['50words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
                            'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee',
                            'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction',
                            'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                            'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
                            'FacesUCR', 'FISH', 'FordA', 'FordB', 'Gun_Point', 'Ham', 'HandOutlines',
                            'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
                            'LargeKitchenAppliances', 'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages',
                            'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
                            'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 'NonInvasiveFatalECG_Thorax2', 'OliveOil',
                            'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
                            'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                            'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface',
                            'SonyAIBORobotSurfaceII', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
                            'synthetic_control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
                            'Two_Patterns', 'UWaveGestureLibraryAll', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
                            'uWaveGestureLibrary_Z', 'wafer', 'Wine', 'WordsSynonyms', 'Worms', 'WormsTwoClass', 'yoga']
UNIVARIATE_DATASET_NAMES = ['50words', 'Adiac', 'ArrowHead', 'Beef']

GROUPED_UNIVARIATE_DATASET_NAMES = [['SonyAIBORobotSurface', 'ItalyPowerDemand', 'SmallKitchenAppliances', 'Adiac', 'SonyAIBORobotSurfaceII'], 
                  ['Computers', 'uWaveGestureLibrary_Y', 'TwoLeadECG', 'MiddlePhalanxTW', 'FacesUCR'], 
                  ['yoga', 'Worms', 'MALLAT', 'OSULeaf', 'WordsSynonyms'], 
                  ['ScreenType', 'Trace', 'Meat', 'FaceAll', 'Two_Patterns'], 
                  ['ProximalPhalanxOutlineAgeGroup', 'ECG5000', 'FordA', 'Symbols', 'DistalPhalanxOutlineAgeGroup'], 
                  ['Plane', 'HandOutlines', 'SwedishLeaf', 'Herring', 'ToeSegmentation2'], 
                  ['synthetic_control', 'StarLightCurves', 'Haptics', 'Cricket_Y', 'wafer'], 
                  ['NonInvasiveFatalECG_Thorax2', '50words', 'ProximalPhalanxTW', 'MedicalImages', 'DistalPhalanxTW'], 
                  ['Wine', 'PhalangesOutlinesCorrect', 'CBF', 'LargeKitchenAppliances', 'ToeSegmentation1'], 
                  ['ArrowHead', 'ECG200', 'FISH', 'uWaveGestureLibrary_X', 'MoteStrain'], 
                  ['Strawberry', 'Cricket_Z', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxOutlineAgeGroup', 'Gun_Point'], 
                  ['BeetleFly', 'CinC_ECG_torso', 'Car', 'Phoneme', 'RefrigerationDevices'], 
                  ['Ham', 'uWaveGestureLibrary_Z', 'ShapeletSim', 'ChlorineConcentration', 'Beef'], 
                  ['Lighting2', 'Cricket_X', 'OliveOil', 'ElectricDevices', 'ShapesAll'], 
                  ['NonInvasiveFatalECG_Thorax1', 'ECGFiveDays', 'Coffee', 'FordB', 'Earthquakes'], 
                  ['UWaveGestureLibraryAll', 'DistalPhalanxOutlineCorrect', 'BirdChicken', 'InsectWingbeatSound', 'Lighting7'], 
                  ['WormsTwoClass', 'InlineSkate', 'FaceFour', 'ProximalPhalanxOutlineCorrect', 'DiatomSizeReduction']]
#UNIVARIATE_DATASET_NAMES = random.sample(UNIVARIATE_DATASET_NAMES, 5)
#UNIVARIATE_DATASET_NAMES = ['DistalPhalanxOutlineCorrect']


#n = 5
#UNIVARIATE_DATASET_NAMES = [UNIVARIATE_DATASET_NAMES[i:i + n] for i in range(0, len(UNIVARIATE_DATASET_NAMES), n)]
#print(UNIVARIATE_DATASET_NAMES)
#print(len(UNIVARIATE_DATASET_NAMES))

#UNIVARIATE_ARCHIVE_NAMES = ['TSC', 'InlineSkateXPs', 'SITS']
UNIVARIATE_ARCHIVE_NAMES = ['TSC']

SITS_DATASETS = ['SatelliteFull_TRAIN_c301', 'SatelliteFull_TRAIN_c200', 'SatelliteFull_TRAIN_c451',
                 'SatelliteFull_TRAIN_c89', 'SatelliteFull_TRAIN_c677', 'SatelliteFull_TRAIN_c59',
                 'SatelliteFull_TRAIN_c133']

InlineSkateXPs_DATASETS = ['InlineSkate-32', 'InlineSkate-64', 'InlineSkate-128',
                           'InlineSkate-256', 'InlineSkate-512', 'InlineSkate-1024',
                           'InlineSkate-2048']

dataset_names_for_archive = {'TSC': UNIVARIATE_DATASET_NAMES,
                             'GROUPED_TSC': GROUPED_UNIVARIATE_DATASET_NAMES, 
                             'SITS': SITS_DATASETS,
                             'InlineSkateXPs': InlineSkateXPs_DATASETS}

