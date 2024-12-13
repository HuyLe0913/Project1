
import numpy as np
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.MLP = nn.Sequential(
            nn.Linear(17,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48,1)
        )

    def forward(self, x):
                    
        return self.MLP(x)
    

model = NeuralNet()

model.load_state_dict(torch.load(r"C:\Users\dmin\HUST\20241\DataScience\Life_Expectancy_Predictor_4_64MSE.pth")) #thay bang file_path dan den model da luu

model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def predict(input_dict : dict):
        with open(self.script_dir + "\mean_max_min_dictionary.json", "r") as json_file:
            mean_max_min_dict = json.load(json_file)
        sgd_region_map = {
            'Central and Southern Asia': 1,
            'Europe and Northern America': 2,
            'Australia and New Zealand': 3,
            'Latin America and the Caribbean': 4,
            'Sub-Saharan Africa': 5,
            'Northern Africa and Western Asia': 6,
            'Oceania': 7,
            'Eastern and South-Eastern Asia': 8
        }
        num_regions = len(sgd_region_map)
        feature_columns = [
            'Young Age Dependency Ratio', 'Retirement Age Dependency Ratio', 'Population',
            'Male Population Ratio', 'Urban Population Ratio', 'Total Area (sq km)',
            'GDP_per_Capita', 'Infant mortality per 1000 live births', 'BMI']
        def one_hot_encode_region(region):
            if not region: #region is none
                return np.zeros(num_regions)
            index = sgd_region_map[region] - 1 
            one_hot_vector = np.zeros(num_regions)
            one_hot_vector[index] = 1
            return one_hot_vector
        value_array = []
        for feature_column in feature_columns:
            if input_dict[feature_column] is None:
                input_dict[feature_column] = 0
                value_array.append(input_dict[feature_column])
            else: 
                mean_value,max_value,min_value = mean_max_min_dict[feature_column]
                old_value = input_dict[feature_column]
                input_dict[feature_column] = (old_value - mean_value) / (max_value - min_value)
                value_array.append(input_dict[feature_column])
            
        region = input_dict["SDG Region"]
        value_array.extend(one_hot_encode_region(region))
        input_for_model = torch.tensor(value_array,dtype=torch.float32).to(device)
        return model(input_for_model) #có thể lỗi ở chỗ này do array ko đủ features input = 17




if __name__ == "__main__":
    sample_input = {
        "Young Age Dependency Ratio": 50,
        "Retirement Age Dependency Ratio": 10,
        "Population": 90000000,
        "Male Population Ratio": 60,
        "Urban Population Ratio": 35,
        "Total Area (sq km)": 300000,
        "GDP_per_Capita": 4000,
        "Infant mortality per 1000 live births": None,
        "BMI": 25,
        "SDG Region" : 'Central and Southern Asia'}
    mean_max_min_dict = {
        "Year": [
            1996.767503858566,
            2016,
            1975
        ],
        "Young Age Dependency Ratio": [
            59.61617370750345,
            114.38828800291,
            15.9547454011501
        ],
        "Retirement Age Dependency Ratio": [
            10.43072321495741,
            46.9608333063349,
            0.199773990122035
        ],
        "Population": [
            32230008.37813947,
            1387790000,
            6381
        ],
        "Male Population Ratio": [
            50.08348249434012,
            76.6057826183619,
            45.5144137746939
        ],
        "Urban Population Ratio": [
            51.315294794443666,
            100.0,
            3.525
        ],
        "Total Area (sq km)": [
            703648.6181321733,
            17098250.0,
            20.0
        ],
        "GDP_per_Capita": [
            7236.891338697621,
            123678.702143275,
            0.0024249251912713
        ],
        "Life Expectancy": [
            65.95674849336258,
            83.9848780487805,
            11.995
        ],
        "Infant mortality per 1000 live births": [
            43.57278282587344,
            254.4442,
            0.7359
        ],
        "BMI": [
            24.219250736635328,
            33.7,
            17.5
        ],
        "BMI Female": [
            24.58917736700997,
            34.1,
            16.5
        ],
        "BMI Male": [
            23.853647523634834,
            33.3,
            17.6
        ]
    }
    print(inference(sample_input,mean_max_min_dict))
    