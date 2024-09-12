import pandas as pd
from tabulate import tabulate

# Supposons que tu aies plusieurs dictionnaires
dict1 = {'key1': 10, 'key2': 20, 'key3': 30}
dict2 = {'key1': 15, 'key2': 25, 'key3': 35}
dict3 = {'key1': 12, 'key2': 22, 'key3': 32}

# Nomme les dictionnaires
dicts = {
    'Measure1': dict1,
    'Measure2': dict2,
    'Measure3': dict3
}

# Créer un DataFrame à partir des dictionnaires
df = pd.DataFrame(dicts)

# Afficher le DataFrame avec des box-drawing characters grâce à tabulate
print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
