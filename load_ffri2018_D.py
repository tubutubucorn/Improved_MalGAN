import re, pandas as pd
from pathlib import Path


# folder_list :   ['<Path>/cleanware', '<Path>/malware']
# api_list_area :   '555*.txt', To limit dimensions.
def make_api_list(folder_list, api_list_area):
    api_list = ['label']
    
    for folder in folder_list:
        file_list = [str(file) for file in Path(folder).glob(api_list_area)]
        # To arrange in numerical order
        file_list = [(int(re.search(r"[0-9]+", file).group()), file) for file in file_list]
        file_list.sort()
        file_list = [x[1] for x in file_list]
        
        for file in file_list:
            with open(file) as f:
                for line in f.readlines():
                    # e.g.) KERNEL32.dll.GetTickCount Hint[469]
                    if 'Hint' in line and '.' in line and not '@' in line and not '?' in line and not '$' in line and not '*' in line:
                        try:
                            api = line.split()[0]
                        except:
                            continue
                        if not api in api_list:
                            api_list.append(api)
    return api_list


# file :   Target malware
# api_list :   label + some APIs
def make_used_api_dataframe_with_malware_file(file, api_list):
    used_api_dict = {api:[0] for api in api_list}
    
    with open(file) as f:
        for line in f.readlines():
            if 'Hint' in line and '.' in line and not '@' in line and not '?' in line and not '$' in line and not '*' in line:
                try:
                    api = line.split()[0]
                except:
                    continue
                if api in used_api_dict.keys():
                    used_api_dict[api][0] = 1
                else:
                    used_api_dict[api] = [1]
    used_api_dict['label'][0] = 1
    return pd.DataFrame.from_dict(used_api_dict)


# folder_list :   ['<Path>/cleanware']
# data_area_list :   ['555*.txt'], To limit 
# api_list :   label + some APIs + Target malware culumns
def make_used_api_dataframe(folder_list, data_area_list, api_list):
    used_api_dict = {api:[] for api in api_list}
    
    for folder in folder_list:
        for data_area in data_area_list:
            file_list = [str(file) for file in Path(folder).glob(data_area)]
            file_list = [(int(re.search(r"[0-9]+", file).group()), file) for file in file_list]
            file_list.sort()
            file_list = [x[1] for x in file_list]

            for file in file_list:
                with open(file) as f:
                    for api in api_list:
                        used_api_dict[api].append(0) 
                    for line in f.readlines():
                        if 'Hint' in line and '.' in line and not '@' in line and not '?' in line and not '$' in line and not '*' in line:
                            try:
                                api = line.split()[0]
                            except:
                                continue
                            if api in used_api_dict.keys():
                                used_api_dict[api][len(used_api_dict['label']) - 1] = 1
    return pd.DataFrame.from_dict(used_api_dict)

