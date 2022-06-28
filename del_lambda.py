import pickle
import sys

assert len(sys.argv) == 3

folder = f'outs/{sys.argv[1]}'
key = sys.argv[2]
if key != "all":
    exec('key='+key)

def del_key(csvs, key):
    del csvs[0][1][key]
    del csvs[0][2][key]
    return csvs

def main():
    with open(f"{folder}/csvs_dict.pkl","rb") as cdh:
        csvs = pickle.load(cdh)
    if key == 'all':
        iter_keys = list(csvs[0][1].keys())
        for k in iter_keys:
            csvs = del_key(csvs, k)

    elif not key in list(csvs[0][1].keys()):
        print(f"{key} not in:")
        print(list(csvs[0][1].keys()))
        return
    
    else:
        csvs = del_key(csvs, key)    
    #del csvs[0][1][key]
    #del csvs[0][2][key]
    
    with open(f"{folder}/csvs_dict.pkl","wb") as cdh:
        pickle.dump(csvs,cdh)

if __name__ == '__main__':
    main()
