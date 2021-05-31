data_path = 'Datasets/MDF/'
df = pd.DataFrame()   
# foreach user folder
for user in range(31): 
    user_dir = data_path + 'user_' + str(user)
    # read running_apps.csv and use it as a starting point
    df1 = pd.read_csv(user_dir + '/running_apps.csv', header=0) 
    df1['time'] = pd.to_datetime(df1['time'], unit='ms') 
    df1.sort_values('time', inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df1.insert(1,'user',user) # insert user ID column
    
    rows = []
    # foreach row in running apps dataframe find the closest row in all other csv file using timestamp
    for dt in df1['time']: 
        row = []
        # foreach csv file in user folder
        for filename, columns in file_dict.items(): 
            file_path = user_dir + '/' + filename
            # single row with all the context features
            row = row + get_closest_row(file_path, columns, dt).tolist() 
        rows.append(row)

    df2 = pd.DataFrame(rows, columns=np.concatenate(list(file_dict.values()))) 
    df3 = pd.concat([df1, df2], axis=1) # concat by column
    df = pd.concat([df, df3], axis=0) # concat by row
    
df.reset_index(drop=True, inplace=True)