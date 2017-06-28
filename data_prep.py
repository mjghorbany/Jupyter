import os
import pandas as pd
import numpy as np

#Importing the Files
def excel_2_df(dir_path, sheet_name):
    for file in os.listdir(dir_path):
        if (file.endswith(".xlsx") or file.endswith(".xls")) and not file.startswith("~$"):
            file_path = os.path.join(dir_path, file)
            print("\n...Extracting from: " + sheet_name+'\n'+ file_path)
            df = pd.ExcelFile(file_path)
            ski=0
            df_sheet = df.parse(sheet_name)
            print(df_sheet.columns)
            #print(df_sheet.head(5))

            while ('Trans Amount' not in df_sheet.columns) and ('Amount' not in df_sheet.columns) and ('Description' not in df_sheet.columns):
                ski=ski+1
                df_sheet = df.parse(sheet_name, skiprows=ski)
                print("Skipped %d lines" %ski)
                #print(df_sheet.columns)
                if ski>10:
                    print('File not readable!')
                    break

            return df_sheet
'''
            if 'Trans Amount' in df_sheet.columns:
                print('yes')
            else:
                df_sheet = df.parse(sheet_name, skiprows=1)
'''

'''
            if sheet_name=="MHGL-Dec":
                df_sheet = df.parse(sheet_name)
            else:
                df_sheet = df.parse(sheet_name, skiprows=1)
'''

