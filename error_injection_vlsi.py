import pandas as pd
import numpy as np
import os
import glob
import struct
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser(description="Default options")
# Following argument is only for NAND chip
#parser.add_argument("--info", default=0)
args = parser.parse_args()

##### Path list
# define weight data root
original_weight_root = '/home/mizushina/python_program/segnet/Weight_data_original/'
extention_all = "*"
extention_csv = '.csv'

# list of weight list
weight_name_list = ['conv1', 'conv2', 'conv3', 'conv4',\
                    'conv_deconv1', 'conv_deconv2', 'conv_deconv3', 'conv_deconv4']

#output_path_dir = 'injected_weight_csv'
current_path = os.getcwd()
output_path_dir = os.path.join(current_path, 'weight4VLSI')

error_rate = [0.001, 0.005, 0.008, 0.01, 0.015, 0.02, 0.022, 0.025, 0.03, 0.035,
              0.04, 0.045, 0.05, 0.055]


def main():
    full_original_weight = os.path.join(original_weight_root, extention_all + extention_csv)
    weight_csv_list = glob.glob(full_original_weight)
    weight_csv_list.sort()
    print("Number of weight : ", len(weight_csv_list))
    print("Path of weight data : ", weight_csv_list)

    target_array = np.loadtxt(weight_csv_list[0], delimiter=',')
    print(target_array[0])

    # 外側のループは8個の重みをすべて調整するときに回す。内側のループはリテンションのデータを扱う。
    for current_weight_num in range(8):  # 0:conv1, ... , 7:deconv4
        for inject_ber in range(13):  # 0:day0,...,16:day2.5
            df_current = pd.read_csv(weight_csv_list[current_weight_num], delimiter=',', header=None, names='A')
            # Null check
            #print(df_current.isnull().any())
            array_current = df_current.values
            parameter_num = array_current.shape[0]
            error_cell_num = int(parameter_num * error_rate[inject_ber])

            print("********" * 5)
            print("weight contains {} params.".format(parameter_num))
            print("Progress... \ncurrently T_BER is {}".format(error_rate[inject_ber]))
            print("...almost {} params causes errors.".format(error_cell_num))
            print("********" * 5)

            for i in range(parameter_num):
                if(random.random() < error_rate[inject_ber]):
                    float_temp = float(array_current[i])
                    #print("original : ", float_temp)
                    bin_data = struct.pack('>d', float_temp)
                    mod_data = struct.unpack('>Q', bin_data)

                    mod_data = bin(mod_data[0])
                    mod_data = (mod_data)
                    mod_data = int(mod_data, 2)

                    #mod_data = random.randint(0, mod_data-1)
                    #if(mod_data == ""):
                    #    print("FAIL")

                    #mod_data = (mod_data - np.random.randint(mod_data))
                    print("original:", mod_data)
                    mod_data = random.randint(0, mod_data)
                    print("random_aft:", mod_data)

                    mod_data = struct.pack('>Q', mod_data)
                    mod_data = struct.unpack('>d', mod_data)
                    #print("mod : ", mod_data)

                    array_current[i] = mod_data

            save_df = pd.DataFrame(array_current)
            # null check
            print(save_df.isnull().any())
            #print(error_rate[inject_ber])
            temp_err_name = str(error_rate[inject_ber])
            #os.makedirs(os.path.join(output_path_dir, temp_err_name), exist_ok=True)
            output_name = os.path.join(output_path_dir, temp_err_name, weight_name_list[current_weight_num] + '_' + str(
                error_rate[inject_ber]) + extention_csv)
            print("See result : ", output_name)
            save_df.to_csv(output_name, header=None, index=None)



if __name__ == "__main__":
    main()
else:
    print(__name__)