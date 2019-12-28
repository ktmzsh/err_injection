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
parser.add_argument("--info", default=0)
args = parser.parse_args()

##### Path list
# define weight data root
original_weight_root = '/home/mizushina/segnet/Weight_data_original/'
extention_all = "*"
extention_csv = '.csv'

# list of weight list
weight_name_list = ['conv1', 'conv2', 'conv3', 'conv4',\
                    'conv_deconv1', 'conv_deconv2', 'conv_deconv3', 'conv_deconv4']


# Define Excel format file and dir
excel_dir = "Excel_file"
excel_file_name = "Bics3_#11_800K_ReadDisturb.csv"
trans_path = os.path.join(excel_dir, excel_file_name)

pe = [1, 2, 10, 30, 100, 300, 1000, 3000, 7000, 10000, 15000, 30000, 50000, 100000,\
          200000, 300000, 400000, 500000, 600000, 700000, 800000]

output_path_dir = 'injected_weight_csv'
##### Path list end

##### Calc trans
# This function calculates BER and plot them w/ matplotlib.
def read_trans_data(trans_path):
    #print("This function reads trans data. And this result is inserted to numpy array.")

    # Read CSV file with plane text.
    trans_data = np.loadtxt(trans_path, delimiter=",")
    trans_divide = np.split(trans_data, 21)

    # Define empty array and PE cycle
    array = []
    array_lower = []
    array_middle = []
    array_upper = []


    # LOOP for calc BER every PEs
    for i in range(21):
        target_trans = trans_divide[i]
        U_BER, M_BER, L_BER = calc_error_rate(target_trans)
        TOTAL_BER = ((U_BER + M_BER + L_BER) / 3)
        #print("TOTAL : ", TOTAL_BER, "%")
        array.append(TOTAL_BER)
        array_lower.append(U_BER)
        array_middle.append(M_BER)
        array_upper.append(U_BER)

    if(args.info != 0):
        plt.plot(pe, array, label="Total BER")
        plt.legend()
        plt.show()

    return array, array_lower, array_middle, array_upper

# This function is called with trans data and return each BER.
def calc_error_rate(trans_data):
    elemnts_sum = np.sum(trans_data)

    # calc middle BER
    lower_part1 = np.sum(trans_data[0][1:4])
    lower_part2 = np.sum(trans_data[1:5, 0])
    lower_part3 = np.sum(trans_data[5:8, 1:5])
    lower_part4 = np.sum(trans_data[1:5, 5:8])
    lower_sum = lower_part1 + lower_part2 + lower_part3 + lower_part4
    Lower_ber = lower_sum / elemnts_sum
    #print("Lower BER : ", Lower_ber, " %")

    #calc middle BER
    middle_part1 = np.sum(trans_data[2:4, 0:2])
    middle_part2 = np.sum(trans_data[6:8, 0:2])
    middle_part3 = np.sum(trans_data[0:2, 2:4])
    middle_part4 = np.sum(trans_data[4:6, 2:4])
    middle_part5 = np.sum(trans_data[2:4, 4:6])
    middle_part6 = np.sum(trans_data[6:8, 4:6])
    middle_part7 = np.sum(trans_data[0:2, 6:8])
    middle_part8 = np.sum(trans_data[4:6, 6:8])
    middle_sum = middle_part1 + middle_part2 + middle_part3 + middle_part4 \
                 + middle_part5 + middle_part6 + middle_part7 + middle_part8
    Middle_ber = middle_sum / elemnts_sum
    #print("Middle BER : ", Middle_ber, " %")

    # calc high BER
    high_part1 = np.sum(trans_data[3:7, 0:3])
    high_part2 = np.sum(trans_data[0:3, 3:7])
    high_part3 = np.sum(trans_data[7:8, 3:7])
    high_part4 = np.sum(trans_data[3:7, 7:8])
    high_sum = high_part1 + high_part2 + high_part3 + high_part4
    Upper_ber = high_sum / elemnts_sum
    #print("High BER : ", Upper_ber, " %")


    return Lower_ber, Middle_ber, Upper_ber


ARR_total_BER, ARR_lower, ARR_middle, ARR_upper = read_trans_data(trans_path)

if(args.info != 0):
    for j in range(len(ARR_total_BER)):
        print("PE cycyle {} : T_BER {}, L_BER : {}, M_BER : {}, U_BER : {}".format(pe[j],
                                                                                   ARR_total_BER[j],
                                                                                   ARR_lower[j],
                                                                                   ARR_middle[j],
                                                                                   ARR_upper[j]))


#####

def main():
    full_original_weight = os.path.join(original_weight_root, extention_all + extention_csv)
    weight_csv_list = glob.glob(full_original_weight)
    print("Number of weight : ", len(weight_csv_list))
    print("Path of weight data : ", weight_csv_list)

    for current_weight_num in range(8): #0:conv1, ... , 7:deconv4
        for pe_cycle in range(21):  #0:pe1,...,20:pe800K
            df = pd.read_csv(weight_csv_list[current_weight_num], delimiter=',', header=None, names='A')
            # Display all data-frame
            arr = df.values
            parameter_num = arr.shape[0]
            error_cell_num = int(parameter_num * ARR_total_BER[pe_cycle])

            print("weight contains {} params.".format(parameter_num))
            print("Progress... \ncurrently pe : {} and T_BER is {}".format(pe[pe_cycle], ARR_total_BER[pe_cycle]))
            print("...almost {} params causes errors.".format(error_cell_num))

            for i in range(parameter_num):
                if (random.random() < ARR_total_BER[pe_cycle]):
                    #print(arr[i])
                    float_temp = float(arr[i])
                    bin_data = struct.pack('>d', float_temp)
                    mod_data = struct.unpack('>Q', bin_data)
                    mod_data = bin(mod_data[0])
                    mod_data = (mod_data)
                    mod_data = int(mod_data, 2)
                    #print("Ordinal int : ", (mod_data))
                    #########################################このパラメータでエラーの調整ができる###########################
                    mod_data = (mod_data - random.randint(0, 100000000000000000))
                    #print("Shifted int : ", (mod_data))
                    mod_data = struct.pack('>Q', mod_data)
                    mod_data = struct.unpack('>d', mod_data)
                    #print("mod test : {}".format((mod_data)))

                    arr[i] = mod_data
                    #print(arr[i])

            save_df = pd.DataFrame(arr)
            output_name = os.path.join(output_path_dir, weight_name_list[current_weight_num]+'_PEA'+str(pe[pe_cycle])+extention_csv)
            print("See result : ",output_name)
            #save_df.to_csv(output_name, header=None, index=None)


if __name__ == "__main__":
    main()