# import os
# from pathlib import Path
# import csv
import openneuro

dataset_id = "ds002721"
openneuro.download(dataset=dataset_id, target_dir='./data/ds002721-raw')




# # for i in range(0, 10):
# #   path = 'data/ds002721-master/sub-3' + str(i) + '/eeg'  # Use forward slashes or double backslashes
# #   out = 'data/ds002721-prep/sub-3' + str(i) + '/eeg'

# #   os.makedirs(os.path.dirname(path), exist_ok=True)

# #   for filename in os.listdir(path):
# #     if 'events.tsv' in filename:
# #         filepath = os.path.join(path, filename)  # Construct the full file path
# #         outpath = os.path.join(out, filename)
# #         print(filename)
# #         os.makedirs(os.path.dirname(outpath), exist_ok=True)
# #         with open(filepath, 'r') as tsvin, open(outpath, 'w+', newline='') as tsvout:
# #             reader = csv.reader(tsvin, delimiter='\t')
# #             writer = csv.writer(tsvout, delimiter='\t')
# #             for line in reader:
# #                 if line[2] == '788' or line[2] in map(str, (range(300, 661))):
# #                   print(line)
# #                   writer.writerow(line)

# for i in range(0, 10):
#   path = 'data/ds002721-prep/sub-3' + str(i) + '/eeg'
#   os.makedirs(os.path.dirname(path), exist_ok=True)
#   for filename in os.listdir(path):
#     filepath = os.path.join(path, filename)

#     # Read the data from the input file
#     data = []
#     with open(filepath, 'r') as file:
#         reader = csv.reader(file, delimiter='\t')
#         for row in reader:
#             data.append(row)

#     # print(data)

#     # Perform the replacement
#     new_data = []
#     for i, row in enumerate(data):
#         # print(i)
#         if i < 2:
#           new_data.append(row)
#         else:
#           if i % 2 != 0:
#             new_data.append(row)
#             # val = row[0]
#           else:
#               row[0] = data[i+1][0]
#               new_data.append(row)
#     # print(new_data)

#     # # Write the updated data to the output file
#     with open(filepath, 'w', newline='') as file:
#         writer = csv.writer(file, delimiter='\t')
#         writer.writerows(new_data)
