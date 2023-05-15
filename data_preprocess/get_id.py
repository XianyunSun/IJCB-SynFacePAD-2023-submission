import os
import pandas as pd

# change the path according to your situation
img_folder = r'/sdata/xianyun.sun/SynthASpoof_data_crop'
label_file = r'/sdata/xianyun.sun/SynthASpoof_data_crop/labels.csv'

label_df = pd.read_csv(label_file)
img_path = list(label_df['image_path'])

id_list = []
id_max = -1
for p in img_path:
    img_name = p.split('/')[-1]
    img_name = img_name.split('.')[0]
    img_id = int(img_name[3:])
    if img_id>id_max:
        id_max = img_id
    id_list.append(img_id)

label_df['id'] = id_list
label_df.to_csv(label_file, index=False)
print(id_max)
