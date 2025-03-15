 import pickle

def get_annotations(txt_file,pkl_file):
    annotations =[]
    if not isinstance(pkl_file,list):
        with open(pkl_file,'rb') as f:
            data = pickle.load(f)
    else:
        data = pkl_file
    for item in data:
        vid_name = item['frame_dir']+'_rgb.mp4'
        vid_label = item['label']
        vid_annot = vid_name +' '+str(vid_label)
        annotations.append(vid_annot)
    print(f" annotations len of {txt_file} is {len(annotations)}")
    # == write it to file ===#
    with open(txt_file, 'w') as f:
        for vid_annot in annotations:
            f.write(f"{vid_annot} \n")
    return annotations
def getCorrespondingAnnots(dir_names,annotations):
    ntu_annot = []
    for video_name in dir_names:
        cor_annot = None
        for item in annotations:
            if item['frame_dir'] == video_name:
                cor_annot = item
                break
        if cor_annot:
            ntu_annot.append(cor_annot)
        else:
            print("error at ", video_name)
    return  ntu_annot

NTU60 = False # NTU120
flag = 'xset'  # 'xsub'

ntu_pkl_path = 'ntu120_2d.pkl'  #
with open(ntu_pkl_path, 'rb') as f:
    data = pickle.load(f)
    annotations = data['annotations']
if flag == 'xsub':
    xsub_train_dir_names = data['split']['xsub_train']
    xsub_val_dir_names = data['split']['xsub_val']
    ntu120_xsub_train = getCorrespondingAnnots(xsub_train_dir_names,annotations)
    ntu120_xsub_val = getCorrespondingAnnots(xsub_val_dir_names,annotations)
    #=============save pkl files========#
    with open('ntu120_xsub_train.pkl', 'wb') as file:
        pickle.dump(ntu120_xsub_train, file)
    with open('ntu120_xsub_val.pkl', 'wb') as file:
        pickle.dump(ntu120_xsub_val, file)
    get_annotations('ntu120_xsub_train.txt', ntu120_xsub_train)
    get_annotations('ntu120_xsub_val.txt', ntu120_xsub_val)
elif flag == 'xset':
    xset_train_dir_names = data['split']['xset_train']
    xset_val_dir_names = data['split']['xset_val']
    ntu120_xset_train = getCorrespondingAnnots(xset_train_dir_names,annotations)
    ntu120_xset_val = getCorrespondingAnnots(xset_val_dir_names,annotations)

    # =============save pkl files========#
    with open('ntu120_xset_train.pkl', 'wb') as file:
        pickle.dump(ntu120_xset_train, file)
    with open('ntu120_xset_val.pkl', 'wb') as file:
        pickle.dump(ntu120_xset_val, file)
    get_annotations('ntu120_xset_train.txt', ntu120_xset_train)
    get_annotations('ntu120_xset_val.txt', ntu120_xset_val)
