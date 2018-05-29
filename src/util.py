import os
import numpy as np
import cv2
import csv
import re
def show_heatmap(mat,type='opencv'):
    """Show heatmap using opencv or matplotlib.

    Args:
        mat: numpy array, can be a greyscaled image or probability map.
        type: show mat in 'opencv' or 'matplotlib', while 'opencv' is default.
    """
    if type=='matplot':
        plt.figure()
        plt.imshow(mat,cmap="jet")
        cb = plt.colorbar()
        cb.ax.set_ylabel('Frequency')
        plt.show()
    else:
        heatmap = colorize(mat)
        cv2.imshow('Heatmap',np.uint8(np.clip(heatmap, 0, 255)))
        cv2.waitKey(20)

def write_mat_to_csv(mat,csvfile,row_by_row=True):
    """Reshape a 2D matrix to a vector and save to csv file.

    Args:
        mat: a 2D numpy array to save.
        csvfile: csv writer used to write file.
        row_by_row(optional): take by row or column.
    """
    assert len(mat.shape)==2
    if row_by_row:
        i_total = mat.shape[0]
        j_total = mat.shape[1]
    else:
        i_total = mat.shape[1]
        j_total = mat.shape[0]
    string_to_save = []

    for i in range(i_total):
        for j in range(j_total):
            if row_by_row:
                string_to_save.append(str(mat[i,j]))
            else:
                string_to_save.append(str(mat[j,i]))
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(string_to_save)

def get_file_fullpath_list(dir_path, file_format):
    """Get a full file path list from directory regarding file format.

    Args:
        dir_path: a string indicating the directory to get file list.
        file_format: a string indicating the file format to filter, e.g. 'mp4'.

    Returns:
        file_list: a string list containing all file names in specified file 
                    format.
    """

    file_list = []
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for file in files:
            if file_format in file:
            # if file_format in file and len(file)<=len(file_format)+3:
                file_list.append(os.path.join(root, file))
    file_list = sorted(file_list,key=lambda x: (int(re.sub('\D','',x)),x))
    return file_list

def get_file_list(dir_path, file_format):
    """Get a file name list from directory regarding file format.

    Args:
        dir_path: a string indicating the directory to get file list.
        file_format: a string indicating the file format to filter, e.g. 'mp4'.

    Returns:
        file_list: a string list containing all file names in specified file 
                    format.
    """

    file_list = []
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for file in files:
            if file_format in file:
                file_list.append(file)
    file_list = sorted(file_list,key=lambda x: (int(re.sub('\D','',x)),x))
    return file_list

def get_dir_list(dir_path):
    """Get a dir name list under the directory.
        
    Args:
        dir_path: a string indicating the directory to get dir list.
    Returns:
        dir_list: a string list containing all dir names under specified folder.
    """ 

    dir_list = []
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for dirr in dirs:
            dir_list.append(dirr)
    return dir_list

def map_mats_to_matl(mat_s,mat_l,drift_x,drift_y):
    """This function is used to copy mat_s to mat_l with drift x,y.

    Args:
        mat_s: numpy array, small matrix to copy from.
        mat_l: numpy array, large matrix to copy to.
        drift_x,drift_y: an integer indicating drift coordinate.
    Returns:
        mat_rst: numpy array, result matrix in the same shape as large matrix.
    """
    
    mat_rst = np.array(mat_l)
    [xp_st,xp_ed,yp_st,yp_ed,xi_st,xi_ed,yi_st,yi_ed] = \
                                        map_mat_XY(mat_s,mat_l,drift_x,drift_y)
    mat_rst[yi_st:yi_ed,xi_st:xi_ed] = mat_s[yp_st:yp_ed,xp_st:xp_ed]
    return mat_rst

def map_matl_to_mats(mat_s,mat_l,drift_x,drift_y):
    """This function is used to extract part of mat_l to mat_s with drift x,y.

    Args:
        mat_s: numpy array, small matrix to copy to.
        mat_l: numpy array, large matrix to copy from.
        drift_x,drift_y: an integer indicating drift coordinate.
    Returns:
        mat_rst: numpy array, result matrix in the same shape as small matrix.        
    """

    mat_rst = np.array(mat_s)
    [xp_st,xp_ed,yp_st,yp_ed,xi_st,xi_ed,yi_st,yi_ed] = \
                                        map_mat_XY(mat_s,mat_l,drift_x,drift_y)
    mat_rst[yp_st:yp_ed,xp_st:xp_ed] = mat_l[yi_st:yi_ed,xi_st:xi_ed] 
    return mat_rst 

def map_mat_XY(mat_s,mat_l,drift_x,drift_y):
    """This function is used as a subfunction for mapping small and large 
        matrix. 
    To map a small matrix to a large matrix, this function will return the start 
    and end coordinates (diagonal) of two mapping matrix.

    Args:
        mat_s,mat_l: numpy array to calculate the coordinates
        drift_x,drift_y: an integer indicating drift coordinate.
    Returns:
        xp_st,xp_ed,yp_st,yp_ed: integers, start and end coordinates for small 
                                    matrix.
        xi_st,xi_ed,yi_st,yi_ed: integers, start and end coordinates for large 
                                    matrix.
    """

    assert len(mat_s.shape)==2 and len(mat_l.shape)==2 and \
            mat_s.shape[0]<=mat_l.shape[0] and mat_s.shape[1]<=mat_l.shape[1]

    xp_st,yp_st = 0,0
    xp_ed,yp_ed = mat_s.shape[1],mat_s.shape[0]
    xi_st = 0 - mat_s.shape[1]/2 + drift_x
    yi_st = 0 - mat_s.shape[0]/2 + drift_y
    xi_ed = mat_s.shape[1]/2 + drift_x
    yi_ed = mat_s.shape[0]/2 + drift_y
    if xi_st<0:
        xp_st = -xi_st
        xi_st = 0
    if yi_st<0:
        yp_st = -yi_st
        yi_st = 0
    if xi_ed>mat_l.shape[1]:
        xp_ed = xp_ed-(xi_ed-mat_l.shape[1])
        xi_ed = mat_l.shape[1]
    if yi_ed>mat_l.shape[0]:
        yp_ed = yp_ed-(yi_ed-mat_l.shape[0])
        yi_ed = mat_l.shape[0]    
    return [xp_st,xp_ed,yp_st,yp_ed,xi_st,xi_ed,yi_st,yi_ed]

def get_jet_color(v, vmin, vmax):
    """Colorize pixel value in heatmap shape.

    Args:
        v: float, pixel value.
        vmin,vmax: floats, range of pixel value.
    Returns:
        c: float, colorized value.
    """
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)): 
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
    return c

def colorize(gray_img):
    """Colorize matrix to heatmap.

    Args:
        gray_img: 2d numpy array(single channel), can be a grey-scaled image of 
                    probability map.
    Returns:
        out_img: numpy array of BGR channels, colorized image.
    """
    out_img = np.zeros(gray_img.shape + (3,))
    for y in range(out_img.shape[0]):
        for x in range(out_img.shape[1]):
            out_img[y,x,:] = get_jet_color(gray_img[y,x], 0, 1)
    return out_img

def pad_right_down_corner(img):
    """Pad the image at its right-down corner, make it divisible for 8.

    Args:
        img: numpy array, image matrix.
    Returns:
        img_padded: numpy array, padded image.
        pad: integer list indicating the number of padding pixel at four 
            corners.
    """
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%8==0) else 8 - (h % 8) # down
    pad[3] = 0 if (w%8==0) else 8 - (w % 8) # right
    
    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + 128, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]* + 128, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]* + 128, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + 128, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def read_data_in_txt(txt_path,row_num):
    """Read in the string data(predction position) in txt file, cast into 'int' type, 
        need to speccify the number of rows in the txt file.
    """

    rawText = open(txt_path)
    data = []
    row = row_num

    for x in range(0,row):
        data.append([])

    line_num = 0
    for line in rawText.readlines():
        for i in line.split():
            data[line_num].append(int(float(i)))
        line_num = line_num + 1

    rawText.close()
    return data

def get_jonit_pos_sequence(txt_top_path,joint_num, type):
    """This function returns the joint positions along time series, for comparing the performance of traackers

    Args:
        txt_top_path: folder path that contains all txt files;
        joint_number: joint num, 4 for right wrist
        type: cpm or annotion; for row number in txt 14 or 1 
    Returns:
        joint_pos : a list that contains all joint postions in the video frames
    """

    # path_list= util.get_file_list(txt_top_path, 'txt')
    path_list= get_file_fullpath_list(txt_top_path, file_format='txt')

    path_list = sorted(path_list,key=lambda x: (int(re.sub('\D','',x)),x))
    
    joint_pos = []
    for x in range(0,len(path_list)):
        joint_pos.append([])

    if (type == "cpm"):
        row_num = 14
    elif (type == "annotion"):
        row_num = 1
    else :
        print "wrong specified type in function: get_jonit_pos_sequence() "

    for i in range(0,len(path_list)):
        joint_pos[i]= read_data_in_txt(path_list[i],row_num)[joint_num]

    return joint_pos

def get_full_path_under_folder(top_path):
    """
    this function returns the fisrt order subdirectories and files with a full path in a folder
    top_path should end with '/', else will not find the subdirectories.
    """
    if not os.path.isdir(top_path):
        print 'Error: top_path '

    all_folder_list=[]
    folder_names_list =  os.listdir(top_path)
    folder_num = len(folder_names_list)

    for i in range(0,folder_num):
        full_path = top_path + folder_names_list[i]
        if not os.path.isfile(full_path):
            # then folder
            full_path = full_path +'/' 
        all_folder_list.append(full_path)

    return all_folder_list


