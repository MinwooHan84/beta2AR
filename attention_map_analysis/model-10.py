import numpy as np
import nibabel as nib
import pandas as pd
import sys
import os

# define nii_loader(n=number of trajectory, cutoff=number of top signal pixel)
def nii_loader(n,m,cutoff):
    # read nii images
    img = nib.load('layer4/attention_map_%d_%d_0.nii.gz' %(n, m))
    img_data = img.get_fdata()
    # flattening and sorting pixels
    flat = np.ravel(img_data, order='C')
    sflat = sorted(flat, reverse=True)
    # indexing lighter pixels
    idx = np.empty((0,3), dtype = int)
    for m in range(cutoff):
        idx_add = np.argwhere(img_data == sflat[m])
        idx = np.append(idx, idx_add, axis = 0)
    # remove frame column
    res_inter = np.delete(idx, 2, axis = 1)
    # residue number modification
    total_residue = 312 # loop = 326, noloop = 312
    #starting_residue = 21
    #loop_start_residue = 243 # noloop case
    #missing_residues = 14
    for m in range(len(res_inter)):
        # start from residue number 1
        res_inter[m][0] += 1
        res_inter[m][1] += 1
        # y axis modification
        res_inter[m][1] = total_residue - res_inter[m][1] + 1
    #    # renumbering residue
    #    res_inter[m][0] = res_inter[m][0] + starting_residue
    #    res_inter[m][1] = res_inter[m][1] + starting_residue
    #    # for noloop case
    #    if res_inter[m][0] >= loop_start_residue :
    #        res_inter[m][0] = res_inter[m][0] + missing_residues
    #    if res_inter[m][1] >= loop_start_residue :
    #        res_inter[m][1] = res_inter[m][1] + missing_residues
    #remove duplicate
    res_xy = remove_dupl(res_inter)
    #vmd_format(res_xy)
    return res_xy

# function removing duplicate in 2-D list
def remove_dupl(dupl):
    no_dupl = list(set([tuple(set(dupl)) for dupl in dupl]))
    #sorting and transform tuple to list
    xy = np.empty((len(no_dupl), 2), int)
    for i in range(len(no_dupl)):
        xy[i][0] = no_dupl[i][0]
        xy[i][1] = no_dupl[i][1]
        if xy[i][0] > xy[i][1]:
            tmp = xy[i][1]
            xy[i][1] = xy[i][0]
            xy[i][0] = tmp
    #sorting xy
    xy = sorted(xy, key = lambda x:x[1])
    xy = sorted(xy, key = lambda x:x[0])
    return xy

# fucntion for multiple trajectories
def multiple_trj(start_trj, no_trj):
    res_xy_all = np.empty((0,2), dtype = int)
    end_trj = start_trj + no_trj
    cutoff = 1 # number of pixels(interactions) collecting from the brightest one
    print("\ncalculating", start_trj, "-",end_trj-1, "trajectories")
    for n in range(start_trj, end_trj):
        for m in range(0,3): # the range is determined by the inference stride
                #print("trajectory %d"%n)
                res_xy_all = np.append(res_xy_all, nii_loader(n,m,cutoff), axis = 0)
                #print("\n")
    #removing duplicate
    res_xy_all = remove_dupl(res_xy_all)
    return res_xy_all

# printing in VMD format
def vmd_format(res_xy_all):
    print(len(res_xy_all),"interactions")
    for i in range(len(res_xy_all)):
        print("resid", res_xy_all[i][0], res_xy_all[i][1])
    #print("\n")
    #flattening, sorting, removing duplicates
    flat = np.ravel(res_xy_all, order='C')
    #VMD format
    rm_dupl = str(sorted(list(set(flat))))[1:-1]
    rm_dupl = rm_dupl.replace(',', '')
    print("\none line")
    print("resid", rm_dupl)

# collect duplicate
def collect_dupl(arr1,arr2):
    dupl = np.empty((0,2), dtype = int)
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            if arr1[i][0] == arr2[j][0] and arr1[i][1] == arr2[j][1]:
                dupl = np.append(dupl, np.array([arr1[i]]), axis = 0)
    print(len(dupl), "common interactions")
    #print(dupl)
    #flattening, sorting, removing duplicates
    flat = np.ravel(dupl, order='C')
    #VMD format
    rm_dupl = str(sorted(list(set(flat))))[1:-1]
    rm_dupl = rm_dupl.replace(',', '')
    print("resid", rm_dupl)

# Pymol draw line script
def pymol_script(res_xy_all):
    sys.stdout = open('draw_line.pml','w')
    for i in range(len(res_xy_all)):
        print('select r{}, /3nya///{}/CA'.format(res_xy_all[i][0],res_xy_all[i][0]))
        print('select r{}, /3nya///{}/CA'.format(res_xy_all[i][1],res_xy_all[i][1]))
        print('distance r{}-r{}, r{}, r{}'.format(res_xy_all[i][0],res_xy_all[i][1],res_xy_all[i][0],res_xy_all[i][1]))
        print('color blue, r{}-r{}'.format(res_xy_all[i][0],res_xy_all[i][1]))
        print('hide labels, r{}-r{}'.format(res_xy_all[i][0],res_xy_all[i][1]))
        print('delete r{}'.format(res_xy_all[i][0],res_xy_all[i][0]))
        print('delete r{}\n'.format(res_xy_all[i][1],res_xy_all[i][1]))
    print('set dash_gap, 0.5')
    print('set dash_radius, 0.1')

# Counting interacting residues
def count_residue(res_xy_all):
    f = open('cnt.txt','w')
    cnt_res = [0]*312
    for i in range(len(res_xy_all)):
        cnt_res[res_xy_all[i][0]-1] += 1
        cnt_res[res_xy_all[i][1]-1] += 1
    for i in range(len(cnt_res)):
        f.write('{}\n'.format(cnt_res[i]))
    f.close()

# running multiple_trj(start_trj, no_trj)
#print("\nIA3p0g")
#IA3p0g = multiple_trj(0,50)    # agonist
#pymol_script(IA3p0g)
#count_residue(IA3p0g)
#vmd_format(IA3p0g)

#print("\nIA6mxt")
#IA6mxt = multiple_trj(50,50)   # agonist
#pymol_script(IA6mxt)
#count_residue(IA6mxt)
#vmd_format(IA6mxt)

#print("\nIN3nya")
#IN3nya = multiple_trj(100,50)   # antagonist
#pymol_script(IN3nya)
#count_residue(IN3nya)
#vmd_format(IN3nya)

#print("\nIN6ps6")
#IN6ps6 = multiple_trj(150,50)   # antagonist
#pymol_script(IN6ps6)
#count_residue(IN6ps6)
#vmd_format(IN6ps6)

#print("\nIP3nya")
#IP3nya = multiple_trj(200,50)   # apo
#pymol_script(IP3nya)
#count_residue(IP3nya)
#vmd_format(IP3nya)

#print("\nIA4ldo")
#IA4ldo = multiple_trj(250,50)   # agonist
#pymol_script(IA4ldo)
#count_residue(IA4ldo)
#vmd_format(IA4ldo)

print("\nIN6ps5")
IN6ps5 = multiple_trj(300,50)   # antagonist
pymol_script(IN6ps5)
count_residue(IN6ps5)
#vmd_format(IN6ps5)

#II2rh1 = multiple_trj(100,50) # inverse agonist
#II3ny8 = multiple_trj(150,50) # inverse agonist
#IN3nya = multiple_trj(200,50) # antagonist
#IN6ps6 = multiple_trj(250,50) # antagonist
#IP3nya = multiple_trj(300,50) # apo

# collecting common interaction
#print("\nagonist : IA3p0g and IA6mxt")
#collect_dupl(IA3p0g,IA6mxt)
#print("\nagonist : IA3p0g and IA4ldo")
#collect_dupl(IA3p0g,IA4ldo)
#print("\nagonist : IA6mxt and IA4ldo")
#collect_dupl(IA6mxt,IA4ldo)
#print("\nantagonist : IN3nya and IN6ps6")
#collect_dupl(IN3nya,IN6ps6)
#print("\nantagonist : IN3nya and IN6ps5")
#collect_dupl(IN3nya,IN6ps5)
#print("\nantagonist : IN6ps6 and IN6ps5")
#collect_dupl(IN6ps6,IN6ps5)
#print("\ninverse agonist : II2rh1 and II3ny8")
#collect_dupl(II2rh1,II3ny8)
#print("\nantagonist : IN3nya and IN6ps6")
#collect_dupl(IN3nya,IN6ps6)
#print("\nIA3p0g and IN3nya")
#collect_dupl(IA3p0g,IN6ps6)
