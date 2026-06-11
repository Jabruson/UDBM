import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_dataset_all, make_dataset_all_text, make_dataset_3, make_dataset_5, make_dataset_6, make_dataset_4, make_dataset_2
from PIL import Image
from pathlib import Path
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import Augmentor
import cv2
import glob
class AlignedDataset_all(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory 'data train folder' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory 'data test folder'.
    """

    def __init__(self, opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = image_size
        self.opt = opt
        #origin----------------------------------------------------------------------------------------------------------
        self.dir_Arain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/rainy_image')
        self.dir_Brain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/ground_truth')
        self.dir_Alsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/low')
        self.dir_Blsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/high')
        self.dir_Alol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/low')
        self.dir_Blol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/high')
        
        if opt.phase == 'train':
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L'+'/synthetic')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase +'/Snow100K-L'+ '/gt')
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/input')
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/target')
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/target')

            flog_prefix = os.path.join(opt.dataroot, 'RESIDE/OTS_ALPHA/')
            self.dir_Afog = flog_prefix + 'haze/OTS'
            self.dir_Bfog = flog_prefix + 'clear/clear_images'
        else:
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')


            self.dir_Asnow1 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow1 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')
            self.dir_Asnow2 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow2 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/gt')
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic')


            self.dir_Arain_syn1 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn1 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/target')  #Test2800
            self.dir_Arain_syn2 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn2 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/target')  #Test2800
            self.dir_Arain_syn3 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100L/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn3 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100L/target')  #Test2800
            self.dir_Arain_syn4 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test100/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn4 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test100/target')  #Test2800
            self.dir_Arain_syn5 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test1200/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn5 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test1200/target')  #Test2800



            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/target')
            self.dir_Afog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy')
            self.dir_Bfog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt')
            self.dir_Aasd = os.path.join(opt.dataroot, 'temp')
            self.dir_Basd = os.path.join(opt.dataroot, 'temp')



            #real dark:
            self.dir_A_real_dark_mef=os.path.join(opt.dataroot,"real_dark/real_dark/MEF")
            self.dir_A_real_dark_npe=os.path.join(opt.dataroot,"real_dark/real_dark/NPE")
            self.dir_A_real_dark_dice=os.path.join(opt.dataroot,"real_dark/real_dark/DICE")

            # real rain: <dataroot>/real_rain/real_rain/Practical
            self.dir_A_real_rain=os.path.join(opt.dataroot,"real_rain/real_rain/Practical")

            # real snow: <dataroot>/Snow100K/realistic
            self.dir_A_real_snow=os.path.join(opt.dataroot,"Snow100K/realistic")


            self.dir_Ablur_hide = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/HIDE/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur_hide = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/HIDE/target')

            self.dir_Ablur_j = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/RealBlur_J/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur_j = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/RealBlur_J/target')

            self.dir_Ablur_r = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/RealBlur_R/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur_r = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/RealBlur_R/target')
            


            ############################CD 11############################################
            cd11_prefix = os.path.join(opt.dataroot, 'cd11')
            self.dir_Bcd11=os.path.join(cd11_prefix, "clear")
            # h hr hs; l lh lhr lhs lr ls r s;
            self.dir_Acd11_h=os.path.join(cd11_prefix, "haze")
            self.dir_Acd11_hr=os.path.join(cd11_prefix, "haze_rain")
            self.dir_Acd11_hs=os.path.join(cd11_prefix, "haze_snow")
            self.dir_Acd11_l=os.path.join(cd11_prefix, "low")
            self.dir_Acd11_lh=os.path.join(cd11_prefix, "low_haze")
            self.dir_Acd11_lhr=os.path.join(cd11_prefix, "low_haze_rain")
            self.dir_Acd11_lhs=os.path.join(cd11_prefix, "low_haze_snow")
            self.dir_Acd11_lr=os.path.join(cd11_prefix, "low_rain")
            self.dir_Acd11_ls=os.path.join(cd11_prefix, "low_snow")
            self.dir_Acd11_r=os.path.join(cd11_prefix, "rain")
            self.dir_Acd11_s=os.path.join(cd11_prefix, "snow")
            #
            self.dir_Adense_haze=""
            self.dir_Bdense_haze=""
            #real udc
            # self.dir_A_real_udc=os.path.join(opt.dataroot,"real_rain/real_rain/Practical")
        #test
        if task == 'light':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset_2(self.dir_Alol, self.dir_Alsrw, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset_2(self.dir_Blol, self.dir_Blsrw, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'light_only':
            self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'rain':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset(self.dir_Arain_syn, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Brain_syn, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Arain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn5, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Brain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn5, opt.max_dataset_size))
        
                if len(self.A_paths) != len(self.B_paths):
                    raise ValueError(
                        f"Mismatched file list lengths: A has {len(self.A_paths)} files, "
                        f"B has {len(self.B_paths)} files."
                    )

                print("Checking paired file names between input and target folders...")
                mismatched_files = []
                for path_a, path_b in zip(self.A_paths, self.B_paths):
                    # os.path.basename('path/to/file.png') -> 'file.png'
                    # os.path.splitext('file.png') -> ('file', '.png')
                    # [0] -> 'file'
                    basename_a = os.path.splitext(os.path.basename(path_a))[0]
                    
                    basename_b = os.path.splitext(os.path.basename(path_b))[0]

                    if basename_a != basename_b:
                        mismatched_files.append((path_a, path_b))

                if mismatched_files:
                    error_msg = f"Error: detected {len(mismatched_files)} filename mismatches.\n"
                    error_msg += "Filename mismatch examples, up to 5:\n"
                    for a, b in mismatched_files[:5]:
                        error_msg += f"  - A: {os.path.basename(a)} (from {a})\n"
                        error_msg += f"  - B: {os.path.basename(b)} (from {b})\n"
                        error_msg += "---\n"
                    raise ValueError(error_msg)
                
                print(f"Pair check passed for {len(self.A_paths)} files.")
        elif task=='mix_train':
            self.A_paths = sorted(make_dataset_2(self.dir_Alol, self.dir_Alsrw, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_2(self.dir_Blol, self.dir_Blsrw, opt.max_dataset_size))

            self.A_paths = sorted(make_dataset(self.dir_Arain_syn, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn, opt.max_dataset_size))

            self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
        elif task == 'rain1':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn1, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn1, opt.max_dataset_size))
        elif task == 'rain2':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn2, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn2, opt.max_dataset_size))
        elif task == 'rain3':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn3, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn3, opt.max_dataset_size))
        elif task == 'rain4':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn4, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn4, opt.max_dataset_size))
        elif task == 'rain5':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn5, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn5, opt.max_dataset_size))
        
        elif task == 'snow1':
            if opt.phase == 'train':
                 self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
                 self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Asnow1, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Bsnow1, opt.max_dataset_size))
        elif task == 'snow2':
            if opt.phase == 'train':
                 self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
                 self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Asnow2, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Bsnow2, opt.max_dataset_size))
        elif task == 'snow':
            if opt.phase == 'train':
                 self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
                 self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Asnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Asnow2, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Bsnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Bsnow2, opt.max_dataset_size))
        elif task == 'blur':
            self.A_paths = sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))
        elif task == 'fog':
            self.A_paths = sorted(make_dataset(self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bfog, opt.max_dataset_size))
            if opt.phase!="train":
                files_a = sorted(glob.glob(os.path.join(self.dir_Bfog, "*.png")))
                files_b = sorted(glob.glob(os.path.join(self.dir_Afog, "*.jpg")))
                self.A_paths, self.B_paths = [], []

                dict_b = {}
                for fb in files_b:
                    name = os.path.basename(fb).split("_")[0]
                    dict_b.setdefault(name, []).append(fb)

                for fa in files_a:
                    name = os.path.splitext(os.path.basename(fa))[0]
                    if name in dict_b:
                        for fb in dict_b[name]:
                            self.A_paths.append(fb)
                            self.B_paths.append(fa)
        elif task=="all":
            if opt.phase!="train":
                # --- 1. 'light' (test) ---
                self.A_paths=[]
                self.B_paths=[]
                self.A_paths += sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
                
                # --- 2. 'light_only' (test) --- 
                # In test mode, light_only and light share the same fallback branch.

                # --- 3. 'rain' (test) ---
                self.A_paths += sorted(make_dataset(self.dir_Arain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn5, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Brain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn5, opt.max_dataset_size))

                # --- 4. 'snow' (test) ---
                self.A_paths += sorted(make_dataset(self.dir_Asnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Asnow2, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Bsnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Bsnow2, opt.max_dataset_size))

                self.A_paths += sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))

                # --- 6. 'fog' (test) ---
                files_a_fog = sorted(glob.glob(os.path.join(self.dir_Bfog, "*.png")))
                files_b_fog = sorted(glob.glob(os.path.join(self.dir_Afog, "*.jpg")))
                
                A_paths_fog_test = []
                B_paths_fog_test = []

                dict_b_fog = {}
                for fb in files_b_fog:
                    name = os.path.basename(fb).split("_")[0]
                    dict_b_fog.setdefault(name, []).append(fb)

                for fa in files_a_fog:
                    name = os.path.splitext(os.path.basename(fa))[0]
                    if name in dict_b_fog:
                        for fb in dict_b_fog[name]:
                            A_paths_fog_test.append(fb)
                            B_paths_fog_test.append(fa)
                
                self.A_paths += A_paths_fog_test
                self.B_paths += B_paths_fog_test
        elif task == 'mix':
            
            raw_A_fog = sorted(make_dataset(self.dir_Afog, float("inf")))
            raw_B_fog = sorted(make_dataset(self.dir_Bfog, float("inf")))
            
            # Low-light restoration uses the LOL training split.
            raw_A_light = sorted(make_dataset(self.dir_Alol, float("inf")))
            raw_B_light = sorted(make_dataset(self.dir_Blol, float("inf")))
            
            raw_A_snow = sorted(make_dataset(self.dir_Asnow, float("inf")))
            raw_B_snow = sorted(make_dataset(self.dir_Bsnow, float("inf")))
            
            raw_A_rain = sorted(make_dataset(self.dir_Arain_syn, float("inf")))
            raw_B_rain = sorted(make_dataset(self.dir_Brain_syn, float("inf")))
            
            raw_A_blur = sorted(make_dataset(self.dir_Ablur, float("inf")))
            raw_B_blur = sorted(make_dataset(self.dir_Bblur, float("inf")))

            target_total = opt.max_dataset_size
            if target_total == float('inf'):
                target_total = len(raw_A_fog) + len(raw_A_light) + len(raw_A_snow) + len(raw_A_rain) + len(raw_A_blur)
            
            print(f"Synthesis Mix Dataset with Total Size: {target_total}")

            # Keep the 4:1:2:2:1 task ratio used by UDBM training.
            weights = {'fog': 4, 'light': 1, 'snow': 2, 'rain': 2, 'blur': 1}
            w_sum = sum(weights.values()) # 10
            
            n_fog = int(target_total * (weights['fog'] / w_sum))
            n_light = int(target_total * (weights['light'] / w_sum))
            n_snow = int(target_total * (weights['snow'] / w_sum))
            n_rain = int(target_total * (weights['rain'] / w_sum))
            n_blur = int(target_total * (weights['blur'] / w_sum))

            def _resize_dataset(paths_list, target_n):
                if len(paths_list) == 0: return []
                if len(paths_list) >= target_n:
                    return paths_list[:target_n]
                else:
                    # Oversample small subsets to preserve the task ratio.
                    repeat_factor = (target_n // len(paths_list)) + 1
                    return (paths_list * repeat_factor)[:target_n]

            self.A_paths = []
            self.B_paths = []

            self.A_paths += _resize_dataset(raw_A_fog, n_fog)
            self.B_paths += _resize_dataset(raw_B_fog, n_fog)

            self.A_paths += _resize_dataset(raw_A_light, n_light)
            self.B_paths += _resize_dataset(raw_B_light, n_light)

            self.A_paths += _resize_dataset(raw_A_snow, n_snow)
            self.B_paths += _resize_dataset(raw_B_snow, n_snow)

            self.A_paths += _resize_dataset(raw_A_rain, n_rain)
            self.B_paths += _resize_dataset(raw_B_rain, n_rain)

            self.A_paths += _resize_dataset(raw_A_blur, n_blur)
            self.B_paths += _resize_dataset(raw_B_blur, n_blur)

            print(f"Mix Dataset Composition: Fog:{n_fog}, Light:{n_light}, Snow:{n_snow}, Rain:{n_rain}, Blur:{n_blur}")
        
        elif task == '4':
            self.A_paths = sorted(make_dataset_4(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_4(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, opt.max_dataset_size))
        elif task == '5':
            self.A_paths = sorted(make_dataset_5(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_5(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, opt.max_dataset_size))
        elif task == '6':
            self.A_paths = sorted(make_dataset_6(self.dir_Arain_syn, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_6(self.dir_Brain_syn, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, self.dir_Bfog, opt.max_dataset_size))
        elif task=='real_dark_mef':
            self.A_paths=make_dataset(self.dir_A_real_dark_mef, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_dark_dice':
            self.A_paths=make_dataset(self.dir_A_real_dark_dice, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_dark_npe':
            self.A_paths=make_dataset(self.dir_A_real_dark_npe, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_rain':
            self.A_paths=make_dataset(self.dir_A_real_rain, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_snow':
            self.A_paths=make_dataset(self.dir_A_real_snow, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_hide':
            self.A_paths=make_dataset(self.dir_Ablur_hide, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bblur_hide, opt.max_dataset_size)
        elif task=='real_j':
            self.A_paths=make_dataset(self.dir_Ablur_j, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bblur_j, opt.max_dataset_size)
        elif task=='real_r':
            self.A_paths=make_dataset(self.dir_Ablur_r, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bblur_r, opt.max_dataset_size)
        elif task=='h':
            self.A_paths=make_dataset(self.dir_Acd11_h, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='hr':
            self.A_paths=make_dataset(self.dir_Acd11_hr, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='hs':
            self.A_paths=make_dataset(self.dir_Acd11_hs, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='l':
            self.A_paths=make_dataset(self.dir_Acd11_l, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='lh':
            self.A_paths=make_dataset(self.dir_Acd11_lh, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='lhr':
            self.A_paths=make_dataset(self.dir_Acd11_lhr, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='lhs':
            self.A_paths=make_dataset(self.dir_Acd11_lhs, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='lr':
            self.A_paths=make_dataset(self.dir_Acd11_lr, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='ls':
            self.A_paths=make_dataset(self.dir_Acd11_ls, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='r':
            self.A_paths=make_dataset(self.dir_Acd11_r, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='s':
            self.A_paths=make_dataset(self.dir_Acd11_s, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bcd11, opt.max_dataset_size)
        elif task=='rni15':
            self.A_paths=make_dataset(os.path.join(opt.dataroot, "RNI15"), opt.max_dataset_size)
            self.B_paths=make_dataset(os.path.join(opt.dataroot, "RNI15"), opt.max_dataset_size)
        elif task=='dense_haze':
            self.A_paths=make_dataset(os.path.join(opt.dataroot, "tmp_dataset/hazy"), opt.max_dataset_size)
            self.B_paths=make_dataset(os.path.join(opt.dataroot, "tmp_dataset/GT"), opt.max_dataset_size)
        else:
            self.A_paths = sorted(make_dataset(self.dir_Aasd, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Basd, opt.max_dataset_size))
        # elif tas

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(f"Dataset [{task}]: {self.A_size} input images, {self.B_size} target images")
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]
        condition = Image.open(A_path).convert('RGB') #condition
        gt = Image.open(B_path).convert('RGB') #gt
        
        if 'LOL' in A_path or 'LSRW' in A_path or 'dark' in A_path:
            condition = cv2.cvtColor(np.asarray(condition), cv2.COLOR_RGB2BGR)
            gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
        
            if self.crop_patch:
                gt, condition = self.get_patch([gt, condition], self.image_size)
            if 'LOL' in A_path or 'dark' in A_path:
                condition = self.cv2equalizeHist(condition) if self.equalizeHist else condition
            else:
                condition = condition

            images = [[gt, condition]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            gt = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            condition = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
        
            gt = self.to_tensor(gt)
            condition = self.to_tensor(condition)
        else:
            w, h = condition.size
            transform_params = get_params(self.opt, condition.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=False)
            B_transform = get_transform(self.opt, transform_params, grayscale=False)
            condition = A_transform(condition)
            gt = B_transform(gt)
            if self.opt.phase == 'train':
                if h < self.opt.crop_size or w < self.opt.crop_size:
                    osize = [self.opt.crop_size, self.opt.crop_size]
                    resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
                    condition = resi(condition)
                    gt = resi(gt)
        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
    
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]
class AlignedDataset_mix(BaseDataset):
    """
    A dataset class that mixes 5 degradation types with a fixed ratio of 4:1:2:2:1.
    Fog : Light : Rain : Snow : Blur = 4 : 1 : 2 : 2 : 1
    """

    def __init__(self, opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = image_size
        self.opt = opt
        
        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
        # Light (LOL + LSRW)
        self.dir_Alsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/low')
        self.dir_Blsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/high')
        self.dir_Alol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/low')
        self.dir_Blol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/high')
        
        if opt.phase == 'train':
            # Snow (Snow100K-L)
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L'+'/synthetic')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase +'/Snow100K-L'+ '/gt')
            
            # Rain (syn_rain)
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/input')
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/target')
            
            # Blur (Deblur)
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/target')

            # Fog (RESIDE/OTS)
            flog_prefix = os.path.join(opt.dataroot, 'RESIDE/OTS_ALPHA/')
            self.dir_Afog = flog_prefix + 'haze/OTS'
            self.dir_Bfog = flog_prefix + 'clear/clear_images'
        else:
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/synthetic')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/gt')
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/input')
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/target')
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/target')
            self.dir_Afog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy')
            self.dir_Bfog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt')

        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
        # Fog
        raw_A_fog = sorted(make_dataset(self.dir_Afog, opt.max_dataset_size))
        raw_B_fog = sorted(make_dataset(self.dir_Bfog, opt.max_dataset_size))
        if opt.phase != 'train' and task == 'fog':
             pass 

        # Light (LOL + LSRW)
        if opt.phase == 'train':
            raw_A_light = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
            raw_B_light = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        else:
            raw_A_light = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
            raw_B_light = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))

        # Rain
        raw_A_rain = sorted(make_dataset(self.dir_Arain_syn, opt.max_dataset_size))
        raw_B_rain = sorted(make_dataset(self.dir_Brain_syn, opt.max_dataset_size))

        # Snow
        raw_A_snow = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
        raw_B_snow = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))

        # Blur
        raw_A_blur = sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
        raw_B_blur = sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))

        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
        weights = {
            'fog': 4,
            'light': 1,
            'rain': 2,
            'snow': 2,
            'blur': 1
        }
        
        lens = {
            'fog': len(raw_A_fog),
            'light': len(raw_A_light),
            'rain': len(raw_A_rain),
            'snow': len(raw_A_snow),
            'blur': len(raw_A_blur)
        }
        
        print(f"Original counts: {lens}")

        if sum(lens.values()) > 0:
            base_unit = max(
                lens['fog'] / weights['fog'],
                lens['light'] / weights['light'],
                lens['rain'] / weights['rain'],
                lens['snow'] / weights['snow'],
                lens['blur'] / weights['blur']
            )
            base_unit = int(base_unit)
        else:
            base_unit = 0

        target_lens = {k: v * base_unit for k, v in weights.items()}
        print(f"Target counts (Ratio 4:1:2:2:1): {target_lens}")

        def adjust_list(data_list, target_len):
            if len(data_list) == 0: return []
            if len(data_list) >= target_len:
                return data_list[:target_len]
            else:
                # Cycle small subsets to meet the target length.
                repeat_count = target_len // len(data_list)
                remainder = target_len % len(data_list)
                return data_list * repeat_count + data_list[:remainder]

        self.A_paths = []
        self.B_paths = []

        # Fog (4)
        self.A_paths += adjust_list(raw_A_fog, target_lens['fog'])
        self.B_paths += adjust_list(raw_B_fog, target_lens['fog'])

        # Light (1)
        self.A_paths += adjust_list(raw_A_light, target_lens['light'])
        self.B_paths += adjust_list(raw_B_light, target_lens['light'])

        # Rain (2)
        self.A_paths += adjust_list(raw_A_rain, target_lens['rain'])
        self.B_paths += adjust_list(raw_B_rain, target_lens['rain'])

        # Snow (2)
        self.A_paths += adjust_list(raw_A_snow, target_lens['snow'])
        self.B_paths += adjust_list(raw_B_snow, target_lens['snow'])

        # Blur (1)
        self.A_paths += adjust_list(raw_A_blur, target_lens['blur'])
        self.B_paths += adjust_list(raw_B_blur, target_lens['blur'])

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        print(f"Final Dataset Size: {self.A_size}")
        
        if self.A_size != self.B_size:
            raise ValueError("A and B paths length mismatch after mixing!")

        assert(self.opt.load_size >= self.opt.crop_size)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        
        Using the exact same logic as AlignedDataset_all to handle different data types 
        (LOL/Dark require BGR conversion and specific processing).
        """
        # read a image given a random integer index
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        
        condition = Image.open(A_path).convert('RGB') #condition
        gt = Image.open(B_path).convert('RGB') #gt
        
        if 'LOL' in A_path or 'LSRW' in A_path or 'dark' in A_path:
            condition = cv2.cvtColor(np.asarray(condition), cv2.COLOR_RGB2BGR)
            gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
        
            if self.crop_patch:
                gt, condition = self.get_patch([gt, condition], self.image_size)
            
            if 'LOL' in A_path or 'dark' in A_path:
                condition = self.cv2equalizeHist(condition) if self.equalizeHist else condition
            else:
                condition = condition

            images = [[gt, condition]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            gt = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            condition = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
        
            gt = self.to_tensor(gt)
            condition = self.to_tensor(condition)
        else:
            w, h = condition.size
            transform_params = get_params(self.opt, condition.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=False)
            B_transform = get_transform(self.opt, transform_params, grayscale=False)
            condition = A_transform(condition)
            gt = B_transform(gt)
            if self.opt.phase == 'train':
                if h < self.opt.crop_size or w < self.opt.crop_size:
                    osize = [self.opt.crop_size, self.opt.crop_size]
                    resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
                    condition = resi(condition)
                    gt = resi(gt)
        
        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
    
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]
            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = TF.to_tensor(img).float()
        return img_t

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list
class AlignedDataset_all_for_npy_save(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory 'data train folder' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory 'data test folder'.
    """

    def __init__(self, opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = image_size
        opt.phase='train'
        self.opt = opt
        
        #origin----------------------------------------------------------------------------------------------------------
        self.dir_Arain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/rainy_image')
        self.dir_Brain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/ground_truth')
        self.dir_Alsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/low')
        self.dir_Blsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/high')
        self.dir_Alol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/low')
        self.dir_Blol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/high')
        
        if opt.phase == 'train':
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L'+'/synthetic')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase +'/Snow100K-L'+ '/gt')
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/input')
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/target')
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/target')

            flog_prefix = os.path.join(opt.dataroot, 'RESIDE/OTS_ALPHA/')
            self.dir_Afog = flog_prefix + 'haze/OTS'
            self.dir_Bfog = flog_prefix + 'clear/clear_images'
        else:
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')


            self.dir_Asnow1 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow1 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')
            self.dir_Asnow2 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow2 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/gt')
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic')


            self.dir_Arain_syn1 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn1 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/target')  #Test2800
            self.dir_Arain_syn2 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn2 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/target')  #Test2800
            self.dir_Arain_syn3 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100L/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn3 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100L/target')  #Test2800
            self.dir_Arain_syn4 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test100/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn4 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test100/target')  #Test2800
            self.dir_Arain_syn5 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test1200/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn5 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test1200/target')  #Test2800



            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/target')
            self.dir_Afog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy')
            self.dir_Bfog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt')
            self.dir_Aasd = os.path.join(opt.dataroot, 'temp')
            self.dir_Basd = os.path.join(opt.dataroot, 'temp')



            #real dark:
            self.dir_A_real_dark_mef=os.path.join(opt.dataroot,"real_dark/real_dark/MEF")
            self.dir_A_real_dark_npe=os.path.join(opt.dataroot,"real_dark/real_dark/NPE")
            self.dir_A_real_dark_dice=os.path.join(opt.dataroot,"real_dark/real_dark/DICE")

            # real rain: <dataroot>/real_rain/real_rain/Practical
            self.dir_A_real_rain=os.path.join(opt.dataroot,"real_rain/real_rain/Practical")

            # real snow: <dataroot>/Snow100K/realistic
            self.dir_A_real_snow=os.path.join(opt.dataroot,"Snow100K/realistic")


            self.dir_Ablur_hide = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/HIDE/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur_hide = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/HIDE/target')

            self.dir_Ablur_j = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/RealBlur_J/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur_j = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/RealBlur_J/target')

            self.dir_Ablur_r = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/RealBlur_R/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur_r = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/RealBlur_R/target')

            #

            #real udc
            # self.dir_A_real_udc=os.path.join(opt.dataroot,"real_rain/real_rain/Practical")
        #test
        if task == 'light':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset_2(self.dir_Alol, self.dir_Alsrw, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset_2(self.dir_Blol, self.dir_Blsrw, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'light_only':
            self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'rain':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset(self.dir_Arain_syn, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Brain_syn, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Arain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn5, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Brain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn5, opt.max_dataset_size))
        
                if len(self.A_paths) != len(self.B_paths):
                    raise ValueError(
                        f"Mismatched file list lengths: A has {len(self.A_paths)} files, "
                        f"B has {len(self.B_paths)} files."
                    )

                print("Checking paired file names between input and target folders...")
                mismatched_files = []
                for path_a, path_b in zip(self.A_paths, self.B_paths):
                    # os.path.basename('path/to/file.png') -> 'file.png'
                    # os.path.splitext('file.png') -> ('file', '.png')
                    # [0] -> 'file'
                    basename_a = os.path.splitext(os.path.basename(path_a))[0]
                    
                    basename_b = os.path.splitext(os.path.basename(path_b))[0]

                    if basename_a != basename_b:
                        mismatched_files.append((path_a, path_b))

                if mismatched_files:
                    error_msg = f"Error: detected {len(mismatched_files)} filename mismatches.\n"
                    error_msg += "Filename mismatch examples, up to 5:\n"
                    for a, b in mismatched_files[:5]:
                        error_msg += f"  - A: {os.path.basename(a)} (from {a})\n"
                        error_msg += f"  - B: {os.path.basename(b)} (from {b})\n"
                        error_msg += "---\n"
                    raise ValueError(error_msg)
                
                print(f"Pair check passed for {len(self.A_paths)} files.")
        elif task == 'rain1':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn1, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn1, opt.max_dataset_size))
        elif task == 'rain2':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn2, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn2, opt.max_dataset_size))
        elif task == 'rain3':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn3, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn3, opt.max_dataset_size))
        elif task == 'rain4':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn4, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn4, opt.max_dataset_size))
        elif task == 'rain5':
            self.A_paths = sorted(make_dataset(self.dir_Arain_syn5, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Brain_syn5, opt.max_dataset_size))
        
        elif task == 'snow1':
            if opt.phase == 'train':
                 self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
                 self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Asnow1, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Bsnow1, opt.max_dataset_size))
        elif task == 'snow2':
            if opt.phase == 'train':
                 self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
                 self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Asnow2, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Bsnow2, opt.max_dataset_size))
        elif task == 'snow':
            if opt.phase == 'train':
                 self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
                 self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Asnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Asnow2, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Bsnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Bsnow2, opt.max_dataset_size))
        elif task == 'blur':
            self.A_paths = sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))
        elif task == 'fog':
            self.A_paths = sorted(make_dataset(self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bfog, opt.max_dataset_size))
            if opt.phase!="train":
                files_a = sorted(glob.glob(os.path.join(self.dir_Bfog, "*.png")))
                files_b = sorted(glob.glob(os.path.join(self.dir_Afog, "*.jpg")))
                self.A_paths, self.B_paths = [], []

                dict_b = {}
                for fb in files_b:
                    name = os.path.basename(fb).split("_")[0]
                    dict_b.setdefault(name, []).append(fb)

                for fa in files_a:
                    name = os.path.splitext(os.path.basename(fa))[0]
                    if name in dict_b:
                        for fb in dict_b[name]:
                            self.A_paths.append(fb)
                            self.B_paths.append(fa)
        elif task=="all":
            if opt.phase!="train":
                # --- 1. 'light' (test) ---
                self.A_paths=[]
                self.B_paths=[]
                self.A_paths += sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
                
                # --- 2. 'light_only' (test) --- 
                # In test mode, light_only and light share the same fallback branch.

                # --- 3. 'rain' (test) ---
                self.A_paths += sorted(make_dataset(self.dir_Arain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn5, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Brain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn5, opt.max_dataset_size))

                # --- 4. 'snow' (test) ---
                self.A_paths += sorted(make_dataset(self.dir_Asnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Asnow2, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Bsnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Bsnow2, opt.max_dataset_size))

                self.A_paths += sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))

                # --- 6. 'fog' (test) ---
                files_a_fog = sorted(glob.glob(os.path.join(self.dir_Bfog, "*.png")))
                files_b_fog = sorted(glob.glob(os.path.join(self.dir_Afog, "*.jpg")))
                
                A_paths_fog_test = []
                B_paths_fog_test = []

                dict_b_fog = {}
                for fb in files_b_fog:
                    name = os.path.basename(fb).split("_")[0]
                    dict_b_fog.setdefault(name, []).append(fb)

                for fa in files_a_fog:
                    name = os.path.splitext(os.path.basename(fa))[0]
                    if name in dict_b_fog:
                        for fb in dict_b_fog[name]:
                            A_paths_fog_test.append(fb)
                            B_paths_fog_test.append(fa)
                
                self.A_paths += A_paths_fog_test
                self.B_paths += B_paths_fog_test
        elif task == '4':
            self.A_paths = sorted(make_dataset_4(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_4(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, opt.max_dataset_size))
        elif task == '5':
            self.A_paths = sorted(make_dataset_5(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_5(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, opt.max_dataset_size))
        elif task == '6':
            self.A_paths = sorted(make_dataset_6(self.dir_Arain_syn, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_6(self.dir_Brain_syn, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, self.dir_Bfog, opt.max_dataset_size))
        elif task=='real_dark_mef':
            self.A_paths=make_dataset(self.dir_A_real_dark_mef, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_dark_dice':
            self.A_paths=make_dataset(self.dir_A_real_dark_dice, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_dark_npe':
            self.A_paths=make_dataset(self.dir_A_real_dark_npe, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_rain':
            self.A_paths=make_dataset(self.dir_A_real_rain, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_snow':
            self.A_paths=make_dataset(self.dir_A_real_snow, opt.max_dataset_size)
            self.B_paths=self.A_paths[:]
        elif task=='real_hide':
            self.A_paths=make_dataset(self.dir_Ablur_hide, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bblur_hide, opt.max_dataset_size)
        elif task=='real_j':
            self.A_paths=make_dataset(self.dir_Ablur_j, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bblur_j, opt.max_dataset_size)
        elif task=='real_r':
            self.A_paths=make_dataset(self.dir_Ablur_r, opt.max_dataset_size)
            self.B_paths=make_dataset(self.dir_Bblur_r, opt.max_dataset_size)
        else:
            self.A_paths = sorted(make_dataset(self.dir_Aasd, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Basd, opt.max_dataset_size))
    
        self.A_paths=self.A_paths[:200]
        self.B_paths=self.B_paths[:200]
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(f"Dataset [{task}]: {self.A_size} input images, {self.B_size} target images")
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]
        condition = Image.open(A_path).convert('RGB') #condition
        gt = Image.open(B_path).convert('RGB') #gt
        
        if 'LOL' in A_path or 'LSRW' in A_path or 'dark' in A_path or 'low' in A_path:
            condition = cv2.cvtColor(np.asarray(condition), cv2.COLOR_RGB2BGR)
            gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
            gt, condition = self.get_patch([gt, condition], self.image_size)
            condition = self.cv2equalizeHist(condition)
            images = [[gt, condition]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            gt = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            condition = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
        
            gt = self.to_tensor(gt)
            condition = self.to_tensor(condition)
        else:
            w, h = condition.size
            transform_params = get_params(self.opt, condition.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=False)
            B_transform = get_transform(self.opt, transform_params, grayscale=False)
            condition = A_transform(condition)
            gt = B_transform(gt)
            # if self.opt.phase == 'train':
            if h < 256 or w < 256:
                osize = [256, 256]
                resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
                condition = resi(condition)
                gt = resi(gt)
        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
    
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]
import glob

class AlignedDataset_all_weather(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory 'data train folder' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory 'data test folder'.
    """

    def __init__(self, opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = image_size
        self.opt = opt
        #origin----------------------------------------------------------------------------------------------------------
        
        
        if opt.phase == 'train':
            self.dir_lq=os.path.join(opt.dataroot, "input", "*")
            self.dir_hq=os.path.join(opt.dataroot, "gt", "*")
            self.A_paths=sorted(glob.glob(self.dir_lq))
            self.B_paths=sorted(glob.glob(self.dir_hq))
        else:
            self.dir_snow_s_gt=os.path.join(opt.dataroot, "test/Snow100K-S/gt", "*")
            self.dir_snow_s_lq=os.path.join(opt.dataroot, "test/Snow100K-S/synthetic", "*")

            self.dir_snow_l_gt=os.path.join(opt.dataroot, "test/Snow100K-L/gt", "*")
            self.dir_snow_l_lq=os.path.join(opt.dataroot, "test/Snow100K-L/synthetic", "*")

            self.dir_val_test_gt=os.path.join(opt.dataroot, "test/Test1/gt", "*")
            self.dir_val_test_lq=os.path.join(opt.dataroot, "test/Test1/input", "*")

            self.dir_raindrop_gt=os.path.join(opt.dataroot, "test/raindrop/test_a/gt", "*")
            self.dir_raindrop_lq=os.path.join(opt.dataroot, "test/raindrop/test_a/data", "*")
        
            if task=="snow_s":
                self.A_paths=sorted(glob.glob(self.dir_snow_s_lq))
                self.B_paths=sorted(glob.glob(self.dir_snow_s_gt))
            elif task=="snow_l":
                self.A_paths=sorted(glob.glob(self.dir_snow_l_lq))
                self.B_paths=sorted(glob.glob(self.dir_snow_l_gt))
            elif task=="rain_out":
                self.A_paths=sorted(glob.glob(self.dir_val_test_lq))
                self.B_paths=sorted(glob.glob(self.dir_val_test_gt))
            elif task=="raindrop":
                self.A_paths=sorted(glob.glob(self.dir_raindrop_lq))
                self.B_paths=sorted(glob.glob(self.dir_raindrop_gt))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(f"Dataset size: {self.A_size} input images, {self.B_size} target images")
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]
        condition = Image.open(A_path).convert('RGB') #condition
        gt = Image.open(B_path).convert('RGB') #gt
        
        w, h = condition.size
        transform_params = get_params(self.opt, condition.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=False)
        condition = A_transform(condition)
        gt = B_transform(gt)
        if self.opt.phase == 'train':
            if h < 256 or w < 256:
                osize = [256, 256]
                resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
                condition = resi(condition)
                gt = resi(gt)
                
        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
    
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]


class AlignedDataset_all_plot_predict(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory 'data train folder' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory 'data test folder'.
    """

    def __init__(self, opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = image_size
        self.opt = opt
        #origin----------------------------------------------------------------------------------------------------------
        self.dir_Arain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/rainy_image')
        self.dir_Brain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/ground_truth')
        self.dir_Alsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/low')
        self.dir_Blsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/high')
        self.dir_Alol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/low')
        self.dir_Blol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/high')
        
        if opt.phase == 'train':
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L'+'/synthetic')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase +'/Snow100K-L'+ '/gt')
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/input')
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/target')
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/target')

            flog_prefix = os.path.join(opt.dataroot, 'RESIDE/OTS_ALPHA/')
            self.dir_Afog = flog_prefix + 'haze/OTS'
            self.dir_Bfog = flog_prefix + 'clear/clear_images'
        else:
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')


            self.dir_Asnow1 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow1 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')
            self.dir_Asnow2 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow2 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/gt')
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic')


            self.dir_Arain_syn1 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn1 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/target')  #Test2800
            self.dir_Arain_syn2 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn2 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/target')  #Test2800
            self.dir_Arain_syn3 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100L/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn3 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100L/target')  #Test2800
            self.dir_Arain_syn4 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test100/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn4 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test100/target')  #Test2800
            self.dir_Arain_syn5 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test1200/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn5 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test1200/target')  #Test2800



            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/target')

            self.dir_Ablur_hide = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/HIDE/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur_hide = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/HIDE/target')

            self.dir_Ablur_j = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/Reblur_J/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur_j = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/Reblur_J/target')

            self.dir_Ablur_r = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/Reblur_R/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur_r = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/Reblur_R/target')

            self.dir_Afog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy')
            self.dir_Bfog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt')
            self.dir_Aasd = os.path.join(opt.dataroot, 'temp')
            self.dir_Basd = os.path.join(opt.dataroot, 'temp')
        
        #test
        if task == 'light':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset_2(self.dir_Alol, self.dir_Alsrw, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset_2(self.dir_Blol, self.dir_Blsrw, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'light_only':
            self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'rain':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset(self.dir_Arain_syn, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Brain_syn, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Arain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn5, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Brain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn5, opt.max_dataset_size))
        elif task == 'snow':
            if opt.phase == 'train':
                 self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
                 self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Asnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Asnow2, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Bsnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Bsnow2, opt.max_dataset_size))
        elif task == 'blur':
            self.A_paths = sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))
        elif task == 'fog':
            self.A_paths = sorted(make_dataset(self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bfog, opt.max_dataset_size))
            if opt.phase!="train":
                files_a = sorted(glob.glob(os.path.join(self.dir_Bfog, "*.png")))
                files_b = sorted(glob.glob(os.path.join(self.dir_Afog, "*.jpg")))
                self.A_paths, self.B_paths = [], []

                dict_b = {}
                for fb in files_b:
                    name = os.path.basename(fb).split("_")[0]
                    dict_b.setdefault(name, []).append(fb)

                for fa in files_a:
                    name = os.path.splitext(os.path.basename(fa))[0]
                    if name in dict_b:
                        for fb in dict_b[name]:
                            self.A_paths.append(fb)
                            self.B_paths.append(fa)
        elif task == '4':
            self.A_paths = sorted(make_dataset_4(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_4(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, opt.max_dataset_size))
        elif task == '5':
            self.A_paths = sorted(make_dataset_5(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_5(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, opt.max_dataset_size))
        elif task == '6':
            self.A_paths = sorted(make_dataset_6(self.dir_Arain_syn, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_6(self.dir_Brain_syn, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, self.dir_Bfog, opt.max_dataset_size))
        else:
            self.A_paths = sorted(make_dataset(self.dir_Aasd, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Basd, opt.max_dataset_size))
    

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(f"Dataset size: {self.A_size} input images, {self.B_size} target images")
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range lq
        B_path = self.B_paths[index % self.B_size] # hq
        
        B_path=A_path.replace("all_in_one","all_in_one_test_result")
        A_path=B_path
        condition = Image.open(A_path).convert('RGB') #condition
        gt = Image.open(B_path).convert('RGB') #gt
        # gt = Image.open(B_path).convert('RGB') #gt
        if 'LOL' in A_path or 'LSRW' in A_path:
            condition = cv2.cvtColor(np.asarray(condition), cv2.COLOR_RGB2BGR)
            gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
        
            if self.crop_patch:
                gt, condition = self.get_patch([gt, condition], self.image_size)
            if 'LOL' in A_path:
                condition = self.cv2equalizeHist(condition) if self.equalizeHist else condition
            else:
                condition = condition

            images = [[gt, condition]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            gt = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            condition = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
        
            gt = self.to_tensor(gt)
            condition = self.to_tensor(condition)
        else:
            w, h = condition.size
            transform_params = get_params(self.opt, condition.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=False)
            B_transform = get_transform(self.opt, transform_params, grayscale=False)
            condition = A_transform(condition)
            gt = B_transform(gt)
            if self.opt.phase == 'train':
                if h < 256 or w < 256:
                    osize = [256, 256]
                    resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
                    condition = resi(condition)
                    gt = resi(gt)
                
        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
    
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]
class AlignedDataset_all_part(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory 'data train folder' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory 'data test folder'.
    """

    def __init__(self, opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = image_size
        self.part=200
        self.opt = opt
        #origin----------------------------------------------------------------------------------------------------------
        self.dir_Arain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/rainy_image')
        self.dir_Brain = os.path.join(opt.dataroot, 'rain1400/' + opt.phase + '/ground_truth')
        self.dir_Alsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/low')
        self.dir_Blsrw = os.path.join(opt.dataroot, 'LSRW/' + opt.phase + '/high')
        self.dir_Alol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/low')
        self.dir_Blol = os.path.join(opt.dataroot, 'LOL/' + opt.phase + '/high')
        
        if opt.phase == 'train':
            self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L'+'/synthetic')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase +'/Snow100K-L'+ '/gt')
            self.dir_Arain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/input')
            self.dir_Brain_syn = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/target')
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/target')

            flog_prefix = os.path.join(opt.dataroot, 'RESIDE/OTS_ALPHA/')
            self.dir_Afog = flog_prefix + 'haze/OTS'
            self.dir_Bfog = flog_prefix + 'clear/clear_images'
        else:
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')


            self.dir_Asnow1 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow1 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-S/gt')
            self.dir_Asnow2 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/synthetic') #Snow100K-S Snow100K-L
            self.dir_Bsnow2 = os.path.join(opt.dataroot, 'Snow100K/' + opt.phase + '/Snow100K-L/gt')
            # self.dir_Asnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic') #Snow100K-S Snow100K-L
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'Snow100K/' + 'realistic')


            self.dir_Arain_syn1 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn1 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test2800/target')  #Test2800
            self.dir_Arain_syn2 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn2 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100H/target')  #Test2800
            self.dir_Arain_syn3 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100L/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn3 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Rain100L/target')  #Test2800
            self.dir_Arain_syn4 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test100/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn4 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test100/target')  #Test2800
            self.dir_Arain_syn5 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test1200/input') #Rain100H, Rain100L, Test100, Test1200,
            self.dir_Brain_syn5 = os.path.join(opt.dataroot, 'syn_rain/' + opt.phase + '/Test1200/target')  #Test2800



            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/input')  #GoPro, HIDE,  Reblur_J, Reblur_R
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/' + opt.phase + '/GoPro/target')
            self.dir_Afog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy')
            self.dir_Bfog = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt')
            self.dir_Aasd = os.path.join(opt.dataroot, 'temp')
            self.dir_Basd = os.path.join(opt.dataroot, 'temp')
        
        #test
        if task == 'light':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset_2(self.dir_Alol, self.dir_Alsrw, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset_2(self.dir_Blol, self.dir_Blsrw, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'light_only':
            self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'rain':
            if opt.phase == 'train':
                self.A_paths = sorted(make_dataset(self.dir_Arain_syn, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Brain_syn, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Arain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn5, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Brain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn5, opt.max_dataset_size))
        elif task == 'snow':
            if opt.phase == 'train':
                 self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
                 self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
            else:
                self.A_paths = sorted(make_dataset(self.dir_Asnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Asnow2, opt.max_dataset_size))
                self.B_paths = sorted(make_dataset(self.dir_Bsnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Bsnow2, opt.max_dataset_size))
        elif task == 'blur':
            self.A_paths = sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))
        elif task == 'fog':
            self.A_paths = sorted(make_dataset(self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bfog, opt.max_dataset_size))
            if opt.phase!="train":
                files_a = sorted(glob.glob(os.path.join(self.dir_Bfog, "*.png")))
                files_b = sorted(glob.glob(os.path.join(self.dir_Afog, "*.jpg")))
                self.A_paths, self.B_paths = [], []

                dict_b = {}
                for fb in files_b:
                    name = os.path.basename(fb).split("_")[0]
                    dict_b.setdefault(name, []).append(fb)

                for fa in files_a:
                    name = os.path.splitext(os.path.basename(fa))[0]
                    if name in dict_b:
                        for fb in dict_b[name]:
                            self.A_paths.append(fb)
                            self.B_paths.append(fa)
        elif task=="all":
            if opt.phase!="train":
                # --- 1. 'light' (test) ---
                self.A_paths=[]
                self.B_paths=[]
                self.A_paths += sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
                
                # --- 2. 'light_only' (test) --- 
                # In test mode, light_only and light share the same fallback branch.

                # --- 3. 'rain' (test) ---
                self.A_paths += sorted(make_dataset(self.dir_Arain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Arain_syn5, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Brain_syn1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn2, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn3, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn4, opt.max_dataset_size))+sorted(make_dataset(self.dir_Brain_syn5, opt.max_dataset_size))

                # --- 4. 'snow' (test) ---
                self.A_paths += sorted(make_dataset(self.dir_Asnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Asnow2, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Bsnow1, opt.max_dataset_size))+sorted(make_dataset(self.dir_Bsnow2, opt.max_dataset_size))

                self.A_paths += sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
                self.B_paths += sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))

                # --- 6. 'fog' (test) ---
                files_a_fog = sorted(glob.glob(os.path.join(self.dir_Bfog, "*.png")))
                files_b_fog = sorted(glob.glob(os.path.join(self.dir_Afog, "*.jpg")))
                
                A_paths_fog_test = []
                B_paths_fog_test = []

                dict_b_fog = {}
                for fb in files_b_fog:
                    name = os.path.basename(fb).split("_")[0]
                    dict_b_fog.setdefault(name, []).append(fb)

                for fa in files_a_fog:
                    name = os.path.splitext(os.path.basename(fa))[0]
                    if name in dict_b_fog:
                        for fb in dict_b_fog[name]:
                            A_paths_fog_test.append(fb)
                            B_paths_fog_test.append(fa)
                
                self.A_paths += A_paths_fog_test
                self.B_paths += B_paths_fog_test
        elif task == '4':
            self.A_paths = sorted(make_dataset_4(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_4(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, opt.max_dataset_size))
        elif task == '5':
            self.A_paths = sorted(make_dataset_5(self.dir_Arain_syn, self.dir_Alsrw, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_5(self.dir_Brain_syn, self.dir_Blsrw, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, opt.max_dataset_size))
        elif task == '6':
            self.A_paths = sorted(make_dataset_6(self.dir_Arain_syn, self.dir_Alol, self.dir_Asnow, self.dir_Ablur, self.dir_Afog, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset_6(self.dir_Brain_syn, self.dir_Blol, self.dir_Bsnow, self.dir_Bblur, self.dir_Bfog, opt.max_dataset_size))
        else:
            self.A_paths = sorted(make_dataset(self.dir_Aasd, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Basd, opt.max_dataset_size))
    
        self.A_paths=self.A_paths[0:self.part]
        self.B_paths=self.B_paths[0:self.part]
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(f"Dataset [{task}]: {self.A_size} input images, {self.B_size} target images")
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]
        condition = Image.open(A_path).convert('RGB') #condition
        gt = Image.open(B_path).convert('RGB') #gt
        
        if 'LOL' in A_path or 'LSRW' in A_path:
            condition = cv2.cvtColor(np.asarray(condition), cv2.COLOR_RGB2BGR)
            gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)
        
            if self.crop_patch:
                gt, condition = self.get_patch([gt, condition], self.image_size)
            if 'LOL' in A_path:
                condition = self.cv2equalizeHist(condition) if self.equalizeHist else condition
            else:
                condition = condition

            images = [[gt, condition]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            gt = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            condition = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
        
            gt = self.to_tensor(gt)
            condition = self.to_tensor(condition)
        else:
            w, h = condition.size
            transform_params = get_params(self.opt, condition.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=False)
            B_transform = get_transform(self.opt, transform_params, grayscale=False)
            condition = A_transform(condition)
            gt = B_transform(gt)
            if self.opt.phase == 'train':
                if h < 256 or w < 256:
                    osize = [256, 256]
                    resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
                    condition = resi(condition)
                    gt = resi(gt)

        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
    
    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]
