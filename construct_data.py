# coding:utf-8
import os
import numpy as np
import tensorflow as tf
dir_modes = ['val', 'train', 'test']
modes = ['valid', 'train', 'test']
refs = ['', '_ref']
fields = ['type', 'value', 'associated']
y_fields = ['y_aux']
old_data_dir = 'e2e_data'
new_data_dir = 'e2e_0512_max3'

if __name__ == '__main__':
    ref = refs[1]
    for i, mode in enumerate(modes):
        with open(os.path.join(old_data_dir, "{}/x_type.{}.txt".format(dir_modes[i], mode)), 'r') as f_type, \
            open(os.path.join(old_data_dir, "{}/x_type.{}.txt".format(dir_modes[i], mode)), 'r') as f_type_ref:
            lines_type = f_type.readlines()
            lines_type_ref = f_type_ref.readlines()
            # ref_flag =
            ref_tag = np.zeros(len(lines_type_ref))
            for idx, line_type in enumerate(lines_type):
                reverse_flag = False
                line_set = set(line_type.strip().split())
                match_flag = False
                # if idx % 2:
                #     reverse_flag = False
                # else:
                #     reverse_flag = True
                #     lines_type_ref.reverse()
                for ref_idx, line_type_ref in enumerate(lines_type_ref):
                    if not ref_tag[ref_idx]:
                        line_set_ref = set(line_type_ref.strip().split())
                        if (len(line_set) == 8) and (len(line_set & line_set_ref) == max(len(line_set_ref), len(line_set))) and (abs(len(line_set_ref) - len(line_set)) == 0):
                            for field in fields:
                                with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                    ref_tag[ref_idx] = 1
                            with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                    open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                lines = f_r.readlines()
                                f_o.write(lines[ref_idx])
                            match_flag = True
                            break
                        elif (len(line_set) == 7) and (len(line_set & line_set_ref) == 6) and (len(line_set_ref) - len(line_set) == 0):
                            for field in fields:
                                with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                    ref_tag[ref_idx] = 1
                            with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                    open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                lines = f_r.readlines()
                                f_o.write(lines[ref_idx])
                            match_flag = True
                            break
                        elif (len(line_set) == 6) and (len(line_set & line_set_ref) == 5) and (len(line_set_ref) - len(line_set) == 1):
                            for field in fields:
                                with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                    ref_tag[ref_idx] = 1
                            with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                    open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                lines = f_r.readlines()
                                f_o.write(lines[ref_idx])
                            match_flag = True
                            break
                        elif (len(line_set) == 5) and (len(line_set & line_set_ref) == 4) and len(line_set) == len(line_set_ref) - 1:
                            for field in fields:
                                with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                    ref_tag[ref_idx] = 1
                            with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                    open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                lines = f_r.readlines()
                                f_o.write(lines[ref_idx])
                            match_flag = True
                            break
                        elif (len(line_set) == 4) and len(line_set) == len(line_set_ref) - 1  and len(line_set & line_set_ref) == 3:
                            for field in fields:
                                with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                    ref_tag[ref_idx] = 1
                            with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                    open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                lines = f_r.readlines()
                                f_o.write(lines[ref_idx])
                            match_flag = True
                            break
                        elif (len(line_set) == 3) and (len(line_set & line_set_ref) == 2) and (abs(len(line_set_ref) - len(line_set)) == 1):
                            for field in fields:
                                with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                    ref_tag[ref_idx] = 1
                            with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                    open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                lines = f_r.readlines()
                                f_o.write(lines[ref_idx])
                            match_flag = True
                            break
                        else:
                            continue
                    else:
                        continue
                if not match_flag:
                    print('[info] mismatch is:{}, len is:{}'.format(idx, len(line_set)))
                    ref_tag = np.zeros(len(lines_type_ref))
                    for ref_idx, line_type_ref in enumerate(lines_type_ref):
                        if not ref_tag[ref_idx]:
                            line_set_ref = set(line_type_ref.strip().split())
                            if (len(line_set) == 4) and len(line_set) == len(line_set_ref) + 1  and len(line_set & line_set_ref) == len(line_set) - 2:
                                for field in fields:
                                    with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                            open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                        lines = f_r.readlines()
                                        f_o.write(lines[ref_idx])
                                        ref_tag[ref_idx] = 1
                                with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                match_flag = True
                                break
                            elif (len(line_set) == 8) and (len(line_set & line_set_ref) == max(len(line_set_ref), len(line_set))) and (abs(len(line_set_ref) - len(line_set)) == 0):
                                for field in fields:
                                    with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                            open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                        lines = f_r.readlines()
                                        f_o.write(lines[ref_idx])
                                        ref_tag[ref_idx] = 1
                                with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                match_flag = True
                                break
                            elif (len(line_set) == 7) and (len(line_set & line_set_ref) == max(len(line_set_ref), len(line_set)) - 1) and (abs(len(line_set_ref) - len(line_set)) == 0):
                                for field in fields:
                                    with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                            open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                        lines = f_r.readlines()
                                        f_o.write(lines[ref_idx])
                                        ref_tag[ref_idx] = 1
                                with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                match_flag = True
                                break
                            elif (len(line_set) == 6) and (len(line_set & line_set_ref) == max(len(line_set_ref), len(line_set)) - 1) and (abs(len(line_set_ref) - len(line_set)) == 0):
                                for field in fields:
                                    with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                            open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                        lines = f_r.readlines()
                                        f_o.write(lines[ref_idx])
                                        ref_tag[ref_idx] = 1
                                with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                match_flag = True
                                break
                            elif (len(line_set) == 5) and (len(line_set & line_set_ref) == max(len(line_set_ref), len(line_set)) - 2) and len(line_set) == len(line_set_ref) + 1:
                                for field in fields:
                                    with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                            open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                        lines = f_r.readlines()
                                        f_o.write(lines[ref_idx])
                                        ref_tag[ref_idx] = 1
                                with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                match_flag = True
                                break
                            elif (len(line_set) == 3) and (len(line_set & line_set_ref) == 2) and (abs(len(line_set_ref) - len(line_set)) == 1):
                                for field in fields:
                                    with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/x_ref_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                                        lines = f_r.readlines()
                                        f_o.write(lines[ref_idx])
                                        ref_tag[ref_idx] = 1
                                with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                                        open(os.path.join(new_data_dir, "{}/y_ref.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                                    lines = f_r.readlines()
                                    f_o.write(lines[ref_idx])
                                match_flag = True
                                break
                            else:
                                continue
                        else:
                            continue

                for field in fields:
                    with open(os.path.join(old_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i], field, mode)), 'r') as f_r, \
                            open(os.path.join(new_data_dir, "{}/x_{}.{}.txt".format(dir_modes[i],field, mode)), 'a') as f_o:
                        lines = f_r.readlines()
                        f_o.write(lines[idx])
                with open(os.path.join(old_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i], mode)), 'r') as f_r, \
                        open(os.path.join(new_data_dir, "{}/y_aux.{}.txt".format(dir_modes[i],mode)), 'a') as f_o:
                    lines = f_r.readlines()
                    f_o.write(lines[idx])

                # print('idx is:{} and len is:{}'.format(idx, len(line_set)))
                    # for field in fields:
                    #     with open(os.path.join(old_data_dir, "e2e.{}_ref.{}.txt".format(field, mode)), 'r') as f_r, \
                    #             open(os.path.join(new_data_dir, "e2e.{}_ref.{}.txt".format(field, mode)), 'a') as f_o:
                    #         f_o.write('\n')

    # with open(os.path.join(old_data_dir, "e2e.entry{}.valid.txt".format(ref)), 'r') as f_entry:
        #     lines_entry = f_entry.readlines()
        #     with open(os.path.join(old_data_dir, "e2e.entry{}.valid.txt".format(refs[0])), 'r') as f_entry_x:
        #         lines_entry_x = f_entry_x.readlines()
        #         with open("e2ev14_output/rule/ckpt/hypos.step0.val.txt", 'r') as f_sent:#SST-2/hypos.v10_rule_mask.val.txt"), 'r') as f_sent:
        #             lines_sent = f_sent.readlines()
        #             for (idx_line, line_type) in enumerate(lines_type):
        #                 line_type = line_type.strip('\n').split(' ')
        #                 for (idx_val, attr) in enumerate(line_type):
        #                     entry_list = lines_entry[idx_line].strip('\n').split(' ')
        #                     if(lines_entry_x[idx_line].find(entry_list[idx_val]) == -1):  #if the record not exit in x
        #                         pos_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
        #                         with open("e2ev14_output/rule.2.txt", 'a') as f_w:
        #                             f_w.write(pos_samp)
