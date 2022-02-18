import numpy as np
import pandas as pd
import torch


class AttrPredictor(object):

    def __init__(self, cfg, tops_type=[10]):
        """Create the empty array to count true positive(tp),
            true negative(tn), false positive(fp) and false negative(fn).

        Args:
            class_num : number of classes in the dataset
            tops_type : default calculate top3, top5 and top10
        """

        attr_cloth_file = open(cfg.attr_cloth_file).readlines()
        self.attr_idx2name = {}
        self.attr_name2idx = {}
        for i, line in enumerate(attr_cloth_file[2:]):
            attr_name = line.strip('\n').split()[:-1]
            if len(attr_name) > 1:
                attr_name = " ".join(attr_name)
            else:
                attr_name = attr_name[0]

            self.attr_idx2name[i] = attr_name
            self.attr_name2idx[attr_name] = i

        shorten_df = pd.read_csv(cfg.shorten_csv)
        self.shorten_list_idx = []
        for attr_name in shorten_df["attribute_name"]:
            self.shorten_list_idx.append(self.attr_name2idx[attr_name])

        self.tops_type = tops_type

    def print_attr_name(self, pred_idx):
        for idx in pred_idx:
            print(self.attr_idx2name[idx])

    def get_shortlist_name(self):
        shortlist_names = []
        for attr_name in self.shorten_list_idx:
            shortlist_names.append(self.attr_idx2name[attr_name])
        return shortlist_names

    def show_prediction(self, image_ids, pred, print=False):
        if isinstance(pred, torch.Tensor):
            data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
            data = pred
        else:
            raise TypeError('type {} cannot be calculated.'.format(type(pred)))

        if print:
            for i in range(pred.size(0)):
                image_id = image_ids[i]
                indexes = np.argsort(data[i])[::-1]
                for topk in self.tops_type:
                    idxes = indexes[:topk]
                    print('[ Top%d Attribute Prediction for Image %d]' % (topk, image_id))
                    self.print_attr_name(idxes)

        # return the subset of attributes (542 attrs)
        data = data[:, self.shorten_list_idx]
        return data

    def save_prediction(self, image_ids, pred, save_path):
        output = pd.DataFrame(np.concatenate([
            np.expand_dims(image_ids, axis=1), 
            pred], axis=1))
        output.columns = ["image_id"] + self.get_shortlist_name()
        output = output.sort_values("image_id")
        output.to_csv(save_path, index=False)