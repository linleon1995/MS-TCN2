import matplotlib.pyplot as plt
import numpy as np


class VisActionSeg():
    def __init__(self, data_actions):
        pass

    def single_vis(self, actions: list):
        pass
        vis_acts = self.actions_to_colors(actions)
        plt.imshow(vis_acts)
        plt.show()

    def compare_vis(self, gt_actions: list, pred_actions: list):
        pass

    def actions_to_colors(self, actions):
        actions = np.array(actions)
        aa = actions[-1]
        switchs = np.int32(actions[:-1] != actions[1:])
        switchs = np.append(switchs, 1)
        switch_idxs = np.where(switchs==1)[0]
        temp = np.concatenate([np.int32([0]), switch_idxs])[:-1]
        seg_lens = switch_idxs - temp
        seg_lens[0] = seg_lens[0] + 1
        seg_acts = np.take(actions, switch_idxs)
        
        
        for idx, l in enumerate(seg_lens):
            if idx == 0:
                plt.barh(0, l, height=0.25)
            else:
                plt.barh(0, l, left=last_l, height=0.25)
            last_l = l
        plt.ylim([-1, 1])
        plt.show()

        return actions

def vis_action_seg(actions):
    pass
    # Get unique labels
    # Assign colors