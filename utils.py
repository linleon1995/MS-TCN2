# XXX: temp
import os
from datetime import datetime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import cv2
# from eval import get_labels_start_end_time



def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def mstcn_loss(batch_target, predictions, mask):
	cls_loss = 0
	smooth_loss = 0
	_lambda = 1 # classification loss weight
	_tau = 0.15 # smooth loss weight
	num_classes = predictions.shape[2]
	# TODO: why clamp to 0 16?
	for p in predictions:
        # classification loss
		cls_loss += nn.CrossEntropyLoss(ignore_index=-100)(
			p.transpose(2, 1).contiguous().view(-1, num_classes), batch_target.view(-1))

        # smooth loss
		temp_smooth_loss = nn.MSELoss(reduction='none')(
            F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1))
		smooth_loss += torch.mean(
			torch.clamp(
				temp_smooth_loss, min=0, max=16
			)*mask[:, :, 1:]
		)
	loss = _lambda*cls_loss + _tau*smooth_loss
	return loss


def survey(results, category_names, all_category):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    all_category = np.array(all_category)

    labels = list(results.keys())
    # data = np.array([r['width'] for r in list(results.values())])
    # data_cum = data.cumsum(axis=1)
    # XXX: temp
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.0, 1.0, len(all_category)))
    # category_colors = plt.get_cmap('RdYlGn')(
    #     np.linspace(0.15, 0.85, len(all_category)))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    width_list = [sum(r['width']) for r in list(results.values())]
    print(width_list)
    x_range = max(width_list)
    ax.set_xlim(0, x_range)

    for result_name, result in results.items():
        left = 0
        for width, label in zip(result['width'], result['label']):
            color_idx = np.where(all_category==label)[0][0]
            color = category_colors[color_idx]
            ax.barh(result_name, width, left=left, height=0.2,
                    label=label, color=color)
            left += width

    # TODO: legend
    def legend_without_duplicate_labels(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), loc='center right')
        
    legend_without_duplicate_labels(ax)
    # ax.text(5, 0.5, f'Precision 0.9', color='black')
    # for i, colname in enumerate(category_names):
    #     color_idx = np.where(all_category==colname)[0][0]
    #     color = category_colors[color_idx]
    #     widths = data[:, i]
    #     starts = data_cum[:, i] - widths
    #     ax.barh(labels, widths, left=starts, height=0.5,
    #             label=colname, color=color)
    #     xcenters = starts + widths / 2

        # r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # for y, (x, c) in enumerate(zip(xcenters, widths)):
        #     ax.text(x, y, str(int(c)), ha='center', va='center',
        #             color=text_color)


    # category_names = list(np.unique(np.array(all_category)))
    # ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
    #           loc='lower left', fontsize='small')

    return fig, ax


class VisActionSeg():
    def __init__(self, data_actions):
        self.action_dict = data_actions

    def single_vis(self, correct_actions: list, pred_actions: list):
        vis_acts = self.actions_to_colors(correct_actions, pred_actions)
        # plt.imshow(vis_acts)
        # plt.show()

    def compare_vis(self, correct_actions: list, pred_actions: list):
        pass

    def actions_to_colors(self, correct_actions: list, pred_actions: list):
        # actions = np.array(actions)
        # switchs = np.int32(actions[:-1] != actions[1:])
        # switchs = np.append(switchs, 1)
        # switch_idxs = np.where(switchs==1)[0]
        # temp = np.concatenate([np.int32([0]), switch_idxs])[:-1]
        # seg_lens = switch_idxs - temp
        # seg_lens[0] = seg_lens[0] + 1
        # seg_acts = np.take(actions, switch_idxs)

        
        correct_labels, correct_starts, correct_ends = get_labels_start_end_time(correct_actions, bg_class=[])
        correct_widths = np.array(correct_ends) - np.array(correct_starts)
        pred_labels, pred_starts, pred_ends = get_labels_start_end_time(pred_actions, bg_class=[])
        pred_widths = np.array(pred_ends) - np.array(pred_starts)
        results = {
            'Prediction': {'width': pred_widths.tolist(), 'label': pred_labels},
            'Correct': {'width': correct_widths.tolist(), 'label': correct_labels}
        }
        labels = correct_labels + pred_labels
        # all_category = np.arange(min(labels), max(labels)+1)
        all_category = np.unique(labels)
        fig, ax = survey(results, labels, all_category)
        
        # for idx, l in enumerate(seg_lens):
        #     if idx == 0:
        #         plt.barh(0, l, height=0.25)
        #     else:
        #         plt.barh(0, l, left=last_l, height=0.25)
        #     last_l = l
        # plt.ylim([-1, 1])
        plt.show()
        # fig.savefig()

        # return actions

def vis_action_seg(actions):
    pass
    # Get unique labels
    # Assign colors



def video_vis():
    # Create a VideoCapture object and read from input file
    f = r'C:\Users\test\Desktop\Leon\Datasets\Breakfast\BreakfastII_15fps_qvga_sync\P03\cam01\P03_cereals.avi'
    cap = cv2.VideoCapture(f)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")
    
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.rectangle(frame, (100, 150), (500, 600),
                          (0, 255, 0), -1)
            cv2.imshow('Frame', frame)
            
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
        # Break the loop
        else:
            break
    
    # When everything done, release
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # category_names = ['Strongly disagree', 'Disagree', 'Disagree',
    #                 'Neither agree nor disagree', 'Agree', 'Strongly agree', 'Agree']
    # results = {
    #     'Prediction': [10, 15, 17, 32, 26, 15, 17],
    #     'Correct': [26, 22, 29, 10, 13, 15, 17],
    #     'Question 3': [35, 37, 7, 2, 19, 15, 17],
    #     'Question 4': [32, 11, 9, 15, 33, 15, 17],
    #     'Question 5': [21, 29, 5, 5, 40, 15, 17],
    #     'Question 6': [8, 19, 5, 30, 38, 15, 17]
    # }
    # all_category = ['Strongly disagree', 'Disagree',
    #                 'Neither agree nor disagree', 'Agree', 'Strongly agree']

    # survey(results, category_names, all_category)
    # plt.show()


    video_vis()