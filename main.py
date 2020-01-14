import glob
import cv2
import copy
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


class TrackedObject:

    """
        TrackedObject: Stores all information about tracked objects by the class Tracker
    """

    def __init__(self, name, start_img_name, start_bbox, start_mask, color):
        self.name = name
        self.imgs_names = []
        self.imgs_names.append(start_img_name)
        self.bboxes = []
        self.bboxes.append(start_bbox)
        self.confirmed = 0
        self.lost_track = 0
        self.masks = {}
        self.masks[start_img_name] = start_mask
        self.print_color = color

    def add_update(self, new_img_name, new_bbox):
        self.imgs_names.append(new_img_name)
        self.bboxes.append(list(new_bbox))


    def add_measurement(self, new_img_name, new_bbox, new_mask):
        self.imgs_names.append(new_img_name)
        self.bboxes.append(new_bbox)
        self.masks[new_img_name] = new_mask

    def replace_measurement(self, img_name, new_bbox, new_mask):
        self.bboxes[self.imgs_names.index(img_name)] = new_bbox
        self.masks[img_name] = new_mask

    def set_lost_track(self):
        self.lost_track = 1

    def get_lost_track(self):
        return self.lost_track

    def get_last_box(self):
        return self.bboxes[-1]

    def get_all_boxes(self):
        return self.bboxes

    def get_all_imgs_names(self):
        return self.imgs_names

    def get_color(self):
        return self.print_color


class Tracker:

    """
        Wrapper around OpenCV cv2.TrackerMedianFlow_create tracker that enables to track multiple objects, track objects
        inbetween objects detections and evaluates tracking performance by IoU metric
    """

    def __init__(self, start_img, start_img_name, start_bboxes, start_mask):

        self.tracked_objects = {}
        self.trackers = {}
        self.imgs_names = []
        self.imgs = []
        self.corr_threshold = 0.2
        self.colors = [(255, 0, 0),(255, 128, 0),(255, 255, 0), (191, 255, 0), (64, 255, 0), (0, 255, 191), \
                       (0, 191, 255), (0, 64, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128)]

        for idx, obj in enumerate(start_bboxes):
            a = start_bboxes[obj].split(" ")
            curr_box = [float(val) for val in a]
            self.tracked_objects[idx] = TrackedObject(str(idx), start_img_name, curr_box, start_mask[obj], self.colors[idx])
            self.trackers[idx] = cv2.TrackerMedianFlow_create()
            self.trackers[idx].init(start_img, tuple(curr_box))

        self.imgs_names.append(start_img_name)
        self.imgs.append(start_img)

    def update(self, new_img, new_img_name):

        self.imgs.append(new_img)
        self.imgs_names.append(new_img_name)
        for obj, tracker in self.trackers.items():
            status, bbox = tracker.update(new_img)
            if status is True:
                self.tracked_objects[obj].add_update(new_img_name, bbox)
            else:
                self.tracked_objects[obj].set_lost_track()

    def calc_IoU(self, box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[0]+box1[2], box2[0]+box2[2])
        y_bottom = min(box1[1]+box1[3], box2[1]+box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        return intersection / float(box1_area + box2_area - intersection)

    def get_curr_tracked_objs(self):
        objs_return = {}
        for name, obj in self.tracked_objects.items():
            if not obj.get_lost_track():
                objs_return[name] = obj
        return objs_return

    def new_detection(self, img, img_name, bboxes, mask):

        objs = self.get_curr_tracked_objs()
        if objs:
            for box_name, box in bboxes.items():
                box = [float(val) for val in box.split(" ")]
                IoU_scores = {}
                for obj_name, obj in objs.items():
                    IoU_scores[obj_name] = (self.calc_IoU(box, obj.get_last_box()))
                obj_detected_name = max(IoU_scores, key=IoU_scores.get)
                if self.corr_threshold < float(IoU_scores[obj_detected_name]):
                    self.tracked_objects[obj_detected_name].replace_measurement(img_name, box, mask[box_name])
                    #self.tracked_objects[obj_detected_name].add_measurement(img_name, box, mask[box_name])
                    self.trackers[obj_detected_name].init(img, tuple(box))
                else:
                    new_idx = len(self.tracked_objects)
                    self.tracked_objects[new_idx] = \
                        TrackedObject(str(new_idx), img_name, box, mask[box_name], self.colors[new_idx])
                    self.trackers[new_idx] = cv2.TrackerMedianFlow_create()
                    self.trackers[new_idx].init(img, tuple(box))

    def draw_last_tracking_frame(self):

        last_img = copy.deepcopy(self.imgs[-1])
        objs = self.get_curr_tracked_objs()
        colors = [(255,0,0),(0,255,0),(0,0,255)]
        for i, obj_name in enumerate(objs):
            bbox = objs[obj_name].get_last_box()
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.putText(last_img, "Name: "+ str(obj_name), (int(bbox[0]), int(bbox[1])+int(bbox[3])+20), \
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, objs[obj_name].get_color())
            cv2.rectangle(last_img, p1, p2, objs[obj_name].get_color(), 5, 1)

        return last_img

    def save_all_frames(self, path):
        output = copy.deepcopy(self.imgs[0])
        for img in self.imgs[1:]:
            output = np.vstack((output, img))
        cv2.imwrite(path, output)

    def draw_save_all_frames(self, path):

        output = copy.deepcopy(self.imgs[0])

        for i, obj_name in enumerate(self.tracked_objects):
            try:
                curr_img_position = self.tracked_objects[obj_name].get_all_imgs_names().index(self.imgs_names[0])
            except ValueError:
                continue
            bbox = (self.tracked_objects[obj_name].get_all_boxes())[curr_img_position]
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.putText(output, "Name: "+ str(obj_name), (int(bbox[0])-40, int(bbox[1])+int(bbox[3])+40), \
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, self.tracked_objects[obj_name].get_color(), 2)
            cv2.rectangle(output, p1, p2, self.tracked_objects[obj_name].get_color(), 5, 1)

        gap = np.zeros((30, output.shape[1], 3))
        output = np.vstack((output, gap))

        for n, img in enumerate(self.imgs[1:10]):
            img = np.vstack((img, gap))
            for i, obj_name in enumerate(self.tracked_objects):
                try:
                    curr_img_position = self.tracked_objects[obj_name].get_all_imgs_names().index(self.imgs_names[n+1])
                except ValueError:
                    continue
                bbox = (self.tracked_objects[obj_name].get_all_boxes())[curr_img_position]
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.putText(img, "Name: " + str(obj_name), (int(bbox[0])-40, int(bbox[1]) + int(bbox[3]) + 40),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, self.tracked_objects[obj_name].get_color(), 2)
                cv2.rectangle(img, p1, p2, self.tracked_objects[obj_name].get_color(), 5, 1)

            output = np.vstack((output, img))
        cv2.imwrite(path, output)

    def evaluate_ground_truth(self, ground_truth):

        x = np.arange(len(self.imgs_names))
        width = 0.7

        fig, ax = plt.subplots()
        shift_i=0
        shift = np.linspace(-width/2,width/2, len(self.tracked_objects))
        for obj_name, obj in self.tracked_objects.items():
            IoU = [0]*len(self.imgs_names)
            for idx, box in enumerate(obj.get_all_boxes()):
                img_name = obj.get_all_imgs_names()[idx]
                IoU_scores_temp = []
                gt_img_bbox = ground_truth[img_name]
                for name, gt_bbox in gt_img_bbox[0].items():
                    IoU_scores_temp.append(self.calc_IoU(box, [float(n) for n in gt_bbox.split(" ")]))
                IoU[self.imgs_names.index(img_name)] = max(IoU_scores_temp)
            ax.bar(x+shift[shift_i], IoU, width/len(self.tracked_objects), label=obj_name)
            shift_i = shift_i + 1
        ax.set_xticks(x)
        ax.legend()
        ax.set_ylabel('IoU score')
        ax.set_title('IoU scores of each tracked object in an image sequence 0...'+str(len(self.imgs_names)-1)\
                     +' with 2 trees tracked')

        plt.tight_layout()
        plt.show()



def load_annotated_data(path_annotations):

    bbox = defaultdict(list)
    masks = defaultdict(list)

    for name in img_names:
        bbox_temp = {}
        print(path_annotations + "*_box_" + name + "..txt")
        for file in glob.glob(path_annotations + "*_box_" + name + "..txt"):

            input_box = (open(file, "r").read())
            # TO DO: use regex to find object name
            bbox_temp[file[-(len(name)+11):-(len(name)+10)]] = input_box[:-1]
        bbox[name].append(bbox_temp)

        mask_temp = {}
        for file in glob.glob(path_annotations + "*_mask_" + name + ".bmp"):

            # TO DO: use regex to find object name
            mask_temp[file[-(len(name)+11):-(len(name)+10)]] = cv2.imread(file)
        masks[name].append(mask_temp)

    return bbox, masks


def dataset_paths(dataset_num):

    """
        dataset_paths: Creates paths to the image's location and annotated data
                        dataset_num - provides which dataset to use, possible values: 1,2,3
    """
    path_src_imgs = os.path.abspath(os.getcwd()) + "\\Data\\seq" + str(dataset_num) + "\\"
    path_annotations = os.path.abspath(os.getcwd()) + "\\Data\\seq" + str(dataset_num) + "_annotation\\"

    return path_src_imgs, path_annotations


if __name__ == "__main__":

    path_src_imgs, path_annotations = dataset_paths(1)

    obj_names = ["ID1", "ID2", "ID3"]
    img_names = [os.path.basename(file)[:-4] for file in glob.glob(path_src_imgs + "*.bmp")]
    images = {os.path.basename(file)[:-4]: cv2.imread(file) for file in glob.glob(path_src_imgs+"*.bmp")}
    bboxes, masks = load_annotated_data(path_annotations)

    for i, img_name in enumerate(img_names):

        if i == 0:
            tracker = Tracker(images[img_name], img_name, bboxes[img_name][0], masks[img_name][0])
        elif i % 5 == 0:
            # Update positions of the previously detected objects
            tracker.update(images[img_name], img_name)
            # Every 5th image there are new detected objects to track
            tracker.new_detection(images[img_name], img_name, bboxes[img_name][0], masks[img_name][0])
        else:
            # Update positions of the previously detected objects
            tracker.update(images[img_name], img_name)

        img_with_boxes = tracker.draw_last_tracking_frame()
        cv2.imshow("Current frame", img_with_boxes)
        #cv2.imwrite("X:\\final sem\\Tracking course\\Data2\\seq3_tracking\\"+str(i)+".jpg", img_with_boxes)

        # Exit if ESC pressed
        k = cv2.waitKey(1000) & 0xff
        if k == 27:
            break

    #tracker.draw_save_all_frames("seq3.jpg")
    tracker.evaluate_ground_truth(bboxes)
