import xml.dom.minidom
import numpy as np
import numpy as np
import pandas as pd
from itertools import product
from scipy import ndimage as nd
from skimage import measure
from tqdm import tqdm


def get_coordinates(annotation_file):
    DOMTree = xml.dom.minidom.parse(annotation_file)
    collection = DOMTree.documentElement
    coordinatess = collection.getElementsByTagName("Coordinates")
    polygons = []
    for coordinates in coordinatess:
        coordinate = coordinates.getElementsByTagName("Coordinate")
        poly_coordinates = []
        for point in coordinate:
            x = point.getAttribute("X")
            y = point.getAttribute("Y")
            poly_coordinates.append([float(x), float(y)])
        polygons.append(np.array(poly_coordinates, dtype=int))
    return polygons


def read_shapes(shape_file):
    shape_dict = {}
    with open(shape_file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line_records = line.split(',')
            shape_dict[line_records[0].split('.')[0]] = [int(
                line_records[1]), int(line_records[2])]
    return shape_dict


def circular_mask(candidate_region, center, radius):
    # for candidate_region, every row is a coordinate pair (x, y)
    distance = np.sqrt(np.sum(np.power(candidate_region - center, 2), axis=1))
    mask = distance <= radius
    return candidate_region[mask]


def argmax2d(array):
    return np.unravel_index(np.argmax(array, axis=None), array.shape)


# the evaluation code is borrowed from camelyon16 challenge official github.
EVALUATION_MASK_LEVEL = 5  # Image level at which the evaluation is done
L0_RESOLUTION = 0.243  # pixel resolution at level 0


def computeEvaluationMask(mask, resolution, level):
    """Computes the evaluation mask.

    Args:
        mask:    numpy array of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made

    Returns:
        evaluation_mask
    """
    distance = nd.distance_transform_edt(1 - mask)
    # 75µm is the equivalent size of 5 tumor cells
    Threshold = 75/(resolution * pow(2, level) * 2)
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity=2)
    return evaluation_mask


def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)

    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object 
        should be less than 275µm to be considered as ITC (Each pixel is 
        0.243µm*0.243µm in level 0). Therefore the major axis of the object 
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made

    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
    return Isolated_Tumor_Cells


# def readCSVContent(csvDIR):
#     """Reads the data inside CSV file

#     Args:
#         csvDIR:    The directory including all the .csv files containing the results.
#         Note that the CSV files should have the same name as the original image

#     Returns:
#         Probs:      list of the Probabilities of the detected lesions
#         Xcoor:      list of X-coordinates of the lesions
#         Ycoor:      list of Y-coordinates of the lesions
#     """
#     df = pd.read_csv(csvDIR)
#     Probs = df["confidence"]
#     Xcoor, Ycoor = df["x"], df["y"]
#     return Probs, Xcoor, Ycoor


def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells):
    """Generates true positive and false positive stats for the analyzed image

    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells

    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections

        TP_probs:   A list containing the probabilities of the True positive detections

        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)

        detection_summary:   A python dictionary object with keys that are the labels 
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate]. 
        Lesions that are missed by the algorithm have an empty value.

        FP_summary:   A python dictionary object with keys that represent the 
        false positive finding number and values that contain detection 
        details [confidence score, X-coordinate, Y-coordinate]. 
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1, max_label+1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []

    FP_counter = 0
    if (is_tumor):
        for i in range(0, len(Xcorr)):
            HittedLabel = evaluation_mask[Xcorr[i], Ycorr[i]]
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter += 1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i] > TP_probs[HittedLabel-1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel-1] = Probs[i]
    else:
        for i in range(0, len(Xcorr)):
            FP_probs.append(Probs[i])
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
            FP_counter += 1

    num_of_tumors = max_label - len(Isolated_Tumor_Cells)
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary


def non_maxima_suppression(heatmap, threshold=0.5, radius=64):
    center_list = []
    prob = []
    x_max, y_max = heatmap.shape

    while np.max(heatmap) > threshold:
        center = argmax2d(heatmap)
        center_list.append(center)
        prob.append(heatmap[center])
        candidate_x = range(
            max(0, center[0] - radius), min(x_max, center[0] + radius + 1))
        candidate_y = range(
            max(0, center[1] - radius), min(y_max, center[1] + radius + 1))
        candidate_region = list(product(candidate_x, candidate_y))
        candidate_region = np.array(candidate_region)
        masked_region = circular_mask(candidate_region, center, radius)
        heatmap[masked_region[:, 0], masked_region[:, 1]] = 0
        heatmap[center] = 0

    df = pd.DataFrame({
        "x": [c[0] for c in center_list],
        "y": [c[1] for c in center_list],
        "confidence": prob
    })
    return df


def computeFROC(slide_names, all_dfs, all_img_annos, all_labels):
    '''
    compute FROC score
        slide_names: str
        all_dfs: dataframes for all slides. 
        all_img_annos: xxx
        all_labels: xxx
    '''
    FROC_data = np.zeros((4, len(slide_names)), dtype=object)
    FP_summary = np.zeros((2, len(slide_names)), dtype=object)
    detection_summary = np.zeros((2, len(slide_names)), dtype=object)

    for idx, slide_name in enumerate(slide_names):
        df = all_dfs[idx]
        img_anno = all_img_annos[idx]
        slide_name = slide_name + ".tif" if ".tif" not in slide_name else slide_name
        Probs = df["confidence"]
        Xcorr = df["x"]
        Ycorr = df["y"]
        tumor_flag = all_labels[idx]

        if tumor_flag:
            ground_truth_mask = img_anno.copy()
            evaluation_mask = computeEvaluationMask(ground_truth_mask,
                                                    resolution=L0_RESOLUTION,
                                                    level=EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(
                evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)

        else:
            evaluation_mask = 0
            ITC_labels = []

        FROC_data[0][idx] = slide_name
        FP_summary[0][idx] = slide_name
        detection_summary[0][idx] = slide_name
        FROC_data[1][idx], FROC_data[2][idx], \
            FROC_data[3][idx], detection_summary[1][idx], \
            FP_summary[1][idx] = \
            compute_FP_TP_Probs(Ycorr,
                                Xcorr,
                                Probs,
                                tumor_flag,
                                evaluation_mask,
                                ITC_labels)

    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs)/float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs)/float(sum(FROC_data[3]))

    eval_threshold = np.array([.25, .5, 1, 2, 4, 8])
    eval_TPs = np.interp(
        eval_threshold, total_FPs[::-1], total_sensitivity[::-1])

    return np.mean(eval_TPs)
