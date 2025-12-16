import cv2
import numpy as np


def check_local_extremum(rdf_array, index, margin, mode="max"):
    num_points = len(rdf_array)
    current_val = rdf_array[index]
    for i in range(1, margin + 1):
        prev_idx = (index - i) % num_points
        next_idx = (index + i) % num_points
        if mode == "max":
            if rdf_array[prev_idx] > current_val or rdf_array[next_idx] > current_val:
                return False
        elif mode == "min":
            if rdf_array[prev_idx] < current_val or rdf_array[next_idx] < current_val:
                return False
    return True


def extract_peaks_and_valleys(binary_img, pref_point, margin=30):
    """提取波峰波谷，margin 可配"""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], None, None

    contour = max(contours, key=cv2.contourArea).squeeze()
    dists_to_pref = np.linalg.norm(contour - np.array(pref_point), axis=1)
    start_index = np.argmin(dists_to_pref)
    ordered_contour = np.roll(contour, -start_index, axis=0)
    rdf = np.linalg.norm(ordered_contour - np.array(pref_point), axis=1)

    found_points = []
    total_points = len(rdf)
    search_state = 0
    ignore_zone = 50

    for i in range(ignore_zone, total_points - ignore_zone):
        if search_state == 0:
            if check_local_extremum(rdf, i, margin, mode="max"):
                if rdf[i] > 50:
                    found_points.append(
                        {
                            "type": "peak",
                            "index": i,
                            "point": tuple(ordered_contour[i]),
                            "dist": rdf[i],
                        }
                    )
                    search_state = 1
        elif search_state == 1:
            if check_local_extremum(rdf, i, margin, mode="min"):
                if rdf[i] > 20:
                    found_points.append(
                        {
                            "type": "valley",
                            "index": i,
                            "point": tuple(ordered_contour[i]),
                            "dist": rdf[i],
                        }
                    )
                    search_state = 0

    return found_points, rdf, ordered_contour


def analyze_hand_geometry(found_points, rdf_data, ordered_contour):
    peaks = [p for p in found_points if p["type"] == "peak"]
    valleys = [p for p in found_points if p["type"] == "valley"]
    roi_ref_points = {}

    if len(peaks) >= 6:
        return "Anomaly (Too Many Peaks)", [], {}
    if len(peaks) < 5:
        return "Insufficient Peaks", [], {}

    hand_type = "Unknown"
    max_idx = len(ordered_contour)

    if len(peaks) == 5 and len(valleys) >= 4:
        first_valley = valleys[0]
        last_valley = valleys[-1]

        if first_valley["dist"] < last_valley["dist"]:
            hand_type = "Left Hand"
        else:
            hand_type = "Right Hand"

        if hand_type == "Left Hand":
            p_index_t = peaks[1]
            p_index_v2_data = valleys[1]
            p_little_t = peaks[4]
            p_little_v2_data = valleys[3]
            dist_idx_index = abs(p_index_v2_data["index"] - p_index_t["index"])
            idx_v1 = p_index_t["index"] - dist_idx_index
            dist_real_little = abs(p_little_t["index"] - p_little_v2_data["index"])
            idx_lv1 = p_little_t["index"] + dist_real_little
        else:
            p_index_t = peaks[3]
            p_index_v2_data = valleys[2]
            p_little_t = peaks[0]
            p_little_v2_data = valleys[0]
            dist_idx_index = abs(p_index_t["index"] - p_index_v2_data["index"])
            idx_v1 = p_index_t["index"] + dist_idx_index
            dist_real_little = abs(p_little_v2_data["index"] - p_little_t["index"])
            idx_lv1 = p_little_t["index"] - dist_real_little

        idx_v1 = np.clip(idx_v1, 0, max_idx - 1)
        idx_lv1 = np.clip(idx_lv1, 0, max_idx - 1)

        pt_index_v1 = ordered_contour[idx_v1]
        pt_index_v2 = np.array(p_index_v2_data["point"])
        pt_little_v1 = ordered_contour[idx_lv1]
        pt_little_v2 = np.array(p_little_v2_data["point"])

        P1_np = (pt_index_v1 + pt_index_v2) / 2
        P1 = tuple(P1_np.astype(int))
        P2_np = (pt_little_v1 + pt_little_v2) / 2
        P2 = tuple(P2_np.astype(int))

        roi_ref_points = {"P1": P1, "P2": P2}

    return hand_type, [], roi_ref_points
