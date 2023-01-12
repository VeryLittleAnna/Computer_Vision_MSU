
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    x5, y5, x6, y6 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
    inter_area = max(0, (x6 - x5)) * max(0, (y6 - y5))
    iou = (inter_area) / ((x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter_area)
    return iou



def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        detections = {x[0] : x for x in frame_obj}
        hyp_detections = {x[0]: x for x in frame_hyp}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        for real_id, hyp_id in matches.items():
            if real_id in detections and hyp_id in hyp_detections:
                cur_iou = iou_score(detections[real_id][1:], hyp_detections[hyp_id][1:])
                if cur_iou > threshold:
                    dist_sum += cur_iou
                    match_count += 1
                    del(detections[real_id])
                    del(hyp_detections[hyp_id])

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ious = []
        for real_id in detections.keys():
            for hyp_id in hyp_detections.keys():
                cur_iou = iou_score(hyp_detections[hyp_id][1:], detections[real_id][1:])
                if cur_iou > threshold:
                    ious.append([cur_iou, real_id, hyp_id])
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: Update matches with current matched IDs
        ious.sort(key=lambda x: x[0], reverse=True)
        for new_matches in ious:
            if (new_matches[1] not in matches.keys()) and (new_matches[2] not in matches.values()):
                dist_sum += new_matches[0]
                match_count += 1
                matches[new_matches[1]] = new_matches[2]
                del detections[new_matches[1]]
                del hyp_detections[new_matches[2]]
    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count
    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    gt_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs
    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        detections = {x[0] : x for x in frame_obj}
        hyp_detections = {x[0]: x for x in frame_hyp}
        gt_count += len(detections)
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for real_id, hyp_id in matches.items():
            if real_id in detections and hyp_id in hyp_detections:
                cur_iou = iou_score(detections[real_id][1:], hyp_detections[hyp_id][1:])
                if cur_iou > threshold:
                    dist_sum += cur_iou
                    match_count += 1
                    del(detections[real_id])
                    del(hyp_detections[hyp_id])

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        ious = []
        for real_id in detections.keys():
            for hyp_id in hyp_detections.keys():
                cur_iou = iou_score(hyp_detections[hyp_id][1:], detections[real_id][1:])
                if cur_iou > threshold:
                    ious.append([cur_iou, real_id, hyp_id])
        # Step 4: Iterate over sorted pairwise IOU
        ious.sort(key=lambda x: x[0], reverse=True)
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
        # Step 6: Update matches with current matched IDs
        for new_matches in ious:
            if (new_matches[1] in detections) and (new_matches[2] in hyp_detections):
                dist_sum += new_matches[0]
                match_count += 1
                if new_matches[1] in matches and matches[new_matches[1]] != new_matches[2]:
                    mismatch_error += 1
                matches[new_matches[1]] = new_matches[2]
                del detections[new_matches[1]]
                del hyp_detections[new_matches[2]]
        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(hyp_detections)
        missed_count += len(detections)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (false_positive + missed_count + mismatch_error) / gt_count

    return MOTP, MOTA
