"""Session counting semantics: no duplicate counts across consecutive frames.

Covers the dashboard requirements:
- Frames Analysed, Faces Seen, Mask and No Mask are separate metrics.
- The same face visible across many frames is counted once.
- Mask + No Mask always equals Faces Seen.
- Mask ratio = (mask detections / total face detections) * 100, zero-safe.
"""

from app import DetectionTracker, bbox_iou


def detection(label="with_mask", bbox=(10, 10, 40, 40), confidence=0.9):
    display = "Mask" if label == "with_mask" else "No Mask"
    return {
        "label": label,
        "display_label": display,
        "confidence": confidence,
        "bbox": list(bbox),
    }


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    partial = bbox_iou((0, 0, 10, 10), (5, 0, 10, 10))
    assert 0.3 < partial < 0.4


def test_same_face_across_consecutive_frames_counted_once():
    tracker = DetectionTracker()
    events_total = 0
    for _ in range(115):
        events_total += len(tracker.update([detection()]))
    assert tracker.frames == 115
    assert tracker.faces == 1
    assert tracker.with_mask == 1
    assert tracker.without_mask == 0
    assert events_total == 1


def test_face_leaving_and_returning_is_counted_again():
    tracker = DetectionTracker()
    tracker.update([detection(bbox=(10, 10, 40, 40))])
    assert tracker.faces == 1

    for _ in range(DetectionTracker.MAX_MISSED_FRAMES):
        tracker.update([])
        assert tracker.faces == 1  # brief occlusion: still the same face

    tracker.update([])  # gone for good: the track is dropped
    events = tracker.update([detection(bbox=(10, 10, 40, 40))])
    assert len(events) == 1
    assert tracker.faces == 2


def test_two_faces_in_one_frame_are_both_counted():
    tracker = DetectionTracker()
    events = tracker.update(
        [
            detection(bbox=(0, 0, 40, 40)),
            detection(label="without_mask", bbox=(300, 300, 40, 40)),
        ]
    )
    assert len(events) == 2
    assert tracker.faces == 2
    assert tracker.with_mask == 1
    assert tracker.without_mask == 1


def test_classification_flip_counts_a_new_detection_and_keeps_totals_consistent():
    tracker = DetectionTracker()
    tracker.update([detection(label="with_mask")])
    events = tracker.update([detection(label="without_mask")])
    assert len(events) == 1
    assert events[0]["label"] == "without_mask"
    assert tracker.faces == 2
    assert tracker.with_mask + tracker.without_mask == tracker.faces


def test_totals_always_add_up():
    tracker = DetectionTracker()
    sequence = [
        [detection(bbox=(0, 0, 40, 40))],
        [],
        [detection(bbox=(0, 0, 40, 40))],
        [detection(label="without_mask", bbox=(400, 0, 40, 40))],
        [],
        [],
        [],
        [],
        [detection(bbox=(0, 0, 40, 40))],
    ]
    for frame in sequence:
        tracker.update(frame)
        assert tracker.with_mask + tracker.without_mask == tracker.faces
    # Face A, then face B, then face A returning after a long absence.
    assert tracker.faces == 3
    assert tracker.with_mask == 2
    assert tracker.without_mask == 1
    assert tracker.frames == len(sequence)


def test_mask_ratio_matches_reference_example():
    """Reference test case from the requirements (not stored in the app):
    115 frames, 48 face detections, 46 mask, 2 no mask, ratio 95.8%."""
    tracker = DetectionTracker()
    for index in range(48):
        label = "with_mask" if index < 46 else "without_mask"
        # Distinct, non-overlapping positions so each appearance is a new face.
        x = 100 * (index % 5)
        y = 100 * (index // 5)
        tracker.update([detection(label=label, bbox=(x, y, 40, 40))])
        tracker.update([])
    while tracker.frames < 115:
        tracker.update([])

    assert tracker.frames == 115
    assert tracker.faces == 48
    assert tracker.with_mask == 46
    assert tracker.without_mask == 2
    assert tracker.with_mask + tracker.without_mask == tracker.faces

    ratio = (tracker.with_mask / tracker.faces) * 100
    assert round(ratio, 4) == 95.8333
    assert f"{ratio:.1f}%" == "95.8%"


def test_zero_detections_never_divide_by_zero():
    tracker = DetectionTracker()
    for _ in range(5):
        tracker.update([])
    assert tracker.frames == 5
    assert tracker.faces == 0
    ratio = round(tracker.with_mask / tracker.faces, 4) if tracker.faces else None
    assert ratio is None


def test_reset_clears_everything():
    tracker = DetectionTracker()
    tracker.update([detection()])
    tracker.update([detection()])
    tracker.reset()
    assert tracker.frames == 0
    assert tracker.faces == 0
    assert tracker.with_mask == 0
    assert tracker.without_mask == 0
    events = tracker.update([detection()])
    assert len(events) == 1  # after reset the face is new again
