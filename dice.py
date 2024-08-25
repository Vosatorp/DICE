#!/usr/bin/env python3

import argparse
from ultralytics import YOLO
import os
import cv2
import numpy as np
import json
import math
import re

def is_image(path):
    return path.endswith(".jpg") or path.endswith(".png") or path.endswith(".jpeg")

def gen_result_path(output_path, prefix):
    i = 0
    while True:
        path = os.path.join(output_path, f"{prefix}_{i}")
        if not os.path.exists(path):
            return path
        i += 1


def convert_to_list_of_images(image_paths):
    result = []
    
    for path in image_paths:
        if os.path.isdir(path):
            result.extend([os.path.join(path, filename) for filename in os.listdir(path) if is_image(filename)])
        elif is_image(path):
            result.append(path)

    return result

def parse_single_image(detected_image):
    img = detected_image.orig_img.copy()

    result = {
        "path": "",
        "total": 0,
        "count": 0,
        "boxes": [],
    }

    for box in detected_image.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = int(box.cls.item()) + 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        mid = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.putText(img, str(label), mid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        result["path"] = detected_image.path
        result["count"] += 1
        result["total"] += label
        result["boxes"].append({
            "xyxy": [x1, y1, x2, y2],
            "mid": mid,
            "label": label,
            "confidence": box.conf.item(),
            "wh": [x2 - x1, y2 - y1],
        })

    return img, result

def validate_single_image(detected_image):
    img, parse_result = parse_single_image(detected_image)

    path = detected_image.path

    val_data_path = re.sub("\\.(jpg|jpeg|png)$", ".txt", path)

    val_data = []
    expected_total = 0
    expected_count = 0
    with open(val_data_path, "r") as f:
        for line in f:
            lbl, x, y, w, h = line.split()
            lbl, x, y, w, h = int(lbl) + 1, float(x), float(y), float(w), float(h)
            x, y = int(x * img.shape[1]), int(y * img.shape[0])

            expected_total += lbl
            expected_count += 1

            val_data.append((lbl, x, y))


    validation_result = {
        "expected_total": expected_total,
        "expected_count": expected_count,
        "total": parse_result["total"],
        "count": parse_result["count"],
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "f1": 0,
    }

    for box in parse_result["boxes"]:
        box["matched"] = False

    for expected_box in val_data:
        lbl, x, y = expected_box

        found = False
        for box in parse_result["boxes"]:
            x1, y1, x2, y2 = box["xyxy"]
            mid_x, mid_y = box["mid"]
            w, h = box["wh"]
            if box["label"] == lbl and x1 <= x <= x2 and y1 <= y <= y2 and not box["matched"]: # and math.hypot(mid_x - x, mid_y - y) < 0.3 * min(w, h):
                box["matched"] = True
                validation_result["true_positive"] += 1
                found = True
                break

        if not found:
            validation_result["false_negative"] += 1
            cv2.circle(img, (x, y), 5, (0, 0, 255), 2)
        else:
            cv2.circle(img, (x, y), 3, (0, 255, 0), 2)

    for box in parse_result["boxes"]:
        if not box["matched"]:
            validation_result["false_positive"] += 1
            x1, y1, x2, y2 = box["xyxy"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    precision = validation_result["true_positive"] / (validation_result["true_positive"] + validation_result["false_positive"])
    recall = validation_result["true_positive"] / (validation_result["true_positive"] + validation_result["false_negative"])
    validation_result["f1"] = 2 * 1 / (1 / precision + 1 / recall)

    return img, validation_result

def validate(model_path, validation_path, conf, iou, output_path):
    path = gen_result_path(output_path, "validate")
    try:
        os.makedirs(path)
    except FileExistsError:
        print(f"Can't create directory {path}")
        exit(1)

    model = YOLO(model_path)
    image_paths = convert_to_list_of_images([validation_path])
    detected = model(image_paths, conf=conf, iou=iou)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for detected_image in detected:
        print(f"Validating {detected_image.path}")

        basename = os.path.basename(detected_image.path).split(".")[0]

        img, validation_result = validate_single_image(detected_image)
        img_path = os.path.join(path, f"{basename}.jpg")
        cv2.imwrite(img_path, img)

        result_path = os.path.join(path, f"{basename}_validation.json")
        with open(result_path, "w") as f:
            json.dump(validation_result, f, indent=4)

        true_positive += validation_result["true_positive"]
        false_positive += validation_result["false_positive"]
        false_negative += validation_result["false_negative"]
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * 1 / (1 / precision + 1 / recall)

    print(f"Total precision: {precision}")
    print(f"Total recall: {recall}")
    print(f"Total F1: {f1}")

    print(f"Detailed validation results saved to {path}")


def parse(model_path, image_paths, conf, iou, output_path):
    path = gen_result_path(output_path, "parse")

    try:
        os.makedirs(path)
    except FileExistsError:
        print(f"Can't create directory {path}")
        exit(1)


    model = YOLO(model_path)
    image_paths = convert_to_list_of_images(image_paths)
    detected = model(image_paths, conf=conf, iou=iou)

    for detected_image in detected:
        print(f"Parsing {detected_image.path}")

        basename = os.path.basename(detected_image.path).split(".")[0]

        img, result = parse_single_image(detected_image)

        img_path = os.path.join(path, f"{basename}.jpg")
        cv2.imwrite(img_path, img)

        result_path = os.path.join(path, f"{basename}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
        

    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Parse dice images")
    parser.add_argument("paths", help="Paths to the images", nargs="*")
    parser.add_argument("-m", "--model", help="Path to the model", default="best.pt")
    parser.add_argument("-v", "--validate", help="Path to the validation dataset", default=None)
    parser.add_argument("--conf", help="Confidence threshold", default=0.25)
    parser.add_argument("--iou", help="IOU threshold", default=0.5)
    parser.add_argument("-o", "--output", help="Output path", default="res")

    args = parser.parse_args()

    if args.validate:
        print("Validation")
        validate(args.model, args.validate, args.conf, args.iou, args.output)
    else:
        print("Parsing")
        parse(args.model, args.paths, args.conf, args.iou, args.output)

if __name__ == '__main__':
    main()