from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2

LEFT_KEYS  = {81, 2424832}  # frame back
UP_KEYS    = {82, 2490368}  # forehand
RIGHT_KEYS = {83, 2555904}  # frame forward
DOWN_KEYS  = {84, 2621440}  # backhand
SPACEBAR   = 32             # serve
ESCAPE     = 27             # exit

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Annotate a video and write a csv file containing tennis shots"
    )
    parser.add_argument("video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Error opening video file")

    df = pd.DataFrame(columns=["Frame", "Shot"])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    shots_list = []

    while cap.isOpened():
        frame_id = max(0, min(frame_id, max(0, total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()

        if not ret:
            print("End of video reached")
            break 


        cv2.imshow("Frame", frame)
        k = cv2.waitKeyEx(30)

        if k in LEFT_KEYS:
            frame_id -= 1
            continue
        elif k in RIGHT_KEYS:
            frame_id += 1
            continue
        elif k in UP_KEYS:
            shots_list.append({"Shot": "Forehand", "Frame": frame_id})
            df = pd.DataFrame.from_records(shots_list)
            print("Added Forehand at frame", frame_id)
        elif k in DOWN_KEYS:
            shots_list.append({"Shot": "Backhand", "Frame": frame_id})
            df = pd.DataFrame.from_records(shots_list)
            print("Added Backhand at frame", frame_id)
        elif k == SPACEBAR:
            shots_list.append({"Shot": "Serve", "Frame": frame_id})
            df = pd.DataFrame.from_records(shots_list)
            print("Added Serve at frame", frame_id)
        elif k == ESCAPE:
            break

        frame_id += 1

    out_file = f"annotation_{Path(args.video).stem}.csv"
    df.to_csv(out_file, index=False)
    print(f"Annotation file was written to {out_file}")
