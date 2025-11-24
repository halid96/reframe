#!/usr/bin/env python3
import cv2
import numpy as np
import pickle
import argparse
import os
import ffmpeg
from ultralytics import YOLO
from pathlib import Path

# Model loading: check models folder first, download there if needed
MODEL_NAME = "yolov8l.pt"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

model_path = MODELS_DIR / MODEL_NAME
if model_path.exists():
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
else:
    print(f"Model not found in {MODELS_DIR}, downloading...")
    import shutil
    # Download model - ultralytics will download it
    temp_model = YOLO(MODEL_NAME)
    
    # Try to find the downloaded model file and copy to models folder
    # Check multiple possible locations
    possible_locations = [
        Path(MODEL_NAME),  # Current directory
        Path.home() / ".ultralytics" / "weights" / MODEL_NAME,  # Ultralytics cache
    ]
    
    copied = False
    for source_path in possible_locations:
        if source_path.exists():
            shutil.copy2(source_path, model_path)
            print(f"Model copied to: {model_path}")
            model = YOLO(str(model_path))
            copied = True
            break
    
    if not copied:
        # If we can't find the file, the model is loaded in memory
        # Save it for next time by getting the weights path
        try:
            weights_path = getattr(temp_model.model, 'yaml_file', None) or getattr(temp_model, 'ckpt_path', None)
            if weights_path and Path(weights_path).exists():
                shutil.copy2(weights_path, model_path)
                print(f"Model saved to: {model_path}")
                model = YOLO(str(model_path))
            else:
                model = temp_model
                print(f"Using model from cache (will check {MODELS_DIR} next time)")
        except:
            model = temp_model
            print(f"Using model from cache (will check {MODELS_DIR} next time)")

def _compute_output_dimensions(w, h):
    # desired aspect ratio (portrait): width / height = 9/16
    desired_ar = 9.0 / 16.0
    # prefer full height if possible
    out_w = int(h * desired_ar)
    out_h = h
    if out_w > w:
        # fallback to full width and adjust height
        out_w = w
        out_h = int(w / desired_ar)
        # ensure we don't exceed original height
        out_h = min(out_h, h)
    # round to multiple of 16
    out_w = max(16, (out_w // 16) * 16)
    out_h = max(16, (out_h // 16) * 16)
    return out_w, out_h

def analyze_fast(video_path):
    """
    Analyzes the video to track subjects, calculate group width, and record the 
    number of people detected. It applies a stability filter to the mode decision.
    
    Returns:
        frames_data: List of tuples [(target_x, target_y, group_w, num_people, is_split_mode_stable), ...]
        original_w: Video width
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate output dimensions to determine the split threshold
    out_w, _ = _compute_output_dimensions(w, h)
    padding = 60 
    split_threshold = out_w - padding
    
    # State variables for stability filtering
    MIN_DURATION_SECONDS = 0.4
    MIN_FRAMES_FOR_SPLIT = max(1, int(fps * MIN_DURATION_SECONDS)) # e.g., 10 frames at 25 FPS
    
    split_condition_count = 0
    is_split_mode_stable = False
    
    # Stores (target_center, group_width, num_people, is_split_mode_stable) per frame
    frames_data = [] 
    main_speaker_center = (w//2, h//2)
    override_target = None
    override_frames = 0
    i = 0

    # Helper target variables for smoothing (tracking the last *detected* values)
    last_target = main_speaker_center
    last_group_w = 0
    last_num_people = 0
    last_is_split_mode_stable = False

    print("Meme-perfect analysis — tracking subjects frame by frame...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Check if the frame array is valid before proceeding with analysis
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            print(f"\rWarning: Invalid frame read at analysis frame {i}. Skipping.", end="")
            i += 1
            continue

        if i % 6 == 0:  # analyze every 6th frame
            small = cv2.resize(frame, (640, int(640 * h / w)))
            results = model.track(small, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

            sx = w / 640
            sy = h / (640 * h / w)

            person_boxes = []
            new_appearance = None
            new_appearance_size = 0

            for b in results.boxes:
                # require reasonable confidence
                if b.conf[0] < 0.45:
                    continue

                # ensure this is a person (class 0)
                cls = int(b.cls[0]) if hasattr(b, 'cls') else None
                if cls is not None and cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                X1, X2 = int(x1 * sx), int(x2 * sx)
                Y1, Y2 = int(y1 * sy), int(y2 * sy)

                # clamp to frame bounds
                X1 = max(0, min(X1, w-1))
                X2 = max(0, min(X2, w-1))
                Y1 = max(0, min(Y1, h-1))
                Y2 = max(0, min(Y2, h-1))
                if X2 <= X1 or Y2 <= Y1:
                    continue

                person_boxes.append((X1, Y1, X2, Y2))

                size = (X2 - X1) * (Y2 - Y1)
                center = ((X1 + X2)//2, (Y1 + Y2)//2)

                track_id = int(b.id[0]) if b.id is not None else -1
                # sudden new object / meme-appearance logic
                if track_id == -1 or b.id is None:
                    if size > new_appearance_size and size > 80000:
                        new_appearance_size = size
                        new_appearance = center

            # group logic
            num_people = len(person_boxes) # Capture number of people

            if num_people == 0:
                group_center = main_speaker_center
                group_w = 0
            elif num_people == 1:
                l, t, r, b = person_boxes[0]
                group_center = ((l + r)//2, (t + b)//2)
                group_w = r - l
            else:
                left   = min(b[0] for b in person_boxes)
                right  = max(b[2] for b in person_boxes)
                top    = min(b[1] for b in person_boxes)
                bottom = max(b[3] for b in person_boxes)

                group_center = ((left + right)//2, (top + bottom)//2)
                group_w = right - left

                # Clamp center so it doesn't drift too far off edge
                half = group_w // 2
                group_center = (
                    max(half, min(w - half, group_center[0])),
                    group_center[1]
                )

            # TARGET selection with override handling
            if new_appearance and new_appearance_size > 100000:
                override_target = new_appearance
                override_frames = 90
                target = new_appearance
            elif override_frames > 0:
                target = override_target
                override_frames -= 1
            else:
                target = group_center
                main_speaker_center = group_center

            # --- STABILITY CHECK FOR SPLIT MODE ---
            should_split_now = num_people >= 2 and group_w > split_threshold
            
            # Since we sample every 6th frame, we increment the counter by 6
            if should_split_now:
                split_condition_count += 6 
                if split_condition_count >= MIN_FRAMES_FOR_SPLIT:
                    is_split_mode_stable = True
            else:
                # Condition lost: Reset count and mode
                split_condition_count = 0
                is_split_mode_stable = False
                
            # Store the current detected values for smoothing
            last_target = target
            last_group_w = group_w
            last_num_people = num_people
            last_is_split_mode_stable = is_split_mode_stable

        else:
            # --- INTERPOLATION FRAME (i % 6 != 0) ---
            if override_frames > 0:
                target = override_target
                override_frames -= 1
            else:
                # Smooth move toward last detected main speaker
                tx = int(main_speaker_center[0] * 0.8 + last_target[0] * 0.2)
                ty = int(main_speaker_center[1] * 0.8 + last_target[1] * 0.2)
                target = (tx, ty)
                last_target = target
            
            # Use the last stable values for group attributes
            group_w = last_group_w
            num_people = last_num_people
            is_split_mode_stable = last_is_split_mode_stable

            # Since this is an interpolated frame, we still need to check and update the counter
            if split_condition_count > 0 and split_condition_count < MIN_FRAMES_FOR_SPLIT:
                split_condition_count += 1
                if split_condition_count >= MIN_FRAMES_FOR_SPLIT:
                     is_split_mode_stable = True
                     last_is_split_mode_stable = True


        frames_data.append((target, group_w, num_people, is_split_mode_stable))
        i += 1
        if i % 100 == 0:
            print(f"\rProcessing frame {i}/{total}...", end="", flush=True)

    cap.release()
    print(f"\n✓ Analysis complete! Processed {len(frames_data)} frames.")
    return frames_data, w

def reframe(video_path, frames_data, preview=False, output_path=None):
    """
    Reframes the video frame-by-frame using the pre-calculated stable mode decision.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)

    # Base output dims (portrait 9:16)
    out_w, out_h = _compute_output_dimensions(w, h)
    
    # Prepare Output
    tmp = "tmp_final.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    # State for smoothing center crop movement
    pos_x = w//2 - out_w//2
    pos_y = h//2 - out_h//2
    SMOOTH = 0.12 

    frame_count = 0
    total_frames = len(frames_data)

    for (target, group_w, num_people, is_split_mode_stable) in frames_data:
        ret, frame = cap.read()
        if not ret: 
            break
            
        # --- ROBUST FRAME VALIDITY CHECK (CRITICAL FOR CV2.ERROR) ---
        # Ensures 'frame' is a valid array before any slicing or processing
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            print(f"\rWarning: Invalid frame read at reframing frame {frame_count}. Skipping.")
            frame_count += 1
            continue

        # --- MODE DECISION: Use the pre-calculated STABLE flag ---
        if is_split_mode_stable:
            # --- SPLIT SCREEN MODE ---
            split_point = w // 2
            
            # Use safety checks for slicing in case 'frame' is unexpectedly small
            if split_point > frame.shape[1] or (w - split_point) <= 0:
                print(f"\rWarning: Split dimensions invalid at frame {frame_count}. Skipping.")
                frame_count += 1
                continue
                
            left_half = frame[0:h, 0:split_point]
            right_half = frame[0:h, split_point:w]
            
            # Since width of halves might be uneven due to 'w' being odd, resize
            if left_half.shape[1] != right_half.shape[1]:
                 right_half = cv2.resize(right_half, (left_half.shape[1], h)) 

            # Stack: Right (Speaker 2) usually goes top, or Left (Speaker 1) bottom
            stacked_frame = cv2.vconcat([right_half, left_half]) 
            
            # Resize to exact output dimensions
            final_frame = cv2.resize(stacked_frame, (out_w, out_h))
            
            if preview:
                cv2.putText(final_frame, "MODE: SPLIT (STABLE)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        else:
            # --- CENTER CROP MODE ---
            # compute requested top-left so target is centered
            tx = target[0] - out_w//2
            ty = target[1] - out_h//2

            # clamp to image bounds
            tx = max(0, min(int(tx), w - out_w))
            ty = max(0, min(int(ty), h - out_h))

            # apply smoothing
            pos_x += SMOOTH * (tx - pos_x)
            pos_y += SMOOTH * (ty - pos_y)

            x, y = int(round(pos_x)), int(round(pos_y))

            # final safety clamp
            x = max(0, min(x, w - out_w))
            y = max(0, min(y, h - out_h))

            crop = frame[y:y+out_h, x:x+out_w]
            
            # --- CROP DIMENSION CHECK (Prevents the OpenCV resize error) ---
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                print(f"\rWarning: Crop resulted in zero dimension at frame {frame_count}. Skipping.")
                frame_count += 1
                continue
            
            # Resize just in case
            if crop.shape[1] != out_w or crop.shape[0] != out_h:
                final_frame = cv2.resize(crop, (out_w, out_h))
            else:
                final_frame = crop

            if preview:
                cv2.putText(final_frame, "MODE: CENTER", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        writer.write(final_frame)

        if preview:
            cv2.imshow("Auto Reframe Preview", final_frame)
            if cv2.waitKey(1) == ord('q'): break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"\rReframing frame {frame_count}/{total_frames}...", end="", flush=True)

    cap.release()
    writer.release()
    if preview: cv2.destroyAllWindows()

    # Determine output file path
    if output_path:
        out_file = output_path
        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        out_file = f"{os.path.splitext(video_path)[0]}_reframed.mp4"

    print(f"\nMerging audio...")
    video = ffmpeg.input(tmp)
    audio = ffmpeg.input(video_path).audio
    # Use h.264 video codec for wide compatibility
    ffmpeg.output(video, audio, out_file, vcodec='libx264', acodec='aac', s=f"{out_w}x{out_h}").overwrite_output().run(quiet=True)
    os.remove(tmp)
    print(f"✓ Complete! Output: {out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video_path")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--output", type=str, help="Custom output path")
    p.add_argument("--cleanup", action="store_true", help="Delete cache files after processing")
    args = p.parse_args()

    CACHE_DIR = Path(".cache")
    CACHE_DIR.mkdir(exist_ok=True)
    
    video_path_obj = Path(args.video_path)
    # Append '_v6' to cache name due to new safety checks and stable logic
    cache_filename = f"{video_path_obj.stem}_{hash(str(video_path_obj.absolute())) % 100000}_v6.pkl"
    cache = CACHE_DIR / cache_filename
    
    frames_data = []

    try:
        # Note: We force a re-analysis if the new cache file doesn't exist to ensure the latest logic is run.
        if cache.exists():
            print("Loading cached analysis...")
            frames_data, original_w = pickle.load(open(cache, "rb"))
        else:
            frames_data, original_w = analyze_fast(args.video_path)
            pickle.dump((frames_data, original_w), open(cache, "wb"))
            print(f"Analysis cached: {cache}")

        reframe(args.video_path, frames_data, args.preview, args.output)
        
        if args.cleanup and cache.exists():
            cache.unlink()
            print(f"Cache file removed: {cache}")
    except Exception as e:
        if args.cleanup and cache.exists():
            cache.unlink()
        raise