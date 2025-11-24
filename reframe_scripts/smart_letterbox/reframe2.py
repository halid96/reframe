#!/usr/bin/env python3
import cv2
import numpy as np
import pickle
import argparse
import os
import ffmpeg
from ultralytics import YOLO
from pathlib import Path

# --- MODEL INITIALIZATION ---
MODEL_NAME = "yolov8l.pt"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

model_path = MODELS_DIR / MODEL_NAME
try:
    if model_path.exists():
        print(f"Loading model from: {model_path}")
        model = YOLO(str(model_path))
    else:
        print(f"Model not found in {MODELS_DIR}, downloading...")
        import shutil
        temp_model = YOLO(MODEL_NAME)
        
        # Logic to locate and copy the downloaded model to the standard folder
        possible_locations = [Path(MODEL_NAME), Path.home() / ".ultralytics" / "weights" / MODEL_NAME]
        copied = False
        for source_path in possible_locations:
            if source_path.exists():
                shutil.copy2(source_path, model_path)
                model = YOLO(str(model_path))
                copied = True
                break
        
        if not copied:
            model = temp_model
            print(f"Using model from cache (will check {MODELS_DIR} next time)")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    # Initialize a placeholder model to allow execution to continue if possible
    class MockYOLO:
        def track(self, *args, **kwargs): return [MockResults()]
    class MockResults:
        boxes = []
    model = MockYOLO()


def _compute_output_dimensions(w, h):
    """Calculates the target 9:16 portrait dimensions based on input video size."""
    desired_ar = 9.0 / 16.0
    out_w = int(h * desired_ar)
    out_h = h
    if out_w > w:
        out_w = w
        out_h = int(w / desired_ar)
        out_h = min(out_h, h)
    # Ensure dimensions are multiples of 16 for better encoding compatibility
    out_w = max(16, (out_w // 16) * 16) 
    out_h = max(16, (out_h // 16) * 16)
    return out_w, out_h

def get_cluster_center(boxes):
    """Calculates the center of the overall bounding box for a list of boxes."""
    if not boxes:
        return None
    
    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[2] for b in boxes) 
    y_max = max(b[3] for b in boxes) 

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    return center_x, center_y

def analyze_fast(video_path):
    """
    Analyzes the video to track subjects, calculate group width, and apply
    the stability filter for mode switching.
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
    
    # --- Strict Threshold for Split Screen (Podcast/Studio Mode) ---
    # We require the subjects to spread across 85% of the portrait output width 
    STRICT_THRESHOLD_MULTIPLIER = 0.85 
    split_threshold = int(out_w * STRICT_THRESHOLD_MULTIPLIER) 
    split_threshold = max(split_threshold, 600) # Absolute minimum for safety
    
    # Mode stability settings (set to instant switch, 1 frame)
    MIN_DURATION_SECONDS = 0.0 
    MIN_FRAMES_FOR_SPLIT = max(1, int(fps * MIN_DURATION_SECONDS)) 
    
    split_condition_count = 0
    is_split_mode_stable = False
    
    # Stores (target_center, num_people, is_split_mode_stable, target_L, target_R) per frame
    frames_data = [] 
    main_speaker_center = (w//2, h//2)

    # State variables for interpolation
    last_target = main_speaker_center
    last_num_people = 0
    last_is_split_mode_stable = False
    last_target_L = (w//3, h//2)
    last_target_R = (2*w//3, h//2)

    i = 0

    print("Analyzing video and caching mode decisions...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Only run YOLO detection every 6th frame to speed up analysis
        if i % 6 == 0:  
            small = cv2.resize(frame, (640, int(640 * h / w)))
            
            try:
                results = model.track(small, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            except:
                results = MockResults() 

            sx = w / 640
            sy = h / (640 * h / w)

            person_boxes = []
            
            # 1. Gather all 'person' bounding boxes
            for b in results.boxes:
                if b.conf is None or b.conf[0] < 0.45: continue
                cls = int(b.cls[0]) if hasattr(b, 'cls') else None
                if cls is not None and cls != 0: continue 

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                X1, X2 = int(x1 * sx), int(x2 * sx)
                Y1, Y2 = int(y1 * sy), int(y2 * sy)

                # Clamp bounds and check validity
                X1, X2 = max(0, min(X1, w-1)), max(0, min(X2, w-1))
                Y1, Y2 = max(0, min(Y1, h-1)), max(0, min(Y2, h-1))
                if X2 <= X1 or Y2 <= Y1: continue

                person_boxes.append((X1, Y1, X2, Y2))

            # --- GROUP METRICS CALCULATION ---
            num_people = len(person_boxes)
            group_w = 0
            group_center = main_speaker_center
            target_L = last_target_L
            target_R = last_target_R


            if num_people >= 2:
                # Calculate overall bounds
                left   = min(b[0] for b in person_boxes)
                right  = max(b[2] for b in person_boxes)
                group_w = right - left
                group_center = ((left + right)//2, group_center[1])
                main_speaker_center = group_center # Update overall target

                # 2. Split Box Logic: Calculate two distinct targets
                mid_x = (left + right) // 2
                
                left_boxes = [b for b in person_boxes if (b[0] + b[2])/2 < mid_x]
                right_boxes = [b for b in person_boxes if (b[0] + b[2])/2 >= mid_x]

                if left_boxes:
                    target_L = get_cluster_center(left_boxes)
                if right_boxes:
                    target_R = get_cluster_center(right_boxes)
                
            elif num_people == 1:
                # Single person: update center and width
                l, t, r, b = person_boxes[0]
                group_center = ((l + r)//2, (t + b)//2)
                group_w = r - l
                main_speaker_center = group_center

            # --- STABILITY CHECK FOR SPLIT MODE (Strict Trigger) ---
            should_split_now = num_people >= 2 and group_w > split_threshold
            
            if should_split_now:
                split_condition_count += 6 
            else:
                split_condition_count = 0

            # Apply the stability filter
            if split_condition_count >= MIN_FRAMES_FOR_SPLIT:
                is_split_mode_stable = True
            elif split_condition_count == 0:
                is_split_mode_stable = False
                
            # Update interpolation state variables
            last_target = main_speaker_center
            last_num_people = num_people
            last_is_split_mode_stable = is_split_mode_stable
            if target_L: last_target_L = target_L
            if target_R: last_target_R = target_R


        else:
            # --- INTERPOLATION FRAME ---
            # Smooth move toward last detected main speaker center
            tx = int(main_speaker_center[0] * 0.8 + last_target[0] * 0.2)
            ty = int(main_speaker_center[1] * 0.8 + last_target[1] * 0.2)
            main_speaker_center = (tx, ty)
            last_target = main_speaker_center
            
            # Use the last stable values for mode and split targets
            num_people = last_num_people
            is_split_mode_stable = last_is_split_mode_stable
            target_L = last_target_L
            target_R = last_target_R

            # Update the counter even on interpolated frames to maintain stability filter
            if split_condition_count > 0 and split_condition_count < MIN_FRAMES_FOR_SPLIT:
                split_condition_count += 1
                if split_condition_count >= MIN_FRAMES_FOR_SPLIT:
                     is_split_mode_stable = True
                     last_is_split_mode_stable = True


        frames_data.append((main_speaker_center, num_people, is_split_mode_stable, target_L, target_R))
        i += 1
        if i % 100 == 0:
            print(f"\rProcessing frame {i}/{total}...", end="", flush=True)

    cap.release()
    print(f"\n✓ Analysis complete! Processed {len(frames_data)} frames.")
    return frames_data, w

def calculate_center_crop_coords(center_x, center_y, frame_w, frame_h, crop_w, crop_h):
    """Calculates the top-left (x, y) coordinates for a crop centered on a point."""
    target_x = center_x - crop_w // 2
    target_y = center_y - crop_h // 2

    # Clamp X: Ensure the crop stays within the original video bounds 
    target_x = max(0, min(target_x, frame_w - crop_w))

    # Clamp Y: Ensure the crop stays within the original video bounds
    target_y = max(0, min(target_y, frame_h - crop_h))
    
    return target_x, target_y


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

    # --- DAMPING/SMOOTHING FACTOR for Center Crop Mode ---
    pos_x = w//2 - out_w//2
    pos_y = h//2 - out_h//2
    SMOOTH = 0.05 

    frame_count = 0
    total_frames = len(frames_data)
    half_out_w = out_w // 2

    for (target_overall, num_people, is_split_mode_stable, target_L, target_R) in frames_data:
        ret, frame = cap.read()
        if not ret: break

        final_frame = None

        # --- MODE DECISION: Use the pre-calculated STABLE flag ---
        if is_split_mode_stable and num_people >= 2:
            # --- SPLIT BOX (ZOOMED) MODE RESTORED ---
            
            # 1. Calculate Crop for Left Subject/Group
            L_center_x, L_center_y = target_L
            L_tx, L_ty = calculate_center_crop_coords(L_center_x, L_center_y, w, h, half_out_w, out_h)
            L_crop = frame[L_ty:L_ty + out_h, L_tx:L_tx + half_out_w]

            # 2. Calculate Crop for Right Subject/Group
            R_center_x, R_center_y = target_R
            R_tx, R_ty = calculate_center_crop_coords(R_center_x, R_center_y, w, h, half_out_w, out_h)
            R_crop = frame[R_ty:R_ty + out_h, R_tx:R_tx + half_out_w]
            
            # 3. Stitch them together (Horizontal Stack)
            if L_crop.size > 0 and R_crop.size > 0 and L_crop.shape == R_crop.shape:
                final_frame = np.hstack((L_crop, R_crop))
            # Fallback if crops are bad (shouldn't happen with clamping)
            elif L_crop.size > 0:
                final_frame = cv2.resize(L_crop, (out_w, out_h))
            elif R_crop.size > 0:
                final_frame = cv2.resize(R_crop, (out_w, out_h))

            if preview and final_frame is not None:
                cv2.putText(final_frame, "MODE: SPLIT BOX (ZOOMED)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        else:
            # --- CENTER CROP MODE ---
            target_center_x, target_center_y = target_overall
            
            # Calculate target crop position without smoothing
            tx, ty = calculate_center_crop_coords(target_center_x, target_center_y, w, h, out_w, out_h)

            # Apply smoothing (Lerp)
            pos_x += SMOOTH * (tx - pos_x)
            pos_y += SMOOTH * (ty - pos_y)

            x, y = int(round(pos_x)), int(round(pos_y))

            # Final safety clamp
            x = max(0, min(x, w - out_w))
            y = max(0, min(y, h - out_h))

            # Create crop slice
            crop = frame[y:y+out_h, x:x+out_w]
            
            if crop.size > 0 and crop.shape[1] == out_w and crop.shape[0] == out_h:
                final_frame = crop
            elif crop.size > 0:
                # Resize if the crop size was slightly off due to rounding near the edge
                final_frame = cv2.resize(crop, (out_w, out_h))
            
            if preview and final_frame is not None:
                cv2.putText(final_frame, "MODE: CENTER (DAMPED)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # 5. Write Frame to Output
        if final_frame is not None and final_frame.shape[0] == out_h and final_frame.shape[1] == out_w:
            writer.write(final_frame)
        
        if preview and final_frame is not None:
            cv2.imshow("Auto Reframe Preview", final_frame)
            if cv2.waitKey(1) == ord('q'): break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"\rReframing frame {frame_count}/{total_frames}...", end="", flush=True)

    cap.release()
    writer.release()
    if preview: cv2.destroyAllWindows()

    # Determine output file path and merge audio
    if output_path:
        out_file = output_path
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    else:
        out_file = f"{os.path.splitext(video_path)[0]}_reframed.mp4"

    print(f"\nMerging audio...")
    try:
        video = ffmpeg.input(tmp)
        audio = ffmpeg.input(video_path).audio
        ffmpeg.output(video, audio, out_file, vcodec='libx264', acodec='aac', s=f"{out_w}x{out_h}").overwrite_output().run(quiet=True)
        os.remove(tmp)
        print(f"✓ Complete! Output: {out_file}")
    except Exception as e:
        print(f"Warning: Could not merge audio using ffmpeg. Output video file {tmp} remains. Error: {e}")
        if os.path.exists(tmp):
             os.rename(tmp, out_file)
             print(f"Saving temporary video file as final output: {out_file}")


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
    # Cache version v8 reflects the restored Split Box (Zoomed) logic
    cache_filename = f"{video_path_obj.stem}_{hash(str(video_path_obj.absolute())) % 100000}_v8.pkl"
    cache = CACHE_DIR / cache_filename
    
    frames_data = []

    try:
        # Check if the cache exists and load it
        if cache.exists():
            print("Loading cached analysis...")
            # Unpack the new 5-element tuple structure
            frames_data, original_w = pickle.load(open(cache, "rb")) 
        else:
            # Run analysis and save cache
            frames_data, original_w = analyze_fast(args.video_path)
            pickle.dump((frames_data, original_w), open(cache, "wb"))
            print(f"Analysis cached: {cache}")

        reframe(args.video_path, frames_data, args.preview, args.output)
        
        if args.cleanup and cache.exists():
            cache.unlink()
            print(f"Cache file removed: {cache}")
    except Exception as e:
        print(f"A major error occurred: {e}")
        if args.cleanup and cache.exists():
            cache.unlink()