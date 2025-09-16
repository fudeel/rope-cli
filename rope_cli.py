"""
Rope CLI - Command line interface for Rope deepfake face swapping
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import cv2
from math import ceil
import torchvision
from tqdm import tqdm

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rope.VideoManager import VideoManager
from rope.Models import Models

# Define device for CUDA operations
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RopeCLI:
    def __init__(self):
        """Initialize the Rope CLI with VideoManager"""
        # Create Models instance first
        self.models = Models()
        # Pass models to VideoManager
        self.video_manager = VideoManager(self.models)
        self.source_embeddings = []
        self.source_images = []
        self.found_faces = []
        print("Initializing Rope CLI...")

    def findCosineDistance(self, vector1, vector2):
        """
        Calculate cosine similarity between two vectors
        Same method used in GUI.py and VideoManager.py
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        cosine_distance = dot_product / (norm_vec1 * norm_vec2)
        return cosine_distance

    def load_source_faces(self, faces_dir):
        """
        Load all face images from a directory and extract embeddings

        Args:
            faces_dir: Path to directory containing face images
        """
        faces_path = Path(faces_dir)
        if not faces_path.exists():
            raise ValueError(f"Faces directory does not exist: {faces_dir}")

        print(f"Loading source faces from: {faces_dir}")

        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        face_files = [f for f in faces_path.iterdir()
                     if f.suffix.lower() in image_extensions]

        if not face_files:
            raise ValueError(f"No image files found in {faces_dir}")

        print(f"Found {len(face_files)} face images")

        # Process each face image
        for face_file in face_files:
            print(f"Processing: {face_file.name}")

            # Load image using cv2 (as done in GUI.py)
            img = cv2.imread(str(face_file))
            if img is None:
                print(f"  ✗ Could not load image")
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to tensor and move to CUDA (following VideoManager pattern)
            img_tensor = torch.from_numpy(img.astype('uint8')).cuda()
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW

            # Detect faces using the same method as GUI.py
            try:
                kpss = self.models.run_detect(img_tensor, max_num=1)

                if hasattr(kpss, 'size') and kpss.size > 0:
                    # Get face embedding using the same method as GUI.py
                    face_emb, cropped_img = self.models.run_recognize(img_tensor, kpss[0])

                    if face_emb is not None:
                        self.source_embeddings.append({
                            'embedding': face_emb,
                            'file': str(face_file.name)
                        })
                        print(f"  ✓ Face embedding extracted")
                    else:
                        print(f"  ✗ Could not extract embedding")
                else:
                    print(f"  ✗ No face detected")
            except Exception as e:
                print(f"  ✗ Error processing face: {e}")

        if not self.source_embeddings:
            raise ValueError("No valid face embeddings could be extracted")

        print(f"Successfully loaded {len(self.source_embeddings)} face embeddings")

    def find_faces_in_video(self, video_path):
        """
        Find all unique faces in the video

        Args:
            video_path: Path to input video

        Returns:
            Number of unique faces found
        """
        print(f"Finding faces in video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        # Sample frames to find faces
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total_frames // 20)  # Sample more frames for better detection

        # TWEAK: Lowered threshold for more robust face matching across frames
        threshold = 0.60

        for i in range(0, min(total_frames, 20 * sample_interval), sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB and tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(frame_rgb.astype('uint8')).cuda()
            img_tensor = img_tensor.permute(2, 0, 1)

            # Detect faces
            kpss = self.models.run_detect(img_tensor, max_num=10)

            for face_kps in kpss:
                face_emb, cropped_img = self.models.run_recognize(img_tensor, face_kps)

                # Check if this face is already found
                found = False
                for existing_face in self.found_faces:
                    if self.findCosineDistance(existing_face['embedding'], face_emb) >= threshold:
                        found = True
                        break

                # If new face, add to found faces
                if not found:
                    self.found_faces.append({
                        'embedding': face_emb,
                        'kps': face_kps
                    })

        cap.release()

        print(f"Found {len(self.found_faces)} unique face(s) in video")

        # Create assignments - assign all source faces to all target faces
        # This mimics selecting all faces in the GUI
        for found_face in self.found_faces:
            # Use all source embeddings cyclically
            found_face['SourceFaceAssignments'] = self.source_embeddings
            if self.source_embeddings:
                # Use the first source as the assigned embedding
                found_face['AssignedEmbedding'] = self.source_embeddings[0]['embedding']

        # Assign found faces to video manager (like GUI does)
        self.video_manager.assign_found_faces(self.found_faces)

        return len(self.found_faces)

    def process_video(self, video_path, output_dir, quality=14, threads=2, codec='libx264', preset='slow'):
        """
        Process video with face swapping - WORKING VERSION
        Uses the original swapping logic with enhanced quality settings
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"{video_path.stem}_deepfake_{timestamp}{video_path.suffix}"
        output_file = output_dir / output_name
        temp_file = output_dir / f"temp_{output_name}"

        print(f"\n🎬 Processing: {video_path.name}")
        print(f"  Output: {output_file.name}")
        print(f"  Quality: CRF={quality} (lower=better)")
        print(f"  Found {len(self.found_faces)} target faces")

        # Set up video manager for processing
        self.video_manager.load_target_video(str(video_path))

        # Assign the found faces to video manager with source embeddings
        for i, face in enumerate(self.found_faces):
            # Assign average of source embeddings to each target face
            if self.source_embeddings:
                face['AssignedEmbedding'] = np.mean(self.source_embeddings, axis=0)
                face['SourceFaceAssignments'] = list(range(len(self.source_embeddings)))

        self.video_manager.assign_found_faces(self.found_faces)

        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Set parameters for swapping
        self.video_manager.parameters = {
            'SwapFacesButton': True,
            'StrengthSlider': 1.0,
            'ThreadsSlider': threads,
            'NumFacesSlider': 5,
            'FaceSearchSlider': 0.6,
            'MaskViewButton': False,
            'CLIP_text': '',
            'CLIPSwitch': False,
            'MergeTextSel': 'Mean'
        }

        self.video_manager.control = {
            'SwapFacesButton': True,
            'AudioButton': False
        }

        # Create FFMPEG process for output with enhanced quality
        ffmpeg_args = [
            "ffmpeg",
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-an',
            '-c:v', codec,
            '-preset', preset,
            '-crf', str(quality),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-tune', 'film',
            str(temp_file)
        ]

        process = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE)

        # Process frames
        print("\nProcessing frames...")
        start_time = time.time()

        with tqdm(total=frame_count, desc="Deepfaking", unit="frames") as pbar:
            for frame_num in range(frame_count):
                # Get frame through video manager
                self.video_manager.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = self.video_manager.capture.read()

                if not ret:
                    break

                # Convert BGR to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    # Apply face swapping using video manager's swap logic
                    swapped_frame = self.video_manager.swap_video(frame_rgb, frame_num, marker=False)

                    # Convert RGB back to BGR for FFMPEG
                    output_frame = cv2.cvtColor(swapped_frame, cv2.COLOR_RGB2BGR)

                except Exception as e:
                    # If swapping fails, use original frame
                    print(f"\n  Warning: Error processing frame {frame_num}: {e}")
                    output_frame = frame

                # Write to FFMPEG
                process.stdin.write(output_frame.tobytes())

                pbar.update(1)
                # Update progress stats
                if frame_num % 30 == 0 and frame_num > 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_num / elapsed
                    eta = (frame_count - frame_num) / fps_current if fps_current > 0 else 0
                    pbar.set_postfix({
                        'FPS': f'{fps_current:.1f}',
                        'ETA': f'{eta:.0f}s'
                    })

        # Close FFMPEG stdin and wait for it to finish
        process.stdin.close()
        process.wait()

        # Release video manager capture
        if self.video_manager.capture:
            self.video_manager.capture.release()

        # Add audio back with high quality
        print("\n🎵 Merging audio track...")
        audio_args = [
            "ffmpeg",
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            "-i", str(temp_file),
            "-i", str(video_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-movflags", "+faststart",
            "-shortest",
            str(output_file)
        ]

        subprocess.run(audio_args, capture_output=False, text=True)

        # Remove temp file
        if temp_file.exists():
            temp_file.unlink()

        elapsed_total = time.time() - start_time
        print(f"\n✓ Processing complete!")
        print(f"  Output: {output_file}")
        print(f"  Total time: {elapsed_total:.1f} seconds")
        print(f"  Average FPS: {frame_count / elapsed_total:.1f}")

        return str(output_file)


def swap_faces_in_frame(self, frame, frame_num):
    """
    Swap faces in a single frame

    Args:
        frame: The video frame (BGR format from OpenCV)
        frame_num: Frame number for tracking

    Returns:
        Processed frame with swapped faces
    """
    # Convert BGR to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in current frame
    faces = self.models.analyze_frame(frame_rgb)

    if not faces:
        # No faces detected in this frame, return original
        return frame

    # Process each detected face
    for face in faces:
        try:
            # Find the best matching target face from our found_faces
            best_match_idx = -1
            best_similarity = -1

            # Extract embedding for current face
            current_embedding = self.models.get_embedding_from_face(frame_rgb, face)

            # Find best matching target face
            for idx, target_face in enumerate(self.found_faces):
                similarity = self.findCosineDistance(current_embedding, target_face['embedding'])
                if similarity > best_similarity and similarity > 0.5:  # Threshold for matching
                    best_similarity = similarity
                    best_match_idx = idx

            if best_match_idx >= 0:
                # We found a matching face to swap
                # Use the average of source embeddings for swapping
                if self.source_embeddings:
                    # Calculate average embedding from all source faces
                    avg_embedding = np.mean(self.source_embeddings, axis=0)

                    # Perform the face swap
                    swapped_face = self.models.run_swapper(
                        frame_rgb,
                        face,
                        avg_embedding
                    )

                    # Blend the swapped face back into the frame
                    frame_rgb = self.models.blend_face(
                        frame_rgb,
                        swapped_face,
                        face
                    )

        except Exception as e:
            print(f"  Warning: Could not swap face {face.get('index', 'unknown')}: {e}")
            continue

    # Convert back to BGR for OpenCV/FFMPEG
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


# Alternative simpler implementation that uses video_manager directly
def swap_faces_in_frame_simple(self, frame, frame_num):
    """
    Simpler version that delegates to video_manager's swap logic
    """
    # Set up video manager state for this frame
    self.video_manager.target_media = [None, str(self.current_video_path)]
    self.video_manager.current_frame = frame_num

    # Assign found faces to video manager
    self.video_manager.assign_found_faces(self.found_faces)

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply swapping through video manager
    try:
        # The video_manager expects RGB input and returns RGB output
        swapped_frame = self.video_manager.swap_video(frame_rgb, frame_num, marker=False)

        # Convert back to BGR for OpenCV
        return cv2.cvtColor(swapped_frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  Warning: Frame {frame_num} swap failed: {e}")
        return frame


def main():
    parser = argparse.ArgumentParser(description='Rope Deepfake CLI - Command line interface for face swapping')

    parser.add_argument('-v', '--video', required=True, help='Input video file path')
    parser.add_argument('-f', '--faces', required=True, help='Directory containing source face images')
    parser.add_argument('-o', '--output', required=True, help='Output directory for processed videos')
    parser.add_argument('-q', '--quality', type=int, default=16,
                       help='Video quality (CRF value, lower=better, default: 16)')
    parser.add_argument('-t', '--threads', type=int, default=2, help='Number of processing threads (default: 2)')
    parser.add_argument('--find-faces-only', action='store_true', help='Only find faces without processing')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file does not exist: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.faces):
        print(f"Error: Faces directory does not exist: {args.faces}")
        sys.exit(1)

    # Create CLI instance
    cli = RopeCLI()

    try:
        # Load source faces
        cli.load_source_faces(args.faces)

        # Find faces in video
        num_faces = cli.find_faces_in_video(args.video)

        if args.find_faces_only:
            print(f"\nFound {num_faces} face(s) in video. Use without --find-faces-only to process.")
            sys.exit(0)

        if num_faces == 0:
            print("\nWarning: No faces found in the video to swap. Exiting.")
            sys.exit(0)

        # Process video
        output_file = cli.process_video(
            args.video,
            args.output,
            quality=args.quality,
            threads=args.threads
        )

        print(f"\n✅ Success! Output saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

