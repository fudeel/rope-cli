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
import torchvision
from tqdm import tqdm

torchvision.disable_beta_transforms_warning()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rope.VideoManager import VideoManager
from rope.Models import Models

# Define device for CUDA operations
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RopeCLI:
    def __init__(self, progress_callback=None):
        """Initialize the Rope CLI with VideoManager

        Args:
            progress_callback: Optional callback function for progress updates
                              Signature: callback(progress: float, message: str)
        """
        # Create Models instance first
        self.models = Models()
        # Pass models to VideoManager
        self.video_manager = VideoManager(self.models)
        self.source_embeddings = []
        self.source_images = []
        self.found_faces = []
        self.main_face_index = -1  # Track the main actor's face
        self.progress_callback = progress_callback
        print("Initializing Rope CLI...")

    def update_progress(self, progress, message):
        """Update progress through callback if available"""
        if self.progress_callback:
            self.progress_callback(progress, message)

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
        # Progress: 0% - 10%
        self.update_progress(0, "Loading source faces...")

        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        face_files = [f for f in faces_path.iterdir()
                     if f.suffix.lower() in image_extensions]

        if not face_files:
            raise ValueError(f"No image files found in {faces_dir}")

        print(f"Found {len(face_files)} face images")

        # Process each face image
        for idx, face_file in enumerate(face_files):
            print(f"  Processing: {face_file.name}")

            # Load image
            img = cv2.imread(str(face_file))
            if img is None:
                print(f"  Warning: Could not read {face_file.name}")
                continue

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to tensor
            img_tensor = torch.from_numpy(img_rgb.astype('uint8')).cuda()
            img_tensor = img_tensor.permute(2, 0, 1)

            # Detect faces - FIXED: removed det_thresh parameter
            kpss = self.models.run_detect(img_tensor, max_num=1)

            # --- FIX ---
            # Check if kpss is not None and has items before proceeding.
            # This avoids the "ValueError: The truth value of an array is ambiguous" error.
            if kpss is not None and len(kpss) > 0:
                # Get embedding
                face_emb, cropped_img = self.models.run_recognize(img_tensor, kpss[0])

                self.source_embeddings.append({
                    'embedding': face_emb,
                    'file': face_file.name
                })

                # Store the actual image for GUI compatibility
                self.source_images.append(img_rgb)

                print(f"    ✓ Face extracted from {face_file.name}")
            else:
                print(f"    Warning: No face detected in {face_file.name}")

            # Update progress
            progress = 0 + (10 * (idx + 1) / len(face_files))
            self.update_progress(progress, f"Loaded {idx + 1}/{len(face_files)} faces")

        if not self.source_embeddings:
            raise ValueError("No valid faces found in source images")

        print(f"Successfully loaded {len(self.source_embeddings)} face embeddings")
        self.update_progress(10, f"Loaded {len(self.source_embeddings)} source faces")

    def find_faces_in_video(self, video_path):
        """
        Find faces in target video and identify the main actor (largest face)

        Args:
            video_path: Path to the video file

        Returns:
            Number of unique faces found (but only main actor will be swapped)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file does not exist: {video_path}")

        print(f"\n🔍 Analyzing video for faces: {video_path.name}")
        # Progress: 10% - 20%
        self.update_progress(10, "Analyzing video for faces...")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Sample frames for face detection
        sample_interval = max(1, int(fps * 2))  # Sample every 2 seconds
        self.found_faces = []
        face_sizes = []  # Track face sizes to identify main actor

        # Variables for tracking faces
        threshold = 0.60
        frames_analyzed = 0
        max_frames_to_analyze = min(total_frames, 20 * sample_interval)

        for i in range(0, max_frames_to_analyze, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            frames_analyzed += 1

            # Convert to RGB and tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(frame_rgb.astype('uint8')).cuda()
            img_tensor = img_tensor.permute(2, 0, 1)

            # Detect faces
            kpss = self.models.run_detect(img_tensor, max_num=10)

            for face_kps in kpss:
                face_emb, cropped_img = self.models.run_recognize(img_tensor, face_kps)

                # Calculate face size (bounding box area)
                try:
                    # Attempt to access bbox as a dictionary key, which is the expected format.
                    bbox = face_kps['bbox']
                except (IndexError, TypeError):
                    # FALLBACK: If face_kps is a raw numpy array of keypoints (which causes an IndexError),
                    # we derive the bounding box from the keypoints' min/max coordinates.
                    # This handles inconsistencies in the data structure returned by the model.
                    x_coords = face_kps[:, 0]
                    y_coords = face_kps[:, 1]
                    bbox = np.array([
                        np.min(x_coords),
                        np.min(y_coords),
                        np.max(x_coords),
                        np.max(y_coords)
                    ])

                face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                # Check if this face is already found
                found = False
                face_idx = -1
                for idx, existing_face in enumerate(self.found_faces):
                    if self.findCosineDistance(existing_face['Embedding'], face_emb) >= threshold:
                        found = True
                        face_idx = idx
                        # Update average face size for this face
                        face_sizes[idx].append(face_area)
                        break

                # If new face, add to found faces
                if not found:
                    self.found_faces.append({
                        'Embedding': face_emb,
                        'kps': face_kps
                    })
                    face_sizes.append([face_area])

            # Update progress
            progress = 10 + (10 * frames_analyzed / (max_frames_to_analyze / sample_interval))
            self.update_progress(progress, f"Analyzing faces... ({frames_analyzed} samples)")

        cap.release()

        # Identify the main actor (face with largest average size)
        if self.found_faces:
            avg_sizes = []
            for idx, sizes in enumerate(face_sizes):
                avg_size = np.mean(sizes)
                avg_sizes.append((idx, avg_size))
                print(f"  Face {idx}: Average size = {avg_size:.0f} pixels²")

            # Sort by average size and get the largest
            avg_sizes.sort(key=lambda x: x[1], reverse=True)
            self.main_face_index = avg_sizes[0][0]

            print(f"\n✓ Found {len(self.found_faces)} unique face(s) in video")
            print(f"★ Main actor identified: Face #{self.main_face_index} (largest average size)")

            # Only assign source embeddings to the main actor's face
            for idx, found_face in enumerate(self.found_faces):
                if idx == self.main_face_index:
                    # This is the main actor - assign source embeddings
                    found_face['SourceFaceAssignments'] = self.source_embeddings
                    if self.source_embeddings:
                        # Calculate average embedding from all source faces
                        embeddings_list = [d['embedding'] for d in self.source_embeddings]
                        found_face['AssignedEmbedding'] = np.mean(embeddings_list, axis=0)
                    print(f"  → Source face assigned to Face #{idx} (main actor)")
                else:
                    # Not the main actor - don't assign any source
                    found_face['SourceFaceAssignments'] = []
                    found_face['AssignedEmbedding'] = None
                    print(f"  → Face #{idx} will not be swapped (supporting actor)")

            # Assign found faces to video manager
            self.video_manager.assign_found_faces(self.found_faces)

            self.update_progress(20, f"Identified main actor from {len(self.found_faces)} faces")
            return len(self.found_faces)
        else:
            print("  No faces found in video")
            self.update_progress(20, "No faces found in video")
            return 0

    def process_video(self, video_path, output_dir, quality=14, threads=2, codec='libx264', preset='slow', progress_callback=None):
        """
        Process video with face swapping - only swaps the main actor's face
        """
        # Update progress callback if provided
        if progress_callback:
            self.progress_callback = progress_callback

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
        print(f"  Main actor face will be swapped")

        # Progress: 20%
        self.update_progress(20, "Setting up video processing...")

        # Set up video manager for processing
        self.video_manager.load_target_video(str(video_path))

        # The found_faces are already assigned with only main actor having source embeddings
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
            'StrengthSlider': 100,
            'ThreadsSlider': threads,
            'DetectTypeTextSel': 'Retinaface',
            'DetectScoreSlider': 50,
            'ThresholdSlider': 55,
            'OrientSwitch': False,
            'OrientSlider': 0,
            'RestorerSwitch': False,
            'RestorerTypeTextSel': 'GFPGAN',
            'RestorerDetTypeTextSel': 'Blend',
            'RestorerSlider': 100,
            'StrengthSwitch': False,
            'BorderTopSlider': 10,
            'BorderSidesSlider': 10,
            'BorderBottomSlider': 10,
            'BorderBlurSlider': 10,
            'BlendSlider': 5,
            'ColorSwitch': False,
            'ColorRedSlider': 0,
            'ColorGreenSlider': 0,
            'ColorBlueSlider': 0,
            'ColorGammaSlider': 1.0,
            'FaceAdjSwitch': False,
            'KPSXSlider': 0,
            'KPSYSlider': 0,
            'KPSScaleSlider': 0,
            'FaceScaleSlider': 0,
            'DiffSwitch': False,
            'DiffSlider': 4,
            'OccluderSwitch': False,
            'OccluderSlider': 0,
            'FaceParserSwitch': False,
            'FaceParserSlider': 0,
            'MouthParserSlider': 0,
            'CLIPSwitch': False,
            'CLIPTextEntry': '',
            'CLIPSlider': 50,
            'SwapperTypeTextSel': '128',
            'RecordTypeTextSel': 'FFMPEG',
            'VideoQualSlider': quality,
            'MergeTextSel': 'Mean',
        }

        self.video_manager.control = {
            'SwapFacesButton': True,
            'AudioButton': False,
            'MaskViewButton': False
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
        print("\nProcessing frames (swapping main actor only)...")
        # Progress: 21% - 95%
        self.update_progress(21, "Processing video frames...")
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
                    # This will only swap faces that have AssignedEmbedding (main actor only)
                    swapped_frame = self.video_manager.swap_video(frame_rgb, frame_num, use_markers=False)

                    # Convert RGB back to BGR for FFMPEG
                    output_frame = cv2.cvtColor(swapped_frame, cv2.COLOR_RGB2BGR)

                except Exception as e:
                    # If swapping fails, use original frame
                    print(f"\n  Warning: Error processing frame {frame_num}: {e}")
                    output_frame = frame

                # Write to FFMPEG
                process.stdin.write(output_frame.tobytes())

                pbar.update(1)

                # Update progress through callback (21% to 95%)
                video_progress = 21 + (74 * (frame_num + 1) / frame_count)
                if frame_num % 10 == 0:  # Update every 10 frames to avoid too many updates
                    self.update_progress(video_progress, f"Processing frame {frame_num + 1}/{frame_count}")

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

        # Progress: 95%
        self.update_progress(95, "Adding audio track...")

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

        # Progress: 100%
        self.update_progress(100, "Processing completed successfully!")

        return str(output_file)


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
            print(f"\nFound {num_faces} face(s) in video. Main actor identified.")
            print("Use without --find-faces-only to process.")
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
