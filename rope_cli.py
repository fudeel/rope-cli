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

    def process_video(self, input_video, output_dir, quality=18, threads=2):
        """
        Process video with face swapping

        Args:
            input_video: Path to input video
            output_dir: Directory for output
            quality: Video encoding quality (CRF value)
            threads: Number of processing threads

        Returns:
            Path to output video
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        input_name = Path(input_video).stem
        timestamp = int(time.time())
        output_file = output_path / f"{input_name}_swapped_{timestamp}.mp4"

        print(f"\nProcessing video: {input_video}")
        print(f"Output directory: {output_dir}")
        print(f"Quality: {quality}, Threads: {threads}")

        # Open video
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height} @ {fps}fps, {frame_count} frames")

        # Create temporary output file (without audio)
        temp_file = output_path / f"temp_{timestamp}.mp4"

        # Setup video writer with x264 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_file), fourcc, fps, (width, height))

        # TWEAK: Lowered detection score to be less strict, helps with motion blur or angled faces.
        parameters = {
            # Detection settings
            'DetectTypeTextSel': 'Retinaface',
            'DetectScoreSlider': 30, # Lowered from 50
            'ThresholdSlider': 60, # Lowered from 65 for better matching

            # Basic swapper settings
            'SwapperTypeTextSel': '128',

            # Feature switches - start with minimal settings
            'FaceAdjSwitch': False,
            'StrengthSwitch': True, # Enabled for potentially better results
            'ColorSwitch': False,
            'RestorerSwitch': False,
            'FaceParserSwitch': False,
            'MouthParserSwitch': False,
            'OccluderSwitch': False,
            'DiffSwitch': False,
            'CLIPSwitch': False,
            'OrientSwitch': False,

            # Sliders with default values
            'BorderTopSlider': 0,
            'BorderBottomSlider': 0,
            'BorderSidesSlider': 0,
            'BorderBlurSlider': 6, # Added slight blur for better blending
            'BlendSlider': 10, # Added some blending
            'ColorRedSlider': 0,
            'ColorGreenSlider': 0,
            'ColorBlueSlider': 0,
            'ColorGammaSlider': 1.0,
            'DiffSlider': 0,
            'FaceParserSlider': 0,
            'MouthParserSlider': 0,
            'FaceScaleSlider': 0,
            'KPSScaleSlider': 0,
            'KPSXSlider': 0,
            'KPSYSlider': 0,
            'OccluderSlider': 0,
            'OrientSlider': 0,
            'RestorerSlider': 0,
            'StrengthSlider': 100,
            'CLIPSlider': 0,
            'CLIPTextEntry': '',
            'RestorerTypeTextSel': 'GFPGAN',
            'RestorerDetTypeTextSel': 'Reference'
        }

        control = {
            'MaskViewButton': False
        }

        print("\nProcessing frames...")
        start_time = time.time()

        # Process each frame
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to tensor (following VideoManager pattern)
            img = torch.from_numpy(frame_rgb.astype('uint8')).cuda()
            img = img.permute(2, 0, 1)  # HWC to CHW

            # Detect faces in current frame
            kpss = self.models.run_detect(img, max_num=20, score=parameters['DetectScoreSlider']/100.0)

            # Get embeddings for all faces in frame
            faces_in_frame = []
            for face_kps in kpss:
                face_emb, _ = self.models.run_recognize(img, face_kps)
                faces_in_frame.append({'kps': face_kps, 'embedding': face_emb})

            if faces_in_frame and self.found_faces:
                # Process faces (following VideoManager logic)
                for fface in faces_in_frame:
                    # Find a matching face from the initial scan
                    for found_face in self.found_faces:
                        if self.findCosineDistance(fface['embedding'], found_face['embedding']) >= parameters['ThresholdSlider']/100.0:
                            # Use the assigned source embedding for this found face
                            if found_face.get('AssignedEmbedding') is not None:
                                s_e = found_face['AssignedEmbedding']

                                # Swap the face using swap_core
                                try:
                                    img = self.video_manager.swap_core(img, fface['kps'], s_e, parameters, control)
                                    # Once swapped, move to the next detected face in the frame
                                    break
                                except Exception as e:
                                    print(f"\nWarning: Swap failed for a face in frame {frame_idx}: {e}")
                                    pass

            # Convert back to HWC for writing
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Write frame
            out.write(frame_bgr)

            # Progress update
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                progress = (frame_idx + 1) / frame_count * 100
                print(f"  Progress: {progress:.1f}% ({frame_idx+1}/{frame_count}) - {fps_actual:.1f} fps", end='\r')

        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("\n\nAdding audio from original video...")

        # Use ffmpeg to add audio from original video
        audio_args = [
            "ffmpeg",
            "-y", # Overwrite output file if it exists
            "-i", str(temp_file),
            "-i", str(input_video),
            "-c:v", "libx264",
            "-crf", str(quality),
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
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


def main():
    parser = argparse.ArgumentParser(description='Rope Deepfake CLI - Command line interface for face swapping')

    parser.add_argument('-v', '--video', required=True, help='Input video file path')
    parser.add_argument('-f', '--faces', required=True, help='Directory containing source face images')
    parser.add_argument('-o', '--output', required=True, help='Output directory for processed videos')
    parser.add_argument('-q', '--quality', type=int, default=18,
                       help='Video quality (CRF value, lower=better, default: 18)')
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

