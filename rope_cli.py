import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
import time
import subprocess
from math import floor

# Import Rope modules
import rope.Models as Models
import rope.VideoManager as VM

torchvision.disable_beta_transforms_warning()


class RopeCLI:
    """Command-line interface for Rope deepfake processing"""

    def __init__(self):
        self.models = Models.Models()
        self.video_manager = VM.VideoManager(self.models)
        self.source_embeddings = []
        self.found_faces = []

    def load_source_faces(self, faces_dir):
        """Load and process source face embeddings from directory"""
        print(f"Loading source faces from: {faces_dir}")

        faces_path = Path(faces_dir)
        if not faces_path.exists():
            raise ValueError(f"Faces directory does not exist: {faces_dir}")

        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        face_files = []
        for ext in image_extensions:
            face_files.extend(faces_path.glob(f'*{ext}'))
            face_files.extend(faces_path.glob(f'*{ext.upper()}'))

        if not face_files:
            raise ValueError(f"No face images found in {faces_dir}")

        print(f"Found {len(face_files)} face images")

        # Process each face image
        for face_file in face_files:
            print(f"Processing: {face_file.name}")

            # Read and process image
            img = cv2.imread(str(face_file))
            if img is None:
                print(f"  Warning: Could not load {face_file.name}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to tensor
            img_tensor = torch.from_numpy(img).cuda()
            img_tensor = img_tensor.permute(2, 0, 1)

            # Detect face
            try:
                kpss = self.models.run_detect(img_tensor, max_num=1)[0]

                # Get face embedding
                face_emb, _ = self.models.run_recognize(img_tensor, kpss)

                self.source_embeddings.append({
                    'file': str(face_file),
                    'embedding': face_emb
                })
                print(f"  ✓ Face embedding extracted")

            except (IndexError, Exception) as e:
                print(f"  Warning: No face detected in {face_file.name}")
                continue

        if not self.source_embeddings:
            raise ValueError("No valid face embeddings could be extracted")

        print(f"Successfully loaded {len(self.source_embeddings)} face embeddings")

    def find_faces_in_video(self, video_path):
        """
        Find faces in the target video (first frame)

        Robust face detection with proper numpy array handling to avoid
        boolean context ambiguity errors. Follows Single Responsibility Principle.

        Args:
            video_path (str): Path to the video file

        Returns:
            int: Number of faces detected

        Raises:
            ValueError: If video cannot be opened, no frames can be read, or no faces found
        """
        print(f"Finding faces in video: {video_path}")

        # Initialize video capture with error handling
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame from video")

            # Convert to RGB and prepare tensor
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame).cuda()
            frame_tensor = frame_tensor.permute(2, 0, 1)

            # Detect faces using the model
            kpss = self.models.run_detect(frame_tensor, max_num=1)

            # CRITICAL FIX: Properly handle numpy array boolean evaluation
            # This prevents "truth value of array is ambiguous" error
            if kpss is None or (hasattr(kpss, '__len__') and len(kpss) == 0):
                raise ValueError("No faces found in video")

            print(f"Found {len(kpss)} face(s) in video")

            # Map detected faces to source face embeddings (1:1 mapping)
            self.found_faces = []
            for i, kps in enumerate(kpss):
                if i < len(self.source_embeddings):
                    self.found_faces.append({
                        'kps': kps,
                        'source_embedding': self.source_embeddings[i]['embedding']
                    })

            return len(kpss)

        finally:
            # Ensure video capture is always released (Dependency Inversion Principle)
            cap.release()

    def process_video(self, input_video, output_dir, quality=18, threads=2):
        """Process the video with face swapping"""
        print(f"\nProcessing video: {input_video}")
        print(f"Output directory: {output_dir}")
        print(f"Quality: {quality}, Threads: {threads}")

        # Setup paths
        video_path = Path(input_video)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        timestamp = str(time.time())[:10]
        output_file = output_path / f"{video_path.stem}_swapped_{timestamp}.mp4"
        temp_file = output_path / f"{video_path.stem}_temp_{timestamp}.mp4"

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

        # Setup FFmpeg writer
        ffmpeg_args = [
            "ffmpeg",
            '-hide_banner',
            '-loglevel', 'error',
            "-y",  # Overwrite output files
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{frame_width}x{frame_height}",
            "-r", str(fps),
            "-i", "pipe:",
            "-vf", "format=yuvj420p",
            "-c:v", "libx264",
            "-crf", str(quality),
            "-r", str(fps),
            str(temp_file)
        ]

        ffmpeg_process = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Process frames
        frame_count = 0
        start_time = time.time()

        print("\nProcessing frames...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).cuda()
                frame_tensor = frame_tensor.permute(2, 0, 1)

                # Detect faces in current frame
                try:
                    kpss = self.models.run_detect(frame_tensor, max_num=1)

                    # CRITICAL FIX: Properly handle numpy array boolean evaluation
                    if kpss is not None and hasattr(kpss, '__len__') and len(kpss) > 0:
                        # Swap face using the first detected face and first source embedding
                        for kps in kpss:
                            # Use first source embedding for swapping
                            source_emb = self.source_embeddings[0]['embedding']

                            # Use the proper face swap method
                            swapped_tensor = self.swap_faces_simple(frame_tensor, source_emb, kps)
                            frame_tensor = swapped_tensor
                            break  # Only process first face

                except Exception as e:
                    print(f"Warning: Face swap failed on frame {frame_count}: {e}")

                # Convert back to numpy and BGR for output
                frame_rgb_output = frame_tensor.permute(1, 2, 0).cpu().numpy()
                frame_bgr_output = cv2.cvtColor(frame_rgb_output.astype(np.uint8), cv2.COLOR_RGB2BGR)

                # Write frame to FFmpeg
                try:
                    ffmpeg_process.stdin.write(frame_bgr_output.tobytes())
                except BrokenPipeError:
                    print("FFmpeg pipe broken, stopping...")
                    break

                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames) * 100
                    print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_actual:.1f} fps")

        finally:
            # Cleanup video capture
            cap.release()

            # Close FFmpeg stdin and wait for completion
            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close()
            ffmpeg_process.wait()

        # Add audio from original video
        print("\nAdding audio from original video...")
        audio_args = [
            "ffmpeg",
            '-hide_banner',
            '-loglevel', 'error',
            "-y",  # Overwrite output files
            "-i", str(temp_file),
            "-i", str(video_path),
            "-c", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output_file)
        ]
        subprocess.run(audio_args)

        # Remove temp file
        if temp_file.exists():
            temp_file.unlink()

        elapsed_total = time.time() - start_time
        print(f"\n✓ Processing complete!")
        print(f"  Output: {output_file}")
        print(f"  Total time: {elapsed_total:.1f} seconds")
        print(f"  Average FPS: {frame_count / elapsed_total:.1f}")

        return str(output_file)

    def swap_faces_simple(self, frame_tensor, source_embedding, target_kps):
        """
        Simple face swap using the models

        Args:
            frame_tensor: Input frame as tensor
            source_embedding: Face embedding from source image
            target_kps: Target face keypoints

        Returns:
            Tensor: Frame with swapped face
        """
        try:
            # Calculate swapper latent from source embedding
            latent = self.models.calc_swapper_latent(source_embedding)

            # Prepare face crop from target
            cropped_face = self.models.crop_face(frame_tensor, target_kps)

            # Create output tensor for swap
            output = torch.zeros((1, 3, 128, 128), dtype=torch.float32, device='cuda')

            # Run the face swapper
            self.models.run_swapper(cropped_face, latent, output)

            # Paste swapped face back to frame
            result = self.models.paste_face(frame_tensor, output, target_kps)

            return result

        except Exception as e:
            print(f"Swap error: {e}")
            return frame_tensor


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
        sys.exit(1)


if __name__ == "__main__":
    main()