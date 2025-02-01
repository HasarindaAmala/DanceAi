import numpy as np
from moviepy import VideoFileClip


class VocalSync:
    def __init__(self, reference_path, test_path):
        self.ref = reference_path
        self.test = test_path

    def extract_audio_array(self,duration, video_path, sr=8000):
        """
        Extract audio from the video as a float32 NumPy array (mono), resampled to 'sr'.
        Returns (audio_array, sr).
        """
        clip = VideoFileClip(video_path).subclipped(0, 0 + duration)
        audio_array = clip.audio.to_soundarray(fps=sr)  # <--- replaces 'clip.audio.set_fps(sr)'
        clip.close()

        if audio_array.ndim == 2 and audio_array.shape[1] == 2:
            # stereo => average to mono
            audio_array = audio_array.mean(axis=1)

        return audio_array.astype(np.float32), sr

    def find_audio_offset(self, y_ref, y_test, sr):
        if len(y_ref) == 0 or len(y_test) == 0:
            return 0.0
        corr = np.correlate(y_ref, y_test, mode='full')
        best_idx = np.argmax(corr)
        shift_in_samples = best_idx - (len(y_test) - 1)
        offset_seconds = shift_in_samples / float(sr)
        return offset_seconds

    def shift_or_trim_clip(self, clip, clip_test, offset_seconds):
        if offset_seconds > 0:
            start_trim = offset_seconds
            if start_trim >= clip.duration:
                from moviepy.video.fx import Freeze
                w, h = clip.size
                black_clip = (clip.fx(Freeze, t=0)
                              .subclipped(0, 0.01)
                              .resize((w, h))
                              .set_duration(1.0)
                              .volumex(0))
                return black_clip
            else:
                return clip.subclipped(start_trim)
        elif offset_seconds < 0:
            start_trim = abs(offset_seconds)
            return clip_test.subclipped(start_trim)
        else:
            return clip

    def sync_videos_by_audio(self, ref_path, test_path,
                             synced_ref_output="ref_synced.mp4",
                             synced_test_output="test_synced.mp4",
                             sr=44100):

        print("Extracting audio from reference...")
        y_ref, sr_ref = self.extract_audio_array(10,ref_path, sr=sr)
        print("Extracting audio from test...")
        y_test, sr_test = self.extract_audio_array(10,test_path, sr=sr)

        print("Finding audio offset via cross-correlation...")
        offset_seconds = self.find_audio_offset(y_ref, y_test, sr)
        print(f"Offset found (test - ref) = {offset_seconds:.2f} seconds")

        ref_clip = VideoFileClip(ref_path)
        test_clip = VideoFileClip(test_path)

        print("Shifting/Trimming test clip based on offset...")
        if offset_seconds < 0:
            ref_shifted = ref_clip
            test_shifted = self.shift_or_trim_clip(ref_clip, test_clip, offset_seconds)
            final_duration = min(ref_shifted.duration, test_shifted.duration)
        else:
            ref_shifted = self.shift_or_trim_clip(ref_clip, test_clip, offset_seconds)
            final_duration = min(test_clip.duration, ref_shifted.duration)

        print("Trimming both to the same final duration...")

        if final_duration == test_clip.duration and test_clip.duration == ref_shifted.duration:
            ref_final = ref_shifted
            test_final = test_clip
        elif final_duration < ref_shifted.duration:
            ref_final = ref_shifted.subclipped(0, final_duration)
            test_final = test_clip
        else:
            ref_final = ref_shifted
            test_final = test_clip.subclipped(0, final_duration)

        print(f"Writing out final synced videos => {synced_ref_output}, {synced_test_output}")
        ref_final.write_videofile(synced_ref_output, codec="libx264", audio_codec="aac")
        test_final.write_videofile(synced_test_output, codec="libx264", audio_codec="aac")

        ref_clip.close()
        test_clip.close()
        ref_final.close()
        test_final.close()
        print("Done syncing videos.")
        return synced_ref_output, synced_test_output

    def start_vocalSync(self):

        synced_ref, synced_test = self.sync_videos_by_audio(
            ref_path=self.ref,
            test_path=self.test,
            synced_ref_output="reference_synced.mp4",
            synced_test_output="test_synced.mp4",
            sr=44100
        )
        print("Synced files:", synced_ref, synced_test)
        return synced_ref,synced_test
