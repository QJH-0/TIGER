import os
import subprocess
import uuid
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import gradio as gr
import spaces
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import look2hear.models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
dnr_model = look2hear.models.TIGERDNR.from_pretrained("JusperLee/TIGER-DnR", cache_dir="cache").to(device).eval()
sep_model = look2hear.models.TIGER.from_pretrained("JusperLee/TIGER-speech", cache_dir="cache").to(device).eval()

TARGET_SR = 16000
MAX_SPEAKERS = 4

def extract_audio_from_video(video_path, freq):
    video = VideoFileClip(video_path)
    session_id = uuid.uuid4().hex[:8]
    audio_path = f"temp_audio/{session_id}.wav"
    os.makedirs("temp_audio", exist_ok=True)
    video.audio.write_audiofile(audio_path, fps=freq, verbose=False, logger=None)
    return audio_path, video

def attach_audio_to_video(original_video, audio_path, out_path):
    new_audio = AudioFileClip(audio_path)
    new_video = original_video.set_audio(new_audio)
    new_video.write_videofile(out_path, audio_codec='aac', verbose=False, logger=None)
    return out_path


def separate_speakers_core(audio_path):
    waveform, original_sr = torchaudio.load(audio_path)
    if original_sr != TARGET_SR:
        waveform = T.Resample(orig_freq=original_sr, new_freq=TARGET_SR)(waveform)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Ensure shape is (1, samples)
    audio_input = waveform.unsqueeze(0).to(device)  # Shape: (1, 1, samples)

    with torch.no_grad():
        ests_speech = sep_model(audio_input).squeeze(0)  # Shape: (num_speakers, samples)

    session_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join("output_sep", session_id)
    os.makedirs(output_dir, exist_ok=True)

    output_files = []
    for i in range(ests_speech.shape[0]):
        path = os.path.join(output_dir, f"speaker_{i+1}.wav")
        speaker_waveform = ests_speech[i].cpu()

        if speaker_waveform.dim() == 1:
            speaker_waveform = speaker_waveform.unsqueeze(0)  # (1, samples)

        # Ensure correct dtype and save in a widely compatible format
        speaker_waveform = speaker_waveform.to(torch.float32)
        torchaudio.save(path, speaker_waveform, TARGET_SR, format="wav", encoding="PCM_S", bits_per_sample=16)
        output_files.append(path)

    print(output_files)

    return output_files




@spaces.GPU()
def separate_dnr(audio_file):
    audio, sr = torchaudio.load(audio_file)
    audio = audio.to(device)

    with torch.no_grad():
        dialog, effect, music = dnr_model(audio[None])

    session_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join("output_dnr", session_id)
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "dialog": os.path.join(output_dir, "dialog.wav"),
        "effect": os.path.join(output_dir, "effect.wav"),
        "music": os.path.join(output_dir, "music.wav"),
    }

    torchaudio.save(paths["dialog"], dialog.cpu(), sr)
    torchaudio.save(paths["effect"], effect.cpu(), sr)
    torchaudio.save(paths["music"], music.cpu(), sr)

    return paths["dialog"], paths["effect"], paths["music"]

@spaces.GPU()
def separate_speakers(audio_path):
    output_files = separate_speakers_core(audio_path)
    updates = []
    for i in range(MAX_SPEAKERS):
        if i < len(output_files):
            updates.append(gr.update(value=output_files[i], visible=True, label=f"Speaker {i+1}"))
        else:
            updates.append(gr.update(value=None, visible=False))
    return updates

@spaces.GPU()
def separate_dnr_video(video_path):
    audio_path, video = extract_audio_from_video(video_path, 44100)
    dialog_path, effect_path, music_path = separate_dnr(audio_path)

    session_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join("output_dnr_video", session_id)
    os.makedirs(output_dir, exist_ok=True)

    dialog_video = attach_audio_to_video(video, dialog_path, os.path.join(output_dir, "dialog_video.mp4"))
    effect_video = attach_audio_to_video(video, effect_path, os.path.join(output_dir, "effect_video.mp4"))
    music_video = attach_audio_to_video(video, music_path, os.path.join(output_dir, "music_video.mp4"))

    return dialog_video, effect_video, music_video

def convert_to_ffmpeg_friendly(input_wav, output_wav):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_wav,
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_wav
    ], check=True)


@spaces.GPU()
def separate_speakers_video(video_path):
    audio_path, video = extract_audio_from_video(video_path, 16000)
    output_files = separate_speakers_core(audio_path)

    session_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join("output_sep_video", session_id)
    os.makedirs(output_dir, exist_ok=True)

    output_videos = []
    for i, audio_file in enumerate(output_files):
        speaker_video_path = os.path.join(output_dir, f"speaker_{i+1}_video.mp4")
        video_with_sep_audio = attach_audio_to_video(video, audio_file, speaker_video_path)
        output_videos.append(video_with_sep_audio)

    updates = []
    for i in range(MAX_SPEAKERS):
        if i < len(output_videos):
            updates.append(gr.update(value=output_videos[i], visible=True, label=f"Speaker {i+1}"))
        else:
            updates.append(gr.update(value=None, visible=False))
    return updates





# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# TIGER: 轻量级语音分离模型")
    gr.Markdown("更多前沿AI技术和应用，访问：https://deepfaces.cc")


    with gr.Tabs():
        with gr.Tab("人声伴奏分离"):
            dnr_input = gr.Audio(type="filepath", label="上传音频")
            dnr_btn = gr.Button("分离")
            gr.Examples(
                examples = ["./test/test_mixture_466.wav"],
                inputs = dnr_input
            )
            dnr_output = [gr.Audio(label=l) for l in ["人声", "音效", "伴奏"]]
            dnr_btn.click(separate_dnr, inputs=dnr_input, outputs=dnr_output)

        with gr.Tab("音频说话人分离"):
            sep_input = gr.Audio(type="filepath", label="上传语音音频")
            sep_btn = gr.Button("分离说话人")
            gr.Examples(
                examples = ["./test/mix.wav"],
                inputs = sep_input
            )
            sep_outputs = [gr.Audio(label=f"说话人 {i+1}", visible=(i==0)) for i in range(MAX_SPEAKERS)]
            sep_btn.click(separate_speakers, inputs=sep_input, outputs=sep_outputs)

        with gr.Tab("视频人声音效分离"):
            vdnr_input = gr.Video(label="上传视频")
            vdnr_btn = gr.Button("分离音频轨道")
            vdnr_output = [gr.Video(label=l) for l in ["人声视频", "音效视频", "伴奏视频"]]
            vdnr_btn.click(separate_dnr_video, inputs=vdnr_input, outputs=vdnr_output)

        with gr.Tab("视频语音分离"):
            vsep_input = gr.Video(label="上传视频")
            vsep_btn = gr.Button("分离说话人")
            vsep_outputs = [gr.Video(label=f"说话人 {i+1}", visible=(i==0)) for i in range(MAX_SPEAKERS)]
            vsep_btn.click(separate_speakers_video, inputs=vsep_input, outputs=vsep_outputs)

if __name__ == "__main__":
    demo.launch(ssr_mode=False,inbrowser=True)