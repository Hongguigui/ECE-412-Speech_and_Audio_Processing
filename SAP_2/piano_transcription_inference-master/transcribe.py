from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import librosa
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

audio_path = 'resources/clip_1.wav'

# Load audio
audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
print(audio)

# Transcriptor
transcriptor = PianoTranscription(device='cuda', checkpoint_path='downsize_combined.pth')  # device: 'cuda' | 'cpu'

# Transcribe and write out to MIDI file
transcribed_dict = transcriptor.transcribe(audio, 'clip_1_small.mid')
