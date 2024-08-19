import torch
import pyaudio
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

#ty
# Setup device and model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Audio capture setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024)

print("Listening...")

try:
    while True:
        # Read audio data from the microphone
        audio_data = stream.read(1024, exception_on_overflow=False)
        # Convert the audio data to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        # Convert numpy array to a format suitable for the model
        input_features = processor.feature_extractor(audio_np, sampling_rate=16000, return_tensors="pt")
        input_features = input_features.to(device, dtype=torch.float16)

        # Prepare the decoder input with English language token
        language_token = processor.tokenizer("<|en|>", add_special_tokens=False, return_tensors="pt").input_ids
        language_token = language_token.to(device, dtype=torch.long)

        # Generate the transcription with the forced language token
        with torch.no_grad():
            outputs = model.generate(input_features["input_features"], decoder_input_ids=language_token)
            transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Print the transcription
        print("You said:", transcription)

except KeyboardInterrupt:
    print("Stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
