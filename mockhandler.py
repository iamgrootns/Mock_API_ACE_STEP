import os
import time
import uuid
import base64
import traceback

import torch
import torchaudio
import soundfile as sf
import runpod

# IMPORTANT — correct import
from acestep.pipeline_ace_step import ACEStepPipeline


INIT_ERROR_FILE = "/tmp/init_error.log"
model_demo = None


# ------------------------------------------------------------
#               MODEL INITIALIZATION (GLOBAL)
# ------------------------------------------------------------
try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)

    print("Loading ACEStep model...")

    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/runpod-volume/checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_file = os.path.join(checkpoint_path, "ace_step_v1_3.5b.safetensors")

    # Download model if not already cached
    if not os.path.exists(checkpoint_file):
        from huggingface_hub import snapshot_download

        print(f"📥 Downloading ACEStep checkpoints to {checkpoint_path}...")
        snapshot_download(
            repo_id="ACE-Step/ACE-Step-v1-3.5B",
            local_dir=checkpoint_path,
            local_dir_use_symlinks=False,
        )
        print("✅ Checkpoints downloaded successfully")
    else:
        print(f"✅ Using cached checkpoints from {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=True,
        overlapped_decode=True,
    )

    if not model_demo.loaded:
        model_demo.load_checkpoint()

    print("✅ ACEStep model loaded successfully")

except Exception as e:
    error_trace = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize ACEStep model:\n{error_trace}")
    print(f"❌ Initialization error:\n{error_trace}")
    model_demo = None


# ------------------------------------------------------------
#              PATCH SAVE FUNCTION (RUNTIME)
# ------------------------------------------------------------
def patch_save_method(model):
    original_save = model.save_wav_file

    def patched_save(target_wav, idx, save_path=None, sample_rate=48000, format="wav"):
        base_path = "/tmp/outputs"
        os.makedirs(base_path, exist_ok=True)

        if save_path is None:
            output_path = f"{base_path}/output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
        else:
            if os.path.isdir(save_path):
                output_path = os.path.join(
                    save_path, f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
                )
            else:
                output_path = save_path

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Move tensor -> CPU
        target_wav = target_wav.float().cpu()
        audio_data = target_wav.numpy() if target_wav.ndim == 1 else target_wav.T.numpy()

        sf.write(output_path, audio_data, sample_rate)
        return output_path

    return original_save, patched_save


# ------------------------------------------------------------
#                         HANDLER
# ------------------------------------------------------------
def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            return {"error": f"Worker initialization failed:\n{f.read()}", "status": "failed"}

    job_input = event.get("input", {})
    endpoint = job_input.get("endpoint")

    if endpoint not in {"generate", "inpaint"}:
        return {"error": "Invalid or missing 'endpoint'", "status": "failed"}

    try:
        # =====================================================
        #                    GENERATE ENDPOINT
        # =====================================================
        if endpoint == "generate":
            prompt = job_input.get("prompt")
            audio_duration = job_input.get("audio_duration")

            if not prompt or audio_duration is None:
                return {"error": "Missing 'prompt' or 'audio_duration'", "status": "failed"}

            output_path = f"/tmp/output_{uuid.uuid4().hex}.wav"
            original_save, patched = patch_save_method(model_demo)
            model_demo.save_wav_file = patched

            start = time.time()
            try:
                model_demo(
                    format="wav",
                    audio_duration=audio_duration,
                    prompt=prompt,
                    lyrics="[inst]",
                    infer_step=60,
                    guidance_scale=14.0,
                    scheduler_type="euler",
                    cfg_type="apg",
                    omega_scale=10.0,
                    manual_seeds=[42, 99],
                    guidance_interval=0.5,
                    min_guidance_scale=3.0,
                    use_erg_tag=True,
                    use_erg_lyric=True,
                    use_erg_diffusion=True,
                    guidance_scale_text=3.0,
                    save_path=output_path,
                )
            finally:
                model_demo.save_wav_file = original_save

            elapsed = time.time() - start

            with open(output_path, "rb") as f:
                data = f.read()
            os.remove(output_path)

            return {
                "audio_base64": base64.b64encode(data).decode(),
                "sample_rate": 48000,
                "format": "wav",
                "duration": audio_duration,
                "generation_time": elapsed,
                "status": "completed",
            }

        # =====================================================
        #                      INPAINT ENDPOINT
        # =====================================================
        elif endpoint == "inpaint":
            prompt = job_input.get("prompt")
            start_s = job_input.get("start_time")
            end_s = job_input.get("end_time")
            audio_b64 = job_input.get("audio_base64")

            if None in (prompt, start_s, end_s, audio_b64):
                return {"error": "Missing one of required fields", "status": "failed"}

            # Decode input audio
            input_path = f"/tmp/input_{uuid.uuid4().hex}.wav"
            with open(input_path, "wb") as f:
                f.write(base64.b64decode(audio_b64))

            try:
                info = torchaudio.info(input_path)
                total_dur = info.num_frames / info.sample_rate
            except:
                total_dur = 30.0

            output_path = f"/tmp/inpainted_{uuid.uuid4().hex}.wav"
            original_save, patched = patch_save_method(model_demo)
            model_demo.save_wav_file = patched

            start = time.time()
            try:
                model_demo(
                    format="wav",
                    audio_duration=total_dur,
                    prompt=prompt,
                    lyrics="[inst]",
                    infer_step=60,
                    guidance_scale=10.0,
                    scheduler_type="euler",
                    cfg_type="apg",
                    omega_scale=10.0,
                    manual_seeds=[43],
                    guidance_interval=0.5,
                    min_guidance_scale=3.0,
                    use_erg_tag=True,
                    use_erg_lyric=True,
                    use_erg_diffusion=True,
                    guidance_scale_text=6.0,
                    save_path=output_path,
                    task="repaint",
                    repaint_start=int(start_s),
                    repaint_end=int(end_s),
                    retake_variance=0.75,
                    src_audio_path=input_path,
                )
            finally:
                model_demo.save_wav_file = original_save

            elapsed = time.time() - start

            with open(output_path, "rb") as f:
                out = f.read()

            os.remove(input_path)
            os.remove(output_path)

            return {
                "audio_base64": base64.b64encode(out).decode(),
                "sample_rate": 48000,
                "format": "wav",
                "inpainted_section": f"{start_s}s-{end_s}s",
                "processing_time": elapsed,
                "status": "completed",
            }

    except Exception as e:
        err = traceback.format_exc()
        print(f"❌ Error:\n{err}")
        return {"error": err, "status": "failed"}


runpod.serverless.start({"handler": handler})
