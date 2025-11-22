import os
import torch
import runpod
import base64
import traceback
import uuid
import soundfile as sf
import time
import torchaudio

INIT_ERROR_FILE = "/tmp/init_error.log"
model_demo = None

try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)

    print("Loading ACEStep model...")
    from ace_step.pipeline_ace_step import ACEStepPipeline  # <--- Correct import for PyPI/GitHub

    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/runpod-volume/checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_file = os.path.join(checkpoint_path, "ace_step_v1_3.5b.safetensors")
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
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize ACEStep model: {tb_str}")
    print(f"❌ Initialization error: {tb_str}")
    model_demo = None

def patch_save_method(model):
    original_save = model.save_wav_file
    def patched_save(target_wav, idx, save_path=None, sample_rate=48000, format="wav"):
        if save_path is None:
            base_path = "/tmp/outputs"
            os.makedirs(base_path, exist_ok=True)
            output_path_wav = f"{base_path}/output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
        else:
            if os.path.isdir(save_path):
                output_path_wav = os.path.join(save_path, f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}")
            else:
                output_path_wav = save_path
            output_dir = os.path.dirname(os.path.abspath(output_path_wav))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        target_wav = target_wav.float().cpu()
        if target_wav.dim() == 1:
            audio_data = target_wav.numpy()
        else:
            audio_data = target_wav.transpose(0, 1).numpy()
        sf.write(output_path_wav, audio_data, sample_rate)
        return output_path_wav
    return original_save, patched_save

def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            error_msg = f"Worker initialization failed: {f.read()}"
        return {"error": error_msg, "status": "failed"}

    job_input = event.get("input", {})
    endpoint = job_input.get("endpoint")
    if not endpoint or endpoint not in ["generate", "inpaint"]:
        return {"error": "Invalid or missing 'endpoint'. Must be 'generate' or 'inpaint'", "status": "failed"}

    try:
        if endpoint == "generate":
            prompt = job_input.get("prompt")
            audio_duration = job_input.get("audio_duration")
            if prompt is None or audio_duration is None:
                return {"error": "Missing 'prompt' or 'audio_duration' for generate", "status": "failed"}
            lyrics = "[inst]"
            output_path = f"/tmp/output_{uuid.uuid4().hex}.wav"
            original_save, patched_save = patch_save_method(model_demo)
            model_demo.save_wav_file = patched_save
            start_time = time.time()
            try:
                model_demo(
                    format="wav",
                    audio_duration=audio_duration,
                    prompt=prompt,
                    lyrics=lyrics,
                    infer_step=60,
                    guidance_scale=14.0,
                    scheduler_type="euler",
                    cfg_type="apg",
                    omega_scale=10.0,
                    manual_seeds=[42, 99],
                    guidance_interval=0.5,
                    guidance_interval_decay=0.0,
                    min_guidance_scale=3.0,
                    use_erg_tag=True,
                    use_erg_lyric=True,
                    use_erg_diffusion=True,
                    oss_steps=None,
                    guidance_scale_text=3.0,
                    guidance_scale_lyric=0.0,
                    save_path=output_path,
                )
            finally:
                model_demo.save_wav_file = original_save
            generation_time = time.time() - start_time
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            if os.path.exists(output_path):
                os.remove(output_path)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return {
                "audio_base64": audio_base64,
                "sample_rate": 48000,
                "format": "wav",
                "duration": audio_duration,
                "generation_time": generation_time,
                "status": "completed"
            }

        elif endpoint == "inpaint":
            prompt = job_input.get("prompt")
            start_time_val = job_input.get("start_time")
            end_time_val = job_input.get("end_time")
            audio_base64 = job_input.get("audio_base64")
            if None in [prompt, start_time_val, end_time_val, audio_base64]:
                return {"error": "Missing 'prompt', 'start_time', 'end_time' or 'audio_base64' for inpaint", "status": "failed"}
            input_audio_path = f"/tmp/input_{uuid.uuid4().hex}.wav"
            audio_data = base64.b64decode(audio_base64)
            with open(input_audio_path, "wb") as f:
                f.write(audio_data)
            try:
                audio_info = torchaudio.info(input_audio_path)
                total_duration = audio_info.num_frames / audio_info.sample_rate
            except Exception:
                total_duration = 30.0
            output_path = f"/tmp/inpainted_{uuid.uuid4().hex}.wav"
            original_save, patched_save = patch_save_method(model_demo)
            model_demo.save_wav_file = patched_save
            start_process = time.time()
            try:
                model_demo(
                    format="wav",
                    audio_duration=total_duration,
                    prompt=prompt,
                    lyrics="[inst]",
                    infer_step=60,
                    guidance_scale=10.0,
                    scheduler_type="euler",
                    cfg_type="apg",
                    omega_scale=10.0,
                    manual_seeds=[43],
                    guidance_interval=0.5,
                    guidance_interval_decay=0.0,
                    min_guidance_scale=3.0,
                    use_erg_tag=True,
                    use_erg_lyric=True,
                    use_erg_diffusion=True,
                    oss_steps=None,
                    guidance_scale_text=6.0,
                    guidance_scale_lyric=0.0,
                    save_path=output_path,
                    task="repaint",
                    repaint_start=int(start_time_val),
                    repaint_end=int(end_time_val),
                    retake_variance=0.75,
                    src_audio_path=input_audio_path,
                )
            finally:
                model_demo.save_wav_file = original_save
            processing_time = time.time() - start_process
            with open(output_path, "rb") as f:
                output_bytes = f.read()
            if os.path.exists(input_audio_path):
                os.remove(input_audio_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            audio_base64_output = base64.b64encode(output_bytes).decode('utf-8')
            return {
                "audio_base64": audio_base64_output,
                "sample_rate": 48000,
                "format": "wav",
                "inpainted_section": f"{start_time_val}s-{end_time_val}s",
                "processing_time": processing_time,
                "status": "completed"
            }
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"❌ Error: {error_msg}")
        return {"error": error_msg, "status": "failed"}

runpod.serverless.start({"handler": handler})
