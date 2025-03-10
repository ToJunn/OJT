from fastapi import FastAPI, File, UploadFile, Form, Response
import numpy as np
import io
from PIL import Image
import tritonclient.http as httpclient
from transformers import CLIPTokenizer

app = FastAPI()
triton_client = httpclient.InferenceServerClient("localhost:8000")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

@app.post("/generate/")
async def generate_image(file: UploadFile = File(...), prompt: str = Form(...)):
    """Nhận ảnh và prompt, xử lý với Triton Server"""
    
    # 1️⃣ Mã hóa văn bản với Text Encoder
    tokens = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="np")["input_ids"].astype(np.int64)
    text_input = httpclient.InferInput("input_ids", tokens.shape, "INT64")
    text_input.set_data_from_numpy(tokens)

    text_response = triton_client.infer("text_encoder", inputs=[text_input],
                                         outputs=[httpclient.InferRequestedOutput("text_embeddings")])
    text_embedding = text_response.as_numpy("text_embeddings").astype(np.float16)

    # 2️⃣ Mã hóa ảnh với VAE Encoder
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((512, 512))
    image_np = np.array(image).astype(np.float16) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))[None, :, :, :]

    vae_input = httpclient.InferInput("image", image_np.shape, "FP16")
    vae_input.set_data_from_numpy(image_np)

    vae_response = triton_client.infer("vae_encoder", inputs=[vae_input],
                                    outputs=[httpclient.InferRequestedOutput("latents")])
    latent = vae_response.as_numpy("latents").astype(np.float16)

    # 3️⃣ Gửi dữ liệu vào UNet để xử lý nhiễu
    extra_noise = np.random.randn(*latent.shape).astype(latent.dtype) * 0.1  # Thêm nhiễu nhẹ
    latents = np.concatenate([latent, extra_noise], axis=1)  # Tạo 8 kênh thay vì lặp lại

    timestep = np.array([1], dtype=np.int64)

    latent_input = httpclient.InferInput("latents", latents.shape, "FP16")
    latent_input.set_data_from_numpy(latents)

    timestep_input = httpclient.InferInput("time_steps", timestep.shape, "INT64")
    timestep_input.set_data_from_numpy(timestep)

    text_emb_input = httpclient.InferInput("text_embeddings", text_embedding.shape, "FP16")
    text_emb_input.set_data_from_numpy(text_embedding)

    unet_response = triton_client.infer("unet",
                                        inputs=[latent_input,
                                                timestep_input,
                                                text_emb_input],
                                                outputs=[httpclient.InferRequestedOutput("predicted_noise")])
    denoised_latents = unet_response.as_numpy("predicted_noise")

    # 4️⃣ Decode latent space thành ảnh
    vae_dec_input = httpclient.InferInput("latents", denoised_latents.shape, "FP16")
    vae_dec_input.set_data_from_numpy(denoised_latents)

    vae_dec_response = triton_client.infer("vae_decoder",
                                        inputs=[vae_dec_input],
                                        outputs=[httpclient.InferRequestedOutput("decoded_image")])
    generated_image = vae_dec_response.as_numpy("decoded_image")[0]
    generated_image = np.clip(np.transpose(generated_image, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)

    # 5️⃣ Trả về ảnh kết quả
    pil_img = Image.fromarray(generated_image)
    img_io = io.BytesIO()
    pil_img.save(img_io, format="PNG")
    img_io.seek(0)

    return Response(img_io.getvalue(), media_type="image/png")
