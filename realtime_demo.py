import time
import torch
import numpy as np
import cv2 as cv
import os
import re
import glob
import argparse
from cuda import cudart
from quantization import QParam
from polygraphy import cuda
from tensorrt_acceleration import Engine


def slerp(a, b, t, eps=1e-5):
    """球面线性插值, 沿最后一维计算."""
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    cos_theta = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    use_lerp = sin_theta < eps
    factor_a = torch.sin((1.0 - t) * theta) / sin_theta
    factor_b = torch.sin(t * theta) / sin_theta
    result = factor_a * a + factor_b * b
    lerp_result = (1.0 - t) * a + t * b
    result = torch.where(use_lerp.expand_as(result), lerp_result, result)
    return result


## Real-time generation engine, from prompt to video.
class Generator():
    def __init__(self, batch=1, device='cuda:0'):
        self.cuda_stream = cuda.Stream()
        self.use_cuda_graph = False
        self.batch = batch # the number of frames generated at once, needs to match the engine file.
        self.device = device
        self.noise = self.seeded_randn(shape=(1,4,64,64), seed=88) # the random seed needs to be consistent with the inversion.
        self.sigma = torch.Tensor([0.05]).float().cuda()
        self.prev_frame = None
        self.timesteps = torch.Tensor([999]).long().to(self.device)
        self.timesteps = self.timesteps.repeat(self.batch)
        self.c_in = torch.Tensor([[[[0.0683]]]]).float().to(self.device)
        self.c_out = torch.Tensor([[[[-14.6146]]]]).float().to(self.device)
        self.denoise_engine= None
        self.decoder_engine = None
        self.denoise_engine_load()
        self.decoder_engine_load()

    def build_engine(self, input_profile, onnx_path, engine_path):
        _, free_mem, _ = cudart.cudaMemGetInfo()
        GiB = 2 ** 30
        if free_mem > 6 * GiB:
            activation_carveout = 4 * GiB
            max_workspace_size = free_mem - activation_carveout
        else:
            max_workspace_size = 0
        engine = Engine(engine_path)
        engine.build(
            onnx_path,
            fp16=True,
            input_profile=input_profile,
            enable_refit=False,
            enable_all_tactics=False,
            workspace_size=max_workspace_size,
        )
        return engine

    def denoise_engine_build(self):
        print('Start building the denoise engine for this machine.')
        engine_path = 'engine/denoise_batch_{}.engine'.format(self.batch)
        onnx_path = 'engine/denoise_batch_{}.onnx'.format(self.batch)
        input_profile = {'x': [(self.batch, 4, 64, 64), (self.batch, 4, 64, 64), (self.batch, 4, 64, 64)], 'timesteps': [(self.batch,), (self.batch,), (self.batch,)], 'context': [(self.batch,77,1024), (self.batch,77,1024), (self.batch,77,1024)]}
        self.build_engine(input_profile, onnx_path, engine_path)
        print('Build completed successfully.')

    def denoise_engine_load(self):
        engine_path = 'engine/denoise_batch_{}.engine'.format(self.batch)
        if not os.path.exists(engine_path):
            print('The denoise engine is not found.')
            self.denoise_engine_build()
        self.denoise_engine = Engine(engine_path)
        self.denoise_engine.load()
        self.denoise_engine.activate()
        self.denoise_engine.allocate_buffers(
            shape_dict={
                "x": [self.batch, 4, 64, 64],
                "timesteps": [self.batch],
                "context": [self.batch, 77, 1024],
            },
            device=self.device,
        )

    def decoder_engine_build(self):
        print('Start building the decoder engine for this machine.')
        engine_path = 'engine/decoder_batch_{}.engine'.format(self.batch)
        onnx_path = 'engine/decoder_batch_{}.onnx'.format(self.batch)
        input_profile = {'latent': [(self.batch, 4, 64, 64), (self.batch, 4, 64, 64), (self.batch, 4, 64, 64)]}
        self.build_engine(input_profile, onnx_path, engine_path)
        print('Build completed successfully.')

    def decoder_engine_load(self):
        engine_path = 'engine/decoder_batch_{}.engine'.format(self.batch)
        if not os.path.exists(engine_path):
            print('The decoder engine is not found.')
            self.decoder_engine_build()
        self.decoder_engine = Engine(engine_path)
        self.decoder_engine.load()
        self.decoder_engine.activate()
        self.decoder_engine.allocate_buffers(
            shape_dict={
                "latent": [self.batch, 4, 64, 64],
            },
            device=self.device,
        )

    def run_denoise_engine(self, x, timesteps, context):
        output = self.denoise_engine.infer(
            {
                "x": x,
                "timesteps": timesteps,
                "context": context,
            },
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        return output['out']

    def run_decoder_engine(self, latent):
        output = self.decoder_engine.infer(
            {
                "latent": latent,
            },
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        return output['images']

    def normalization(self, images_tensor, to_numpy=False):
        images_tensor = torch.clamp((images_tensor + 1.0) / 2.0, min=0.0, max=1.0)
        images_tensor = images_tensor.permute(0, 2, 3, 1)
        images_tensor = (images_tensor * 255).byte()
        if to_numpy:
            return images_tensor.detach().cpu().numpy()
        else:
            return images_tensor.detach()

    def seeded_randn(self, shape, seed):
        randn = np.random.RandomState(seed).randn(*shape)
        randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
        return randn

    def add_noise(self, prev_frame):
        noised_prev_frame = (prev_frame * self.sigma + self.noise * (1 - self.sigma)).detach()
        noised_prev_frame = noised_prev_frame.repeat(batch,1,1,1)
        return noised_prev_frame

    def generate(self, cond):
        z = self.add_noise(self.prev_frame)
        latent_noise = self.run_denoise_engine(z, self.timesteps, cond)
        latent = latent_noise * self.c_out + z / self.c_in
        images = self.run_decoder_engine(latent)
        self.prev_frame = latent[-1:,...]
        return images

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-prompt_dir', type=str, default="data/sky/results/rank8_interval10")
    parser.add_argument('-batch', type=int, default="10")
    parser.add_argument('-visualize', type=bool, default=True)
    parser.add_argument('--slerp', action='store_true', default=True,
                        help='Use spherical linear interpolation (default: True)')
    parser.add_argument('--no-slerp', action='store_false', dest='slerp',
                        help='Use linear interpolation instead of slerp')
    args = parser.parse_args()

    prompt_dir = args.prompt_dir
    batch = args.batch
    Generator_RT = Generator(batch=batch)
    result_dir = prompt_dir
    result_frames = None
    speed_warm_up = False
    generation_speed = []
    prompts = sorted(glob.glob(os.path.join(prompt_dir, 'frame_*.prompt')))
    for prompt_pair in zip(prompts[::], prompts[1::]):
        prompt_curr = prompt_pair[0]
        id_curr = int(re.search(r'frame_(\d{5})\.prompt', prompt_curr).group(1))
        prompt_next = prompt_pair[1]
        id_next = int(re.search(r'frame_(\d{5})\.prompt', prompt_next).group(1))
        # interpolation interval
        interval = id_next - id_curr

        prompt_curr = torch.load(prompt_curr, weights_only=True)
        prompt_next = torch.load(prompt_next, weights_only=True)
        # low-rank factors
        U_curr, V_curr, U_next, V_next = prompt_curr['U'], prompt_curr['V'], prompt_next['U'], prompt_next['V']

        torch.cuda.synchronize()
        t_begin = time.time()
        # prompt dequantization
        Quant_Param_U_curr = QParam(num_bits=8)
        Quant_Param_U_curr.scale = prompt_curr['U_scale']
        Quant_Param_U_curr.zero_point = prompt_curr['U_zero_point']
        U_curr = Quant_Param_U_curr.dequantize_tensor(U_curr)
        Quant_Param_V_curr = QParam(num_bits=8)
        Quant_Param_V_curr.scale = prompt_curr['V_scale']
        Quant_Param_V_curr.zero_point = prompt_curr['V_zero_point']
        V_curr = Quant_Param_V_curr.dequantize_tensor(V_curr)

        Quant_Param_U_next = QParam(num_bits=8)
        Quant_Param_U_next.scale = prompt_next['U_scale']
        Quant_Param_U_next.zero_point = prompt_next['U_zero_point']
        U_next = Quant_Param_U_next.dequantize_tensor(U_next)
        Quant_Param_V_next = QParam(num_bits=8)
        Quant_Param_V_next.scale = prompt_next['V_scale']
        Quant_Param_V_next.zero_point = prompt_next['V_zero_point']
        V_next = Quant_Param_V_next.dequantize_tensor(V_next)

        # 加载逐秩活跃度权重 (向后兼容: 旧 prompt 无 weights 字段)
        weights = prompt_curr.get('weights', None)
        if weights is not None:
            weights = weights.cuda()

        rank = U_curr.shape[1]
        prompt = []
        # 逐秩差异化插值
        for step in range(1, interval + 1):
            t = step / interval  # 归一化进度 [0, 1]
            if weights is not None:
                t_i = t ** (1.0 / weights)  # [rank], 逐秩差异化
            else:
                t_i = torch.tensor(t, device=U_curr.device)  # 标量张量, 统一速率

            if args.slerp:
                # Slerp: U 转置后沿最后一维, V 直接沿最后一维
                u = slerp(U_curr.T, U_next.T, t_i.view(-1, 1)).T
                v = slerp(V_curr, V_next, t_i.view(-1, 1))
            else:
                # 线性插值
                u = (1 - t_i.view(1, -1)) * U_curr + t_i.view(1, -1) * U_next
                v = (1 - t_i.view(-1, 1)) * V_curr + t_i.view(-1, 1) * V_next
            # prompt composition
            c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)
            prompt.append(c)
        prompt = torch.concatenate(prompt, dim=0)
        # generating frames from prompts
        if Generator_RT.prev_frame is None:
            # initialize for the first frame
            Generator_RT.prev_frame = torch.load(os.path.join(prompt_dir, 'init.pth'), weights_only=True)
            c0 = (U_curr @ V_curr / np.sqrt(rank)).unsqueeze(dim=0)
            images = Generator_RT.generate(c0)
            images = Generator_RT.normalization(images, to_numpy=True)
            result_frames = images[0:1,...]
        images = Generator_RT.generate(prompt)
        images = Generator_RT.normalization(images, to_numpy=True)
        torch.cuda.synchronize()
        t_end = time.time()
        if speed_warm_up:
            # generation speed in FPS
            fps = 1 / (t_end - t_begin) * batch
            generation_speed.append(fps)
            print('Generation Speed: {} FPS'.format(fps))
        else:
            # the first batch is used for warming up
            speed_warm_up = True
        result_frames = np.append(result_frames, images, axis=0)

    average_speed = int(np.mean(generation_speed))
    for i in range(result_frames.shape[0]):
        image = result_frames[i,...]
        image = image[:, :, ::-1]
        image = np.ascontiguousarray(image)
        # Save the generated frames
        cv.imwrite(os.path.join(result_dir, '{:05d}.png'.format(i)), image)
        if args.visualize:
            # Visualize the generated frames
            cv.putText(image, 'Generation Speed {} FPS'.format(average_speed), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.imshow('Real-time Generation', image)
            cv.waitKey(10)



