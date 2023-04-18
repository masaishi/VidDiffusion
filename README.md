# 🎥 VidDiffusion

VidDiffusion is a Python library that provides vid2vid pipeline by using Hugging Face's `diffusers`.


# Sharing Your Work
Although it is not required, we would appreciate if you could share your generated videos and the configurations you used on SNS, etc. Please include the GitHub URL of this library in your post:
[VidDiffusion](https://github.com/masaishi/viddiffusion)

Also, please post config values you used to generate the video by `instance`.show_config(). I hope we can all find the best value and produce great videos!


# Installation

To install the library, run the following commands:
```bash
pip install --upgrade transformers diffusers
pip install viddiffusion
```

# Usage

First, import the library:
```python	
from viddiffusion import VidDiffusionPipeline
```


Next, create a configuration dictionary and initialize the pipeline:
```python
config = {
    'pretrained_model_name_or_path': 'your_pretrained_model_name_or_path',
		'hf_auth_token': 'your_hf_auth_token', #Optional

    'input_video_path': 'your_input_video_path',
    'output_video_path': 'your_output_video_path'
}
vid_pipe = VidDiffusionPipeline(config)
```

Optionally, you can change the path of temp save directory and other parameters:
```python
vid_pipe.set_config({
    'init_image_dir': './vid_temp/init',
    'output_image_dir': './vid_temp/output',

    'fps': 20,
    'end_time': 5.0,

		'prompt': 'Fantasy, dynamic, cinamatic, cinematic light, hyperrealism',
		'negative_prompt': 'deformed, out of focus, blurry, low resolution, low quality',
})
vid_pipe.show_config()
```


Finally, apply the diffusion model to the video:
```python
vid_pipe.vid2vid()
```


# Configuration:

## Required parameters:
- `pretrained_model_name_or_path`: The name or path of the pretrained diffusion model to use. The model must be a Hugging Face model. The model can be a local path or a model name from the [Hugging Face model hub](https://huggingface.co/models). E.g. `pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2'`
- `input_video_path`: The path to the input video file. E.g. `input_video_path = 'input_video.mp4'`
- `output_video_path`: The path to the output video file. E.g. `output_video_path = 'output_video.mp4'`

## Optional parameters:
* huggingface parameters:
	- `hf_auth_token`: Hugging Face auth token. Default: None.

* Folder parameters:
	- `init_image_dir`: Path to images separated from the video directory. Default: None.
	- `output_image_dir`: Path to regenerated images directory. Default: /tmp/viddiffusion.

* Video parameters:
	- `fps`: FPS of output video. Default: fps of input video.
	- `duration`: Duration of output video. Default: duration of input video.
	- `start_time`: Start time of output video. Default: 0.
	- `end_time`: End time of output video. Default: duration of input video.

* Diffusion parameters:
	- `custom_pipeline`: Custom pipeline. Default: lpw_stable_diffusion.
	- `torch_dtype`: Torch dtype. Default: torch.float.
	- `safety_checker`: Safety checker. Callable. Default: None.
	- `image_size`: Image size. Default: 512 * 512 = 262144.

* Prompt parameters:
	- `prompt`: Prompt. Default: 'Cyberpunk style, intricate details, high-quality'
	- `negative_prompt`: Negative prompt. Default: 'Low-quality, blurry, pixelated, low-resolution'

* Scheduler parameters:
	- `scheduler`: Scheduler. Default: DPMSolverMultistepScheduler.
	- `beta_start`: Beta start. Default: 0.00085 * 0.72.
	- `beta_end`: Beta end. Default: 0.012 * 0.72.
	- `num_train_timesteps`: Number of train timesteps. Default: int(1000 * 0.72).
	- `blend_alph`: Blend alph. Default: 0.27.
	- `blend_decay_rate`: Blend decay rate. Default: 0.9995.
	- `strength`: Strength. Default: 0.58.
	- `num_inference_steps`: Number of inference steps. Default: 30.
	- `guidance_scale`: Guidance scale. Default: 7.5.
	- `seed`: Seed for generator. Default: 2023.


* Other parameters:
	- `preprocess_image`: Preprocess image. Default: lambda image: image.


# How to Improve Video Quality

This library's feature is using generated image one frame before and after the current frame to improve video quality. This feature is enabled by default.
I guess this feature is good for videos that have a lot of motion.

## Parameters to change to improve video quality
- config['scheduler'] : Try various scheduler such as DPMSolverMultistepScheduler, DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, PNDMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler.
- config['beta_start'] : 0.00085 * 0.65~0.75 is good, but I don't know the best value.
- config['beta_end'] : 0.012 * 0.65~0.75 is good, but I don't know the best value.
- config['num_train_timesteps'] : int(1000 * 0.65~0.75) is good, but I don't know the best value.

- config['strength'] : 0.5 ~ 0.9 is good, but I don't know the best value.
- config['guidance_scale'] 7.5 is good, but I don't know the best value.

- config['blend_alph'] : 0.22 ~ 0.29 is good, but I don't know the best value.
- config['blend_decay_rate'] : 0.999 ~ 1.0 is good, but I don't know the best value.

- config['prompt'] : Try various prompt such as 'Cyberpunk style, intricate details, high-quality', 'Fantasy, dynamic, cinamatic, cinematic light, hyperrealism', 'Cyberpunk, cyberpunk style, cyberpunk aesthetic, cyber punk city, cyberpunk cityscape, cyberpunk cityscape aesthetic, cyberpunk cityscape style, cyberpunk cityscape light, cyberpunk cityscape light aesthetic, cyberpunk cityscape light style, cyberpunk cityscape light
- config['negative_prompt'] : Try various negative_prompt such as 'Low-quality, blurry, pixelated, low-resolution', 'deformed, out of focus, blurry, low resolution, low quality', 'Low-quality, blurry, pixelated, low-resolution, deformed, out of focus, blurry, low resolution, low quality'

- config['preprocess_image'] : Try various preprocess_image function such as torchvision.


# License

VidDiffusion is licensed under the Apache License 2.0.


# Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue on the GitHub repository.


# Contact

If you have any questions, please feel free to contact me. Twitter DM is best option for me

- Twitter: https://twitter.com/masaishi2001
- LinkedIn: https://www.linkedin.com/in/masamune-ishihara-31b27b232/
- Mastodon: @masaishi@mastodon.social
- Email: masaishi_masa at yahoo.co.jp (Change at to @ and remove spaces)


# Citation

```bibtex
@misc{masaishi2023viddiffusion,
			title={VidDiffusion: Vid2Vid Diffuser Pipeline for Video Filter}, 
			author={Masamune Ishihara},
			year={2023},
			publisher = {GitHub},
			journal = {GitHub repository},
			howpublished = {\url{https://github.com/masaishi/VidDiffusion}}
}
```