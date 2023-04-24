import os
import glob
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from tqdm import tqdm
from .video_utils import get_video_fps, get_video_duration, split_video_to_images, combine_images_to_video
from .image_utils import load_image, shrink_image


class VidDiffusionPipeline:
    def __init__(self, config: dict):
        self.config = self.get_default_config()
        self.set_config(config)
        self.set_unset_config_values()

        self.preprocess_image = self.config['preprocess_image']
        self.tqdm = tqdm if self.config['show_tqdm'] else lambda x: x


    def set_config(self, config: dict):
        '''
        Set config dict and update pipeline and callable if needed
        :param config: config dict
        '''
        for key, value in config.items():
            self.config[key] = value
        
        _is_pipe_updated = False
        if any(k in config for k in ('pretrained_model_name_or_path', 'custom_pipeline', 'torch_dtype', 'hf_auth_token')):
            self.pipe = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=self.config['pretrained_model_name_or_path'],
                custom_pipeline=self.config['custom_pipeline'],
                torch_dtype=self.config['torch_dtype'],
                use_auth_token=self.config['hf_auth_token'],
                force_download=True,
                resume_download=False,
            ).to(self.config['device'])
            self.pipe.safety_checker = self.config['safety_checker']

            _is_pipe_updated = True
        
        if any(k in config for k in ('scheduler', 'beta_start', 'beta_end', 'num_train_timesteps')) or _is_pipe_updated:
            self.pipe.scheduler = self.config['scheduler'](beta_start=self.config['beta_start'], beta_end=self.config['beta_end'], num_train_timesteps=self.config['num_train_timesteps'])

        if 'preprocess_image' in config:
            self.preprocess_image = self.config['preprocess_image']

        if 'show_tqdm' in config:
            self.tqdm = tqdm if self.config['show_tqdm'] else lambda x: x


    def get_default_config(self):
        '''
        Get default config dict
        :return: default config dict
        '''
        return {
            'init_image_dir': '/tmp/viddiffusion/init',
            'output_image_dir': '/tmp/viddiffusion/output',

            'fps': None,
            'duration': None,
            'start_time': None,
            'end_time': None,

            'hf_auth_token': None,
            'custom_pipeline': 'lpw_stable_diffusion',
            'torch_dtype': torch.float,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'safety_checker': None,
            'image_size': 512 * 512,

            'prompt': 'Cyberpunk style, intricate details, high-quality',
            'negative_prompt': 'Low-quality, blurry, pixelated, low-resolution',
            
            'scheduler': DPMSolverMultistepScheduler,
            'beta_start': 0.00085 * 0.75,
            'beta_end': 0.012 * 0.75,
            'num_train_timesteps': int(1000 * 0.75),

            'seed': 2023,
            'strength': 0.65,
            'num_inference_steps': 30,
            'guidance_scale': 7.5,

            'is_blend': True,
            'blend_alph': 0.23,
            'blend_decay_rate': 0.997,

            'preprocess_image': lambda image: image,

            'show_tqdm': True,
        }
    

    def set_unset_config_values(self):
        '''
        Set unset config values
        '''
        # Set default config values
        for key, value in self.get_default_config().items():
            if key not in self.config:
                self.config[key] = value

        # Set video config values
        if self.config['fps'] is None:
            self.config['fps'] = get_video_fps(self.config['input_video_path'])
        if self.config['duration'] is None:
            self.config['duration'] = get_video_duration(self.config['input_video_path'])
        if self.config['start_time'] is None:
            self.config['start_time'] = 0
        if self.config['end_time'] is None:
            self.config['end_time'] = self.config['duration']


    def show_config(self):
        '''
        Return config dict string except hf_auth_token
        '''
        config = self.config.copy()
        config['hf_auth_token'] = '***'
        return str(config)


    def split_video_to_images(self):
        '''
        Convert video to images
        If self.config['init_image_dir'] does not exist, create it.
        Use self.config['input_video_path'] and self.config['init_image_dir']
        '''
        if not os.path.exists(self.config['input_video_path']):
            raise FileNotFoundError('Input video path does not exist.')
        
        if os.path.exists(self.config['init_image_dir']):
            print(f"Removing {self.config['init_image_dir']}")
            os.system(f"rm -rf {self.config['init_image_dir']}")
        os.makedirs(self.config['init_image_dir'])
        
        if self.config['show_tqdm']:
            print(f"Splitting video to images: {self.config['input_video_path']}")
        
        split_video_to_images(
            input_video_path=self.config['input_video_path'],
            output_dir_path=self.config['init_image_dir'],
            fps=self.config['fps'],
            start_time=self.config['start_time'],
            end_time=self.config['end_time']
        )

  
    def combine_images_to_video(self):
        '''
        Convert images to video
        Use self.config['output_image_dir'] and self.config['output_video_path'] 
        ''' 
        if not os.path.exists(self.config['output_image_dir']):
            raise FileNotFoundError('Output image dir does not exist.')
        
        if self.config['show_tqdm']:
            print(f"Combining images to video: {self.config['output_video_path']}")
        
        combine_images_to_video(
            input_dir_path=self.config['output_image_dir'],
            output_video_path=self.config['output_video_path'],
            fps=self.config['fps']
        )
  

    def images2images(self):
        '''
        Convert images to images
        Use self.config['init_image_dir'] and self.config['output_image_dir']
        '''
        if not os.path.exists(self.config['init_image_dir']):
            raise FileNotFoundError('Input image dir does not exist.')
        
        if os.path.exists(self.config['output_image_dir']):
            print(f"Removing {self.config['output_image_dir']}")
            os.system(f"rm -rf {self.config['output_image_dir']}")
        os.makedirs(self.config['output_image_dir'])
        
        if self.config['show_tqdm']:
            print(f"Converting images to images: {self.config['init_image_dir']} -> {self.config['output_image_dir']}")
        
        image_paths = sorted(glob.glob(os.path.join(self.config['init_image_dir'], '*.png')))
        for i_image in self.tqdm(range(len(image_paths))):
            loaded_image = load_image(image_paths[i_image])
            loaded_image = shrink_image(loaded_image, self.config['image_size'])

            if self.config['is_blend'] and i_image > 0:
                image = Image.blend(loaded_image, stack_image, self.config['blend_alph'])
                self.config['blend_alph'] *= self.config['blend_decay_rate']
            else:
                image = loaded_image.copy()
            
            image = self.preprocess_image(image)

            generator = torch.Generator(device="cuda").manual_seed(self.config['seed']) 
            image = self.pipe.img2img(image=image, prompt=self.config['prompt'], negative_prompt=self.config['negative_prompt'], strength=self.config['strength'], num_inference_steps=self.config['num_inference_steps'], guidance_scale=self.config['guidance_scale'], generator=generator).images[0]
            image.save(os.path.join(self.config['output_image_dir'], f'{i_image:05d}.png'))

            if self.config['is_blend']:
                stack_image = image.copy()


    def vid2vid(self):
        '''
        Convert video to video
        '''
        self.show_config()
        self.split_video_to_images()
        self.images2images()
        self.combine_images_to_video()
    

    def images2vid(self):
        '''
        Convert images to video
        This function maight be used for experiments by changing config values.
        '''
        self.show_config()
        self.images2images()
        self.combine_images_to_video()