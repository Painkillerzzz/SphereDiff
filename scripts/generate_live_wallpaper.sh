#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spherediff
export HF_HOME=~/.hf_home/
cd /home/xiangyuz22/workspace/SphereDiff

task_name="LiveWallpapers" ;
pipeline_name="SphericalHunyuanVideoPipeline" ;
default_config="
pipeline_cls=${pipeline_name}
pretrained_model_name_or_path=hunyuanvideo-community/HunyuanVideo
mixed_precision=bf16
enable_model_cpu_offload=False
call_kwargs.n_spherical_points=14400
" ;
subdir="" ;
txt_name="air_balloons"             ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="aurora"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="blossom"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="city"                     ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="clouds_over_grass"        ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="desert"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="desert_sandstorm"         ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="firefly"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="firework"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="forest"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="grand_canyon"             ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="lavender"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="moonlit_snowy_village"    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="ocean"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="rock_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="rock_mountain_with_star"  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="ruins"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="snow_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="storm"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
# txt_name="underwater"               ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;