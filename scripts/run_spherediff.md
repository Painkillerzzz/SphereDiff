# Static Wallpaper Generation

## FLUX

```bash
task_name="StaticWallpapers" ;
pipeline_name="SphericalFluxPipeline" ;
default_config="
pipeline_cls=${pipeline_name}
pretrained_model_name_or_path=black-forest-labs/FLUX.1-dev
variant=None
mixed_precision=bf16
enable_model_cpu_offload=False
call_kwargs.n_spherical_points=26500
" ;
subdir=""
txt_name="air_balloons"             ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="aurora"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="blossom"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="city"                     ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="clouds_over_grass"        ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="desert"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="desert_sandstorm"         ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="firefly"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="firework"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="forest"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="grand_canyon"             ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="lavender"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="moonlit_snowy_village"    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="ocean"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="rock_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="rock_mountain_with_star"  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="ruins"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="snow_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="storm"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="underwater"               ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
```

## SANA

```bash
task_name="StaticWallpapers" ;
pipeline_name="SphericalSanaPipeline" ;
default_config="
pipeline_cls=${pipeline_name}
pretrained_model_name_or_path=Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers
variant=bf16
mixed_precision=bf16
enable_model_cpu_offload=False
call_kwargs.n_spherical_points=2600
" ;
subdir=""
txt_name="air_balloons"             ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="aurora"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="blossom"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="city"                     ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="clouds_over_grass"        ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="desert"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="desert_sandstorm"         ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="firefly"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="firework"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="forest"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="grand_canyon"             ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="lavender"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="moonlit_snowy_village"    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="ocean"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="rock_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="rock_mountain_with_star"  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="ruins"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="snow_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="storm"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="underwater"               ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_static_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
```

# Live Wallpaper Generation

## HunyuanVideo

- Using n_spherical_points = 26,500 yields the highest quality, but the computation time is prohibitively long.
- Using n_spherical_points = 14,400 takes about 12 hours.
- Using n_spherical_points = 7,000 provides reasonably good quality and completes in around 3 hours.

```bash
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
txt_name="aurora"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="blossom"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="city"                     ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="clouds_over_grass"        ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="desert"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="desert_sandstorm"         ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="firefly"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="firework"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="forest"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="grand_canyon"             ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="lavender"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="moonlit_snowy_village"    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="ocean"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="rock_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="rock_mountain_with_star"  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="ruins"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="snow_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="storm"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="underwater"               ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
```

## LTX-Video

```bash
task_name="LiveWallpapers" ;
pipeline_name="SphericalLTXPipeline" ;
default_config="
pipeline_cls=${pipeline_name}
pretrained_model_name_or_path=a-r-r-o-w/LTX-Video-0.9.1-diffusers
mixed_precision=bf16
enable_model_cpu_offload=False
call_kwargs.n_spherical_points=2600
" ;
subdir="n_spherical_points=2600" ;
txt_name="air_balloons"             ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="aurora"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="blossom"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="city"                     ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="clouds_over_grass"        ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="desert"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="desert_sandstorm"         ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="firefly"                  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="firework"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="forest"                   ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="grand_canyon"             ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="lavender"                 ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="moonlit_snowy_village"    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="ocean"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="rock_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="rock_mountain_with_star"  ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="ruins"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="snow_mountain"            ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="storm"                    ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
txt_name="underwater"               ; prompt_txt_path="data/prompts/${txt_name}.txt" ; save_path="./outputs/${task_name}/${pipeline_name}/${subdir}/${txt_name}" ; python generate_live_wallpaper.py --config_add ${default_config} call_kwargs.prompt_txt_path=${prompt_txt_path} save_path=${save_path} ;
```
