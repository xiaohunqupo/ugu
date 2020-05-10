# **UGU**: **U**nclearness **G**eometry **U**tility
**UGU** is a small geometry library which implements camera, image, mesh, etc.

# Features
## Texture Mapping

<img src="https://raw.githubusercontent.com/wiki/unclearness/ugu/images/texture_mapping.png" width="640">

- Comparizon table

||Vertex Color|Projective UV| Tile UV| Triangle UV|Predefined UV|
|---|---|---|---|---|---|
|Runtime Speed|✅|❌||||
|Runtime Memory||❌||||
|UV Space Efficiency|||❌||❓|
|Quality at Rendering|❌|✅|✅|❌|❓|


## CPU Rendering

|color|depth|
|---|---|
|![](https://raw.githubusercontent.com/wiki/unclearness/ugu/images/renderer/front_color.png)|![](https://raw.githubusercontent.com/wiki/unclearness/ugu/images/renderer/front_vis_depth.png)|


|normal|mask|face id|
|---|---|---|
|![](https://raw.githubusercontent.com/wiki/unclearness/ugu/images/renderer/front_vis_normal.png)|![](https://raw.githubusercontent.com/wiki/unclearness/ugu/images/renderer/front_mask.png)|![](https://raw.githubusercontent.com/wiki/unclearness/ugu/images/renderer/front_vis_face_id.png)|

- Original code: https://github.com/unclearness/currender

## Shape-from-Silhouette (SfS)

<img src="https://raw.githubusercontent.com/wiki/unclearness/vacancy/images/how_it_works.gif" width="640">

- Original code: https://github.com/unclearness/vacancy


# Dependencies
## Mandatory
- Eigen
    https://github.com/eigenteam/eigen-git-mirror
    - Math
## Optional
- OpenCV
    - cv::Mat_ as Image class. Image I/O
- stb
    https://github.com/nothings/stb
    - Image I/O
- LodePNG
    https://github.com/lvandeve/lodepng
    - .png I/O particularly for 16bit writing that is not supported by stb
- tinyobjloader
    https://github.com/syoyo/tinyobjloader
    - Load .obj
- tinycolormap
    https://github.com/yuki-koyama/tinycolormap
    - Colorization of depth, face id, etc.
- OpenMP
    (if supported by your compiler)
    - Multi-thread accelaration


# Build
- `git submodule update --init --recursive`
  - To pull dependencies registered as git submodule. 
- Use CMake with `CMakeLists.txt`.
  -  `reconfigure.bat` and `rebuild.bat` are command line CMake utilities for Windows 10 and Visual Studio 2017.


# Data
 Borrowed .obj from [Zhou, Kun, et al. "TextureMontage." ACM Transactions on Graphics (TOG) 24.3 (2005): 1148-1155.](http://www.kunzhou.net/tex-models.htm) for testing purposes.
