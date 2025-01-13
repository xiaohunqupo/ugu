/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>

#include "ugu/cuda/image.h"
#include "ugu/image.h"
#include "ugu/image_io.h"
#include "ugu/image_proc.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  {
    ugu::Image3b img = ugu::imread("../data/color_transfer/reference_00.jpg");
    ugu::Image3b img_org = img.clone();
    ugu::Timer timer;
    int kernel = 51;
    ugu::BoxFilterCuda(img, kernel);
    ugu::imwrite("box_blur_cuda.jpg", img);
    img = img_org.clone();
    timer.Start();
    for (size_t i = 0; i < 1000; i++) {
      ugu::BoxFilterCuda(img, kernel);
    }
    timer.End();
    std::cout << "BoxFilterCuda: " << timer.elapsed_msec() << " / "
              << timer.elapsed_msec() / 1000 << std::endl;

    img = img_org.clone();
    ugu::BoxFilter(img.clone(), &img, kernel);
    ugu::imwrite("box_blur_cpu.jpg", img);
    img = img_org.clone();
    timer.Start();
    for (size_t i = 0; i < 1000; i++) {
      ugu::BoxFilter(img.clone(), &img, kernel);
    }
    timer.End();
    std::cout << "BoxFilter: " << timer.elapsed_msec() << " / "
              << timer.elapsed_msec() / 1000 << std::endl;
  }

  std::string data_dir = "../data/bunny/";
  std::string mask_path = data_dir + "00000_mask.png";

  ugu::Image1b mask = ugu::Imread<ugu::Image1b>(mask_path, -1);

  // 2D SDF
  ugu::Image1f sdf;
  ugu::MakeSignedDistanceField(mask, &sdf, true, false, -1.0f);
  ugu::Image3b vis_sdf;
  ugu::SignedDistance2Color(sdf, &vis_sdf, -1.0f, 1.0f);
  ugu::imwrite(data_dir + "00000_sdf.png", vis_sdf);

  ugu::circle(vis_sdf, {200, 200}, 20, {255, 0, 255}, 3);
  ugu::circle(vis_sdf, {100, 100}, 10, {0, 0, 0}, -1);
  ugu::line(vis_sdf, {0, 0}, {50, 50}, {255, 0, 0}, 1);
  ugu::line(vis_sdf, {10, 200}, {100, 200}, {0, 0, 255}, 5);
  ugu::imwrite(data_dir + "00000_sdf_circle.png", vis_sdf);

  // GIF load
  auto [images, delays] = ugu::LoadGif("../data/gif/dancing.gif");
  for (size_t i = 0; i < images.size(); i++) {
    ugu::imwrite("../data/gif/" + std::to_string(i) + "_" +
                     std::to_string(delays[i]) + "ms.png",
                 images[i]);
  }

  {
    ugu::ImageBase refer =
        ugu::imread("../data/color_transfer/reference_00.jpg");
    ugu::ImageBase target = ugu::imread("../data/color_transfer/target_00.jpg");
    ugu::Image3b res = ugu::ColorTransfer(refer, target);
    ugu::imwrite("../data/color_transfer/result_00.jpg", res);
  }

  {
    ugu::ImageBase source = ugu::imread("../data/poisson_blending/source.png");
    ugu::ImageBase target = ugu::imread("../data/poisson_blending/target.png");
    ugu::ImageBase mask_ = ugu::imread("../data/poisson_blending/mask.png", 0);
    ugu::Image3b res = ugu::PoissonBlend(mask_, source, target, -35, 35);
    ugu::imwrite("../data/poisson_blending/result.png", res);
  }

  return 0;
}
