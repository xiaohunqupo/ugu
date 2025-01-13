#pragma once

#include "ugu/image.h"

namespace ugu {

void BoxFilterCuda(Image3b& img, int kernel);

void BilateralFilterCuda(const Image3b& src, Image3b& dst, int kernel);

}  // namespace ugu
