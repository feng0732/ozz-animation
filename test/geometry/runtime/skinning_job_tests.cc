//----------------------------------------------------------------------------//
//                                                                            //
// ozz-animation is hosted at http://github.com/guillaumeblanc/ozz-animation  //
// and distributed under the MIT License (MIT).                               //
//                                                                            //
// Copyright (c) Guillaume Blanc                                              //
//                                                                            //
// Permission is hereby granted, free of charge, to any person obtaining a    //
// copy of this software and associated documentation files (the "Software"), //
// to deal in the Software without restriction, including without limitation  //
// the rights to use, copy, modify, merge, publish, distribute, sublicense,   //
// and/or sell copies of the Software, and to permit persons to whom the      //
// Software is furnished to do so, subject to the following conditions:       //
//                                                                            //
// The above copyright notice and this permission notice shall be included in //
// all copies or substantial portions of the Software.                        //
//                                                                            //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    //
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    //
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        //
// DEALINGS IN THE SOFTWARE.                                                  //
//                                                                            //
//----------------------------------------------------------------------------//

#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include "ozz/base/containers/vector.h"
#include "ozz/base/log.h"
#include "ozz/base/maths/gtest_math_helper.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/geometry/runtime/skinning_job.h"

using ozz::geometry::SkinningJob;

TEST(JobValidity, SkinningJob) {
  ozz::math::Float4x4 matrices[2];
  ozz::math::Float4x4 it_matrices[2];
  uint16_t joint_indices[8];
  float joint_weights[6];
  float in_positions[6];
  float in_normals[6];
  float in_tangents[6];
  float out_positions[6];
  float out_normals[6];
  float out_tangents[6];

  {  // Default is invalid.
    SkinningJob job;
    EXPECT_FALSE(job.Validate());
    EXPECT_FALSE(job.Run());
  }
  {  // Valid job with 0 vertex.
    SkinningJob job;
    job.vertex_count = 0;
    job.influences_count = 1;
    job.joint_matrices = matrices;
    job.joint_indices = {joint_indices, 2};
    job.joint_indices_stride = sizeof(uint16_t) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Invalid job with 0 influence
    SkinningJob job;
    job.vertex_count = 0;
    job.influences_count = 0;
    job.joint_matrices = matrices;
    job.joint_indices = {joint_indices, 1};
    job.joint_indices_stride = sizeof(uint16_t) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Valid job with 1 influence
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 1;
    job.joint_matrices = matrices;
    job.joint_indices = {joint_indices, 2};
    job.joint_indices_stride = sizeof(uint16_t) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Valid job with 2 influences
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = {joint_indices, 4};
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = {joint_weights, 2};
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Valid job with 3 influences
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 3;
    job.joint_matrices = matrices;
    job.joint_indices = {joint_indices, 6};
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = {joint_weights, 4};
    job.joint_weights_stride = sizeof(float) * 2;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Valid job with 4 influences
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 4;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = {joint_weights, 6};
    job.joint_weights_stride = sizeof(float) * 2;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Valid job with 1 influence, unneeded weights.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 1;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Valid job with 1 influence, with inverse transposed matrices.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 1;
    job.joint_matrices = matrices;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Invalid job with 2 influences, missing indices.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Invalid job with 2 influences, missing weights.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Invalid job with 2 influences, not enough indices.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = {joint_indices, 3};
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Invalid job with 2 influences, not enough indices.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(joint_indices) + 1;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Invalid job with 2 influences, not enough weights.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = {joint_weights, 1};
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Invalid job with 2 influences, not enough weights.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(joint_weights) + 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Valid job with 2 influences
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Invalid job with 2 influences, mandatory positions.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    EXPECT_FALSE(job.Validate());
  }
  {  // Invalid job with 2 influences, mandatory positions input.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Invalid job with 2 influences, mandatory positions output.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Valid job with 2 influences, with normals.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Valid job with 2 influences, with normals but missing positions.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Valid job with 2 influences, with normals, mandatory output.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Valid job with 2 influences, with normals and tangents.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Validate());
  }
  {  // Invalid job with 2 influences, with normals and tangents,
    // required output.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
  {  // Invalid job with 2 influences, with tangents but missing normals.
    SkinningJob job;
    job.vertex_count = 2;
    job.influences_count = 2;
    job.joint_matrices = matrices;
    job.joint_indices = joint_indices;
    job.joint_indices_stride = sizeof(uint16_t) * 2;
    job.joint_weights = joint_weights;
    job.joint_weights_stride = sizeof(float) * 1;
    job.in_positions = in_positions;
    job.in_positions_stride = sizeof(float) * 3;
    job.out_positions = out_positions;
    job.out_positions_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_FALSE(job.Validate());
  }
}

TEST(JobResult, SkinningJob) {
  ozz::math::Float4x4 matrices[4] = {
      {{ozz::math::simd_float4::Load(-1.f, 0.f, 0.f, 0.f),
        ozz::math::simd_float4::Load(0.f, 1.f, 0.f, 0.f),
        ozz::math::simd_float4::Load(0.f, 0.f, -1.f, 0.f),
        ozz::math::simd_float4::Load(0.f, 0.f, 0.f, 1.f)}},
      ozz::math::Float4x4::Translation(
          ozz::math::simd_float4::Load(1.f, 2.f, 3.f, 0.f)),
      ozz::math::Float4x4::Scaling(
          ozz::math::simd_float4::Load(1.f, 2.f, 3.f, 0.f)),
      ozz::math::Float4x4::Translation(
          ozz::math::simd_float4::Load(1.f, 2.f, 3.f, 0.f))};
  // This isn't the inverse transpose of matrices array, but no mind.
  ozz::math::Float4x4 it_matrices[4] = {
      {{ozz::math::simd_float4::Load(1.f, 0.f, 0.f, 0.f),
        ozz::math::simd_float4::Load(0.f, -1.f, 0.f, 0.f),
        ozz::math::simd_float4::Load(0.f, 0.f, 1.f, 0.f),
        ozz::math::simd_float4::Load(0.f, 0.f, 0.f, 1.f)}},
      ozz::math::Float4x4::identity(),
      {{ozz::math::simd_float4::Load(-1.f, 0.f, 0.f, 0.f),
        ozz::math::simd_float4::Load(0.f, -1.f, 0.f, 0.f),
        ozz::math::simd_float4::Load(0.f, 0.f, -1.f, 0.f),
        ozz::math::simd_float4::Load(0.f, 0.f, 0.f, 1.f)}},
      ozz::math::Float4x4::identity()};
  uint16_t joint_indices[10] = {0, 1, 2, 3, 0, 3, 2, 1, 0, 3};
  float joint_weights[8] = {.5f, .25f, .25f, .1f, .1f, .25f, .25f, .15f};
  float in_positions[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  float in_normals[6] = {.1f, .2f, .3f, .4f, .5f, .6f};
  float in_tangents[6] = {.01f, .02f, .03f, .04f, .05f, .06f};
  float out_positions[6];
  float out_normals[6];
  float out_tangents[6];

  SkinningJob base_job;
  base_job.vertex_count = 2;
  base_job.influences_count = 1;
  base_job.joint_matrices = matrices;
  base_job.joint_indices = joint_indices;
  base_job.joint_indices_stride = sizeof(uint16_t) * 5;
  base_job.in_positions = in_positions;
  base_job.in_positions_stride = sizeof(float) * 3;
  base_job.out_positions = out_positions;
  base_job.out_positions_stride = sizeof(float) * 3;

  {  // P1
    SkinningJob job = base_job;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(-1.f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.f, out_positions[1]);
    EXPECT_FLOAT_EQ(-3.f, out_positions[2]);

    EXPECT_FLOAT_EQ(5.f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.f, out_positions[4]);
    EXPECT_FLOAT_EQ(9.f, out_positions[5]);
  }
  {  // PN1
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(-1.f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.f, out_positions[1]);
    EXPECT_FLOAT_EQ(-3.f, out_positions[2]);
    EXPECT_FLOAT_EQ(-.1f, out_normals[0]);
    EXPECT_FLOAT_EQ(.2f, out_normals[1]);
    EXPECT_FLOAT_EQ(-.3f, out_normals[2]);

    EXPECT_FLOAT_EQ(5.f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.f, out_positions[4]);
    EXPECT_FLOAT_EQ(9.f, out_positions[5]);
    EXPECT_FLOAT_EQ(.4f, out_normals[3]);
    EXPECT_FLOAT_EQ(.5f, out_normals[4]);
    EXPECT_FLOAT_EQ(.6f, out_normals[5]);
  }
  {  // PitN1
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(-1.f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.f, out_positions[1]);
    EXPECT_FLOAT_EQ(-3.f, out_positions[2]);
    EXPECT_FLOAT_EQ(.1f, out_normals[0]);
    EXPECT_FLOAT_EQ(-.2f, out_normals[1]);
    EXPECT_FLOAT_EQ(.3f, out_normals[2]);

    EXPECT_FLOAT_EQ(5.f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.f, out_positions[4]);
    EXPECT_FLOAT_EQ(9.f, out_positions[5]);
    EXPECT_FLOAT_EQ(.4f, out_normals[3]);
    EXPECT_FLOAT_EQ(.5f, out_normals[4]);
    EXPECT_FLOAT_EQ(.6f, out_normals[5]);
  }
  {  // PNT1
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(-1.f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.f, out_positions[1]);
    EXPECT_FLOAT_EQ(-3.f, out_positions[2]);
    EXPECT_FLOAT_EQ(-.1f, out_normals[0]);
    EXPECT_FLOAT_EQ(.2f, out_normals[1]);
    EXPECT_FLOAT_EQ(-.3f, out_normals[2]);
    EXPECT_FLOAT_EQ(-.01f, out_tangents[0]);
    EXPECT_FLOAT_EQ(.02f, out_tangents[1]);
    EXPECT_FLOAT_EQ(-.03f, out_tangents[2]);

    EXPECT_FLOAT_EQ(5.f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.f, out_positions[4]);
    EXPECT_FLOAT_EQ(9.f, out_positions[5]);
    EXPECT_FLOAT_EQ(.4f, out_normals[3]);
    EXPECT_FLOAT_EQ(.5f, out_normals[4]);
    EXPECT_FLOAT_EQ(.6f, out_normals[5]);
    EXPECT_FLOAT_EQ(.04f, out_tangents[3]);
    EXPECT_FLOAT_EQ(.05f, out_tangents[4]);
    EXPECT_FLOAT_EQ(.06f, out_tangents[5]);
  }
  {  // PitNT1
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(-1.f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.f, out_positions[1]);
    EXPECT_FLOAT_EQ(-3.f, out_positions[2]);
    EXPECT_FLOAT_EQ(.1f, out_normals[0]);
    EXPECT_FLOAT_EQ(-.2f, out_normals[1]);
    EXPECT_FLOAT_EQ(.3f, out_normals[2]);
    EXPECT_FLOAT_EQ(.01f, out_tangents[0]);
    EXPECT_FLOAT_EQ(-.02f, out_tangents[1]);
    EXPECT_FLOAT_EQ(.03f, out_tangents[2]);

    EXPECT_FLOAT_EQ(5.f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.f, out_positions[4]);
    EXPECT_FLOAT_EQ(9.f, out_positions[5]);
    EXPECT_FLOAT_EQ(.4f, out_normals[3]);
    EXPECT_FLOAT_EQ(.5f, out_normals[4]);
    EXPECT_FLOAT_EQ(.6f, out_normals[5]);
    EXPECT_FLOAT_EQ(.04f, out_tangents[3]);
    EXPECT_FLOAT_EQ(.05f, out_tangents[4]);
    EXPECT_FLOAT_EQ(.06f, out_tangents[5]);
  }

  base_job.influences_count = 2;
  base_job.joint_weights = joint_weights;
  base_job.joint_weights_stride = sizeof(float) * 4;

  {  // P2
    SkinningJob job = base_job;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.5f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);

    EXPECT_FLOAT_EQ(4.1f, out_positions[3]);
    EXPECT_FLOAT_EQ(9.7f, out_positions[4]);
    EXPECT_FLOAT_EQ(17.1f, out_positions[5]);
  }
  {  // PN2
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.5f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);
    EXPECT_FLOAT_EQ(0.f, out_normals[0]);
    EXPECT_FLOAT_EQ(.2f, out_normals[1]);
    EXPECT_FLOAT_EQ(0.f, out_normals[2]);

    EXPECT_FLOAT_EQ(4.1f, out_positions[3]);
    EXPECT_FLOAT_EQ(9.7f, out_positions[4]);
    EXPECT_FLOAT_EQ(17.1f, out_positions[5]);
    EXPECT_FLOAT_EQ(.4f, out_normals[3]);
    EXPECT_FLOAT_EQ(.95f, out_normals[4]);
    EXPECT_FLOAT_EQ(1.68f, out_normals[5]);
  }
  {  // PitN2
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.5f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);
    EXPECT_FLOAT_EQ(.1f, out_normals[0]);
    EXPECT_FLOAT_EQ(0.f, out_normals[1]);
    EXPECT_FLOAT_EQ(.3f, out_normals[2]);

    EXPECT_FLOAT_EQ(4.1f, out_positions[3]);
    EXPECT_FLOAT_EQ(9.7f, out_positions[4]);
    EXPECT_FLOAT_EQ(17.1f, out_positions[5]);
    EXPECT_FLOAT_EQ(-.32f, out_normals[3]);
    EXPECT_FLOAT_EQ(-.4f, out_normals[4]);
    EXPECT_FLOAT_EQ(-.48f, out_normals[5]);
  }
  {  // PNT2
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.5f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);
    EXPECT_FLOAT_EQ(0.f, out_normals[0]);
    EXPECT_FLOAT_EQ(.2f, out_normals[1]);
    EXPECT_FLOAT_EQ(0.f, out_normals[2]);
    EXPECT_FLOAT_EQ(0.f, out_tangents[0]);
    EXPECT_FLOAT_EQ(.02f, out_tangents[1]);
    EXPECT_FLOAT_EQ(0.f, out_tangents[2]);

    EXPECT_FLOAT_EQ(4.1f, out_positions[3]);
    EXPECT_FLOAT_EQ(9.7f, out_positions[4]);
    EXPECT_FLOAT_EQ(17.1f, out_positions[5]);
    EXPECT_FLOAT_EQ(.4f, out_normals[3]);
    EXPECT_FLOAT_EQ(.95f, out_normals[4]);
    EXPECT_FLOAT_EQ(1.68f, out_normals[5]);
    EXPECT_FLOAT_EQ(.04f, out_tangents[3]);
    EXPECT_FLOAT_EQ(.095f, out_tangents[4]);
    EXPECT_FLOAT_EQ(.168f, out_tangents[5]);
  }
  {  // PitNT2
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.5f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);
    EXPECT_FLOAT_EQ(.1f, out_normals[0]);
    EXPECT_FLOAT_EQ(0.f, out_normals[1]);
    EXPECT_FLOAT_EQ(.3f, out_normals[2]);
    EXPECT_FLOAT_EQ(0.01f, out_tangents[0]);
    EXPECT_FLOAT_EQ(0.f, out_tangents[1]);
    EXPECT_FLOAT_EQ(.03f, out_tangents[2]);

    EXPECT_FLOAT_EQ(4.1f, out_positions[3]);
    EXPECT_FLOAT_EQ(9.7f, out_positions[4]);
    EXPECT_FLOAT_EQ(17.1f, out_positions[5]);
    EXPECT_FLOAT_EQ(-.32f, out_normals[3]);
    EXPECT_FLOAT_EQ(-.4f, out_normals[4]);
    EXPECT_FLOAT_EQ(-.48f, out_normals[5]);
    EXPECT_FLOAT_EQ(-.032f, out_tangents[3]);
    EXPECT_FLOAT_EQ(-.04f, out_tangents[4]);
    EXPECT_FLOAT_EQ(-.048f, out_tangents[5]);
  }

  base_job.influences_count = 3;

  {  // P3
    SkinningJob job = base_job;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);

    EXPECT_FLOAT_EQ(4.75f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.75f, out_positions[4]);
    EXPECT_FLOAT_EQ(11.25f, out_positions[5]);
  }
  {  // PN3
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);
    EXPECT_FLOAT_EQ(0.f, out_normals[0]);
    EXPECT_FLOAT_EQ(.25f, out_normals[1]);
    EXPECT_FLOAT_EQ(.15f, out_normals[2]);

    EXPECT_FLOAT_EQ(4.75f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.75f, out_positions[4]);
    EXPECT_FLOAT_EQ(11.25f, out_positions[5]);
    EXPECT_FLOAT_EQ(.4f, out_normals[3]);
    EXPECT_FLOAT_EQ(.625f, out_normals[4]);
    EXPECT_FLOAT_EQ(.9f, out_normals[5]);
  }
  {  // PitN3
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);
    EXPECT_FLOAT_EQ(.05f, out_normals[0]);
    EXPECT_FLOAT_EQ(-.1f, out_normals[1]);
    EXPECT_FLOAT_EQ(.15f, out_normals[2]);

    EXPECT_FLOAT_EQ(4.75f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.75f, out_positions[4]);
    EXPECT_FLOAT_EQ(11.25f, out_positions[5]);
    EXPECT_FLOAT_EQ(.2f, out_normals[3]);
    EXPECT_FLOAT_EQ(.25f, out_normals[4]);
    EXPECT_FLOAT_EQ(.3f, out_normals[5]);
  }
  {  // PNT3
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);
    EXPECT_FLOAT_EQ(0.f, out_normals[0]);
    EXPECT_FLOAT_EQ(.25f, out_normals[1]);
    EXPECT_FLOAT_EQ(.15f, out_normals[2]);
    EXPECT_FLOAT_EQ(0.f, out_tangents[0]);
    EXPECT_FLOAT_EQ(.025f, out_tangents[1]);
    EXPECT_FLOAT_EQ(.015f, out_tangents[2]);

    EXPECT_FLOAT_EQ(4.75f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.75f, out_positions[4]);
    EXPECT_FLOAT_EQ(11.25f, out_positions[5]);
    EXPECT_FLOAT_EQ(.4f, out_normals[3]);
    EXPECT_FLOAT_EQ(.625f, out_normals[4]);
    EXPECT_FLOAT_EQ(.9f, out_normals[5]);
    EXPECT_FLOAT_EQ(.04f, out_tangents[3]);
    EXPECT_FLOAT_EQ(.0625f, out_tangents[4]);
    EXPECT_FLOAT_EQ(.09f, out_tangents[5]);
  }
  {  // PitN3
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);
    EXPECT_FLOAT_EQ(.05f, out_normals[0]);
    EXPECT_FLOAT_EQ(-.1f, out_normals[1]);
    EXPECT_FLOAT_EQ(.15f, out_normals[2]);
    EXPECT_FLOAT_EQ(.005f, out_tangents[0]);
    EXPECT_FLOAT_EQ(-.01f, out_tangents[1]);
    EXPECT_FLOAT_EQ(.015f, out_tangents[2]);

    EXPECT_FLOAT_EQ(4.75f, out_positions[3]);
    EXPECT_FLOAT_EQ(7.75f, out_positions[4]);
    EXPECT_FLOAT_EQ(11.25f, out_positions[5]);
    EXPECT_FLOAT_EQ(.2f, out_normals[3]);
    EXPECT_FLOAT_EQ(.25f, out_normals[4]);
    EXPECT_FLOAT_EQ(.3f, out_normals[5]);
    EXPECT_FLOAT_EQ(.02f, out_tangents[3]);
    EXPECT_FLOAT_EQ(.025f, out_tangents[4]);
    EXPECT_FLOAT_EQ(.03f, out_tangents[5]);
  }

  base_job.influences_count = 4;

  {  // P4
    SkinningJob job = base_job;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(0.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);

    EXPECT_FLOAT_EQ(1.15f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.95f, out_positions[4]);
    EXPECT_FLOAT_EQ(5.25f, out_positions[5]);
  }
  {  // PN4
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(0.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);
    EXPECT_FLOAT_EQ(0.f, out_normals[0]);
    EXPECT_FLOAT_EQ(.25f, out_normals[1]);
    EXPECT_FLOAT_EQ(.15f, out_normals[2]);

    EXPECT_FLOAT_EQ(1.15f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.95f, out_positions[4]);
    EXPECT_FLOAT_EQ(5.25f, out_positions[5]);
    EXPECT_FLOAT_EQ(.08f, out_normals[3]);
    EXPECT_FLOAT_EQ(.625f, out_normals[4]);
    EXPECT_FLOAT_EQ(.42f, out_normals[5]);
  }
  {  // PitN4
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(0.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);
    EXPECT_FLOAT_EQ(.05f, out_normals[0]);
    EXPECT_FLOAT_EQ(-.1f, out_normals[1]);
    EXPECT_FLOAT_EQ(.15f, out_normals[2]);

    EXPECT_FLOAT_EQ(1.15f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.95f, out_positions[4]);
    EXPECT_FLOAT_EQ(5.25f, out_positions[5]);
    EXPECT_FLOAT_EQ(.2f, out_normals[3]);
    EXPECT_FLOAT_EQ(-.15f, out_normals[4]);
    EXPECT_FLOAT_EQ(.3f, out_normals[5]);
  }
  {  // PNT4
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(0.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);
    EXPECT_FLOAT_EQ(0.f, out_normals[0]);
    EXPECT_FLOAT_EQ(.25f, out_normals[1]);
    EXPECT_FLOAT_EQ(.15f, out_normals[2]);
    EXPECT_FLOAT_EQ(0.f, out_tangents[0]);
    EXPECT_FLOAT_EQ(.025f, out_tangents[1]);
    EXPECT_FLOAT_EQ(.015f, out_tangents[2]);

    EXPECT_FLOAT_EQ(1.15f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.95f, out_positions[4]);
    EXPECT_FLOAT_EQ(5.25f, out_positions[5]);
    EXPECT_FLOAT_EQ(.08f, out_normals[3]);
    EXPECT_FLOAT_EQ(.625f, out_normals[4]);
    EXPECT_FLOAT_EQ(.42f, out_normals[5]);
    EXPECT_FLOAT_EQ(.008f, out_tangents[3]);
    EXPECT_FLOAT_EQ(.0625f, out_tangents[4]);
    EXPECT_FLOAT_EQ(.042f, out_tangents[5]);
  }
  {  // PitNT4
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(0.25f, out_positions[0]);
    EXPECT_FLOAT_EQ(3.f, out_positions[1]);
    EXPECT_FLOAT_EQ(2.25f, out_positions[2]);
    EXPECT_FLOAT_EQ(.05f, out_normals[0]);
    EXPECT_FLOAT_EQ(-.1f, out_normals[1]);
    EXPECT_FLOAT_EQ(.15f, out_normals[2]);
    EXPECT_FLOAT_EQ(.005f, out_tangents[0]);
    EXPECT_FLOAT_EQ(-.01f, out_tangents[1]);
    EXPECT_FLOAT_EQ(.015f, out_tangents[2]);

    EXPECT_FLOAT_EQ(1.15f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.95f, out_positions[4]);
    EXPECT_FLOAT_EQ(5.25f, out_positions[5]);
    EXPECT_FLOAT_EQ(.2f, out_normals[3]);
    EXPECT_FLOAT_EQ(-.15f, out_normals[4]);
    EXPECT_FLOAT_EQ(.3f, out_normals[5]);
    EXPECT_FLOAT_EQ(.02f, out_tangents[3]);
    EXPECT_FLOAT_EQ(-.015f, out_tangents[4]);
    EXPECT_FLOAT_EQ(.03f, out_tangents[5]);
  }

  // 5 influences use their own canonical (non negative, summing to 1) weights.
  // The last weight is restored at runtime from the 4 provided ones.
  const float joint_weights_5[8] = {
      .5f, .25f, .15f, .05f,  // Restored last weight is .05.
      .1f, .1f, .25f, .25f};  // Restored last weight is .3.
  base_job.influences_count = 5;
  base_job.joint_weights = joint_weights_5;

  {  // P5
    SkinningJob job = base_job;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.2f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.9f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);

    EXPECT_FLOAT_EQ(2.65f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.8f, out_positions[4]);
    EXPECT_FLOAT_EQ(6.15f, out_positions[5]);
  }
  {  // PN5
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.2f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.9f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);
    EXPECT_FLOAT_EQ(-.01f, out_normals[0]);
    EXPECT_FLOAT_EQ(.23f, out_normals[1]);
    EXPECT_FLOAT_EQ(.06f, out_normals[2]);

    EXPECT_FLOAT_EQ(2.65f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.8f, out_positions[4]);
    EXPECT_FLOAT_EQ(6.15f, out_positions[5]);
    EXPECT_FLOAT_EQ(.2f, out_normals[3]);
    EXPECT_FLOAT_EQ(.55f, out_normals[4]);
    EXPECT_FLOAT_EQ(.42f, out_normals[5]);
  }
  {  // PitN5
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.2f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.9f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);
    EXPECT_FLOAT_EQ(.07f, out_normals[0]);
    EXPECT_FLOAT_EQ(-.08f, out_normals[1]);
    EXPECT_FLOAT_EQ(.21f, out_normals[2]);

    EXPECT_FLOAT_EQ(2.65f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.8f, out_positions[4]);
    EXPECT_FLOAT_EQ(6.15f, out_positions[5]);
    EXPECT_FLOAT_EQ(.32f, out_normals[3]);
    EXPECT_FLOAT_EQ(.15f, out_normals[4]);
    EXPECT_FLOAT_EQ(.48f, out_normals[5]);
  }
  {  // PNT5
    SkinningJob job = base_job;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.2f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.9f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);
    EXPECT_FLOAT_EQ(-.01f, out_normals[0]);
    EXPECT_FLOAT_EQ(.23f, out_normals[1]);
    EXPECT_FLOAT_EQ(.06f, out_normals[2]);
    EXPECT_FLOAT_EQ(-.001f, out_tangents[0]);
    EXPECT_FLOAT_EQ(.023f, out_tangents[1]);
    EXPECT_FLOAT_EQ(.006f, out_tangents[2]);

    EXPECT_FLOAT_EQ(2.65f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.8f, out_positions[4]);
    EXPECT_FLOAT_EQ(6.15f, out_positions[5]);
    EXPECT_FLOAT_EQ(.2f, out_normals[3]);
    EXPECT_FLOAT_EQ(.55f, out_normals[4]);
    EXPECT_FLOAT_EQ(.42f, out_normals[5]);
    EXPECT_FLOAT_EQ(.02f, out_tangents[3]);
    EXPECT_FLOAT_EQ(.055f, out_tangents[4]);
    EXPECT_FLOAT_EQ(.042f, out_tangents[5]);
  }
  {  // PitNT5
    SkinningJob job = base_job;
    job.joint_inverse_transpose_matrices = it_matrices;
    job.in_normals = in_normals;
    job.in_normals_stride = sizeof(float) * 3;
    job.out_normals = out_normals;
    job.out_normals_stride = sizeof(float) * 3;
    job.in_tangents = in_tangents;
    job.in_tangents_stride = sizeof(float) * 3;
    job.out_tangents = out_tangents;
    job.out_tangents_stride = sizeof(float) * 3;
    EXPECT_TRUE(job.Run());

    EXPECT_FLOAT_EQ(.2f, out_positions[0]);
    EXPECT_FLOAT_EQ(2.9f, out_positions[1]);
    EXPECT_FLOAT_EQ(1.5f, out_positions[2]);
    EXPECT_FLOAT_EQ(.07f, out_normals[0]);
    EXPECT_FLOAT_EQ(-.08f, out_normals[1]);
    EXPECT_FLOAT_EQ(.21f, out_normals[2]);
    EXPECT_FLOAT_EQ(.007f, out_tangents[0]);
    EXPECT_FLOAT_EQ(-.008f, out_tangents[1]);
    EXPECT_FLOAT_EQ(.021f, out_tangents[2]);

    EXPECT_FLOAT_EQ(2.65f, out_positions[3]);
    EXPECT_FLOAT_EQ(6.8f, out_positions[4]);
    EXPECT_FLOAT_EQ(6.15f, out_positions[5]);
    EXPECT_FLOAT_EQ(.32f, out_normals[3]);
    EXPECT_FLOAT_EQ(.15f, out_normals[4]);
    EXPECT_FLOAT_EQ(.48f, out_normals[5]);
    EXPECT_FLOAT_EQ(.032f, out_tangents[3]);
    EXPECT_FLOAT_EQ(.015f, out_tangents[4]);
    EXPECT_FLOAT_EQ(.048f, out_tangents[5]);
  }
}

// Tests weight validity boundaries shared by every influence count code path
// (1 to 4 specialized implementations and the generic n-influences one), with
// and without inverse transpose matrices.

// Identity joint matrices keep input geometry unchanged on success.
static const ozz::math::Float4x4 kIdentityMatrices[1] = {
    ozz::math::Float4x4::identity()};

// Distinct sentinel allows to detect any partial write in an output buffer.
static constexpr float kWeightTestSentinel = -97531.2468f;

TEST(JobWeights, SkinningJob) {
  struct Context {
    int vertex_count = 2;
    int influences = 2;
    ozz::vector<uint16_t> indices;
    ozz::vector<float> weights;
    float in_positions[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float in_normals[6] = {.1f, .2f, .3f, .4f, .5f, .6f};
    float in_tangents[6] = {.01f, .02f, .03f, .04f, .05f, .06f};
    float out_positions[6];
    float out_normals[6];
    float out_tangents[6];

    void FillSentinels() {
      for (float& f : out_positions) f = kWeightTestSentinel;
      for (float& f : out_normals) f = kWeightTestSentinel;
      for (float& f : out_tangents) f = kWeightTestSentinel;
    }

    bool ExpectUntouched() const {
      for (int i = 0; i < vertex_count * 3; ++i) {
        if (out_positions[i] != kWeightTestSentinel ||
            out_normals[i] != kWeightTestSentinel ||
            out_tangents[i] != kWeightTestSentinel) {
          return false;
        }
      }
      return true;
    }

    // Builds a structurally valid job, with positions, normals and tangents.
    SkinningJob Build(bool _use_it) {
      SkinningJob job;
      job.vertex_count = vertex_count;
      job.influences_count = influences;
      job.joint_matrices = ozz::make_span(kIdentityMatrices);
      if (_use_it) {
        job.joint_inverse_transpose_matrices = ozz::make_span(kIdentityMatrices);
      }
      job.joint_indices = ozz::make_span(indices);
      job.joint_indices_stride = sizeof(uint16_t) * influences;
      if (influences > 1) {
        job.joint_weights = ozz::make_span(weights);
        job.joint_weights_stride = sizeof(float) * (influences - 1);
      }
      job.in_positions = in_positions;
      job.in_positions_stride = sizeof(float) * 3;
      job.out_positions = out_positions;
      job.out_positions_stride = sizeof(float) * 3;
      job.in_normals = in_normals;
      job.in_normals_stride = sizeof(float) * 3;
      job.out_normals = out_normals;
      job.out_normals_stride = sizeof(float) * 3;
      job.in_tangents = in_tangents;
      job.in_tangents_stride = sizeof(float) * 3;
      job.out_tangents = out_tangents;
      job.out_tangents_stride = sizeof(float) * 3;
      return job;
    }
  };

  // Canonical valid weights: every provided weight is positive and the
  // restored last weight (1 - sum) is positive too.
  auto valid_weights = [](int _influences, int _vertex) {
    ozz::vector<float> weights(
        static_cast<size_t>(_vertex) * (_influences - 1));
    const float provided = 0.5f / _influences;
    for (float& w : weights) w = provided;
    return weights;
  };

  // Canonical inputs succeed for every influence count and both transform
  // conventions, and produce finite outputs.
  for (int influences = 1; influences <= 5; ++influences) {
    for (int use_it = 0; use_it < 2; ++use_it) {
      Context ctx;
      ctx.influences = influences;
      ctx.indices.resize(static_cast<size_t>(ctx.vertex_count) * influences,
                         0);
      ctx.weights = valid_weights(influences, ctx.vertex_count);
      ctx.FillSentinels();

      SkinningJob job = ctx.Build(use_it != 0);
      EXPECT_TRUE(job.Validate());
      EXPECT_TRUE(job.Run());
      // Identity matrices: outputs are the inputs, untouched sentinels are
      // fully overwritten with finite values.
      for (int i = 0; i < ctx.vertex_count * 3; ++i) {
        EXPECT_NE(ctx.out_positions[i], kWeightTestSentinel);
        EXPECT_NE(ctx.out_normals[i], kWeightTestSentinel);
        EXPECT_NE(ctx.out_tangents[i], kWeightTestSentinel);
        EXPECT_TRUE(std::isfinite(ctx.out_positions[i]));
        EXPECT_TRUE(std::isfinite(ctx.out_normals[i]));
        EXPECT_TRUE(std::isfinite(ctx.out_tangents[i]));
      }
    }
  }

  // Every invalid weight category must fail for every multi-influence code
  // path, with and without inverse transpose matrices, and leave every output
  // position, normal and tangent untouched.
  for (int influences = 2; influences <= 5; ++influences) {
    for (int use_it = 0; use_it < 2; ++use_it) {
      const int weights_per_vertex = influences - 1;

      // Provided weight categories that must fail.
      const float bad_values[] = {-0.01f,
                                  std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity()};
      for (float bad : bad_values) {
        for (int bad_index = 0; bad_index < weights_per_vertex; ++bad_index) {
          Context ctx;
          ctx.influences = influences;
          ctx.indices.resize(
              static_cast<size_t>(ctx.vertex_count) * influences, 0);
          ctx.weights = valid_weights(influences, ctx.vertex_count);
          ctx.weights[bad_index] = bad;  // First vertex, one bad weight.
          ctx.FillSentinels();

          SkinningJob job = ctx.Build(use_it != 0);
          EXPECT_TRUE(job.Validate());  // Structure is still valid.
          EXPECT_FALSE(job.Run());
          EXPECT_TRUE(ctx.ExpectUntouched());
        }
      }

      // Restored last weight negative: provided weights alone are valid, but
      // their sum exceeds 1.
      {
        Context ctx;
        ctx.influences = influences;
        ctx.indices.resize(static_cast<size_t>(ctx.vertex_count) * influences,
                           0);
        ctx.weights = valid_weights(influences, ctx.vertex_count);
        for (int j = 0; j < weights_per_vertex; ++j) {
          ctx.weights[j] = 1.f / weights_per_vertex;  // Sum is 1: last is 0...
        }
        ctx.weights[0] += 0.01f;  // ...now the restored last weight is < 0.
        ctx.FillSentinels();

        SkinningJob job = ctx.Build(use_it != 0);
        EXPECT_TRUE(job.Validate());
        EXPECT_FALSE(job.Run());
        EXPECT_TRUE(ctx.ExpectUntouched());
      }
    }
  }

  // The historical n-influences data set restored a negative last weight
  // (provided weights summing to 1.1): it is an explicit failure now.
  for (int use_it = 0; use_it < 2; ++use_it) {
    Context ctx;
    ctx.vertex_count = 1;
    ctx.influences = 5;
    ctx.indices.assign(5, 0);
    ctx.weights = {.5f, .25f, .25f, .1f};  // Sum 1.1, restored weight -0.1.
    ctx.FillSentinels();

    SkinningJob job = ctx.Build(use_it != 0);
    EXPECT_TRUE(job.Validate());
    EXPECT_FALSE(job.Run());
    EXPECT_TRUE(ctx.ExpectUntouched());
  }

  // A bad weight on any vertex fails the whole job: no earlier vertex may be
  // partially updated either.
  for (int use_it = 0; use_it < 2; ++use_it) {
    Context ctx;
    ctx.influences = 2;
    ctx.indices.assign(static_cast<size_t>(ctx.vertex_count) * ctx.influences,
                       0);
    ctx.weights = valid_weights(ctx.influences, ctx.vertex_count);
    ctx.weights[ctx.influences - 1] = -0.25f;  // Bad weight of the 2nd vertex.
    ctx.FillSentinels();

    SkinningJob job = ctx.Build(use_it != 0);
    EXPECT_TRUE(job.Validate());
    EXPECT_FALSE(job.Run());
    EXPECT_TRUE(ctx.ExpectUntouched());
  }

  // A restored last weight of exactly 0 is a valid boundary, as are zero
  // provided weights.
  for (int use_it = 0; use_it < 2; ++use_it) {
    Context ctx;
    ctx.influences = 2;
    ctx.indices.assign(static_cast<size_t>(ctx.vertex_count) * ctx.influences,
                       0);
    ctx.weights = {1.f, 0.f};  // First vertex: last weight 0; second: w0 0.
    ctx.FillSentinels();

    SkinningJob job = ctx.Build(use_it != 0);
    EXPECT_TRUE(job.Validate());
    EXPECT_TRUE(job.Run());
    for (int i = 0; i < ctx.vertex_count * 3; ++i) {
      EXPECT_NE(ctx.out_positions[i], kWeightTestSentinel);
      EXPECT_NE(ctx.out_normals[i], kWeightTestSentinel);
      EXPECT_NE(ctx.out_tangents[i], kWeightTestSentinel);
    }
  }
}

struct BenchVertexIn {
  float pos[3];
  float normals[3];
  float tangents[3];
  uint16_t indices[8];
  float weights[7];
};

struct BenchVertexOut {
  float pos[3];
  float normals[3];
  float tangents[3];
};

TEST(Benchmark, SkinningJob) {
  const int vertex_count = 10000;
  const int joint_count = 100;

  // Prepares matrices.
  ozz::vector<ozz::math::Float4x4> matrices(joint_count);
  for (int i = 0; i < joint_count; ++i) {
    matrices[i] = ozz::math::Float4x4::identity();
  }

  // Prepares vertices.
  ozz::vector<BenchVertexIn> in_vertices(vertex_count + 1);
  for (int i = 0; i < vertex_count; ++i) {
    BenchVertexIn& vertex = in_vertices[i];
    for (size_t j = 0; j < OZZ_ARRAY_SIZE(vertex.indices); ++j) {
      vertex.indices[j] = j % joint_count;
    }
    for (size_t j = 0; j < OZZ_ARRAY_SIZE(vertex.weights); ++j) {
      // Normalized weights: the (stripped) last influence restores the
      // remaining weight, so every provided and restored weight is positive.
      vertex.weights[j] = 1.f / OZZ_ARRAY_SIZE(vertex.indices);
    }

    const float cpnt = std::sqrt(1.f / 3.f);
    for (int j = 0; j < 3; ++j) {
      vertex.pos[j] = 2.f * j * i;
      vertex.normals[j] = cpnt;
      vertex.tangents[j] = cpnt;
    }
  }
  ozz::vector<BenchVertexOut> out_vertices(vertex_count + 1);

  const float* in_vertices_end =
      reinterpret_cast<const float*>(array_end(in_vertices));
  float* out_vertices_end = reinterpret_cast<float*>(array_end(out_vertices));

  SkinningJob base_job;
  base_job.vertex_count = vertex_count;
  base_job.joint_matrices = make_span(matrices);
  base_job.joint_indices = {in_vertices.data()->indices,
                            reinterpret_cast<const uint16_t*>(in_vertices_end)};
  base_job.joint_indices_stride = sizeof(BenchVertexIn);
  base_job.joint_weights = {in_vertices.data()->weights, in_vertices_end};
  base_job.joint_weights_stride = sizeof(BenchVertexIn);
  base_job.in_positions = {in_vertices.data()->pos, in_vertices_end};
  base_job.in_positions_stride = sizeof(BenchVertexIn);
  base_job.out_positions = {out_vertices.data()->pos, out_vertices_end};
  base_job.out_positions_stride = sizeof(BenchVertexOut);

  for (int i = 1; i <= 8; ++i) {
    SkinningJob job = base_job;
    job.influences_count = i;

    // Pi
    { EXPECT_TRUE(job.Run()); }

    // PNi
    {
      job.in_normals = {in_vertices.data()->normals, in_vertices_end};
      job.in_normals_stride = sizeof(BenchVertexIn);
      job.out_normals = {out_vertices.data()->normals, out_vertices_end};
      job.out_normals_stride = sizeof(BenchVertexOut);

      EXPECT_TRUE(job.Run());
    }

    // PitNi
    {
      job.joint_inverse_transpose_matrices = make_span(matrices);
      job.in_normals = {in_vertices.data()->normals, in_vertices_end};
      job.in_normals_stride = sizeof(BenchVertexIn);
      job.out_normals = {out_vertices.data()->normals, out_vertices_end};
      job.out_normals_stride = sizeof(BenchVertexOut);

      EXPECT_TRUE(job.Run());
    }

    // PNTi
    {
      job.in_tangents = {in_vertices.data()->tangents, in_vertices_end};
      job.in_tangents_stride = sizeof(BenchVertexIn);
      job.out_tangents = {out_vertices.data()->tangents, out_vertices_end};
      job.out_tangents_stride = sizeof(BenchVertexOut);

      EXPECT_TRUE(job.Run());
    }

    // PitNTi
    {
      job.joint_inverse_transpose_matrices = make_span(matrices);
      job.in_tangents = {in_vertices.data()->tangents, in_vertices_end};
      job.in_tangents_stride = sizeof(BenchVertexIn);
      job.out_tangents = {out_vertices.data()->tangents, out_vertices_end};
      job.out_tangents_stride = sizeof(BenchVertexOut);

      EXPECT_TRUE(job.Run());
    }
  }
}
