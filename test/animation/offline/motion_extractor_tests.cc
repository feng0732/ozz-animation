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

#include <algorithm>
#include <cstdlib>

#include "gtest/gtest.h"
#include "ozz/animation/offline/motion_extractor.h"
#include "ozz/animation/offline/raw_animation.h"
#include "ozz/animation/offline/raw_animation_utils.h"
#include "ozz/animation/offline/raw_skeleton.h"
#include "ozz/animation/offline/raw_track.h"
#include "ozz/animation/offline/raw_track_utils.h"
#include "ozz/animation/offline/skeleton_builder.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/maths/gtest_math_helper.h"
#include "ozz/base/maths/math_constant.h"
#include "ozz/base/maths/math_ex.h"
#include "ozz/base/maths/transform.h"
#include "ozz/base/memory/unique_ptr.h"

using ozz::animation::Skeleton;
using ozz::animation::offline::MotionExtractor;
using ozz::animation::offline::RawAnimation;
using ozz::animation::offline::RawFloat3Track;
using ozz::animation::offline::RawQuaternionTrack;
using ozz::animation::offline::RawSkeleton;
using ozz::animation::offline::SkeletonBuilder;

// Compares two quaternions as rotations: q and -q represent the same
// rotation, so both signs are accepted.
static void ExpectQuaternionEqRotation(const ozz::math::Quaternion& _a,
                                       const ozz::math::Quaternion& _b,
                                       float _tol = 1e-4f) {
  const float direct = std::abs(_a.x - _b.x) + std::abs(_a.y - _b.y) +
                       std::abs(_a.z - _b.z) + std::abs(_a.w - _b.w);
  const float opposite = std::abs(_a.x + _b.x) + std::abs(_a.y + _b.y) +
                         std::abs(_a.z + _b.z) + std::abs(_a.w + _b.w);
  EXPECT_LE(std::min(direct, opposite), _tol);
}

// Applies the same rotation component selection (all components) as the motion
// extractor: the rotation is decomposed to Euler angles and rebuilt, which is
// how the extractor masks selected yaw/pitch/roll components.
static ozz::math::Quaternion ExtractRotationAll(
    const ozz::math::Quaternion& _rotation) {
  return ozz::math::Quaternion::FromEuler(
      ozz::math::ToEuler(_rotation) * ozz::math::Float3(1.f, 1.f, 1.f));
}

// Compares two raw joint tracks keyframe by keyframe.
template <typename _Keyframes>
static bool KeyframesEqual(const _Keyframes& _a, const _Keyframes& _b) {
  if (_a.size() != _b.size()) {
    return false;
  }
  for (size_t i = 0; i < _a.size(); ++i) {
    if (_a[i].time != _b[i].time || _a[i].value != _b[i].value) {
      return false;
    }
  }
  return true;
}

static bool JointTrackEqual(const RawAnimation::JointTrack& _a,
                            const RawAnimation::JointTrack& _b) {
  return KeyframesEqual(_a.translations, _b.translations) &&
         KeyframesEqual(_a.rotations, _b.rotations) &&
         KeyframesEqual(_a.scales, _b.scales);
}

TEST(Error, MotionExtractor) {
  RawAnimation input;
  input.tracks.resize(1);
  ASSERT_TRUE(input.Validate());

  RawSkeleton raw_skeleton;
  raw_skeleton.roots.resize(1);
  ASSERT_TRUE(raw_skeleton.Validate());
  const auto skeleton = SkeletonBuilder()(raw_skeleton);
  ASSERT_TRUE(skeleton);

  RawFloat3Track motion_position;
  RawQuaternionTrack motion_rotation;
  RawAnimation output;

  {  // Same input and output.
    MotionExtractor extractor;
    EXPECT_FALSE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                           &input));
  }

  {
    // Invalid animation
    RawAnimation input_invalid;
    input_invalid.tracks.resize(1);
    input_invalid.duration = -1.f;
    EXPECT_FALSE(input_invalid.Validate());

    MotionExtractor extractor;
    EXPECT_FALSE(extractor(input_invalid, *skeleton, &motion_position,
                           &motion_rotation, &output));
  }

  {
    // Non matching animation & skeleton
    RawSkeleton raw_skeleton_2;
    raw_skeleton_2.roots.resize(2);
    const auto skeleton_2 = SkeletonBuilder()(raw_skeleton_2);
    ASSERT_TRUE(skeleton_2);

    MotionExtractor extractor;
    EXPECT_FALSE(extractor(input, *skeleton_2, &motion_position,
                           &motion_rotation, &output));
  }

  {  // Null output
    MotionExtractor extractor;
    EXPECT_FALSE(
        extractor(input, *skeleton, nullptr, &motion_rotation, &output));
    EXPECT_FALSE(
        extractor(input, *skeleton, &motion_position, nullptr, &output));
    EXPECT_FALSE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                           nullptr));
  }

  {  // Invalid root
    MotionExtractor extractor;
    extractor.root_joint = 93;
    EXPECT_FALSE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                           &output));
  }

  {  // Valid
    MotionExtractor extractor;
    EXPECT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &output));
  }
}

TEST(Extract, MotionExtractor) {
  RawAnimation input;
  input.duration = 46.f;
  input.name = "test";
  input.tracks.resize(2);

  input.tracks[0].translations.push_back({.5f, {1.f, 2.f, 3.f}});
  input.tracks[0].translations.push_back({5.f, {4.f, 5.f, 6.f}});
  input.tracks[0].rotations.push_back(
      {1.f, ozz::math::Quaternion::FromEuler({ozz::math::kPi / 2.f,
                                              ozz::math::kPi / 3.f,
                                              ozz::math::kPi / 4.f})});
  input.tracks[0].rotations.push_back(
      {12.f, ozz::math::Quaternion::FromEuler({ozz::math::kPi / 2.f,
                                               1.1f * ozz::math::kPi / 3.f,
                                               ozz::math::kPi / 4.f})});
  input.tracks[0].rotations.push_back(
      {46.f, ozz::math::Quaternion::FromEuler({ozz::math::kPi / 2.f,
                                               1.2f * ozz::math::kPi / 3.f,
                                               ozz::math::kPi / 4.f})});

  input.tracks[1].translations.push_back({2.f, {7.f, 8.f, 9.f}});
  input.tracks[1].translations.push_back({10.f, {10.f, 11.f, 12.f}});
  input.tracks[1].rotations.push_back(
      {23.f, ozz::math::Quaternion::FromEuler({-ozz::math::kPi / 5.f,
                                               ozz::math::kPi / 7.f,
                                               ozz::math::kPi / 2.f})});

  ASSERT_TRUE(input.Validate());

  RawSkeleton raw_skeleton;
  raw_skeleton.roots.resize(1);
  raw_skeleton.roots[0].children.resize(1);
  ASSERT_TRUE(raw_skeleton.Validate());
  const auto skeleton = SkeletonBuilder()(raw_skeleton);

  RawFloat3Track motion_position;
  RawQuaternionTrack motion_rotation;
  RawAnimation baked;

  const auto anim_cmpnt_eq = [](const auto& _a, const auto& _b) {
    if (_a.size() != _b.size()) {
      return false;
    }
    for (size_t i = 0; i < _a.size(); ++i) {
      if (_a[i].time != _b[i].time) {
        return false;
      }
      if (_a[i].value != _b[i].value) {
        return false;
      }
    }
    return true;
  };
  const auto anim_track_eq = [&anim_cmpnt_eq](const auto& _a, const auto& _b) {
    return anim_cmpnt_eq(_a.translations, _b.translations) &&
           anim_cmpnt_eq(_a.rotations, _b.rotations) &&
           anim_cmpnt_eq(_a.scales, _b.scales);
  };

  {  // No extraction
    const MotionExtractor extractor{
        0,  // Joint
        {false, false, false, MotionExtractor::Reference::kAbsolute, true},
        {false, false, false, MotionExtractor::Reference::kAbsolute, true}};
    EXPECT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &baked));

    // Animation
    EXPECT_EQ(input.name, baked.name);
    EXPECT_FLOAT_EQ(input.duration, baked.duration);
    EXPECT_EQ(input.num_tracks(), baked.num_tracks());
    EXPECT_TRUE(anim_track_eq(input.tracks[0], baked.tracks[0]));
    EXPECT_TRUE(anim_track_eq(input.tracks[1], baked.tracks[1]));

    // Motion
    const auto& positions = motion_position.keyframes;
    ASSERT_EQ(positions.size(), 2u);
    EXPECT_FLOAT_EQ(positions[0].ratio, .5f / 46.f);
    EXPECT_FLOAT3_EQ(positions[0].value, 0.f, 0.f, 0.f);
    EXPECT_FLOAT_EQ(positions[1].ratio, 5.f / 46.f);
    EXPECT_FLOAT3_EQ(positions[1].value, 0.f, 0.f, 0.f);

    const auto& rotations = motion_rotation.keyframes;
    ASSERT_EQ(rotations.size(), 3u);
    EXPECT_FLOAT_EQ(rotations[0].ratio, 1.f / 46.f);
    EXPECT_QUATERNION_EQ(rotations[0].value, 0.f, 0.f, 0.f, 1.f);
    EXPECT_FLOAT_EQ(rotations[1].ratio, 12.f / 46.f);
    EXPECT_QUATERNION_EQ(rotations[1].value, 0.f, 0.f, 0.f, 1.f);
    EXPECT_FLOAT_EQ(rotations[2].ratio, 46.f / 46.f);
    EXPECT_QUATERNION_EQ(rotations[2].value, 0.f, 0.f, 0.f, 1.f);
  }

  {  // No baking
    const MotionExtractor extractor{
        0,  // Joint
        {true, true, true, MotionExtractor::Reference::kAbsolute, false},
        {true, true, true, MotionExtractor::Reference::kAbsolute, false}};
    EXPECT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &baked));

    // Animation
    EXPECT_EQ(input.name, baked.name);
    EXPECT_FLOAT_EQ(input.duration, baked.duration);
    EXPECT_EQ(input.num_tracks(), baked.num_tracks());
    EXPECT_TRUE(anim_track_eq(input.tracks[0], baked.tracks[0]));
    EXPECT_TRUE(anim_track_eq(input.tracks[1], baked.tracks[1]));

    // Track
    const auto& positions = motion_position.keyframes;
    ASSERT_EQ(positions.size(), 2u);
    EXPECT_FLOAT_EQ(positions[0].ratio, .5f / 46.f);
    EXPECT_FLOAT3_EQ(positions[0].value, 1.f, 2.f, 3.f);
    EXPECT_FLOAT_EQ(positions[1].ratio, 5.f / 46.f);
    EXPECT_FLOAT3_EQ(positions[1].value, 4.f, 5.f, 6.f);

    const auto& rotations = motion_rotation.keyframes;
    ASSERT_EQ(rotations.size(), 3u);
    EXPECT_FLOAT_EQ(rotations[0].ratio, 1.f / 46.f);
    auto r1 = ozz::math::Quaternion::FromEuler(ozz::math::Float3(
        ozz::math::kPi / 2.f, ozz::math::kPi / 3.f, ozz::math::kPi / 4.f));
    EXPECT_QUATERNION_EQ(rotations[0].value, r1.x, r1.y, r1.z, r1.w);
    auto r2 = ozz::math::Quaternion::FromEuler(
        ozz::math::Float3(ozz::math::kPi / 2.f, 1.1f * ozz::math::kPi / 3.f,
                          ozz::math::kPi / 4.f));
    EXPECT_QUATERNION_EQ(rotations[1].value, r2.x, r2.y, r2.z, r2.w);
    auto r3 = ozz::math::Quaternion::FromEuler(
        ozz::math::Float3(ozz::math::kPi / 2.f, 1.2f * ozz::math::kPi / 3.f,
                          ozz::math::kPi / 4.f));
    EXPECT_QUATERNION_EQ(rotations[2].value, r3.x, r3.y, r3.z, r3.w);
  }

  {  // Extract all
    const MotionExtractor extractor{
        0,  // Joint
        {true, true, true, MotionExtractor::Reference::kAbsolute, true},
        {true, true, true, MotionExtractor::Reference::kAbsolute, true}};
    EXPECT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &baked));

    // Animation
    EXPECT_EQ(input.name, baked.name);
    EXPECT_FLOAT_EQ(input.duration, baked.duration);
    EXPECT_EQ(input.num_tracks(), baked.num_tracks());

    const auto& track0 = baked.tracks[0];
    ASSERT_EQ(track0.translations.size(), 2u);
    EXPECT_FLOAT3_EQ(track0.translations[0].value, 0.f, 0.f, 0.f);
    EXPECT_FLOAT3_EQ(track0.translations[1].value, 0.f, 0.f, 0.f);
    ASSERT_EQ(track0.rotations.size(), 3u);
    EXPECT_QUATERNION_EQ(track0.rotations[0].value, 0.f, 0.f, 0.f, 1.f);
    EXPECT_QUATERNION_EQ(track0.rotations[1].value, 0.f, 0.f, 0.f, 1.f);
    EXPECT_QUATERNION_EQ(track0.rotations[2].value, 0.f, 0.f, 0.f, 1.f);

    EXPECT_TRUE(anim_track_eq(input.tracks[1], baked.tracks[1]));

    // Track
    const auto& positions = motion_position.keyframes;
    ASSERT_EQ(positions.size(), 2u);
    EXPECT_FLOAT_EQ(positions[0].ratio, .5f / 46.f);
    EXPECT_FLOAT3_EQ(positions[0].value, 1.f, 2.f, 3.f);
    EXPECT_FLOAT_EQ(positions[1].ratio, 5.f / 46.f);
    EXPECT_FLOAT3_EQ(positions[1].value, 4.f, 5.f, 6.f);

    const auto& rotations = motion_rotation.keyframes;
    ASSERT_EQ(rotations.size(), 3u);
    EXPECT_FLOAT_EQ(rotations[0].ratio, 1.f / 46.f);
    auto mr1 = ozz::math::Quaternion::FromEuler(
        {ozz::math::kPi / 2.f, ozz::math::kPi / 3.f, ozz::math::kPi / 4.f});
    EXPECT_QUATERNION_EQ(rotations[0].value, mr1.x, mr1.y, mr1.z, mr1.w);
    auto mr2 = ozz::math::Quaternion::FromEuler({ozz::math::kPi / 2.f,
                                                 1.1f * ozz::math::kPi / 3.f,
                                                 ozz::math::kPi / 4.f});
    EXPECT_QUATERNION_EQ(rotations[1].value, mr2.x, mr2.y, mr2.z, mr2.w);
    auto mr3 = ozz::math::Quaternion::FromEuler({ozz::math::kPi / 2.f,
                                                 1.2f * ozz::math::kPi / 3.f,
                                                 ozz::math::kPi / 4.f});
    EXPECT_QUATERNION_EQ(rotations[2].value, mr3.x, mr3.y, mr3.z, mr3.w);
  }

  {  // Extract only y position
    const MotionExtractor extractor{
        0,  // Joint
        {false, true, false, MotionExtractor::Reference::kAbsolute, true},
        {false, false, false, MotionExtractor::Reference::kAbsolute, true}};
    EXPECT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &baked));

    // Animation
    EXPECT_EQ(input.name, baked.name);
    EXPECT_FLOAT_EQ(input.duration, baked.duration);
    EXPECT_EQ(input.num_tracks(), baked.num_tracks());

    const auto& track0 = baked.tracks[0];
    ASSERT_EQ(track0.translations.size(), 2u);
    EXPECT_FLOAT3_EQ(track0.translations[0].value, 1.f, 0.f, 3.f);
    EXPECT_FLOAT3_EQ(track0.translations[1].value, 4.f, 0.f, 6.f);

    EXPECT_TRUE(anim_cmpnt_eq(track0.rotations, track0.rotations));
    EXPECT_TRUE(anim_track_eq(input.tracks[1], baked.tracks[1]));

    // Motion
    const auto& positions = motion_position.keyframes;
    ASSERT_EQ(positions.size(), 2u);
    EXPECT_FLOAT_EQ(positions[0].ratio, .5f / 46.f);
    EXPECT_FLOAT3_EQ(positions[0].value, 0.f, 2.f, 0.f);
    EXPECT_FLOAT_EQ(positions[1].ratio, 5.f / 46.f);
    EXPECT_FLOAT3_EQ(positions[1].value, 0.f, 5.f, 0.f);

    const auto& rotations = motion_rotation.keyframes;
    ASSERT_EQ(rotations.size(), 3u);
    EXPECT_FLOAT_EQ(rotations[0].ratio, 1.f / 46.f);
    EXPECT_QUATERNION_EQ(rotations[0].value, 0.f, 0.f, 0.f, 1.f);
    EXPECT_FLOAT_EQ(rotations[1].ratio, 12.f / 46.f);
    EXPECT_QUATERNION_EQ(rotations[1].value, 0.f, 0.f, 0.f, 1.f);
    EXPECT_FLOAT_EQ(rotations[2].ratio, 46.f / 46.f);
    EXPECT_QUATERNION_EQ(rotations[2].value, 0.f, 0.f, 0.f, 1.f);
  }

  {  // Extract only x position and y/yaw rotation
    const MotionExtractor extractor{
        0,  // Joint
        {true, false, false, MotionExtractor::Reference::kAbsolute, true},
        {false, true, false, MotionExtractor::Reference::kAbsolute, true}};
    EXPECT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &baked));

    EXPECT_EQ(input.name, baked.name);
    EXPECT_FLOAT_EQ(input.duration, baked.duration);
    EXPECT_EQ(input.num_tracks(), baked.num_tracks());

    const auto& track0 = baked.tracks[0];
    ASSERT_EQ(track0.translations.size(), 2u);
    // Inverse of extracted y/yaw applied to translation
    EXPECT_FLOAT3_EQ(track0.translations[0].value, -3.f, 2.f, 0.f);
    EXPECT_FLOAT3_EQ(track0.translations[1].value, -6.f, 5.f, 0.f);

    // Y/yaw extracted
    ASSERT_EQ(track0.rotations.size(), 3u);
    auto r1 = ozz::math::Quaternion::FromEuler(
        {0, ozz::math::kPi / 3.f, ozz::math::kPi / 4.f});
    EXPECT_QUATERNION_EQ(track0.rotations[0].value, r1.x, r1.y, r1.z, r1.w);
    auto r2 = ozz::math::Quaternion::FromEuler(
        {0, 1.1f * ozz::math::kPi / 3.f, ozz::math::kPi / 4.f});
    EXPECT_QUATERNION_EQ(track0.rotations[1].value, r2.x, r2.y, r2.z, r2.w);
    auto r3 = ozz::math::Quaternion::FromEuler(
        {0, 1.2f * ozz::math::kPi / 3.f, ozz::math::kPi / 4.f});
    EXPECT_QUATERNION_EQ(track0.rotations[2].value, r3.x, r3.y, r3.z, r3.w);

    EXPECT_TRUE(anim_track_eq(input.tracks[1], baked.tracks[1]));

    // Motion
    const auto& positions = motion_position.keyframes;
    ASSERT_EQ(positions.size(), 2u);
    EXPECT_FLOAT_EQ(positions[0].ratio, .5f / 46.f);
    EXPECT_FLOAT3_EQ(positions[0].value, 1.f, 0.f, 0.f);
    EXPECT_FLOAT_EQ(positions[1].ratio, 5.f / 46.f);
    EXPECT_FLOAT3_EQ(positions[1].value, 4.f, 0.f, 0.f);

    const auto& rotations = motion_rotation.keyframes;
    ASSERT_EQ(rotations.size(), 3u);
    EXPECT_FLOAT_EQ(rotations[0].ratio, 1.f / 46.f);
    auto mr1 = ozz::math::Quaternion::FromEuler({ozz::math::kPi / 2.f, 0, 0});
    EXPECT_QUATERNION_EQ(rotations[0].value, mr1.x, mr1.y, mr1.z, mr1.w);
    auto mr2 = ozz::math::Quaternion::FromEuler({ozz::math::kPi / 2.f, 0, 0});
    EXPECT_QUATERNION_EQ(rotations[1].value, mr2.x, mr2.y, mr2.z, mr2.w);
    auto mr3 = ozz::math::Quaternion::FromEuler({ozz::math::kPi / 2.f, 0, 0});
    EXPECT_QUATERNION_EQ(rotations[2].value, mr3.x, mr3.y, mr3.z, mr3.w);
  }

  {  // Extract all joint 1, which is not a top level joint. Motion is
     // extracted from the joint model-space transform, so parent joint motion
     // is included.
    const MotionExtractor extractor{
        1,  // Joint
        {true, true, true, MotionExtractor::Reference::kAbsolute, true},
        {true, true, true, MotionExtractor::Reference::kAbsolute, true}};
    EXPECT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &baked));

    // Animation
    EXPECT_EQ(input.name, baked.name);
    EXPECT_FLOAT_EQ(input.duration, baked.duration);
    EXPECT_EQ(input.num_tracks(), baked.num_tracks());

    // Parent (top level) joint track is not modified.
    EXPECT_TRUE(anim_track_eq(input.tracks[0], baked.tracks[0]));

    // Motion keyframes keep the same times and count as the joint local
    // track.
    const auto& positions = motion_position.keyframes;
    ASSERT_EQ(positions.size(), 2u);
    EXPECT_FLOAT_EQ(positions[0].ratio, 2.f / 46.f);
    EXPECT_FLOAT_EQ(positions[1].ratio, 10.f / 46.f);

    const auto& rotations = motion_rotation.keyframes;
    ASSERT_EQ(rotations.size(), 1u);
    EXPECT_FLOAT_EQ(rotations[0].ratio, 23.f / 46.f);

    // Expected model-space transforms of joint 1, computed independently by
    // composing sampled local transforms along the hierarchy.
    const auto expected_model = [&](float _time) {
      ozz::math::Transform parent;
      EXPECT_TRUE(ozz::animation::offline::SampleTrack(input.tracks[0], _time,
                                                       &parent));
      ozz::math::Transform local;
      EXPECT_TRUE(ozz::animation::offline::SampleTrack(input.tracks[1], _time,
                                                       &local));
      return parent * local;
    };

    // Position motion is read from model space, so it differs from the joint
    // local translation and includes parent rotation and translation.
    const auto model_p0 = expected_model(2.f);
    EXPECT_FLOAT3_EQ(positions[0].value, model_p0.translation.x,
                     model_p0.translation.y, model_p0.translation.z);
    const auto model_p1 = expected_model(10.f);
    EXPECT_FLOAT3_EQ(positions[1].value, model_p1.translation.x,
                     model_p1.translation.y, model_p1.translation.z);

    // Rotation motion is the joint model-space rotation.
    const auto model_r0 = expected_model(23.f);
    ExpectQuaternionEqRotation(rotations[0].value, model_r0.rotation);

    // Applying the motion tracks to the character transform reconstructs the
    // joint original model-space pose, at every original animation keyframe
    // time. Translation and rotation are each checked at their own keyframe
    // times, where the baking equations are exact.
    const auto check_reconstruction = [&](float time, bool check_translation) {
      ozz::math::Float3 motion_p;
      ozz::math::Quaternion motion_q;
      ASSERT_TRUE(ozz::animation::offline::SampleTrack(
          motion_position, time / input.duration, &motion_p));
      ASSERT_TRUE(ozz::animation::offline::SampleTrack(
          motion_rotation, time / input.duration, &motion_q));

      ozz::math::Transform baked_local;
      ASSERT_TRUE(ozz::animation::offline::SampleTrack(baked.tracks[1], time,
                                                       &baked_local));
      // Joint 1 parent is joint 0, whose track is untouched.
      ozz::math::Transform baked_parent;
      ASSERT_TRUE(ozz::animation::offline::SampleTrack(baked.tracks[0], time,
                                                       &baked_parent));

      const ozz::math::Transform character{motion_p, motion_q,
                                           ozz::math::Float3::one()};
      const auto reconstructed = character * baked_parent * baked_local;
      const auto original = expected_model(time);

      if (check_translation) {
        EXPECT_NEAR(reconstructed.translation.x, original.translation.x,
                    2e-4f);
        EXPECT_NEAR(reconstructed.translation.y, original.translation.y,
                    2e-4f);
        EXPECT_NEAR(reconstructed.translation.z, original.translation.z,
                    2e-4f);
      } else {
        EXPECT_NEAR(reconstructed.rotation.x, original.rotation.x, 2e-4f);
        EXPECT_NEAR(reconstructed.rotation.y, original.rotation.y, 2e-4f);
        EXPECT_NEAR(reconstructed.rotation.z, original.rotation.z, 2e-4f);
        EXPECT_NEAR(reconstructed.rotation.w, original.rotation.w, 2e-4f);
      }
    };

    // Translation keyframes of the selected joint.
    for (float time : {2.f, 10.f}) {
      check_reconstruction(time, true);
    }
    // Rotation keyframes of the selected joint.
    check_reconstruction(23.f, false);
  }
}

namespace {

// Builds a 3 joints chain skeleton (root -> middle -> leaf), optionally with
// non identity rest poses.
ozz::unique_ptr<Skeleton> BuildChainSkeleton(
    RawSkeleton* _raw_skeleton,
    const ozz::math::Transform& _root_rest = ozz::math::Transform::identity(),
    const ozz::math::Transform& _middle_rest =
        ozz::math::Transform::identity()) {
  _raw_skeleton->roots.resize(1);
  _raw_skeleton->roots[0].transform = _root_rest;
  _raw_skeleton->roots[0].children.resize(1);
  _raw_skeleton->roots[0].children[0].transform = _middle_rest;
  _raw_skeleton->roots[0].children[0].children.resize(1);
  EXPECT_TRUE(_raw_skeleton->Validate());
  return SkeletonBuilder()(*_raw_skeleton);
}

// Samples the model-space transform of _joint of _animation at _time,
// composing local transforms from the skeleton root. Independent from the
// motion extractor implementation.
ozz::math::Transform SampleModel(const RawAnimation& _animation,
                                 const Skeleton& _skeleton, int _joint,
                                 float _time) {
  ozz::vector<int16_t> hierarchy;
  for (int16_t joint = static_cast<int16_t>(_joint);
       joint != Skeleton::kNoParent;
       joint = _skeleton.joint_parents()[joint]) {
    hierarchy.push_back(joint);
  }
  std::reverse(hierarchy.begin(), hierarchy.end());

  ozz::math::Transform model = ozz::math::Transform::identity();
  for (int16_t joint : hierarchy) {
    ozz::math::Transform local;
    EXPECT_TRUE(
        ozz::animation::offline::SampleTrack(_animation.tracks[joint], _time,
                                             &local));
    model = model * local;
  }
  return model;
}

// Checks that applying motion tracks to the character transform reconstructs
// the original model-space pose of _root_joint and its sub-tree. The check
// runs at every original keyframe time of the whole animation. Non top level
// joints are re-keyed on the union of hierarchy and sub-tree keyframe times,
// so every such original keyframe time is an exact baking time.
void ExpectReconstruction(const RawAnimation& _input,
                          const RawAnimation& _baked, const Skeleton& _skeleton,
                          int _root_joint,
                          const RawFloat3Track& _motion_position,
                          const RawQuaternionTrack& _motion_rotation) {
  ozz::vector<float> times;
  for (const auto& track : _input.tracks) {
    for (const auto& key : track.translations) {
      times.push_back(key.time);
    }
    for (const auto& key : track.rotations) {
      times.push_back(key.time);
    }
  }
  std::sort(times.begin(), times.end());
  times.erase(std::unique(times.begin(), times.end()), times.end());

  const auto check = [&](float time) {
    ozz::math::Float3 motion_p;
    ozz::math::Quaternion motion_q;
    ASSERT_TRUE(ozz::animation::offline::SampleTrack(
        _motion_position, time / _input.duration, &motion_p));
    ASSERT_TRUE(ozz::animation::offline::SampleTrack(
        _motion_rotation, time / _input.duration, &motion_q));
    const ozz::math::Transform character{motion_p, motion_q,
                                         ozz::math::Float3::one()};

    for (int joint = _root_joint; joint < _skeleton.num_joints(); ++joint) {
      const auto original = SampleModel(_input, _skeleton, joint, time);
      const auto baked_model = SampleModel(_baked, _skeleton, joint, time);
      const auto reconstructed = character * baked_model;

      EXPECT_NEAR(reconstructed.translation.x, original.translation.x, 1e-4f)
          << "joint " << joint << " time " << time;
      EXPECT_NEAR(reconstructed.translation.y, original.translation.y, 1e-4f)
          << "joint " << joint << " time " << time;
      EXPECT_NEAR(reconstructed.translation.z, original.translation.z, 1e-4f)
          << "joint " << joint << " time " << time;
      EXPECT_NEAR(reconstructed.rotation.x, original.rotation.x, 1e-4f)
          << "joint " << joint << " time " << time;
      EXPECT_NEAR(reconstructed.rotation.y, original.rotation.y, 1e-4f)
          << "joint " << joint << " time " << time;
      EXPECT_NEAR(reconstructed.rotation.z, original.rotation.z, 1e-4f)
          << "joint " << joint << " time " << time;
      EXPECT_NEAR(reconstructed.rotation.w, original.rotation.w, 1e-4f)
          << "joint " << joint << " time " << time;
    }
  };

  for (float time : times) {
    check(time);
  }
}
}  // namespace

// Misaligned parent and selected joint keyframes must not break model-space
// reconstruction, for both translation and rotation, nor hide parent motion.
TEST(ModelSpace, MotionExtractor) {
  RawSkeleton raw_skeleton;
  const auto skeleton = BuildChainSkeleton(&raw_skeleton);
  ASSERT_TRUE(skeleton);

  RawAnimation input;
  input.duration = 10.f;
  input.tracks.resize(3);

  // Root joint keyframes, intentionally not aligned with the middle joint.
  input.tracks[0].translations.push_back({1.f, {1.f, 0.f, 0.f}});
  input.tracks[0].translations.push_back({7.f, {5.f, 0.f, 2.f}});
  input.tracks[0].rotations.push_back(
      {0.f, ozz::math::Quaternion::FromEuler({0.f, 0.f, 0.f})});
  input.tracks[0].rotations.push_back(
      {9.f, ozz::math::Quaternion::FromEuler(
                {ozz::math::kPi / 4.f, 0.f, 0.f})});

  // Middle joint, selected as motion root, with keyframes at different times.
  input.tracks[1].translations.push_back({2.f, {0.f, 1.f, 0.f}});
  input.tracks[1].translations.push_back({8.f, {0.f, 3.f, 1.f}});
  input.tracks[1].rotations.push_back(
      {3.f, ozz::math::Quaternion::FromEuler({0.f, 0.f, 0.f})});
  input.tracks[1].rotations.push_back(
      {6.f, ozz::math::Quaternion::FromEuler(
                {0.f, ozz::math::kPi / 6.f, 0.f})});

  // Leaf joint of the selected joint sub-tree.
  input.tracks[2].translations.push_back({4.f, {0.f, 0.f, 1.f}});
  input.tracks[2].rotations.push_back(
      {5.f, ozz::math::Quaternion::FromEuler({0.f, 0.f,
                                              ozz::math::kPi / 8.f})});

  ASSERT_TRUE(input.Validate());

  MotionExtractor extractor;
  extractor.root_joint = 1;  // Non top level joint.
  extractor.position_settings = {
      true, true, true, MotionExtractor::Reference::kAbsolute, true, false};
  extractor.rotation_settings = {
      true, true, true, MotionExtractor::Reference::kAbsolute, true, false};

  RawFloat3Track motion_position;
  RawQuaternionTrack motion_rotation;
  RawAnimation baked;
  ASSERT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                        &baked));

  // Ancestor joint is not modified, sub-tree joints are not modified: only the
  // selected joint track is baked.
  const auto track_eq = [](const auto& _a, const auto& _b) {
    if (_a.translations.size() != _b.translations.size() ||
        _a.rotations.size() != _b.rotations.size()) {
      return false;
    }
    for (size_t i = 0; i < _a.translations.size(); ++i) {
      if (_a.translations[i].time != _b.translations[i].time ||
          _a.translations[i].value != _b.translations[i].value) {
        return false;
      }
    }
    for (size_t i = 0; i < _a.rotations.size(); ++i) {
      if (_a.rotations[i].time != _b.rotations[i].time ||
          _a.rotations[i].value != _b.rotations[i].value) {
        return false;
      }
    }
    return true;
  };
  EXPECT_TRUE(track_eq(input.tracks[0], baked.tracks[0]));
  EXPECT_TRUE(track_eq(input.tracks[2], baked.tracks[2]));

  // Motion keyframe times match the selected joint keyframe times.
  ASSERT_EQ(motion_position.keyframes.size(), 2u);
  EXPECT_FLOAT_EQ(motion_position.keyframes[0].ratio, 2.f / 10.f);
  EXPECT_FLOAT_EQ(motion_position.keyframes[1].ratio, 8.f / 10.f);
  ASSERT_EQ(motion_rotation.keyframes.size(), 2u);
  EXPECT_FLOAT_EQ(motion_rotation.keyframes[0].ratio, 3.f / 10.f);
  EXPECT_FLOAT_EQ(motion_rotation.keyframes[1].ratio, 6.f / 10.f);

  // The extracted position equals the selected joint model-space
  // translation, including the misaligned parent motion.
  const auto model_p0 = SampleModel(input, *skeleton, 1, 2.f);
  EXPECT_FLOAT3_EQ(motion_position.keyframes[0].value,
                   model_p0.translation.x, model_p0.translation.y,
                   model_p0.translation.z);
  const auto model_p1 = SampleModel(input, *skeleton, 1, 8.f);
  EXPECT_FLOAT3_EQ(motion_position.keyframes[1].value,
                   model_p1.translation.x, model_p1.translation.y,
                   model_p1.translation.z);

  // The extracted rotation equals the selected joint model-space rotation.
  const auto model_r0 = SampleModel(input, *skeleton, 1, 3.f);
  ExpectQuaternionEqRotation(motion_rotation.keyframes[0].value,
                             model_r0.rotation);
  const auto model_r1 = SampleModel(input, *skeleton, 1, 6.f);
  ExpectQuaternionEqRotation(motion_rotation.keyframes[1].value,
                             model_r1.rotation);

  // Model-space pose of the selected joint and its sub-tree is reconstructed
  // at every original keyframe time, even misaligned ones.
  ExpectReconstruction(input, baked, *skeleton, 1, motion_position,
                       motion_rotation);
}

// Component selection semantics are preserved in model space: XZ position
// projection and yaw-only rotation, for a non top level joint.
TEST(ComponentMask, MotionExtractor) {
  RawSkeleton raw_skeleton;
  raw_skeleton.roots.resize(1);
  raw_skeleton.roots[0].children.resize(1);
  ASSERT_TRUE(raw_skeleton.Validate());
  const auto skeleton = SkeletonBuilder()(raw_skeleton);
  ASSERT_TRUE(skeleton);

  RawAnimation input;
  input.duration = 10.f;
  input.tracks.resize(2);

  input.tracks[0].translations.push_back({0.f, {1.f, 2.f, 3.f}});
  input.tracks[0].translations.push_back({10.f, {4.f, 5.f, 6.f}});
  input.tracks[0].rotations.push_back(
      {0.f, ozz::math::Quaternion::FromEuler(
                {0.1f, 0.2f, 0.3f})});
  input.tracks[0].rotations.push_back(
      {10.f, ozz::math::Quaternion::FromEuler(
                {0.4f, 0.5f, 0.6f})});

  input.tracks[1].translations.push_back({1.f, {7.f, 8.f, 9.f}});
  input.tracks[1].translations.push_back({9.f, {10.f, 11.f, 12.f}});
  input.tracks[1].rotations.push_back(
      {2.f, ozz::math::Quaternion::FromEuler(
                {0.2f, 0.3f, 0.4f})});

  ASSERT_TRUE(input.Validate());

  MotionExtractor extractor;
  extractor.root_joint = 1;
  // XZ position only, yaw (y) rotation only, absolute reference, baked.
  extractor.position_settings = {
      true, false, true, MotionExtractor::Reference::kAbsolute, true, false};
  extractor.rotation_settings = {
      false, true, false, MotionExtractor::Reference::kAbsolute, true, false};

  RawFloat3Track motion_position;
  RawQuaternionTrack motion_rotation;
  RawAnimation baked;
  ASSERT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                        &baked));

  // Position motion keeps only X and Z model-space components.
  const auto model_p = SampleModel(input, *skeleton, 1, 1.f);
  EXPECT_FLOAT3_EQ(motion_position.keyframes.front().value,
                   model_p.translation.x, 0.f, model_p.translation.z);

  // Rotation motion is the yaw-only model-space rotation. The extractor
  // rotation "y" setting maps to the yaw component (Euler x).
  const auto model_r = SampleModel(input, *skeleton, 1, 2.f);
  const auto euler = ozz::math::ToEuler(model_r.rotation);
  const auto expected_yaw = ozz::math::Quaternion::FromEuler(
      ozz::math::Float3(euler.x, 0.f, 0.f));
  ExpectQuaternionEqRotation(motion_rotation.keyframes.front().value,
                             expected_yaw);

  // Masked components are still baked so reconstruction is exact for the
  // extracted components, at the selected joint keyframe times.
  ExpectReconstruction(input, baked, *skeleton, 1, motion_position,
                       motion_rotation);
}

// For a non top level joint, components without keyframes, or that are not
// baked, are preserved untouched instead of being re-keyed.
TEST(PreserveComponents, MotionExtractor) {
  RawSkeleton raw_skeleton;
  raw_skeleton.roots.resize(1);
  raw_skeleton.roots[0].children.resize(1);
  ASSERT_TRUE(raw_skeleton.Validate());
  const auto skeleton = SkeletonBuilder()(raw_skeleton);
  ASSERT_TRUE(skeleton);

  RawAnimation input;
  input.duration = 10.f;
  input.tracks.resize(2);

  // Parent joint, misaligned keyframes.
  input.tracks[0].translations.push_back({0.f, {1.f, 0.f, 0.f}});
  input.tracks[0].translations.push_back({10.f, {3.f, 0.f, 0.f}});
  input.tracks[0].rotations.push_back(
      {0.f, ozz::math::Quaternion::identity()});
  input.tracks[0].rotations.push_back(
      {10.f, ozz::math::Quaternion::FromEuler(
                {0.f, ozz::math::kPi / 6.f, 0.f})});

  // Selected non top level joint: translations only, no rotation keyframes.
  input.tracks[1].translations.push_back({1.f, {0.f, 2.f, 0.f}});
  input.tracks[1].translations.push_back({9.f, {0.f, 5.f, 0.f}});

  ASSERT_TRUE(input.Validate());

  {  // Bake translation only: rotation track stays empty, scales untouched.
    MotionExtractor extractor;
    extractor.root_joint = 1;
    extractor.position_settings = {
        true, true, true, MotionExtractor::Reference::kAbsolute, true, false};
    extractor.rotation_settings = {
        false, true, false, MotionExtractor::Reference::kAbsolute, true, false};

    RawFloat3Track motion_position;
    RawQuaternionTrack motion_rotation;
    RawAnimation baked;
    ASSERT_TRUE(extractor(input, *skeleton, &motion_position,
                          &motion_rotation, &baked));

    // No rotation keyframes in the input means none in the baked output and
    // none in the rotation motion track.
    EXPECT_TRUE(baked.tracks[1].rotations.empty());
    EXPECT_TRUE(motion_rotation.keyframes.empty());
    EXPECT_TRUE(KeyframesEqual(input.tracks[1].scales,
                               baked.tracks[1].scales));
    // Translations are re-keyed on the hierarchy keyframe times.
    EXPECT_EQ(baked.tracks[1].translations.size(), 4u);
    for (const auto& key : baked.tracks[1].translations) {
      EXPECT_TRUE(key.time == 0.f || key.time == 1.f || key.time == 9.f ||
                  key.time == 10.f);
    }
  }

  {  // No baking at all: the selected joint track is preserved untouched.
    MotionExtractor extractor;
    extractor.root_joint = 1;
    extractor.position_settings = {
        true, true, true, MotionExtractor::Reference::kAbsolute, false, false};
    extractor.rotation_settings = {
        false, true, false, MotionExtractor::Reference::kAbsolute, false,
        false};

    RawFloat3Track motion_position;
    RawQuaternionTrack motion_rotation;
    RawAnimation baked;
    ASSERT_TRUE(extractor(input, *skeleton, &motion_position,
                          &motion_rotation, &baked));

    EXPECT_TRUE(JointTrackEqual(input.tracks[0], baked.tracks[0]));
    EXPECT_TRUE(JointTrackEqual(input.tracks[1], baked.tracks[1]));
  }
}

// Loop mode distributes the end to begin delta on motion tracks only, without
// changing the baked animation character-space pose.
TEST(Loop, MotionExtractor) {
  RawSkeleton raw_skeleton;
  raw_skeleton.roots.resize(1);
  raw_skeleton.roots[0].children.resize(1);
  ASSERT_TRUE(raw_skeleton.Validate());
  const auto skeleton = SkeletonBuilder()(raw_skeleton);
  ASSERT_TRUE(skeleton);

  RawAnimation input;
  input.duration = 10.f;
  input.tracks.resize(2);

  input.tracks[0].translations.push_back({0.f, {1.f, 0.f, 0.f}});
  input.tracks[0].translations.push_back({10.f, {4.f, 0.f, 0.f}});
  input.tracks[0].rotations.push_back(
      {0.f, ozz::math::Quaternion::identity()});
  input.tracks[0].rotations.push_back(
      {10.f, ozz::math::Quaternion::FromEuler(
                {ozz::math::kPi / 3.f, 0.f, 0.f})});

  // Non top level selected joint, with its own misaligned keyframes.
  input.tracks[1].translations.push_back({1.f, {0.f, 2.f, 0.f}});
  input.tracks[1].translations.push_back({9.f, {0.f, 5.f, 0.f}});
  input.tracks[1].rotations.push_back(
      {2.f, ozz::math::Quaternion::identity()});
  input.tracks[1].rotations.push_back(
      {8.f, ozz::math::Quaternion::FromEuler(
                {0.f, ozz::math::kPi / 5.f, 0.f})});

  ASSERT_TRUE(input.Validate());

  RawFloat3Track motion_position;
  RawQuaternionTrack motion_rotation;
  RawAnimation baked_loop;
  RawAnimation baked_no_loop;

  MotionExtractor loop_extractor;
  loop_extractor.root_joint = 1;
  loop_extractor.position_settings = {
      true, true, true, MotionExtractor::Reference::kAbsolute, true, true};
  loop_extractor.rotation_settings = {
      true, true, true, MotionExtractor::Reference::kAbsolute, true, true};
  ASSERT_TRUE(loop_extractor(input, *skeleton, &motion_position,
                             &motion_rotation, &baked_loop));

  // Looping only affects motion tracks: the baked animation is identical with
  // or without loop enabled.
  MotionExtractor no_loop_extractor = loop_extractor;
  no_loop_extractor.position_settings.loop = false;
  no_loop_extractor.rotation_settings.loop = false;
  RawFloat3Track no_loop_position;
  RawQuaternionTrack no_loop_rotation;
  ASSERT_TRUE(no_loop_extractor(input, *skeleton, &no_loop_position,
                                &no_loop_rotation, &baked_no_loop));

  ASSERT_EQ(baked_loop.tracks.size(), baked_no_loop.tracks.size());
  for (size_t j = 0; j < baked_loop.tracks.size(); ++j) {
    ASSERT_EQ(baked_loop.tracks[j].translations.size(),
              baked_no_loop.tracks[j].translations.size());
    ASSERT_EQ(baked_loop.tracks[j].rotations.size(),
              baked_no_loop.tracks[j].rotations.size());
    for (size_t i = 0; i < baked_loop.tracks[j].translations.size(); ++i) {
      EXPECT_FLOAT3_EQ(baked_loop.tracks[j].translations[i].value,
                       baked_no_loop.tracks[j].translations[i].value.x,
                       baked_no_loop.tracks[j].translations[i].value.y,
                       baked_no_loop.tracks[j].translations[i].value.z);
      EXPECT_FLOAT_EQ(baked_loop.tracks[j].translations[i].time,
                      baked_no_loop.tracks[j].translations[i].time);
    }
    for (size_t i = 0; i < baked_loop.tracks[j].rotations.size(); ++i) {
      ExpectQuaternionEqRotation(baked_loop.tracks[j].rotations[i].value,
                                 baked_no_loop.tracks[j].rotations[i].value);
    }
  }

  // First and last motion keyframes are equal, so the motion loops.
  const auto& lp = motion_position.keyframes;
  ASSERT_GE(lp.size(), 2u);
  EXPECT_FLOAT3_EQ(lp.front().value, lp.back().value.x, lp.back().value.y,
                   lp.back().value.z);
  const auto& lr = motion_rotation.keyframes;
  ASSERT_GE(lr.size(), 2u);
  ExpectQuaternionEqRotation(lr.front().value, lr.back().value);

  // The motion delta distributed over the whole clip equals the original end
  // to begin difference: the non-looped first keyframe is untouched, while the
  // non-looped last keyframe is shifted by the whole delta.
  const ozz::math::Float3 position_delta =
      no_loop_position.keyframes.front().value -
      no_loop_position.keyframes.back().value;
  for (size_t i = 0; i < no_loop_position.keyframes.size(); ++i) {
    const float alpha =
        i / (no_loop_position.keyframes.size() - 1.f);
    const ozz::math::Float3 expected =
        no_loop_position.keyframes[i].value + position_delta * alpha;
    EXPECT_NEAR(motion_position.keyframes[i].value.x, expected.x, 1e-5f);
    EXPECT_NEAR(motion_position.keyframes[i].value.y, expected.y, 1e-5f);
    EXPECT_NEAR(motion_position.keyframes[i].value.z, expected.z, 1e-5f);
  }
}

// References are computed from the selected joint model-space transform.
TEST(Reference, MotionExtractor) {
  const ozz::math::Transform root_rest{{1.f, 2.f, 3.f},
                                       ozz::math::Quaternion::FromEuler(
                                           {ozz::math::kPi / 6.f, 0.f, 0.f}),
                                       ozz::math::Float3::one()};
  const ozz::math::Transform middle_rest{{0.f, 1.f, 0.f},
                                         ozz::math::Quaternion::FromEuler(
                                             {0.f, ozz::math::kPi / 4.f, 0.f}),
                                         ozz::math::Float3::one()};
  RawSkeleton raw_skeleton;
  raw_skeleton.roots.resize(1);
  raw_skeleton.roots[0].transform = root_rest;
  raw_skeleton.roots[0].children.resize(1);
  raw_skeleton.roots[0].children[0].transform = middle_rest;
  ASSERT_TRUE(raw_skeleton.Validate());
  const auto skeleton = SkeletonBuilder()(raw_skeleton);
  ASSERT_TRUE(skeleton);

  RawAnimation input;
  input.duration = 10.f;
  input.tracks.resize(2);

  input.tracks[0].translations.push_back({0.f, {2.f, 0.f, 0.f}});
  input.tracks[0].translations.push_back({10.f, {6.f, 0.f, 0.f}});
  input.tracks[0].rotations.push_back(
      {0.f, ozz::math::Quaternion::FromEuler({0.f, 0.f, 0.f})});
  input.tracks[0].rotations.push_back(
      {10.f, ozz::math::Quaternion::FromEuler(
                {ozz::math::kPi / 8.f, 0.f, 0.f})});

  input.tracks[1].translations.push_back({1.f, {0.f, 2.f, 0.f}});
  input.tracks[1].translations.push_back({9.f, {0.f, 7.f, 0.f}});
  input.tracks[1].rotations.push_back(
      {2.f, ozz::math::Quaternion::FromEuler({0.f, 0.f, 0.f})});
  input.tracks[1].rotations.push_back(
      {8.f, ozz::math::Quaternion::FromEuler(
                {0.f, ozz::math::kPi / 7.f, 0.f})});

  ASSERT_TRUE(input.Validate());

  // Independent model-space references. Position and rotation are sampled at
  // their own keyframe times, matching the per-component extraction logic.
  const auto rest_model = root_rest * middle_rest;
  const auto first_model = SampleModel(input, *skeleton, 1, 1.f);
  const auto last_model = SampleModel(input, *skeleton, 1, 9.f);
  const auto first_rot_model = SampleModel(input, *skeleton, 1, 2.f);
  const auto last_rot_model = SampleModel(input, *skeleton, 1, 8.f);

  {  // Animation reference: first keyframe motion is zero.
    MotionExtractor extractor;
    extractor.root_joint = 1;
    extractor.position_settings = {
        true, true, true, MotionExtractor::Reference::kAnimation, true, false};
    extractor.rotation_settings = {
        true, true, true, MotionExtractor::Reference::kAnimation, true, false};

    RawFloat3Track motion_position;
    RawQuaternionTrack motion_rotation;
    RawAnimation baked;
    ASSERT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &baked));

    EXPECT_FLOAT3_EQ(motion_position.keyframes.front().value, 0.f, 0.f, 0.f);
    EXPECT_QUATERNION_EQ(motion_rotation.keyframes.front().value, 0.f, 0.f,
                         0.f, 1.f);

    const ozz::math::Float3 expected_last_p =
        last_model.translation - first_model.translation;
    EXPECT_FLOAT3_EQ(motion_position.keyframes.back().value,
                     expected_last_p.x, expected_last_p.y, expected_last_p.z);
    const auto expected_last_q = ExtractRotationAll(
        last_rot_model.rotation *
        ozz::math::Conjugate(first_rot_model.rotation));
    ExpectQuaternionEqRotation(motion_rotation.keyframes.back().value,
                               expected_last_q);

    ExpectReconstruction(input, baked, *skeleton, 1, motion_position,
                         motion_rotation);
  }

  {  // Skeleton reference: motion is relative to the model-space rest pose.
    MotionExtractor extractor;
    extractor.root_joint = 1;
    extractor.position_settings = {
        true, true, true, MotionExtractor::Reference::kSkeleton, true, false};
    extractor.rotation_settings = {
        true, true, true, MotionExtractor::Reference::kSkeleton, true, false};

    RawFloat3Track motion_position;
    RawQuaternionTrack motion_rotation;
    RawAnimation baked;
    ASSERT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &baked));

    const ozz::math::Float3 expected_first_p =
        first_model.translation - rest_model.translation;
    EXPECT_FLOAT3_EQ(motion_position.keyframes.front().value,
                     expected_first_p.x, expected_first_p.y,
                     expected_first_p.z);
    const auto expected_first_q = ExtractRotationAll(
        first_rot_model.rotation *
        ozz::math::Conjugate(rest_model.rotation));
    ExpectQuaternionEqRotation(motion_rotation.keyframes.front().value,
                               expected_first_q);

    ExpectReconstruction(input, baked, *skeleton, 1, motion_position,
                         motion_rotation);
  }

  {  // Absolute reference: motion is the model-space transform itself.
    MotionExtractor extractor;
    extractor.root_joint = 1;
    extractor.position_settings = {
        true, true, true, MotionExtractor::Reference::kAbsolute, true, false};
    extractor.rotation_settings = {
        true, true, true, MotionExtractor::Reference::kAbsolute, true, false};

    RawFloat3Track motion_position;
    RawQuaternionTrack motion_rotation;
    RawAnimation baked;
    ASSERT_TRUE(extractor(input, *skeleton, &motion_position, &motion_rotation,
                          &baked));

    EXPECT_FLOAT3_EQ(motion_position.keyframes.front().value,
                     first_model.translation.x, first_model.translation.y,
                     first_model.translation.z);
    ExpectQuaternionEqRotation(
        motion_rotation.keyframes.front().value,
        ExtractRotationAll(first_rot_model.rotation));

    ExpectReconstruction(input, baked, *skeleton, 1, motion_position,
                         motion_rotation);
  }
}
