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

#include "ozz/animation/offline/motion_extractor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "ozz/animation/offline/raw_animation.h"
#include "ozz/animation/offline/raw_animation_utils.h"
#include "ozz/animation/offline/raw_track.h"
#include "ozz/animation/offline/raw_track_utils.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/animation/runtime/skeleton_utils.h"
#include "ozz/base/containers/vector.h"
#include "ozz/base/maths/transform.h"

namespace ozz {
namespace animation {
namespace offline {

namespace {

// Builds the chain of joint indices, ordered from the skeleton root down to
// _joint (included).
ozz::vector<int16_t> BuildHierarchy(const Skeleton& _skeleton, int _joint) {
  ozz::vector<int16_t> hierarchy;
  for (int16_t joint = static_cast<int16_t>(_joint);
       joint != Skeleton::kNoParent;
       joint = _skeleton.joint_parents()[joint]) {
    hierarchy.push_back(joint);
  }
  std::reverse(hierarchy.begin(), hierarchy.end());
  return hierarchy;
}

// Samples the model-space transform of the last joint of _hierarchy at _time,
// walking the joint hierarchy from the skeleton root. Every joint is sampled
// at the requested time, so parent motion is taken into account even when
// parent and selected joint keyframes are not aligned.
bool SampleModelTransform(const RawAnimation& _animation,
                          const ozz::span<const int16_t>& _hierarchy,
                          float _time, ozz::math::Transform* _transform) {
  *_transform = ozz::math::Transform::identity();
  for (int joint : _hierarchy) {
    ozz::math::Transform local;
    if (!SampleTrack(_animation.tracks[joint], _time, &local)) {
      return false;
    }
    *_transform = *_transform * local;
  }
  return true;
}

// Computes the model-space rest pose transform of the last joint of
// _hierarchy.
ozz::math::Transform RestPoseModelTransform(
    const Skeleton& _skeleton, const ozz::span<const int16_t>& _hierarchy) {
  ozz::math::Transform model = ozz::math::Transform::identity();
  for (int joint : _hierarchy) {
    model = model * GetJointRestPoseLocalSpace(_skeleton, joint);
  }
  return model;
}

// Samples the model-space transform of the parent of the selected joint.
// Returns identity if the selected joint has no parent.
bool SampleParentModelTransform(
    const RawAnimation& _animation,
    const ozz::span<const int16_t>& _hierarchy, float _time,
    ozz::math::Transform* _transform) {
  *_transform = ozz::math::Transform::identity();
  // The last hierarchy entry is the selected joint itself.
  for (size_t h = 0; h + 1 < _hierarchy.size(); ++h) {
    ozz::math::Transform local;
    if (!SampleTrack(_animation.tracks[_hierarchy[h]], _time, &local)) {
      return false;
    }
    *_transform = *_transform * local;
  }
  return true;
}

// Builds the model-space value used as extraction reference for one component
// (position or rotation), based on its own reference setting. _rest_model is
// the selected joint model-space rest pose component, _first_model its
// model-space value at the component first keyframe.
math::Float3 BuildComponentReference(MotionExtractor::Reference _reference,
                                     const math::Float3& _rest_component,
                                     const math::Float3& _first_component) {
  switch (_reference) {
    case MotionExtractor::Reference::kSkeleton:
      return _rest_component;
    case MotionExtractor::Reference::kAnimation:
      return _first_component;
    default:  // kAbsolute
      return math::Float3::zero();
  }
}

math::Quaternion BuildComponentReference(
    MotionExtractor::Reference _reference,
    const math::Quaternion& _rest_component,
    const math::Quaternion& _first_component) {
  switch (_reference) {
    case MotionExtractor::Reference::kSkeleton:
      return _rest_component;
    case MotionExtractor::Reference::kAnimation:
      return _first_component;
    default:  // kAbsolute
      return math::Quaternion::identity();
  }
}
}  // namespace

bool MotionExtractor::operator()(const RawAnimation& _input,
                                 const Skeleton& _skeleton,
                                 RawFloat3Track* _motion_position,
                                 RawQuaternionTrack* _motion_rotation,
                                 RawAnimation* _output) const {
  // Cannot read/write from/to the same animation.
  if (&_input == _output) {
    return false;
  }

  // All outputs are expected to be valid.
  if (!_output || !_motion_position || !_motion_rotation) {
    return false;
  }

  // Animation must match skeleton.
  if (_input.num_tracks() != _skeleton.num_joints()) {
    return false;
  }

  // Root index must be within skeleton range.
  if (root_joint < 0 || root_joint >= _skeleton.num_joints()) {
    return false;
  }

  // Validate animation.
  if (!_input.Validate()) {
    return false;
  }

  // Copy output animation
  *_output = _input;

  // Track to extract motion from
  const auto& input_track = _input.tracks[root_joint];
  auto& output_track = _output->tracks[root_joint];

  // Joint hierarchy, from the skeleton root down to the selected joint.
  const auto hierarchy = BuildHierarchy(_skeleton, root_joint);
  const ozz::span<const int16_t> hierarchy_span(hierarchy.data(),
                                                hierarchy.size());

  // Model-space rest pose of the selected joint, used by skeleton references.
  const auto rest_model = RestPoseModelTransform(_skeleton, hierarchy_span);

  // Animation references: model-space transform of the selected joint at the
  // first keyframe of each component. A component without any keyframe keeps
  // using its rest pose, as there is no animated first frame to reference.
  math::Float3 first_translation = rest_model.translation;
  if (!input_track.translations.empty()) {
    ozz::math::Transform first_model;
    if (!SampleModelTransform(_input, hierarchy_span,
                              input_track.translations.front().time,
                              &first_model)) {
      return false;
    }
    first_translation = first_model.translation;
  }

  math::Quaternion first_rotation = rest_model.rotation;
  if (!input_track.rotations.empty()) {
    ozz::math::Transform first_model;
    if (!SampleModelTransform(_input, hierarchy_span,
                              input_track.rotations.front().time,
                              &first_model)) {
      return false;
    }
    first_rotation = first_model.rotation;
  }

  // Compute model-space extraction references.
  const math::Float3 position_reference = BuildComponentReference(
      position_settings.reference, rest_model.translation, first_translation);
  const math::Quaternion rotation_reference = BuildComponentReference(
      rotation_settings.reference, rest_model.rotation, first_rotation);

  // Extract root motion
  // -----------------------------------------------------------------------------

  // Copies animation keyframes to motion keyframes.
  auto extract = [duration = _input.duration](const auto& _keyframes,
                                              auto _extract, auto& output) {
    output.clear();
    for (const auto& joint_key : _keyframes) {
      const auto& motion = _extract(joint_key.time);
      output.push_back({ozz::animation::offline::RawTrackInterpolation::kLinear,
                        joint_key.time / duration, motion});
    }
  };

  // Samples the selected joint model-space transform at a keyframe time.
  auto model_at = [&](float _time) {
    ozz::math::Transform model;
    if (!SampleModelTransform(_input, hierarchy_span, _time, &model)) {
      // Animation has been validated above, sampling a valid track cannot fail.
      assert(false && "Failed to sample model-space transform.");
    }
    return model;
  };

  // Copies root position, selecting only expected components. Position is read
  // from the selected joint model-space transform.
  const math::Float3 position_mask{1.f * position_settings.x,
                                   1.f * position_settings.y,
                                   1.f * position_settings.z};
  extract(
      input_track.translations,
      [&](float _time) {
        const auto model = model_at(_time);
        return (model.translation - position_reference) * position_mask;
      },
      _motion_position->keyframes);

  // Copies root rotation, selecting only expected decomposed rotation
  // components. Rotation is read from the selected joint model-space
  // transform.
  const math::Float3 rotation_mask{1.f * rotation_settings.y,   // Yaw
                                   1.f * rotation_settings.x,   // Pitch
                                   1.f * rotation_settings.z};  // Roll
  extract(
      input_track.rotations,
      [&](float _time) {
        const auto model = model_at(_time);
        const auto euler =
            ToEuler(model.rotation * Conjugate(rotation_reference));
        return math::Quaternion::FromEuler(euler * rotation_mask);
      },
      _motion_rotation->keyframes);

  // Bake
  // -----------------------------------------------------------------------------
  // Baking removes the extracted motion from the selected joint local track,
  // so that applying the motion tracks to the character transform reconstructs
  // the original model-space pose of the selected joint and of every joint in
  // its sub-tree. Ancestor joints' local tracks are left untouched.
  //
  // For a selected joint with parent model-space transform P, local transform
  // L (so joint model transform is P * L) and an extracted motion transform M,
  // the baked local transform L' must satisfy:
  //   M * P * L' = P * L
  // Rotation and translation are baked separately to preserve the component
  // selection semantics.

  // A top level joint has no parent: its local transform is already its model
  // transform, and the original per-component baking path applies unchanged.
  const bool has_parent = hierarchy.size() > 1;

  if (!has_parent) {
    // Extract root motion rotation from the animation, aka bake it.
    if (rotation_settings.bake) {
      assert(output_track.rotations.size() ==
             _motion_rotation->keyframes.size());
      for (size_t i = 0; i < output_track.rotations.size(); i++) {
        const float time = output_track.rotations[i].time;
        const auto& motion_q = _motion_rotation->keyframes[i].value;
        const auto original_q = input_track.rotations[i].value;

        // Model-space rotation of the parent of the selected joint.
        ozz::math::Transform parent_model;
        if (!SampleParentModelTransform(_input, hierarchy_span, time,
                                        &parent_model)) {
          return false;
        }

        // Removes the selected motion rotation from the joint model-space
        // rotation, then converts the result back to local space. The
        // original local rotation is always used as input, independently of
        // other bake stages.
        const math::Quaternion model_q = parent_model.rotation * original_q;
        const math::Quaternion baked_model_q =
            Conjugate(motion_q) * model_q;
        output_track.rotations[i].value =
            Conjugate(parent_model.rotation) * baked_model_q;
      }
    }

    // Extract root motion position from the animation, aka bake it. The
    // motion position is removed from the selected joint model-space
    // translation.
    if (position_settings.bake) {
      assert(output_track.translations.size() ==
             _motion_position->keyframes.size());
      for (size_t i = 0; i < output_track.translations.size(); i++) {
        const float time = output_track.translations[i].time;
        const auto& motion_p = _motion_position->keyframes[i].value;
        const auto original_p = input_track.translations[i].value;

        // Model-space transform of the parent of the selected joint.
        ozz::math::Transform parent_model;
        if (!SampleParentModelTransform(_input, hierarchy_span, time,
                                        &parent_model)) {
          return false;
        }

        // The selected joint model-space translation is the parent
        // model-space transform applied to its original local translation.
        // Baking always uses original animation data, so it is independent of
        // other bake stages.
        const math::Float3 model_p =
            parent_model.translation +
            TransformVector(parent_model.rotation, original_p);
        const math::Float3 baked_model_p = model_p - motion_p;
        output_track.translations[i].value = TransformVector(
            Conjugate(parent_model.rotation),
            baked_model_p - parent_model.translation);
      }
    }
  } else if (rotation_settings.bake || position_settings.bake) {
    // Non top level joint. The baked local transform depends on the parent
    // model-space transform, which keeps changing between the selected joint
    // keyframes as parent keyframes are not necessarily aligned. To keep the
    // model-space pose exact at every original keyframe time of the hierarchy
    // and of the sub-tree, the selected joint is re-keyed on the union of all
    // those keyframe times, every baked value being computed from consistent
    // same-time samples.
    ozz::vector<float> bake_times;
    const auto collect_times = [&bake_times](const RawAnimation::JointTrack&
                                                 _track) {
      for (const auto& key : _track.translations) {
        bake_times.push_back(key.time);
      }
      for (const auto& key : _track.rotations) {
        bake_times.push_back(key.time);
      }
    };

    // Ancestor hierarchy joints, including the selected joint.
    for (int16_t joint : hierarchy_span) {
      collect_times(_input.tracks[joint]);
    }
    // Every joint of the selected joint sub-tree, so their pose stays exact at
    // their own keyframe times as well.
    IterateJointsDF(
        _skeleton,
        [&](int _joint, int) {
          if (_joint != root_joint) {
            collect_times(_input.tracks[_joint]);
          }
        },
        root_joint);

    std::sort(bake_times.begin(), bake_times.end());
    bake_times.erase(std::unique(bake_times.begin(), bake_times.end()),
                     bake_times.end());

    RawAnimation::JointTrack baked_track;
    // Components that are not baked, or that have no keyframe in the input,
    // are preserved untouched. Only baked components with keyframes are
    // re-keyed on the expanded time set.
    if (!rotation_settings.bake) {
      baked_track.rotations = input_track.rotations;
    }
    if (!position_settings.bake) {
      baked_track.translations = input_track.translations;
    }
    baked_track.scales = input_track.scales;

    for (float time : bake_times) {
      // Consistent same-time samples of the input joint and its parent.
      ozz::math::Transform original_local;
      if (!SampleTrack(input_track, time, &original_local)) {
        return false;
      }
      ozz::math::Transform parent_model;
      if (!SampleParentModelTransform(_input, hierarchy_span, time,
                                      &parent_model)) {
        return false;
      }

      // Motion transform at this time.
      const float ratio = time / _input.duration;
      math::Float3 motion_p;
      math::Quaternion motion_q;
      if (!SampleTrack(*_motion_position, ratio, &motion_p) ||
          !SampleTrack(*_motion_rotation, ratio, &motion_q)) {
        return false;
      }

      // Bake rotation: removes the motion rotation from the joint model-space
      // rotation, then converts back to local space.
      if (rotation_settings.bake && !input_track.rotations.empty()) {
        const math::Quaternion model_q =
            parent_model.rotation * original_local.rotation;
        const math::Quaternion baked_model_q =
            Conjugate(motion_q) * model_q;
        baked_track.rotations.push_back(
            {time, Conjugate(parent_model.rotation) * baked_model_q});
      }

      // Bake translation: removes the motion position from the joint
      // model-space translation.
      if (position_settings.bake && !input_track.translations.empty()) {
        math::Float3 baked_translation = original_local.translation;
        const math::Float3 model_p =
            parent_model.translation +
            TransformVector(parent_model.rotation, original_local.translation);
        const math::Float3 baked_model_p = model_p - motion_p;
        baked_translation = TransformVector(
            Conjugate(parent_model.rotation),
            baked_model_p - parent_model.translation);

        // Fixup translation: at runtime the motion rotation is applied before
        // the motion position, hence around the model-space origin. The baked
        // translation is corrected so the motion rotation also acts on the
        // joint model-space translation.
        if (rotation_settings.bake) {
          const math::Float3 fixup_model_p =
              parent_model.translation +
              TransformVector(parent_model.rotation, baked_translation);
          const math::Float3 fixed_model_p =
              TransformVector(Conjugate(motion_q), fixup_model_p);
          baked_translation = TransformVector(
              Conjugate(parent_model.rotation),
              fixed_model_p - parent_model.translation);
        }
        baked_track.translations.push_back({time, baked_translation});
      }
    }
    output_track = std::move(baked_track);
  }

  // Loopify
  // -----------------------------------------------------------------------------
  // Distributes the difference between the first and last keyframes all along
  // animation duration, so the animation can loop. Only motion tracks are
  // modified: the baked animation is untouched, hence the character-space pose
  // is preserved while the whole motion delta is spread over character motion.
  auto loopify = [](auto& _keyframes, auto _diff, auto _lerp) {
    if (_keyframes.size() < 2) {
      return;
    }
    const auto delta = _diff(_keyframes.front().value, _keyframes.back().value);
    for (size_t i = 0; i < _keyframes.size(); i++) {
      const float alpha = i / (_keyframes.size() - 1.f);
      auto& value = _keyframes[i].value;
      value = _lerp(value, delta, alpha);
    }
  };

  // Loopify rotations
  if (rotation_settings.loop) {
    loopify(
        _motion_rotation->keyframes,
        [](auto _front, auto _back) { return _front * Conjugate(_back); },
        [](auto _value, auto _delta, float _alpha) {
          return NLerp(math::Quaternion::identity(), _delta, _alpha) * _value;
        });
  }

  // Loopify translations
  if (position_settings.loop) {
    loopify(
        _motion_position->keyframes,
        [](auto _front, auto _back) { return _front - _back; },
        [](auto _value, auto _delta, float _alpha) {
          return _delta * _alpha + _value;
        });
  }

  // Fixup animation translations.
  // -----------------------------------------------------------------------------
  // At runtime the motion rotation is applied before the motion position,
  // hence around the model-space origin. The baked joint translation only
  // compensates the motion position, so it has to be corrected to also
  // compensate the motion rotation acting on the joint model-space
  // translation. This is only needed for the top level path, as the non top
  // level path applies the correction while re-keying the joint track.
  if (!has_parent &&
      rotation_settings.bake) {  // Considers that if rotation is baked, then
                                 // motion rotation will be applied
    for (size_t i = 0; i < output_track.translations.size(); i++) {
      const auto& motion_p_key = _motion_position->keyframes[i];
      auto& joint_p = output_track.translations[i].value;

      // Sample motion rotation (as it might not have the same number of
      // keyframes as translations)
      math::Quaternion motion_q;
      if (!SampleTrack(*_motion_rotation, motion_p_key.ratio, &motion_q)) {
        return false;
      }

      // Model-space transform of the parent of the selected joint.
      ozz::math::Transform parent_model;
      if (!SampleParentModelTransform(
              _input, hierarchy_span,
              motion_p_key.ratio * _input.duration, &parent_model)) {
        return false;
      }

      // Joint model-space translation before the fixup, computed from its
      // (possibly position baked) local translation.
      const math::Float3 model_p =
          parent_model.translation +
          TransformVector(parent_model.rotation, joint_p);

      // Applies the inverse motion rotation around the model-space origin,
      // then converts the corrected model-space translation back to local
      // space. For a top level joint the parent model transform is identity,
      // which is equivalent to the former behavior.
      const math::Float3 fixed_model_p =
          TransformVector(Conjugate(motion_q), model_p);
      joint_p = TransformVector(Conjugate(parent_model.rotation),
                                fixed_model_p - parent_model.translation);
    }
  }

  // Validate outputs
  bool success = true;
  success &= _motion_position->Validate();
  success &= _motion_rotation->Validate();
  success &= _output->Validate();

  return success;
}

}  // namespace offline
}  // namespace animation
}  // namespace ozz
