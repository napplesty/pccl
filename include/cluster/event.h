#pragma once

namespace engine_c {

enum class ClusterEventType : int {
  ResizeWorld,
  JoinSelf,
  QuitSelf,
  ReplaceOld,
  BeReplaced,
  JoinNode,
  QuitNode,
  UpdateNode,
};

}