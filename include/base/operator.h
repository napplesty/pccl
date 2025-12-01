#pragma once

#include <base/registry.h>

namespace engine_c {

struct RingBuffer {
  ExecutorType producer_;
  ExecutorType consumer_;
  int elem_size_;
  int max_num_elems_;
  int max_flush_time_;
  int tail_;
  int head_;
  void *producer_buffer_;
  void *consumer_buffer_;
};

struct OperationLayoutHeader {
  int op_uid_;
  int required_executors_;
  int remaining_executors_;
  unsigned char pre_dependencies_;
  ExecutorType executor_type_;
  unsigned char op_size_;
  unsigned char num_next_ops_;
  void *op_buffer_;
  int *next_op_uids_head_;
};

struct GraphBufferLayout {
  unsigned int *completed_operator;
  unsigned int *total_operator;
  OperationLayoutHeader **operators;
  unsigned int num_operators;
  RingBuffer **ready_queues;
  unsigned int num_queues;
};

class OperatorManager {
public:
  OperatorManager();

  
};

}