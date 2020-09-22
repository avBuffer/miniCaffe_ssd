#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>
#include "caffe/common.hpp"


namespace caffe {
  
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
  free(ptr);
}


/* @brief Manages memory allocation and synchronization between the host (CPU).
 * TODO(dox): more thorough description. */
class SyncedMemory {
 public:
  SyncedMemory() : cpu_ptr_(NULL), size_(0), head_(UNINITIALIZED), own_cpu_data_(false), cpu_malloc_use_cuda_(false) {}
  explicit SyncedMemory(size_t size) : cpu_ptr_(NULL), size_(size), head_(UNINITIALIZED), own_cpu_data_(false), 
                                       cpu_malloc_use_cuda_(false) {}
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  void* mutable_cpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

 private:
  void to_cpu();
  void* cpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
 
  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe
#endif  // CAFFE_SYNCEDMEM_HPP_
