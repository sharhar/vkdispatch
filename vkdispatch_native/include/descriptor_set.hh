#ifndef SRC_DESCRIPTOR_SET_H
#define SRC_DESCRIPTOR_SET_H

#include "base.hh"

struct DescriptorSet* descriptor_set_create_extern(struct ComputePlan* plan);
void descriptor_set_destroy_extern(struct DescriptorSet* descriptor_set);

void descriptor_set_write_buffer_extern(struct DescriptorSet* descriptor_set, unsigned int binding, void* object, unsigned long long offset, unsigned long long range, int uniform);
void descriptor_set_write_image_extern(struct DescriptorSet* descriptor_set, unsigned int binding, void* object, void* sampler_obj);

#endif // SRC_DESCRIPTOR_SET_H