
#version 430
layout(local_size_x=1) in;
layout(std430, binding=10) buffer SparseIndirect {
    uint sparse_groups_x;
    uint sparse_groups_y;
    uint sparse_groups_z;
};
void main() {
    sparse_groups_x = 0u;
    sparse_groups_y = 1u;
    sparse_groups_z = 1u;
}
