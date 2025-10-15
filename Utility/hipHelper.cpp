#pragma once
#include <thrust/device_vector.h>



template<typename T>
static T* GetPointer(thrust::device_vector<T> &vector)
{
    return thrust::raw_pointer_cast(&vector[0]);
}

template<typename T>
static T* GetPointer(thrust::device_vector<T> &vector, int index)
{
    return thrust::raw_pointer_cast(&vector[index]);
}
