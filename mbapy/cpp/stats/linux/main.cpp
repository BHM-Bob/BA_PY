/*

*/
#include <iostream>
#include <vector>
#include <new>
#include <tuple>

extern "C" {

    unsigned long long* remove_simi(float* arr, unsigned long long* len, float sh);

    void freePtr(void* ptr);
}


unsigned long long* remove_simi(float* arr, unsigned long long* len, float sh)
{
    std::vector<unsigned long long> to_remove_idx = std::vector<unsigned long long>();
    for (unsigned long long i = 0; i < *len - 1; i++) //start from the second number
    {
        if (arr[i + 1] - arr[i] < sh)
        {
            arr[i + 1] = arr[i];
            to_remove_idx.emplace_back(i + 1);
        }
    }
    *len = to_remove_idx.size();
    unsigned long long* result = new unsigned long long[to_remove_idx.size()];
    std::copy(to_remove_idx.begin(), to_remove_idx.end(), result);
    return result;
}

void freePtr(void* ptr)
{
    free(ptr);
}
