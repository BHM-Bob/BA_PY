/*
 * @Date: 2023-08-22 23:30:32
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-01-09 23:03:10
 * @Description: 
 */
/*

*/
#include <cstdlib>
#include <new>
#include <vector>

extern "C" {

    unsigned long long* remove_simi(float* arr, unsigned long long* len, float sh);

    void freePtr(void* ptr);
}


/**
 * Removes similar elements from an array and returns the indices of the removed elements.
 *
 * @param arr Pointer to the array of float values
 * @param len Pointer to the length of the array
 * @param sh Threshold value for similarity check
 *
 * @return Pointer to the array of unsigned long long values containing the indices of the removed elements
 *
 * @throws None
 */
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

/**
 * Frees the memory pointed to by the given pointer.
 *
 * @param ptr a pointer to the memory to be freed
 *
 * @throws None
 */
void freePtr(void* ptr)
{
    free(ptr);
}


int main(void)
{
    float arr[3] = {0., 2., 3.};
    unsigned long long len = 3;
    remove_simi(arr, &len, 0.1);
    return 0;
}
