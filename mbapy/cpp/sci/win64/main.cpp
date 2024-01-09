/*
 * @Date: 2023-11-20 13:03:48
 * @LastEditors: BHM-Bob 2262029386@qq.com
 * @LastEditTime: 2024-01-09 11:32:54
 * @Description: 
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C" {

    long long match_non_sp(char* firstChr, char* nonSpResult, char* fullText, long long matchLen);
    bool test_match_non_sp(void);
    void freePtr(void* ptr);
}

long long match_non_sp(char* firstChr, char* nonSpResult, char* fullText, long long matchLen)
{
    long long resultLen = strlen(nonSpResult), re_i = 1, ft_i = 1;
    for(char* pos = strchr(fullText, *firstChr); pos; pos = strchr(pos+1, *firstChr))
    {
        if (*pos == *firstChr)
        {
            for(re_i = 1, ft_i = 1; re_i < resultLen && ft_i < matchLen; ++re_i, ++ft_i)
            {
                while(pos[ft_i] == ' ' && ft_i < matchLen)
                    ++ft_i;
                if(pos[ft_i] != nonSpResult[re_i])
                {
                    re_i = -1; // set signal
                    break;
                }
            }
            if(re_i != -1)
                return (long long)(pos - fullText);
        }
    }
    return -1;
}

char* _strdup(const char* s) {
    size_t len = strlen(s) + 1; // 计算字符串长度，包括结尾的 null 字符
    char* new_s = (char*)malloc(len); // 分配内存
    if (new_s == NULL) {
        return NULL; // 内存分配失败
    }
    memcpy(new_s, s, len); // 复制字符串
    return new_s;
}

bool test_match_non_sp(void)
{
    char* first_chr = _strdup("f");
    char* non_sp_result = _strdup("fgh");
    char* full_text = _strdup("as dfg hj kl");
    long long pos = match_non_sp(first_chr, non_sp_result, full_text, 6);
    printf("%lld", pos);
    return (bool)(pos != -1);
}

int main(void)
{
    test_match_non_sp();
    return 0;
}