#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

template <typename T>
class print
{
public:
    void operator()(T& num)
    {
        num += 1;
        cout << num << " ";
    }
};

int main()
{
    int a[] = {1, 2, 3};
    vector<int> va(a, a+3);
    for_each(va.begin(), va.end(), print<int>());
    return 0;
}

