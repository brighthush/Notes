#include <iostream>

#include "3mylist.h"
#include "3mylist-iter.h"

using namespace std;

int main()
{
    cout << "Hello world!" << endl;
    List<int> ml;
    ml.insert_front(1);
    ml.insert_front(2);
    ml.insert_front(3);
    ml.display();
    return 0;
}
