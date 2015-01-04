#ifndef __3MYLIST_ITER_H_INCLUDED__
#define __3MYLIST-ITER_H_INCLUDED__

#include "3mylist.h"

template <class Item>
struct ListIter
{
    Item* ptr;
    ListIter(Item* p=0):ptr(p){}

    Item& operator*() const { return *ptr; }
    Item* operator->() const { return ptr; }
    ListIter& operator++() { ptr=ptr->next; return *this; }
    bool operator==(const ListIter& i) const
    { return ptr==i.ptr; }
};


#endif // __3MYLIST-ITER_H_INCLUDED__
