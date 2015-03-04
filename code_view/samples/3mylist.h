// file : 3mylist.h

#ifndef  __3MYLIST__H__INCLUDEED__
#define  __3MYLIST__H__INCLUDEED__

template <typename T>
class ListItem
{
public:
    ListItem(T value) { _value = value; _next = NULL; }
    void setNext(ListItem* p) { _next = p; }
    T value() const { return _value; }
    ListItem *next() const { return _next; }
private:
    T _value;
    ListItem *_next;
};

template <typename T>
class List
{
public:
    List(){ _front = NULL; _end = NULL; }
    void insert_front(T value)
    {
        ListItem<T>* temp = new ListItem<T>(value);
        if(_front == NULL)
            _front = temp;
        else
        {
            temp->setNext(_front);
            _front = temp;
        }
    }
    void insert_end(T value);
    void display(std::ostream &os = std::cout) const
    {
        ListItem<T>* temp = _front;
        while(temp != NULL)
        {
            os << temp->value();
            temp = temp->next();
        }
    }
private:
    ListItem<T> *_front;
    ListItem<T> *_end;
    long _size;
};

#endif // __3MYLIST__H__INCLUDEED__
