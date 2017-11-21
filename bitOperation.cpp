#include<vector>
#include<iostream>
#include<climits>
using namespace std;


//不用额外变量交换两个整数的值
void swap(int &a, int &b)
{
    a = a ^ b;
    b = a ^ b;
    a = a ^ b;
}


///只用位运算不用算数运算符实现整数的加减乘除
int add(int a, int b)
{
    int sum = a;
    while(b != 0)
    {
        sum = a ^ b;
        b = (a &b) << 1;
        a = sum;
    }
    return sum;
}

int minus1(int a, int b)
{
    return add(a, add(~b, 1));
}

int multi(int a, int b)
{
    int res = 0;
    while(b != 0)
    {
        if(b & 1 == 1)
            res = add(res, a);
        a <<= 1;
        b >>= 1;
    }
}

int div(int a, int b)
{
    a = a >= 0? a : add(~a, 1);
    b = b >= 0? b : add(~b, 1);
    int res = 0;
    for(int i=31; i>-1; i=minus1(i, 1))
    {
        if((a>>i) >= b)
        {
            res = res | (1<<i);
            a = minus1(a, b << i);
        }
    }
    return a >= 0 && b >= 0? add(~res, 1) : res;
}

int divide1(int a, int b)
{
    if(b == 0)
    {
        cout<<"Input Error"<<endl;
        return -1;
    }
    else if(a == INT_MIN && b == INT_MIN)
        return 1;
    else if(b == INT_MIN)
        return 0;
    else if(a == INT_MIN)
    {    
        int c = div(a+1, b);
        return add(c, div(minus1(a, multi(c, b)), b));
    }
    else
        return div(a, b);
}


//求整数的N次方
int power(int a,int p)
{
    int res = 1;
    while(p != 0)
    {
        if(p & 1 == 1)
        {
            res = res * a;
        }
        p >>= 1;
        a = a * a;
    }
    return res;
}

//不用任何比较判断找出两个数中较大的数
int flip(int n)
{
    return n ^ 1;
}

int sign(int n)
{
    return flip(n>>31 & 1);
}

int getMax1(int a, int b)
{
    int c = a - b;
    int scA = sign(c);
    int scB = flip(scA);
    return a * scA + b * scB;
}

int getMax2(int a, int b)
{
    int scA = sign(a);
    int scB = sign(b);
    int sc = sign(a - b);
    int difSab = scA ^ scB;
    int sameSab = flip(difSab);
    int returnA = scA * difSab + sc * sameSab;
    int returnB = flip(returnA);
    return a * returnA + b * returnB;
}



//整数的二进制表达中有多少个1
int count1(int n)
{
    unsigned int num = (unsigned)n;
    int res = 0;
    while(num != 0)
    {
        res += num & 1;
        num = num >> 1;
    }
    return res;
}

int count2(int n)
{
    int res = 0;
    while(n != 0)
    {
        n -= n & (~n + 1);
        res++;
    }
    return res;
}

int count3(int n)
{
    int res = 0;
    while(n != 0)
    {
        n = n & (n-1);
        res++;
    }
    return res;
}


//在其他数都出现偶数次的数组中找到出现奇数次的数
int printOddTimesNum1(vector<int> arr)
{
    int e = 0;
    for(int i=0; i<arr.size(); i++)
    {
        e = e ^ arr[i];
    }
    return e;
}

void printOddTimesNum2(vector<int> arr)
{
    int e1 = 0;
    int e2 = 0;
    for(int i=0; i<arr.size(); i++)
    {
        e1 ^= arr[i];
    }
    int rightOne = e1 & (~e1 + 1);
    for(int i=0; i<arr.size(); i++)
    {
        if((arr[i] & rightOne) != 0)
        {
            e2 ^= arr[i];
        }
    }
    cout<<e2<<" "<<(e2^e1)<<endl;
}


//在其他数都出现k次的数组中找到只出现一次的数
int getNumfromKSysNum(int e[],int k)
{
    int res = 0;
    for(int i=0; i<32; i++)
    {
        res = res * k + e[i];
    }
    return res;
}

int *getKSysNumfromNum(int value, int k)
{
    int *res = new int[32];
    int index = 31;
    while(value != 0)
    {
        res[index--] = value % k;
        value /= k;
    }
    return res;
}


void setExclusiveOr(int e[], int value, int k)
{
    int *val = new int[32];
    val = getKSysNumfromNum(value, k);
    for(int i=0; i<32; i++)
    {
        e[i] = (e[i] + val[i]) % k;
    }
}

int onceNum(vector<int> arr, int k)
{
    int e[32] = {0};
    for(int i=0; i<arr.size(); i++)
    {
        setExclusiveOr(e, arr[i], k);
    }
    int res = getNumfromKSysNum(e, k);
    return res;
}


int main(){
    int a = 4;
    int b = 0;
    int arr1[10] = {1,1,2,2,3,3,4,4,5,6};
    int arr3[13] = {1,1,1,2,2,2,3,3,3,7,1,2,3};
    vector<int> arr2(&arr1[0], &arr1[10]);
    vector<int> arr4(&arr3[0], &arr3[13]);
    for(int i=0; i<arr2.size(); i++)
    {
        cout<<arr2[i]<<" ";
    }
    cout<<endl;
    cout<<minus1(a, b<<2)<<endl;
    cout<<divide1(a, b)<<endl;
    cout<<power(2,3)<<endl;
    cout<<getMax1(9,3)<<endl;
    cout<<getMax2(9,3)<<endl;
    cout<<count3(-3)<<endl;
    cout<<printOddTimesNum1(arr2)<<endl;
    printOddTimesNum2(arr2);
    cout<<onceNum(arr4, 4)<<endl;
    return 0;
}
