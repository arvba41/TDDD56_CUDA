/*
 * Placeholder OpenCL kernel
 */
inline void bytt(__global unsigned int *data, unsigned int a, unsigned int b) 
{
  unsigned int temp = data[a];
  data[a] = data[b];
  data[b] = temp;
}

__kernel void bitonic(__global unsigned int *data, const unsigned int length, unsigned int ol, unsigned int il)
{ 
    unsigned int pos = get_global_id(0);
    int ixj=pos^il; // Calculate indexing!
    if ((ixj)>pos)
    {
        if ((pos&ol)==0 && data[pos]>data[ixj]) bytt(data,pos,ixj);
        if ((pos&ol)!=0 && data[pos]<data[ixj]) bytt(data,pos,ixj);
    }
//   data[get_global_id(0)]=get_global_id(0);
}
