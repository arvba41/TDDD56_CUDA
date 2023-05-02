/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length, const unsigned int ii)
{ 
    // unsigned int pos = 0;
    // unsigned int val;

    // getting the index 
    unsigned int pos = get_global_id(0)*ii*2;

    // simple if statement to find the maximum of the array 
    if (pos <= length )
    {
        if (data[pos] <= data[pos+ii])
       {
            data[pos] = data[pos+ii];
       } 
    } 

    // data[get_global_id(0)] = data[get_global_id(0)];
    
}