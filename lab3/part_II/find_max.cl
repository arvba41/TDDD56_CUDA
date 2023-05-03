/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length, const unsigned int loopcount)
{ 
    // unsigned int pos = 0;
    // unsigned int val;

    // getting the index 
    unsigned int pos = get_global_id(0)*loopcount*2;

    // simple if statement to find the maximum of the array 
    if (pos <= length )
    {
        if (data[pos] <= data[pos+loopcount])
       {
            data[pos] = data[pos+loopcount];
       } 
    } 

    // data[get_global_id(0)] = data[get_global_id(0)];
    
}