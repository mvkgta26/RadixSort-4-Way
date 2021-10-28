# RadixSort-4-Way
## 4-Way Radix-Sort On CUDA Parallel Program


## CUDA Implementation of 4-Way Radix Sort for integers


## Core Algorithm: Array arr[] of size n:
### Repeat for i = 0, 2, 4, ...., 30 :
  1. *Compact* all the elements of arr[] array based on following *predicate*: Check if bit-i and bit-i+1 is 1
  2. Produce a *Blelloch Scan* correspondong to each of the 4 Compact arrays
  3. Use the scan arrays as *Scatter-Addresses* to do the *4 Way Split For Bit-i, i+1*:   
        a. All the elements with bit-i,i+1 == 00 are grouped in the beginning. Original order between them is retained  
        b. All elements with bit-i, i+1 == 01 are grouped next. Original order between them is retained
        c. All elements with bit-i, i+1 == 10 are grouped next. Original order between them is retained
        d. All elements with bit-i, i+1 == 10 are grouped in the end. Original order between them is retained
        
        
        
### References: 
  1. This paper by Ha, Krüger, and Silva gives the 4-Way Variant of Radix-Sort : https://vgc.poly.edu/~csilva/papers/cgf.pdf
  2. Udacity CS344: Intro to Parallel Programming (http://www.udacity.com/wiki/CS344)
  3. Bozidar, Darko & Dobravec, Tomaž. (2015). Comparison of parallel sorting algorithms. (https://www.researchgate.net/publication/283761857_Comparison_of_parallel_sorting_algorithms)
        


