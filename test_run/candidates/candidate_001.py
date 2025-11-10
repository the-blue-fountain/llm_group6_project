def max_subarray_sum(n, array):
    max_ending_here = max_so_far = array[0]
    for i in range(1, n):
        max_ending_here = max(array[i], max_ending_here + array[i])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

n = int(input().strip())
array = list(map(int, input().strip().split()))
print(max_subarray_sum(n, array))