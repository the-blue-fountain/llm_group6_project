def max_subarray_sum(n, arr):
    max_ending_here = max_so_far = arr[0]
    for i in range(1, n):
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

n = int(input())
arr = list(map(int, input().split()))
print(max_subarray_sum(n, arr))