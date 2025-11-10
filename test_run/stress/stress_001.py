from concurrent.futures import ThreadPoolExecutor
import sys

def solve(case_number, n, array):
    max_sum = -10**9
    for start in range(n):
        current_sum = 0
        for end in range(start, n):
            current_sum += array[end]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum

def main():
    input = sys.stdin.read
    data = input().strip().split()
    
    index = 0
    num_cases = int(data[index])
    index += 1
    results = [None] * num_cases
    params = []
    
    for i in range(num_cases):
        n = int(data[index])
        index += 1
        array = list(map(int, data[index:index+n]))
        index += n
        params.append((i, n, array))
    
    def process_case(param):
        i, n, array = param
        result = solve(i, n, array)
        results[i] = result
    
    with ThreadPoolExecutor() as executor:
        executor.map(process_case, params)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()