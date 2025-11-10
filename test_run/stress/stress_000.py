import sys
import threading

def solve(n, arr):
    max_sum = -float('inf')
    for start in range(n):
        current_sum = 0
        for end in range(start, n):
            current_sum += arr[end]
            max_sum = max(max_sum, current_sum)
    return max_sum

def process_input(args):
    n = args[0]
    arr = args[1:]
    return solve(n, arr)

def main():
    input = sys.stdin.read
    data = input().split()
    index = 0
    num_cases = int(data[index])
    index += 1
    
    cases = []
    for _ in range(num_cases):
        n = int(data[index])
        index += 1
        arr = list(map(int, data[index:index + n]))
        index += n
        cases.append((n, arr))
    
    results = [0] * num_cases
    threads = []

    def worker(i, args):
        results[i] = process_input(args)

    for i, case in enumerate(cases):
        thread = threading.Thread(target=worker, args=(i, case))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for result in results:
        print(result)

if __name__ == "__main__":
    main()