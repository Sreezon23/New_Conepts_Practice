def get_length(arr):
    count = 0
    for _ in arr:
        count += 1
    return count

def get_max(arr):
    if not arr:
        return 0
    maximum = arr[0]
    for item in arr:
        if item > maximum:
            maximum = item
    return maximum

def counting_sort(arr):
    n = get_length(arr)
    if n == 0:
        return arr
        
    max_val = get_max(arr)
    
    count = []
    i = 0
    while i <= max_val:
        count.append(0)
        i += 1
    
    for num in arr:
        count[num] += 1
        
    i = 1
    count_len = get_length(count)
    while i < count_len:
        count[i] += count[i - 1]
        i += 1
        
    output = []
    i = 0
    while i < n:
        output.append(0)
        i += 1
        
    i = n - 1
    while i >= 0:
        num = arr[i]
        output[count[num] - 1] = num
        count[num] -= 1
        i -= 1
        
    return output

if __name__ == "__main__":
    data_counting = [4, 2, 2, 8, 3, 3, 1]
    print(counting_sort(data_counting))
